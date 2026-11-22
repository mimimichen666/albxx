"""
llm_client.py —— 统一的 LLM 调用封装
================================
作用：所有 Agent 都通过本模块调用大模型，好处是：
  1. 换模型/换服务商只改 config.py 一处
  2. 统一记录 token 用量和调用日志（写报告做成本分析 E4 实验的数据来源）
  3. 统一的重试逻辑（网络波动自动重试3次）

核心函数：
  chat(messages, temperature)      -> 返回纯文本回答
  chat_json(messages, schema_class)-> 返回按 Pydantic 结构解析的 JSON 对象
"""

import json
import time
import hashlib
from openai import OpenAI

import budget
import config
import cache as _cache

# ---------------------------------------------------------------
# 多端点客户端（主 + 备用），按 base_url 缓存，避免重复创建
# ---------------------------------------------------------------
# timeout: 单次调用的硬超时（SDK默认600秒，网络挂起时一次就卡10分钟，
#          这是"程序不结束"的主要根因，必须显式收紧）
# max_retries=0: 关闭SDK内部重试，由下方chat()统一控制重试次数与退避，
#          否则SDK重试×我们的重试相乘，失败场景耗时不可控
_clients = {}


def _ep_key(ep):
    """端点唯一键：同一厂商的两把密钥共用 base_url，
    必须带上密钥才能区分，否则第二个端点会复用第一个的客户端。"""
    return (ep["base_url"], ep["api_key"])


def _endpoint_client(ep):
    """按端点创建（并缓存）OpenAI 客户端。"""
    key = _ep_key(ep)
    if key not in _clients:
        _clients[key] = OpenAI(
            api_key=ep["api_key"], base_url=ep["base_url"],
            timeout=config.LLM_TIMEOUT, max_retries=0,
        )
    return _clients[key]


# 端点冷却表：端点键 -> 恢复可用的时间戳
_down_until = {}

# 判定为"拥塞"的错误特征：限流、服务端 5xx、超时、连接中断
_CONGEST_HINTS = ("429", "500", "502", "503", "504", "rate limit",
                  "too many requests", "overloaded", "timeout",
                  "timed out", "connection")
# 认证类错误：重试同一端点无意义，直接换下一个端点
_AUTH_HINTS = ("401", "403", "api key", "unauthorized", "invalid_api_key")


def _is_hint(err, hints) -> bool:
    s = str(err).lower()
    return any(h in s for h in hints)


def _available_endpoints(apis=None):
    """返回当前可用端点（冷却中的排在后面；全部冷却时仍返回全部，
    避免因为冷却表把系统卡死在"无端点可用"）。

    apis: 端点列表，None = config.LLM_APIS（主生成端点）。
    核验侧(reviewer)传 config.VERIFIER_APIS 可走独立厂商端点（K1异构核验）。"""
    if apis is None:
        apis = config.LLM_APIS
    now = time.time()
    fresh, cooled = [], []
    for e in apis:
        (cooled if _down_until.get(_ep_key(e), 0) > now else fresh).append(e)
    return fresh + cooled


def _mark_down(ep):
    _down_until[_ep_key(ep)] = time.time() + config.LLM_ENDPOINT_COOLDOWN


def _mark_ok(ep):
    _down_until.pop(_ep_key(ep), None)


# 简易的用量统计器（用于结题报告的成本分析实验）
usage_stats = {
    "total_calls": 0,      # 总调用次数（真实API调用）
    "cache_hits": 0,       # 缓存命中次数（省下的调用）
    "total_prompt_tokens": 0,   # 总输入token
    "total_completion_tokens": 0,  # 总输出token
    "failovers": 0,        # 端点切换次数（主→备）
}


def _fingerprint(model: str, temperature: float, messages: list) -> str:
    """
    生成一次LLM调用的"消息指纹"（缓存键）

    指纹 = 模型名 + 温度 + 全部消息内容 的SHA1。
    同一篇论文的提取、同一条声明的审查，输入完全一致 → 指纹一致
    → 直接复用上次输出。消息里含论文全文（10万字符），但SHA1哈希
    300KB文本只需几毫秒，开销可忽略。
    """
    raw = json.dumps(
        {"m": model, "t": temperature, "msgs": messages},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def chat(messages, temperature=0.2, model=None, max_retries=None,
         use_cache=True, apis=None):
    """
    基础对话函数

    参数:
        messages: OpenAI格式的消息列表
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: 温度参数，控制随机性（提取任务用0.1，规划用0.5）
        model: 模型名，默认用 config.MODEL_NAME
        max_retries: 网络失败重试次数（默认取config.LLM_MAX_RETRIES）
        use_cache: 是否启用LLM结果缓存（默认开启）。缓存命中时
                   同样的输入直接返回上次输出，零耗时零花费。
                   chat_json内部调用本函数时会关掉它（结构化缓存
                   在chat_json层做，避免把解析失败的原始输出存进缓存）
        apis: 端点列表（None=config.LLM_APIS）。核验侧传
              config.VERIFIER_APIS 走独立厂商端点（K1异构核验）

    返回:
        模型的纯文本回答 (str)
    """
    requested_model = model  # None = 跟随端点的模型名（切换端点时才不会串模型）
    model = model or config.MODEL_NAME
    max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries

    # ---- 缓存查询（命中则完全跳过API调用）----
    # 指纹带上端点组标识：同一条消息在"主端点"和"核验端点"是两次
    # 不同的调用（模型/厂商可能不同），缓存不能互相串
    cache_key = None
    if config.LLM_CACHE_ENABLED and use_cache:
        # 后缀取端点名: 主端点无后缀, 各核验端点按name隔离
        # （多模型投票时"核验/核验2/核验3"互不串缓存）
        suffix = ("@" + str(apis[0].get("name", "verifier"))
                  if apis else "")
        cache_key = _fingerprint(model + suffix, temperature, messages)
        hit = _cache.get_llm(cache_key)
        if hit is not None:
            usage_stats["cache_hits"] += 1
            print(f"[LLM缓存] 命中: {cache_key[:12]} (跳过API调用, "
                  f"已命中{usage_stats['cache_hits']}次)")
            return hit

    endpoints = _available_endpoints(apis)
    for attempt in range(max_retries):
        ep = endpoints[attempt % len(endpoints)]
        # 全局预算检查: 超预算立即中止（异常在try外抛出，不会被下面的
        # 重试逻辑吞掉），前端据此标注失败阶段
        budget.check("LLM调用")
        try:
            response = _endpoint_client(ep).chat.completions.create(
                model=requested_model or ep["model"],
                messages=messages,
                temperature=temperature,
            )
            _mark_ok(ep)
            # ---- 记录用量（成本分析实验的数据）----
            usage_stats["total_calls"] += 1
            usage_stats["total_prompt_tokens"] += response.usage.prompt_tokens
            usage_stats["total_completion_tokens"] += response.usage.completion_tokens

            content = response.choices[0].message.content

            # ---- 成功响应写入缓存（失败/重试的残次输出不入库）----
            if config.LLM_CACHE_ENABLED and use_cache and content:
                try:
                    _cache.put_llm(cache_key, content)
                except Exception as e:
                    print(f"[LLM缓存] 写入失败(忽略): {e}")

            return content

        except budget.BudgetExceededError:
            raise  # 预算耗尽不重试，直接上抛
        except Exception as e:
            multi = len(endpoints) > 1
            # 认证/鉴权类错误：同一端点重试无意义（换多少次都是401）。
            # 有备用端点时直接换过去，否则立即上抛，避免白等 2+4+8 秒
            if _is_hint(e, _AUTH_HINTS):
                if multi:
                    print(f"[llm_client] {ep['name']}端点鉴权失败: {e}，切换备用端点")
                    usage_stats["failovers"] += 1
                    continue
                raise
            # 拥塞（限流/5xx/超时/连接中断）：标记冷却，切换到另一端点
            if multi and _is_hint(e, _CONGEST_HINTS):
                _mark_down(ep)
                usage_stats["failovers"] += 1
                print(f"[llm_client] {ep['name']}端点拥塞({attempt + 1}/{max_retries}): "
                      f"{e}，冷却{int(config.LLM_ENDPOINT_COOLDOWN)}秒并切换备用端点")
                time.sleep(1)
                continue
            # 重试策略：等待时间指数增长（2秒、4秒、8秒）
            wait = 2 ** (attempt + 1)
            print(f"[llm_client] 调用失败(第{attempt + 1}次): {e}，{wait}秒后重试...")
            time.sleep(wait)

    raise RuntimeError(f"LLM调用连续失败{max_retries}次，请检查网络和API密钥")


def chat_stream(messages, temperature=0.2, model=None, max_retries=2):
    """
    流式对话函数（长文本生成场景专用）

    与chat()的区别: 边生成边返回文本片段（yield），前端可逐字显示进度。
    适用场景: 小白解读、综述正文等输出几千字的任务——一次性生成要等
    2-4分钟且界面无任何反馈，流式输出让用户看到"正在写"。

    参数同chat()。
    返回:
        生成器，逐块yield文本片段(str)
    """
    requested_model = model  # None = 跟随端点的模型名
    model = model or config.MODEL_NAME
    endpoints = _available_endpoints()

    for attempt in range(max_retries):
        ep = endpoints[attempt % len(endpoints)]
        produced = False  # 已经吐字就不能换端点重来，否则内容会重复
        budget.check("LLM流式调用")  # 超预算立即中止，不开始新的长生成
        try:
            stream = _endpoint_client(ep).chat.completions.create(
                model=requested_model or ep["model"],
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    produced = True
                    yield chunk.choices[0].delta.content
            _mark_ok(ep)
            return  # 正常结束
        except budget.BudgetExceededError:
            raise  # 预算耗尽不重试
        except Exception as e:
            if produced:
                raise RuntimeError(
                    f"流式输出中断（已输出部分内容，不能切换端点重来）: {e}")
            multi = len(endpoints) > 1
            if _is_hint(e, _AUTH_HINTS):
                if multi:
                    print(f"[llm_client] {ep['name']}端点鉴权失败: {e}，切换备用端点")
                    usage_stats["failovers"] += 1
                    continue
                raise
            if multi and _is_hint(e, _CONGEST_HINTS):
                _mark_down(ep)
                usage_stats["failovers"] += 1
                print(f"[llm_client] {ep['name']}端点拥塞(流式)，切换备用端点")
                time.sleep(1)
                continue
            wait = 2 ** (attempt + 1)
            print(f"[llm_client] 流式调用失败(第{attempt + 1}次): {e}，{wait}秒后重试...")
            time.sleep(wait)

    raise RuntimeError(f"LLM流式调用连续失败{max_retries}次，请检查网络和API密钥")


def chat_json(messages, schema_class, temperature=0.1, model=None,
              use_cache=True, max_retries=None, parse_rounds=3, apis=None):
    """
    要求模型输出 JSON 并解析为 Pydantic 对象

    参数:
        messages: 消息列表
        schema_class: Pydantic 模型类（如 models.SearchPlan）
        temperature: 默认0.1，保证结构化输出稳定
        use_cache: 是否启用LLM结果缓存（默认开启）。缓存粒度是
                   "最终校验通过的输出"——同一篇论文第二次提取、
                   同一条声明第二次审查时直接秒回，不花一分钱。
                   注意缓存键用"修复重试前"的原始消息计算，因此
                   即使上次经历了JSON修复重试，重跑也只需一次查询
        apis: 端点列表（None=config.LLM_APIS）。核验侧传
              config.VERIFIER_APIS 走独立厂商端点（K1异构核验），
              且缓存键带端点名后缀（如"@核验"）与主端点及
              其他核验端点的结果互相隔离
        max_retries: 透传给 chat() 的网络重试上限。None=用全局默认
                     (config.LLM_MAX_RETRIES)。交互式端点(如词云)应传
                     更小值——默认配置下最坏耗时 =
                     parse_rounds × max_retries × LLM_TIMEOUT(120s) +
                     指数退避，全默认约19分钟，交互场景必须收紧
        parse_rounds: JSON解析失败后的"喂回错误让模型自修"轮数上限。
                      默认3；交互式端点建议2

    返回:
        schema_class 的实例对象

    说明:
        不用 instructor 库，手动实现"强制JSON + 校验 + 修复重试"，
        逻辑透明，方便在报告中解释（也是个小教学点）。
    """
    # 在 system 消息中注入 JSON 格式要求
    # 注意：不能直接贴完整JSON Schema（模型会误把Schema本身当答案返回），
    # 改为生成"字段说明清单 + 嵌套示例骨架"，明确要求填入具体值
    full_schema = schema_class.model_json_schema()
    properties = full_schema.get("properties", {})
    defs = full_schema.get("$defs", {})  # Pydantic把嵌套模型定义放在这里

    def _resolve(info: dict) -> dict:
        """解析 $ref 引用（Pydantic v2 嵌套模型的存储方式）"""
        if "$ref" in info:
            ref_name = info["$ref"].split("/")[-1]
            return defs.get(ref_name, info)
        return info

    def _skeleton(info: dict) -> object:
        """递归生成嵌套结构的示例骨架（对象→子字段骨架，数组→[单个元素骨架]）"""
        info = _resolve(info)
        if "anyOf" in info:  # 处理 Optional字段（如 str | None）
            return _skeleton(info["anyOf"][0])
        t = info.get("type")
        if t == "array":
            item = info.get("items", {})
            return [_skeleton(item)]  # 数组给一个元素示例
        if "properties" in info:
            sub = info.get("properties", {})
            return {name: _skeleton(child) for name, child in sub.items()}
        return "..."

    def _field_help(info: dict, prefix: str = "") -> list[str]:
        """递归生成字段说明清单（含嵌套子字段的说明）"""
        info = _resolve(info)
        lines = []
        if "anyOf" in info:
            return _field_help(info["anyOf"][0], prefix)
        t = info.get("type")
        desc = info.get("description", "")
        if t == "array":
            item = info.get("items", {})
            lines.append(f'  - "{prefix}"(数组): {desc}')
            lines.extend(_field_help(item, prefix + "[]"))
        elif "properties" in info:
            sub = info.get("properties", {})
            lines.append(f'  - "{prefix}"(对象): {desc}')
            for name, child in sub.items():
                lines.extend(_field_help(child, f"{prefix}.{name}"))
        else:
            lines.append(f'  - "{prefix}"({t}): {desc}')
        return lines

    field_list = "\n".join(
        "\n".join(_field_help(info, name)).lstrip()
        for name, info in properties.items()
    )
    example = json.dumps(
        {name: _skeleton(info) for name, info in properties.items()},
        ensure_ascii=False, indent=1,
    )
    json_instruction = (
        "\n\n【输出格式要求】\n"
        "你必须输出一个合法的JSON对象作为答案，字段结构如下：\n"
        f"{field_list}\n"
        f'必须严格遵循此嵌套结构（"..."处填入真实内容，数组按需放多个元素）:\n'
        f"{example}\n"
        "注意：所有字段名必须与上面完全一致；字段值必须是具体内容"
        "（如真实的结论、原文引用），不要输出字段定义或JSON Schema，"
        "不要用markdown代码块包裹，不要输出JSON以外的任何文字。"
    )
    messages = list(messages)  # 复制一份，不修改调用方的原始列表
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + json_instruction,
        }
    else:
        messages.insert(0, {"role": "system", "content": json_instruction})

    # ---- 结构化缓存查询（在注入格式要求之后计算指纹）----
    # 键 = schema名 + 温度 + 模型 + 注入格式要求后的完整消息
    cache_key = None
    if config.LLM_CACHE_ENABLED and use_cache:
        # 后缀取端点名（多核验端点互不串缓存，主端点无后缀）
        suffix = ("@" + str(apis[0].get("name", "verifier"))
                  if apis else "")
        cache_key = _fingerprint(
            (model or config.MODEL_NAME) + suffix,
            temperature, messages)
        hit = _cache.get_llm(cache_key)
        if hit is not None:
            try:
                obj = schema_class.model_validate_json(hit)
                usage_stats["cache_hits"] += 1
                print(f"[LLM缓存] 命中: {schema_class.__name__} "
                      f"(跳过API调用, 已命中{usage_stats['cache_hits']}次)")
                return obj
            except Exception:
                # 缓存的输出居然校验失败（理论不该发生）：
                # 删除脏数据，当作未命中继续走正常流程
                print("[LLM缓存] 缓存内容校验失败, 删除脏数据后重新调用")
                try:
                    import sqlite3
                    conn = _cache._init()
                    with _cache._LOCK:
                        conn.execute("DELETE FROM llm_cache WHERE key=?",
                                     (cache_key,))
                        conn.commit()
                except Exception:
                    pass

    # 最多尝试parse_rounds次解析（第一次解析失败就把错误信息喂回去让模型自己修复）
    for attempt in range(parse_rounds):
        # use_cache=False: 结构化调用的缓存在本函数这一层做，
        # 避免把"解析失败的原始输出"存进chat()层的缓存污染重跑
        raw = chat(messages, temperature=temperature, model=model,
                   max_retries=max_retries, use_cache=False, apis=apis)
        try:
            # 剥离可能的 ```json ``` 包裹
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            obj = schema_class.model_validate_json(text)

            # ---- 校验通过才写缓存（残次输出永远不入库）----
            if config.LLM_CACHE_ENABLED and use_cache and cache_key:
                try:
                    _cache.put_llm(cache_key, text)
                except Exception as e:
                    print(f"[LLM缓存] 写入失败(忽略): {e}")

            return obj
        except Exception as e:
            print(f"[llm_client] JSON解析失败(第{attempt + 1}次): {e}")
            # 把解析错误反馈给模型，让它修复
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"你上次的输出不是合法JSON（错误：{e}）。请重新输出，只输出JSON本身。",
            })

    raise ValueError("模型连续3次未能输出合法JSON，请检查Prompt设计")


def print_usage():
    """打印当前累计的token用量（成本分析实验用）"""
    print("=" * 50)
    print("📊 LLM 用量统计")
    print(f"  真实API调用:    {usage_stats['total_calls']} 次")
    print(f"  缓存命中省下:   {usage_stats['cache_hits']} 次")
    print(f"  端点自动切换:   {usage_stats['failovers']} 次")
    print(f"  输入token总量:  {usage_stats['total_prompt_tokens']}")
    print(f"  输出token总量:  {usage_stats['total_completion_tokens']}")
    print("=" * 50)
