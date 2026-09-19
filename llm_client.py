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
from openai import OpenAI

import budget
import config

# ---------------------------------------------------------------
# 创建全局客户端（整个项目共用一个，避免重复创建）
# ---------------------------------------------------------------
# timeout: 单次调用的硬超时（SDK默认600秒，网络挂起时一次就卡10分钟，
#          这是"程序不结束"的主要根因，必须显式收紧）
# max_retries=0: 关闭SDK内部重试，由下方chat()统一控制重试次数与退避，
#          否则SDK重试×我们的重试相乘，失败场景耗时不可控
client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL,
                timeout=config.LLM_TIMEOUT, max_retries=0)

# 简易的用量统计器（用于结题报告的成本分析实验）
usage_stats = {
    "total_calls": 0,      # 总调用次数
    "total_prompt_tokens": 0,   # 总输入token
    "total_completion_tokens": 0,  # 总输出token
}


def chat(messages, temperature=0.2, model=None, max_retries=None):
    """
    基础对话函数

    参数:
        messages: OpenAI格式的消息列表
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature: 温度参数，控制随机性（提取任务用0.1，规划用0.5）
        model: 模型名，默认用 config.MODEL_NAME
        max_retries: 网络失败重试次数（默认取config.LLM_MAX_RETRIES）

    返回:
        模型的纯文本回答 (str)
    """
    model = model or config.MODEL_NAME
    max_retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries

    for attempt in range(max_retries):
        # 全局预算检查: 超预算立即中止（异常在try外抛出，不会被下面的
        # 重试逻辑吞掉），前端据此标注失败阶段
        budget.check("LLM调用")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            # ---- 记录用量（成本分析实验的数据）----
            usage_stats["total_calls"] += 1
            usage_stats["total_prompt_tokens"] += response.usage.prompt_tokens
            usage_stats["total_completion_tokens"] += response.usage.completion_tokens

            return response.choices[0].message.content

        except budget.BudgetExceededError:
            raise  # 预算耗尽不重试，直接上抛
        except Exception as e:
            # 认证/鉴权类错误重试无意义（换多少次都是401），立即上抛
            # 避免白等2+4+8秒并掩盖真实问题
            if "401" in str(e) or "403" in str(e) or "api key" in str(e).lower():
                raise
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
    model = model or config.MODEL_NAME

    for attempt in range(max_retries):
        budget.check("LLM流式调用")  # 超预算立即中止，不开始新的长生成
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return  # 正常结束
        except budget.BudgetExceededError:
            raise  # 预算耗尽不重试
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"[llm_client] 流式调用失败(第{attempt + 1}次): {e}，{wait}秒后重试...")
            time.sleep(wait)

    raise RuntimeError(f"LLM流式调用连续失败{max_retries}次，请检查网络和API密钥")


def chat_json(messages, schema_class, temperature=0.1, model=None):
    """
    要求模型输出 JSON 并解析为 Pydantic 对象

    参数:
        messages: 消息列表
        schema_class: Pydantic 模型类（如 models.SearchPlan）
        temperature: 默认0.1，保证结构化输出稳定

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

    # 最多尝试3次解析（第一次解析失败就把错误信息喂回去让模型自己修复）
    for attempt in range(3):
        raw = chat(messages, temperature=temperature, model=model)
        try:
            # 剥离可能的 ```json ``` 包裹
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text
                text = text.rsplit("```", 1)[0]
            return schema_class.model_validate_json(text)
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
    print(f"  调用次数:       {usage_stats['total_calls']}")
    print(f"  输入token总量:  {usage_stats['total_prompt_tokens']}")
    print(f"  输出token总量:  {usage_stats['total_completion_tokens']}")
    print("=" * 50)
