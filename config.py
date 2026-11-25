"""
config.py —— 全局配置文件
================================
作用：集中管理 API 密钥、模型名称等配置。
密钥从环境变量读取，避免硬编码泄露（.gitignore 时也安全）。

使用方法（二选一）：
  方式1（推荐，临时设置）:
    PowerShell:  $env:LLM_API_KEY="sk-xxxx"
  方式2（长期）:
    在项目根目录新建 .env 文件，写入:
      LLM_API_KEY=sk-xxxx
      LLM_BASE_URL=https://api.deepseek.com
"""

import os

# ---------------------------------------------------------------
# 一、LLM API 配置
# ---------------------------------------------------------------
def _clean(value: str) -> str:
    """清理粘贴配置时常见的空格、换行和包裹引号。
    Secrets/.env 里多一个空格或引号，服务商就会判定密钥无效（401）。"""
    return (value or "").strip().strip('"').strip("'").strip()


# API 密钥：优先从环境变量读取，读不到再从 .env 文件读取
API_KEY = _clean(os.environ.get("LLM_API_KEY", ""))

# API 地址（兼容 OpenAI 格式的服务都可用）。
# 留空 = 按密钥自动识别厂商，无需手填：
#   DeepSeek: https://api.deepseek.com
#   智谱GLM:  https://open.bigmodel.cn/api/paas/v4
BASE_URL = _clean(os.environ.get("LLM_BASE_URL", ""))

# 默认模型名。留空 = 按服务商自动选（DeepSeek=deepseek-chat, GLM=glm-4-flash）
MODEL_NAME = _clean(os.environ.get("LLM_MODEL", ""))

# 备用服务商（可选）：主服务商拥塞（429/超时/5xx）时自动切换。
# 留空 = 单端点模式，行为与原来完全一致。
API_KEY_2 = _clean(os.environ.get("LLM_API_KEY_2", ""))
BASE_URL_2 = _clean(os.environ.get("LLM_BASE_URL_2", ""))
MODEL_NAME_2 = _clean(os.environ.get("LLM_MODEL_2", ""))

# 异构核验端点（可选，K1·B2优化：核验与生成不同源）：
# 声明核验(reviewer.semantic_verify)如果也用生成侧的同一个模型，
# 存在自我一致性偏差——模型看自己风格的话更容易放行。
# 配置 VERIFIER_API_KEY 后核验走独立厂商（不同预训练语料→错误不相关）。
# 留空 = 核验回落主LLM端点，行为与原来完全一致。
VERIFIER_API_KEY = _clean(os.environ.get("VERIFIER_API_KEY", ""))
VERIFIER_BASE_URL = _clean(os.environ.get("VERIFIER_BASE_URL", ""))
VERIFIER_MODEL = _clean(os.environ.get("VERIFIER_MODEL", ""))

# K1可选扩展——多模型投票核验（三投票）：
# ENSEMBLE_VERIFY=1 且配置 VERIFIER2_API_KEY（推荐再加 VERIFIER3_*）
# 后，语义核验由多个独立厂商模型各判一次：严格多数一致→采纳该判定；
# 无多数（各不相同或两票分歧）→取最严判定并在reason标注"待人工复核"。
# 默认0 = 单核验端点（与K1基础版行为一致，不增加调用成本）。
ENSEMBLE_VERIFY = os.environ.get("ENSEMBLE_VERIFY", "0") == "1"
VERIFIER2_API_KEY = _clean(os.environ.get("VERIFIER2_API_KEY", ""))
VERIFIER2_BASE_URL = _clean(os.environ.get("VERIFIER2_BASE_URL", ""))
VERIFIER2_MODEL = _clean(os.environ.get("VERIFIER2_MODEL", ""))
VERIFIER3_API_KEY = _clean(os.environ.get("VERIFIER3_API_KEY", ""))
VERIFIER3_BASE_URL = _clean(os.environ.get("VERIFIER3_BASE_URL", ""))
VERIFIER3_MODEL = _clean(os.environ.get("VERIFIER3_MODEL", ""))

# 各服务商的默认参数（自动识别时使用）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
GLM_DEFAULT_MODEL = "glm-4-flash"


def _looks_like_glm_key(key: str) -> bool:
    """智谱GLM密钥形如 '32位id.16位secret'（含一个点，不以sk-开头）；
    DeepSeek密钥则固定以 sk- 开头。"""
    return bool(key) and not key.startswith("sk-") and "." in key


# ---------------------------------------------------------------
# 一之二、Semantic Scholar API 密钥（检索Agent用）
# ---------------------------------------------------------------
# 免费申请: https://www.semanticscholar.org/product/api#form
# 有密钥后限流额度大幅提升（1次/秒 独享），匿名模式经常429卡死
# 留空 = 匿名模式（能用但不稳定）
S2_API_KEY = os.environ.get("S2_API_KEY", "")

# ---------------------------------------------------------------
# 一之三、云端部署访问密码（Streamlit Cloud / HF Spaces）
# ---------------------------------------------------------------
# 部署到公网时设一个密码，陌生人必须先输密码才能使用。
# 本地开发留空或不设置 → 密码门不启用。
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# ---------------------------------------------------------------
# 一之四、LLM结果缓存（2026-09新增，流水线提速）
# ---------------------------------------------------------------
# 提取/审查/规划/综述等LLM调用的返回值按"消息指纹"缓存到SQLite，
# 同样的输入30天内直接复用本地结果——重复主题的重跑从分钟级降到秒级。
# 置0关闭（会彻底重新调用LLM，用于想要全新输出的场景）
LLM_CACHE_ENABLED = os.environ.get("LLM_CACHE", "1") == "1"

# 版面感知PDF解析（K3·A3优化，2026-11新增）：
# 1 = 按文本块坐标识别双栏并按阅读顺序重组（默认开，根治双栏论文
#     左右栏交错导致的"声明与原文对不上→误判幻觉"）；
# 0 = 回退旧的纯文本流抽取（灰度开关，解析结果异常时可快速回退）
LAYOUT_AWARE_PDF = os.environ.get("LAYOUT_AWARE_PDF", "1") == "1"

# K3·A3公式清理: 1 = 版面重组时剔除"高符号密度且无正文虚词"的
#     公式/乱码块（保守三重条件，宁漏勿误伤正文）；0 = 保留全部块
FORMULA_FILTER = os.environ.get("FORMULA_FILTER", "1") == "1"

# ---------------------------------------------------------------
# 二、各Agent的温度参数（temperature）
# ---------------------------------------------------------------
# 温度越低输出越稳定。提取/审查任务要求"忠实原文"，用 0.1；
# 规划任务需要一点发散性，用 0.5。
TEMP_PLANNER = 0.5     # 规划Agent：任务分解需要发散
TEMP_EXTRACTOR = 0.1   # 提取Agent：信息抽取要求严格忠实原文
TEMP_REVIEWER = 0.1    # 审查Agent：判定结果要求稳定可复现
TEMP_SYNTHESIZER = 0.3 # 综合Agent：写综述允许少量文采

# ---------------------------------------------------------------
# 二之四、超时 / 重试上限 / 全局时间预算（防流水线无限卡死）
# ---------------------------------------------------------------
# 单次LLM调用超时（秒）。OpenAI SDK默认600秒，网络挂起时单次调用
# 就能卡10分钟，叠加chat重试3次+chat_json解析重试3轮，最坏卡90分钟
# ——这是"程序不结束"的主要根因。120秒覆盖正常生成（长综述走流式）。
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))

# LLM网络失败的重试次数上限（2/4/8秒指数退避，总计最多约4分钟）
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

# 端点冷却时间（秒）：某端点被判定为拥塞后，这段时间内优先走另一个端点，
# 避免每次调用都先去撞一次已限流的主端点（多花一次 429 的等待）
LLM_ENDPOINT_COOLDOWN = float(os.environ.get("LLM_ENDPOINT_COOLDOWN", "90"))

# 全局时间预算（分钟）：流水线从开始计时的硬上限，超预算立即中止
# 并在网页标注失败阶段。40分钟远大于正常运行的10-15分钟，
# 只拦截"限流风暴+多重试叠加"导致的病态长跑。
GLOBAL_BUDGET_MIN = float(os.environ.get("GLOBAL_BUDGET_MIN", "40"))

# ---------------------------------------------------------------
# 三、项目路径
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.join(PROJECT_ROOT, "papers")  # 下载的PDF存放处
DATA_DIR = os.path.join(PROJECT_ROOT, "data")     # 证据库JSON存放处
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output") # 综合Agent的产出目录(表格/图谱/报告)

# 首次运行时目录可能不存在（papers/data/output 被 .gitignore 排除，
# 云端克隆下来只有代码）。app.py 会直接往 DATA_DIR 写状态文件和日志，
# 不先建目录会抛 FileNotFoundError。导入 config 时统一建好。
for _d in (PAPER_DIR, DATA_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------
# 四、如果 .env 文件存在，则加载其中未被环境变量覆盖的配置
# ---------------------------------------------------------------
# 注意: 必须逐项补充而非"LLM密钥存在就跳过整个.env"——否则环境变量
# 里设了LLM_API_KEY时，.env中的S2_API_KEY等其余配置将永远读不到
# （2026-09-09实测踩坑: S2匿名模式持续429，密钥形同虚设）
_env_file = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_file):
    # utf-8-sig 自动剥离BOM——部分编辑器保存.env会带BOM，
    # 导致首行键名变成"\ufeffLLM_API_KEY"而永远匹配失败
    with open(_env_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if key == "LLM_API_KEY":
                    API_KEY = API_KEY or _clean(value)
                elif key == "LLM_BASE_URL":
                    # 环境变量显式设置 > .env > 内置默认
                    if "LLM_BASE_URL" not in os.environ:
                        BASE_URL = _clean(value)
                elif key == "LLM_MODEL":
                    if "LLM_MODEL" not in os.environ:
                        MODEL_NAME = _clean(value)
                elif key == "LLM_API_KEY_2":
                    API_KEY_2 = API_KEY_2 or _clean(value)
                elif key == "LLM_BASE_URL_2":
                    if "LLM_BASE_URL_2" not in os.environ:
                        BASE_URL_2 = _clean(value)
                elif key == "LLM_MODEL_2":
                    if "LLM_MODEL_2" not in os.environ:
                        MODEL_NAME_2 = _clean(value)
                elif key == "S2_API_KEY":
                    S2_API_KEY = S2_API_KEY or value
                elif key == "APP_PASSWORD":
                    APP_PASSWORD = APP_PASSWORD or value

# ---------------------------------------------------------------
# 五、按密钥自动识别服务商（.env 加载完之后再判断）
# ---------------------------------------------------------------
# 场景: 用户只换了密钥（例如换成智谱GLM），但 .env / Secrets 里的
# LLM_BASE_URL 和 LLM_MODEL 还是 DeepSeek 的默认值 → 请求必然401。
# 这里以密钥形态为准，自动纠正地址与模型名；显式配置的自定义代理
# 地址（不含 deepseek 字样）不会被覆盖。


def _resolve_provider(key: str, base: str, model: str):
    """按密钥形态纠正地址与模型名，并补默认值。"""
    if _looks_like_glm_key(key):
        if not base or "deepseek" in base.lower():
            base = GLM_BASE_URL
        if not model or model.startswith("deepseek"):
            model = GLM_DEFAULT_MODEL
    if not base:
        base = DEEPSEEK_BASE_URL
    if not model:
        model = DEEPSEEK_DEFAULT_MODEL
    return base, model


BASE_URL, MODEL_NAME = _resolve_provider(API_KEY, BASE_URL, MODEL_NAME)
BASE_URL_2, MODEL_NAME_2 = _resolve_provider(API_KEY_2, BASE_URL_2, MODEL_NAME_2)

# ---------------------------------------------------------------
# 六、可用端点列表（主端点在前，备用端点在后）
# ---------------------------------------------------------------
# llm_client 按此顺序调用：主端点出错即切换下一个，冷却期内跳过拥塞端点。
# 未配置备用密钥时只有一项，行为与原来的单端点模式一致。
LLM_APIS = [{
    "name": "主",
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "model": MODEL_NAME,
}]
if API_KEY_2:
    LLM_APIS.append({
        "name": "备",
        "api_key": API_KEY_2,
        "base_url": BASE_URL_2,
        "model": MODEL_NAME_2,
    })

# 核验端点列表（与LLM_APIS同构）：未配置VERIFIER_API_KEY时直接复用
# 主端点列表（同源回落，行为不变）；配置后核验与生成彻底异构
VERIFIER_APIS = LLM_APIS
if VERIFIER_API_KEY:
    VERIFIER_APIS = [{
        "name": "核验",
        "api_key": VERIFIER_API_KEY,
        "base_url": VERIFIER_BASE_URL,
        "model": VERIFIER_MODEL,
    }]

# 多模型投票核验的端点组（K1可选扩展）：[核验1, 核验2(, 核验3)]
# 空列表 = 未启用投票，semantic_verify 回落单核验端点
VERIFIER_ENSEMBLE: list = []
if ENSEMBLE_VERIFY and VERIFIER2_API_KEY:
    VERIFIER_ENSEMBLE = [VERIFIER_APIS, [{
        "name": "核验2",
        "api_key": VERIFIER2_API_KEY,
        "base_url": VERIFIER2_BASE_URL,
        "model": VERIFIER2_MODEL,
    }]]
    if VERIFIER3_API_KEY:
        VERIFIER_ENSEMBLE.append([{
            "name": "核验3",
            "api_key": VERIFIER3_API_KEY,
            "base_url": VERIFIER3_BASE_URL,
            "model": VERIFIER3_MODEL,
        }])
elif ENSEMBLE_VERIFY:
    print("[config] ENSEMBLE_VERIFY=1 但未配置VERIFIER2_API_KEY，"
          "投票核验未启用（回落单核验端点）")

