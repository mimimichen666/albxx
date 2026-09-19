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
# API 密钥：优先从环境变量读取，读不到再从 .env 文件读取
API_KEY = os.environ.get("LLM_API_KEY", "")

# API 地址（兼容 OpenAI 格式的服务都可用）：
#   DeepSeek: https://api.deepseek.com         （便宜，推荐开发调试）
#   智谱GLM:  https://open.bigmodel.cn/api/paas/v4
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")

# 默认模型名
#   DeepSeek: deepseek-chat
#   智谱GLM:  glm-4-flash（有免费额度，适合开发）
MODEL_NAME = os.environ.get("LLM_MODEL", "deepseek-chat")

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

# ---------------------------------------------------------------
# 四、如果 .env 文件存在，则加载其中未被环境变量覆盖的配置
# ---------------------------------------------------------------
# 注意: 必须逐项补充而非"LLM密钥存在就跳过整个.env"——否则环境变量
# 里设了LLM_API_KEY时，.env中的S2_API_KEY等其余配置将永远读不到
# （2026-09-09实测踩坑: S2匿名模式持续429，密钥形同虚设）
_env_file = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if key == "LLM_API_KEY":
                    API_KEY = API_KEY or value
                elif key == "LLM_BASE_URL":
                    # 环境变量显式设置 > .env > 内置默认
                    if "LLM_BASE_URL" not in os.environ:
                        BASE_URL = value
                elif key == "LLM_MODEL":
                    if "LLM_MODEL" not in os.environ:
                        MODEL_NAME = value
                elif key == "S2_API_KEY":
                    S2_API_KEY = S2_API_KEY or value
                elif key == "APP_PASSWORD":
                    APP_PASSWORD = APP_PASSWORD or value
