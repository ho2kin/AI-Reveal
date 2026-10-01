"""端点常量与 API Key 读取：--api-key > 环境变量 MAKERS_MODELS_KEY > .env 文件."""

from __future__ import annotations

import os
from pathlib import Path

BASE_URL = "https://ai-gateway.edgeone.link"

# 朱雀 AIGC 文本检测（专用端点，非 chat/completions）
ZHUQUE_CLASSIFY_URL = f"{BASE_URL}/v1/providers/zhuque-text/classify"
# jev 决策模型（专用 System One 端点，POST {state, questions}）
SYSTEMONE_URL = f"{BASE_URL}/v1/systemone"
# OpenAI 兼容端点（本项目暂未用到，保留作扩展）
CHAT_COMPLETIONS_URL = f"{BASE_URL}/v1/chat/completions"

ZHUQUE_MODEL = "@makers/zhuque-text"
JEV_MODEL = "@makers/jev"

REQUEST_TIMEOUT = 120  # 秒
MAX_RETRIES = 3

ENV_VAR = "MAKERS_MODELS_KEY"


class MissingApiKeyError(RuntimeError):
    """未找到 API Key."""


def _read_dotenv(path: Path) -> str | None:
    """极简 .env 解析：只认 KEY=VALUE 行，值可带引号，忽略 # 注释."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != ENV_VAR:
            continue
        value = value.strip().strip("'\"")
        return value or None
    return None


def load_api_key(cli_key: str | None = None, start_dir: Path | None = None) -> str:
    """按优先级解析 API Key：命令行参数 > 环境变量 > .env 文件."""
    if cli_key:
        return cli_key.strip()

    env = os.environ.get(ENV_VAR, "").strip()
    if env:
        return env

    dirs: list[Path] = []
    if start_dir is not None:
        dirs.append(start_dir)
    dirs.append(Path.cwd())
    dirs.extend(Path.cwd().parents)
    dirs.extend(Path(__file__).resolve().parents[1:3])

    seen: set[Path] = set()
    for d in dirs:
        d = d.resolve()
        if d in seen:
            continue
        seen.add(d)
        value = _read_dotenv(d / ".env")
        if value:
            return value

    raise MissingApiKeyError(
        f"未找到 API Key：请通过 --api-key 传入、设置环境变量 {ENV_VAR}，"
        "或在项目根目录 .env 中写入 MAKERS_MODELS_KEY=sk-xxx"
    )
