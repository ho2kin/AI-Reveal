"""本地额度追踪：Makers 内置模型每月免费额度（默认 50 万 token）.

网关没有提供额度查询接口，这里按 API Key 在本地累计每月的
makers_models_usage 计费 token，用于估算当月已用/剩余。
跨会话持久化在 ~/.ai-reveal/usage.json；换 Key 或月底自动重置。
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

# Makers 内置模型每月免费额度（token），可用环境变量覆盖
DEFAULT_MONTHLY_FREE_TOKENS = 500_000
ENV_MONTHLY_TOKENS = "AI_REVEAL_MONTHLY_TOKENS"

_STATE_DIR = Path.home() / ".ai-reveal"
_STATE_FILE = _STATE_DIR / "usage.json"


def monthly_free_tokens() -> int:
    raw = os.environ.get(ENV_MONTHLY_TOKENS, "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_MONTHLY_FREE_TOKENS


def _key_id(api_key: str) -> str:
    return hashlib.sha1(api_key.encode()).hexdigest()[:12]


def _load_state() -> dict:
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def month_used_tokens(api_key: str) -> int:
    """当前月份已累计消耗的 Makers 计费 token（不含本次运行）."""
    state = _load_state()
    entry = state.get(_key_id(api_key)) or {}
    if entry.get("month") != datetime.now().strftime("%Y-%m"):
        return 0
    return int(entry.get("makers_tokens") or 0)


def record_usage(api_key: str, makers_tokens: int) -> int:
    """把本次运行的消耗累加进本地账本，返回累加后的当月总量."""
    if makers_tokens <= 0:
        return month_used_tokens(api_key)
    state = _load_state()
    kid = _key_id(api_key)
    month = datetime.now().strftime("%Y-%m")
    entry = state.get(kid) or {}
    if entry.get("month") != month:
        entry = {"month": month, "makers_tokens": 0}
    entry["makers_tokens"] = int(entry.get("makers_tokens") or 0) + makers_tokens
    state[kid] = entry
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # 账本写失败只影响余量估算，不阻断检测
    return entry["makers_tokens"]


def quota_summary(api_key: str, this_run_tokens: int = 0) -> dict:
    """汇总额度信息：当月已用（含本次）、剩余、使用百分比."""
    used = month_used_tokens(api_key) + this_run_tokens
    free = monthly_free_tokens()
    remaining = max(free - used, 0)
    return {
        "free": free,
        "used": used,
        "remaining": remaining,
        "used_pct": round(used / free * 100, 2) if free > 0 else 0.0,
    }
