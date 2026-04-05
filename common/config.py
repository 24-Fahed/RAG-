"""中央配置加载器。

从项目根目录读取 deploy.yaml，并提供类型化的配置访问接口。
被所有三个程序单元共同使用。
"""

import os
import yaml
from pathlib import Path

# 项目根目录 = 包含 deploy.yaml 的目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_config_cache: dict | None = None


def _load() -> dict:
    global _config_cache
    if _config_cache is None:
        path = PROJECT_ROOT / "deploy.yaml"
        with open(path, "r", encoding="utf-8") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_mode() -> str:
    return _load().get("mode", "mock")


def get(section: str, key: str, default=None):
    """获取单个配置值：get("server", "port") → 8000"""
    data = _load()
    return data.get(section, {}).get(key, default)


def get_section(section: str) -> dict:
    """获取整个配置节：get_section("server") → {...}"""
    return _load().get(section, {})


def is_mock() -> bool:
    return get_mode() == "mock"
