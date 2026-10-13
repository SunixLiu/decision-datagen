#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型配置加载（从 generate_dataset.py 移出）。"""

import os


def load_model_config(config_path: str = "config.json") -> dict:
    """加载配置文件，支持环境变量覆盖。"""
    import json as _json

    config = {
        "state_model": {"name": os.getenv("MODEL_NAME", "Qwen3.6-35B-A3B"),
                        "api_key": os.getenv("OPENAI_API_KEY", ""),
                        "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")},
        "teacher_model": {"name": os.getenv("MODEL_NAME", "Qwen3.6-35B-A3B"),
                          "api_key": os.getenv("OPENAI_API_KEY", ""),
                          "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")},
        "gold_model": {"name": os.getenv("MODEL_NAME", "Qwen3.6-35B-A3B"),
                       "api_key": os.getenv("OPENAI_API_KEY", ""),
                       "base_url": os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")},
    }

    # 环境变量覆盖
    for prefix in ["STATE_", "TEACHER_", "GOLD_"]:
        model_key = prefix.lower().rstrip("_")
        if os.getenv(f"{prefix}MODEL_NAME"):
            config[model_key]["name"] = os.getenv(f"{prefix}MODEL_NAME")
        if os.getenv(f"{prefix}API_KEY"):
            config[model_key]["api_key"] = os.getenv(f"{prefix}API_KEY")
        if os.getenv(f"{prefix}BASE_URL"):
            config[model_key]["base_url"] = os.getenv(f"{prefix}BASE_URL")

    # 从文件加载
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_config = _json.load(f)
        for key in ["state_model", "teacher_model", "gold_model"]:
            if key in file_config:
                fc = file_config[key]
                config[key]["name"] = fc.get("name", config[key]["name"])
                if "api_key" in fc and fc["api_key"]:
                    config[key]["api_key"] = fc["api_key"]
                if "base_url" in fc and fc["base_url"]:
                    config[key]["base_url"] = fc["base_url"]

    return config
