#!/usr/bin/env python3
"""gavel_paths.py — one place for every machine-local path (PR-4).
Override with env vars; defaults are this box. No more absolute D:/ literals
scattered across scripts.
"""
import os
from pathlib import Path

TS = Path(os.environ.get("GAVEL_TS", "D:/gavel/models/training_state"))
GAVEL_URL = os.environ.get("GAVEL_URL", "http://127.0.0.1:7575")
BENCH = Path(os.environ.get("GAVEL_BENCH", "D:/jev-phishing-bench"))
PHISH = Path(os.environ.get("GAVEL_PHISH", "D:/gavel-phish"))
DATA = Path(os.environ.get("GAVEL_DATA", "D:/"))
QA_SEEDS = Path(os.environ.get(
    "GAVEL_QA_SEEDS", str(DATA / "virtual-brain/distillation_seeds/qa_seeds.json")))
CODE_SEEDS = Path(os.environ.get(
    "GAVEL_CODE_SEEDS", str(DATA / "virtual-brain/distillation_seeds/code_seeds.json")))
FRONTIER_QA = Path(os.environ.get(
    "GAVEL_FRONTIER_QA", str(DATA / "virtual-brain/frontier_seeds/frontier_qa.json")))
DOLLY = Path(os.environ.get(
    "GAVEL_DOLLY",
    str(DATA / "research on LLM breakthroughs/data/dolly_knowledge_catalog.json")))
QWEN = Path(os.environ.get("GAVEL_QWEN", str(DATA / "gavel/models/qwen3-4b")))
