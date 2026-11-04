"""Decide from a transcript which app to open, with Laya (a local, Jev-compatible model).

Laya never generates text: it answers a fixed-choice question with a probability
per option, so the result is always one of the apps below or "none".

The loading follows fx-chart's ``tools/system_one.py``: the multilingual
checkpoint (the English one cannot read Japanese), with ``device="mps"`` set
explicitly because ``RLAgent`` only checks for CUDA and silently falls back to CPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

LAYA_REPO = "convaiinnovations/laya"
LAYA_VARIANT = "multilingual"

# option -> (app name for `open -a`, description given to Laya)
APPS = {
    "slack": ("Slack", "the user asks to open or launch Slack (スラック), the chat app"),
    "chrome": (
        "Google Chrome",
        "the user asks to open or launch Google Chrome (クローム), the web browser",
    ),
}
NONE = "none"

# Haas custom vocabulary for command mode: written form -> readings. Without it
# a short "Slack" can come out as "スナック"; with it the app names are written
# in English, which Laya matches most confidently.
VOCABULARY = {
    "Slack": ["スラック"],
    "Google Chrome": ["グーグルクローム"],
    "Chrome": ["クローム"],
}

QUESTIONS = {
    "app": {
        "type": "choice",
        "instructions": "Is this utterance a command to open an application? If so, which one?",
        "criteria": {
            **{key: desc for key, (_, desc) in APPS.items()},
            NONE: "not a command to open an app: ordinary dictated text, a question, "
            "or talk about an app without asking to open it",
        },
    }
}


def load_laya(device: str = "mps"):
    from huggingface_hub import snapshot_download

    root = snapshot_download(LAYA_REPO, allow_patterns=["*.py", f"{LAYA_VARIANT}/*"])
    sys.path.insert(0, root)  # rl_agent_api.py lives at the repo root only
    from rl_agent_api import RLAgent

    agent = RLAgent(os.path.join(root, LAYA_VARIANT), device=device)
    agent.system_one("ウォームアップ", QUESTIONS)  # first MPS pass compiles kernels
    return agent


def decide(agent, text: str) -> tuple[str, float, dict[str, float]]:
    """``(option, probability of that option, all probabilities)``."""
    answer = agent.system_one(text, QUESTIONS)["answers"]["app"]
    probs = answer["probabilities"]
    choice = max(probs, key=probs.get)
    return choice, probs[choice], probs


def open_app(option: str) -> None:
    subprocess.run(["open", "-a", APPS[option][0]], check=True)
