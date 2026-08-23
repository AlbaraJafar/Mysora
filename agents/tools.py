"""
Tools the Mysora agent can call — wraps existing Mysora
functions as LangChain tools.

All tool functions are verified against the actual signatures
in collect_data.py and scripts/eval_harness.py.
"""
import json
import sys
from pathlib import Path

from langchain.tools import tool

# Allow imports when called from the agents/ package inside the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@tool
def get_letter_accuracy(letter: str) -> str:
    """
    Get the current model accuracy for a specific Arabic letter.
    Use when the user asks how well Mysora recognizes a particular
    letter, or which letters are weak or strong.

    Args:
        letter: The Arabic letter to check (e.g. 'ح', 'ب', 'س')
    """
    try:
        from scripts.eval_harness import run_evaluation
        results = run_evaluation()

        if results.get("status") != "ok":
            return json.dumps({
                "status": "no_data",
                "message": "لا توجد بيانات تقييم كافية بعد لهذا الحرف",
            }, ensure_ascii=False)

        letter_data = results.get("per_letter", {}).get(letter)
        if not letter_data:
            return json.dumps({
                "status": "not_found",
                "message": f"لا توجد بيانات لحرف {letter}",
            }, ensure_ascii=False)

        return json.dumps({
            "letter": letter,
            "accuracy": round(letter_data["accuracy"] * 100, 1),
            "samples_tested": letter_data["samples"],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
def get_weakest_letters() -> str:
    """
    Get the Arabic letters the model currently struggles with most.
    Use when the user asks what to practice or which letters need
    improvement.
    """
    try:
        from scripts.eval_harness import run_evaluation
        results = run_evaluation()

        if results.get("status") != "ok":
            return json.dumps({
                "status": "fallback",
                "weak_letters": ["ح", "و", "ق", "ب", "ث", "ز", "ط", "ظ"],
                "message": "بناءً على التقييم اليدوي الحالي، هذه الحروف تحتاج تحسيناً",
            }, ensure_ascii=False)

        per_letter = results.get("per_letter", {})
        sorted_letters = sorted(per_letter.items(), key=lambda x: x[1]["accuracy"])
        weakest = [
            {"letter": ltr, "accuracy": round(data["accuracy"] * 100, 1)}
            for ltr, data in sorted_letters[:8]
        ]
        return json.dumps({"status": "ok", "weakest_letters": weakest}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
def get_data_collection_progress() -> str:
    """
    Get the current community data collection progress — how many
    samples have been collected per letter, and which letter needs
    more contributions most urgently.

    Returns total_clips, next priority letter, and sessions today.
    """
    try:
        from collect_data import get_progress, get_next_letter
        # get_progress() → {total_clips, by_letter, target_per_letter, sessions_today}
        progress = get_progress()
        # get_next_letter() → {letter, current_count, target, priority}
        next_letter = get_next_letter()
        return json.dumps({
            "total_clips": progress.get("total_clips", 0),
            "sessions_today": progress.get("sessions_today", 0),
            "next_priority_letter": next_letter.get("letter", ""),
            "next_letter_count": next_letter.get("current_count", 0),
            "next_letter_target": next_letter.get("target", 100),
            "next_letter_priority": next_letter.get("priority", "normal"),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
def get_model_info() -> str:
    """
    Get information about the current Mysora AI model — version,
    architecture, and technical details. Use when the user asks
    how Mysora works technically.
    """
    import os
    return json.dumps({
        "architecture": "MediaPipe HandLandmarker + ResNet-50 (31-class)",
        "version": os.environ.get("MODEL_VERSION", "v1.0"),
        "inference_backend": "cpu",
        "supported_letters": 31,
    }, ensure_ascii=False)


ALL_TOOLS = [
    get_letter_accuracy,
    get_weakest_letters,
    get_data_collection_progress,
    get_model_info,
]
