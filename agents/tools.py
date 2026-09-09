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


@tool
def get_my_progress(student_id: str) -> str:
    """
    Get the current user's own practice history and accuracy per letter.
    Use when a STUDENT asks about their own performance, progress, or
    what they should practice next.

    Args:
        student_id: The UUID of the student (injected by the system)
    """
    from collections import defaultdict

    try:
        from auth.supabase_client import get_supabase
        supabase = get_supabase()
        result = (
            supabase.table("practice_sessions")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        sessions = result.data
        if not sessions:
            return json.dumps({
                "status": "no_data",
                "message": "لا توجد بيانات تدريب بعد. ابدأ بالتدريب على الحروف!",
            }, ensure_ascii=False)

        by_letter: dict = defaultdict(lambda: {"correct": 0, "total": 0})
        for s in sessions:
            by_letter[s["letter"]]["total"] += 1
            if s["correct"]:
                by_letter[s["letter"]]["correct"] += 1

        letter_stats = [
            {
                "letter": ltr,
                "accuracy": round(d["correct"] / d["total"] * 100, 1),
                "attempts": d["total"],
            }
            for ltr, d in by_letter.items()
        ]
        letter_stats.sort(key=lambda x: x["accuracy"])

        return json.dumps({
            "status": "ok",
            "total_sessions": len(sessions),
            "weakest_letters": letter_stats[:5],
            "strongest_letters": letter_stats[-3:] if len(letter_stats) >= 3 else [],
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@tool
def get_my_students_status(teacher_id: str) -> str:
    """
    Get the practice status of all students assigned to this TEACHER.
    Use when a teacher or parent asks who needs attention, who is
    performing well, or for a class overview.

    Args:
        teacher_id: The UUID of the teacher (injected by the system)
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    try:
        from auth.supabase_client import get_supabase
        supabase = get_supabase()

        classes = (
            supabase.table("classes")
            .select("id")
            .eq("teacher_id", teacher_id)
            .execute()
        )
        class_ids = [c["id"] for c in classes.data]

        if not class_ids:
            return json.dumps({
                "status": "no_classes",
                "message": "لا توجد فصول مرتبطة بحسابك بعد",
            }, ensure_ascii=False)

        students_result = (
            supabase.table("class_students")
            .select("student_id, users(display_name)")
            .in_("class_id", class_ids)
            .execute()
        )

        student_ids = [s["student_id"] for s in students_result.data]
        name_map = {
            s["student_id"]: (s.get("users") or {}).get("display_name", "طالب")
            for s in students_result.data
        }

        if not student_ids:
            return json.dumps({
                "status": "no_students",
                "message": "لا يوجد طلاب مسجلين في فصولك بعد",
            }, ensure_ascii=False)

        sessions_resp = (
            supabase.table("practice_sessions")
            .select("*")
            .in_("student_id", student_ids)
            .order("created_at", desc=True)
            .execute()
        )

        by_student: dict = defaultdict(lambda: {"correct": 0, "total": 0, "last_active": None})
        for s in sessions_resp.data:
            sid = s["student_id"]
            by_student[sid]["total"] += 1
            if s["correct"]:
                by_student[sid]["correct"] += 1
            if not by_student[sid]["last_active"]:
                by_student[sid]["last_active"] = s["created_at"]

        now = datetime.now(timezone.utc)
        needs_attention = []
        doing_well = []

        for sid in student_ids:
            name = name_map.get(sid, "طالب")
            stats = by_student.get(sid)

            if not stats or stats["total"] == 0:
                needs_attention.append({"name": name, "reason": "لم يتدرب بعد"})
                continue

            accuracy = stats["correct"] / stats["total"] * 100
            last_active_str = stats["last_active"]
            try:
                last_active = datetime.fromisoformat(last_active_str.replace("Z", "+00:00"))
                days_inactive = (now - last_active).days
            except Exception:
                days_inactive = 0

            if days_inactive >= 5:
                needs_attention.append({
                    "name": name,
                    "reason": f"لم يتدرب منذ {days_inactive} أيام",
                })
            elif accuracy < 50:
                needs_attention.append({
                    "name": name,
                    "reason": f"دقة منخفضة: {round(accuracy)}%",
                })
            elif accuracy >= 80:
                doing_well.append({"name": name, "accuracy": round(accuracy)})

        return json.dumps({
            "status": "ok",
            "total_students": len(student_ids),
            "needs_attention": needs_attention,
            "doing_well": doing_well,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


ALL_TOOLS = [
    get_letter_accuracy,
    get_weakest_letters,
    get_data_collection_progress,
    get_model_info,
    get_my_progress,
    get_my_students_status,
]
