from collect_data import create_session, get_next_letter, get_progress, save_clip

# 1x1 PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z5BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _patch_collect_dirs(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mod, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(mod, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(mod, "PROGRESS_FILE", tmp_path / "progress.json")


def test_collect_flow(tmp_path, monkeypatch):
    import collect_data as mod

    _patch_collect_dirs(mod, monkeypatch, tmp_path)

    s = create_session(signer_type="deaf", dominant_hand="right", experience_years=5)
    assert "session_id" in s and "signer_id" in s

    r = save_clip(
        frame_data='{"landmarks": []}',
        label="و",
        label_type="letter",
        session_id=s["session_id"],
        signer_id=s["signer_id"],
        hand_orientation="front",
        confidence=0.9,
    )
    assert r["status"] == "saved"

    prog = get_progress()
    assert prog["total_clips"] >= 1
    assert prog["by_letter"].get("و", 0) >= 1

    nxt = get_next_letter()
    assert "letter" in nxt and "priority" in nxt


def test_thumbnail_saved(tmp_path, monkeypatch):
    import collect_data as mod

    _patch_collect_dirs(mod, monkeypatch, tmp_path)

    s = create_session(signer_type="hearing", dominant_hand="left", experience_years=2)
    r = save_clip(
        frame_data='{"landmarks": []}',
        label="ب",
        label_type="letter",
        session_id=s["session_id"],
        signer_id=s["signer_id"],
        hand_orientation="front",
        confidence=0.85,
        thumbnail_b64=_TINY_PNG_B64,
    )
    clip_id = r["clip_id"]
    thumb_path = tmp_path / "clips" / "ب" / "thumbnails" / f"{clip_id}.png"
    assert thumb_path.is_file()
    assert thumb_path.stat().st_size > 0
