import os
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import cv2
from scripts.inference import predict_proba
from scripts.hand_crop import prepare_for_inference
import numpy as np

from gesture_engine import GestureStabilizer
from mysora_letters import letter_map, target_fatiha


def draw_arabic_text(frame, text, position, size=32):

    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(
            os.environ.get("MYSORA_FONT_PATH", "arial.ttf"), size
        )
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            font = ImageFont.load_default()

    draw.text(position, bidi_text, font=font, fill=(0,255,0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ===============================
# Camera
# ===============================

cap = cv2.VideoCapture(1)

word = ""
stabilizer = GestureStabilizer(
    num_classes=31,
    ema_alpha=0.35,
    min_confidence=0.65,
    min_margin=0.10,
    hold_frames=12,
)
last_arabic_letter = ""
# ===============================
# Main Loop
# ===============================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    crop, _hand_ok = prepare_for_inference(frame)
    labels, probs = predict_proba(crop)
    pred = stabilizer.update(labels=list(labels), probs=probs)

    # Only append when stabilizer "emits" a stable label (debounced)
    arabic_letter = ""
    if pred.stable_label is not None:
        arabic_letter = letter_map.get(pred.stable_label, "")
        if arabic_letter:
            word += arabic_letter
            last_arabic_letter = arabic_letter
    else:
        # Keep showing the last confirmed letter to reduce flicker
        arabic_letter = last_arabic_letter

    # ===============================
    # Check progress
    # ===============================

    progress_correct = target_fatiha.startswith(word)

    if progress_correct:
        color = (0,255,0)
    else:
        color = (0,0,255)

    # ===============================
    # Display prediction
    # ===============================
    frame = draw_arabic_text(frame, arabic_letter, (30,50), 60)
    frame = draw_arabic_text(frame, word, (30,140), 40)
    status = "صحيح" if progress_correct else "خطأ"

    frame = draw_arabic_text(frame, status, (30,200), 40)

    if word == target_fatiha:
        cv2.putText(frame,
                    "SURAH COMPLETE",
                    (30,240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,255,0),
                    3)

    cv2.imshow("Mysora - Surah Learning", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        word = ""
        last_arabic_letter = ""
        stabilizer.reset()

cap.release()
cv2.destroyAllWindows()