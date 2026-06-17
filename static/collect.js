(function () {
  const STORAGE_SESSION = "mysora_collect_session_id";
  const STORAGE_SIGNER = "mysora_collect_signer_id";

  function getApiBase() {
    const meta = document.querySelector('meta[name="mysora-api-base"]');
    if (meta && meta.getAttribute("content")) {
      return meta.getAttribute("content").trim().replace(/\/$/, "");
    }
    return window.location.origin;
  }

  const API_BASE = getApiBase();

  async function fetchJson(url, opts = {}) {
    const res = await fetch(url, opts);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }
    return res.json();
  }

  const screens = {
    welcome: document.getElementById("screen-welcome"),
    calibration: document.getElementById("screen-calibration"),
    collect: document.getElementById("screen-collect"),
    thanks: document.getElementById("screen-thanks"),
  };

  function showScreen(name) {
    Object.entries(screens).forEach(([key, el]) => {
      if (el) el.hidden = key !== name;
    });
  }

  const welcomeForm = document.getElementById("form-session");
  const welcomeError = document.getElementById("welcome-error");
  const btnStartRecording = document.getElementById("btn-start-recording");
  const calibrationStatus = document.getElementById("calibration-status");
  const videoCal = document.getElementById("collect-video");
  const videoLoop = document.getElementById("collect-video-loop");
  const canvas = document.getElementById("collect-canvas");
  const targetLetterEl = document.getElementById("target-letter");
  const priorityBadge = document.getElementById("priority-badge");
  const letterProgressText = document.getElementById("letter-progress-text");
  const letterProgressFill = document.getElementById("letter-progress-fill");
  const handStatusLoop = document.getElementById("hand-status-loop");
  const sessionClipsCount = document.getElementById("session-clips-count");
  const collectError = document.getElementById("collect-error");
  const globalProgressText = document.getElementById("global-progress-text");
  const btnRecordClip = document.getElementById("btn-record-clip");
  const btnNextLetter = document.getElementById("btn-next-letter");
  const btnFinish = document.getElementById("btn-finish");
  const btnCollectAgain = document.getElementById("btn-collect-again");
  const thanksClipCount = document.getElementById("thanks-clip-count");
  const orientationRow = document.getElementById("orientation-row");

  let stream = null;
  let hands = null;
  let handsReady = false;
  let lastLandmarks = null;
  let lastConfidence = 0;
  let mpLoopId = null;
  let currentLetter = null;
  let clipsThisSession = 0;
  let recordingClip = false;

  const PRIORITY_LABELS = {
    critical: "أولوية حرجة",
    high: "أولوية عالية",
    medium: "أولوية متوسطة",
    normal: "عام",
  };

  function getSessionIds() {
    return {
      sessionId: localStorage.getItem(STORAGE_SESSION),
      signerId: localStorage.getItem(STORAGE_SIGNER),
    };
  }

  function setHandStatus(el, ok) {
    if (!el) return;
    el.classList.toggle("calibration-status--ok", ok);
    el.classList.toggle("calibration-status--warn", !ok);
    el.textContent = ok ? "يدك واضحة ✓" : "اضبط الإضاءة أو قرّب يدك";
  }

  function initMediaPipe() {
    if (typeof Hands === "undefined") {
      console.error("MediaPipe Hands not loaded");
      return Promise.reject(new Error("MediaPipe unavailable"));
    }
    hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
    hands.onResults((results) => {
      const detected = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
      if (detected) {
        lastLandmarks = results.multiHandLandmarks[0].map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z,
        }));
        lastConfidence = 0.9;
      } else {
        lastLandmarks = null;
        lastConfidence = 0;
      }
      const ok = detected && lastConfidence >= 0.5;
      if (!screens.calibration.hidden) {
        setHandStatus(calibrationStatus, ok);
        if (btnStartRecording) btnStartRecording.disabled = !ok;
      }
      if (!screens.collect.hidden) {
        setHandStatus(handStatusLoop, ok);
        if (btnRecordClip) btnRecordClip.disabled = recordingClip || !ok;
      }
    });
    handsReady = true;
    return Promise.resolve();
  }

  async function ensureCamera(videoEl) {
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });
    }
    if (videoEl && videoEl.srcObject !== stream) {
      videoEl.srcObject = stream;
      await videoEl.play().catch(() => {});
    }
    if (videoCal && videoCal !== videoEl) videoCal.srcObject = stream;
    if (videoLoop && videoLoop !== videoEl) videoLoop.srcObject = stream;
  }

  function activeVideo() {
    if (!screens.calibration.hidden) return videoCal;
    if (!screens.collect.hidden) return videoLoop;
    return videoCal || videoLoop;
  }

  function stopMpLoop() {
    if (mpLoopId != null) {
      cancelAnimationFrame(mpLoopId);
      mpLoopId = null;
    }
  }

  function startMpLoop() {
    stopMpLoop();
    const tick = async () => {
      const video = activeVideo();
      if (handsReady && hands && video && video.readyState >= 2) {
        try {
          await hands.send({ image: video });
        } catch (e) {
          console.warn(e);
        }
      }
      mpLoopId = requestAnimationFrame(tick);
    };
    mpLoopId = requestAnimationFrame(tick);
  }

  function getHandOrientation() {
    const checked = document.querySelector('input[name="hand_orientation"]:checked');
    return checked ? checked.value : "front";
  }

  function updateLetterUI(data) {
    currentLetter = data.letter;
    if (targetLetterEl) targetLetterEl.textContent = data.letter;
    const count = data.current_count || 0;
    const target = data.target || 100;
    const pct = Math.min(100, Math.round((count / target) * 100));
    if (letterProgressText) letterProgressText.textContent = `${count} / ${target}`;
    if (letterProgressFill) letterProgressFill.style.width = `${pct}%`;
    const priority = data.priority || "normal";
    if (priorityBadge) {
      priorityBadge.textContent = PRIORITY_LABELS[priority] || priority;
      priorityBadge.className = `priority-badge priority-badge--${priority}`;
    }
    if (orientationRow) {
      const needsOrientation = data.letter === "ط" || data.letter === "ظ";
      orientationRow.classList.toggle("collect-orientation--highlight", needsOrientation);
    }
  }

  async function loadNextLetter() {
    const data = await fetchJson(`${API_BASE}/collect/next-letter`);
    updateLetterUI(data);
    return data;
  }

  async function loadGlobalProgress() {
    try {
      const p = await fetchJson(`${API_BASE}/collect/progress`);
      const total = p.total_clips || 0;
      const today = p.sessions_today || 0;
      if (globalProgressText) {
        globalProgressText.textContent = `إجمالي التسجيلات: ${total} · جلسات اليوم: ${today}`;
      }
    } catch {
      if (globalProgressText) globalProgressText.textContent = "تعذّر تحميل الإحصائيات.";
    }
  }

  function captureFrameBase64(video) {
    const ctx = canvas.getContext("2d");
    canvas.width = 320;
    canvas.height = 240;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.75);
  }

  function captureThumbnail(videoEl) {
    try {
      const c = document.createElement("canvas");
      c.width = 224;
      c.height = 224;
      c.getContext("2d").drawImage(videoEl, 0, 0, 224, 224);
      return c.toDataURL("image/png").split(",")[1];
    } catch (e) {
      return null;
    }
  }

  async function submitClip() {
    const { sessionId, signerId } = getSessionIds();
    if (!sessionId || !signerId) throw new Error("لا توجد جلسة نشطة");

    const video = videoLoop || videoCal;
    const framePayload = {
      type: "landmarks_sequence",
      landmarks: lastLandmarks,
      thumbnail: video ? captureFrameBase64(video) : null,
      captured_at: new Date().toISOString(),
    };

    const form = new FormData();
    form.append("frame_data", JSON.stringify(framePayload));
    form.append("label", currentLetter || "؟");
    form.append("label_type", "letter");
    form.append("session_id", sessionId);
    form.append("signer_id", signerId);
    form.append("hand_orientation", getHandOrientation());
    form.append("confidence", String(lastConfidence));
    const thumb = video ? captureThumbnail(video) : null;
    if (thumb) form.append("thumbnail_b64", thumb);

    const res = await fetch(`${API_BASE}/collect/clip`, { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  welcomeForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    welcomeError.hidden = true;
    const signerType = document.getElementById("signer-type").value;
    const dominantHand = document.getElementById("dominant-hand").value;
    const experienceYears = Number(document.getElementById("experience-years").value);

    try {
      const data = await fetchJson(`${API_BASE}/collect/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signer_type: signerType,
          dominant_hand: dominantHand,
          experience_years: experienceYears,
        }),
      });
      localStorage.setItem(STORAGE_SESSION, data.session_id);
      localStorage.setItem(STORAGE_SIGNER, data.signer_id);
      clipsThisSession = 0;
      showScreen("calibration");
      await initMediaPipe();
      await ensureCamera(videoCal);
      startMpLoop();
    } catch (err) {
      welcomeError.textContent = "تعذّر بدء الجلسة. تحقق من الاتصال بالخادم.";
      welcomeError.hidden = false;
      console.error(err);
    }
  });

  btnStartRecording?.addEventListener("click", async () => {
    showScreen("collect");
    await ensureCamera(videoLoop);
    startMpLoop();
    try {
      await loadNextLetter();
      await loadGlobalProgress();
    } catch (err) {
      collectError.textContent = "تعذّر تحميل الحرف التالي.";
      collectError.hidden = false;
      console.error(err);
    }
  });

  btnRecordClip?.addEventListener("click", async () => {
    if (!lastLandmarks) return;
    recordingClip = true;
    btnRecordClip.disabled = true;
    collectError.hidden = true;
    try {
      await submitClip();
      clipsThisSession += 1;
      if (sessionClipsCount) {
        sessionClipsCount.textContent = `تسجيلاتك في هذه الجلسة: ${clipsThisSession}`;
      }
      await loadNextLetter();
      await loadGlobalProgress();
    } catch (err) {
      collectError.textContent = "فشل حفظ التسجيل. حاول مرة أخرى.";
      collectError.hidden = false;
      console.error(err);
    } finally {
      recordingClip = false;
    }
  });

  btnNextLetter?.addEventListener("click", () => {
    void loadNextLetter().catch(console.error);
  });

  btnFinish?.addEventListener("click", () => {
    stopMpLoop();
    if (thanksClipCount) thanksClipCount.textContent = String(clipsThisSession);
    showScreen("thanks");
  });

  btnCollectAgain?.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_SESSION);
    localStorage.removeItem(STORAGE_SIGNER);
    clipsThisSession = 0;
    showScreen("welcome");
    welcomeForm?.reset();
  });

  document.addEventListener("DOMContentLoaded", () => {
    showScreen("welcome");
  });
})();
