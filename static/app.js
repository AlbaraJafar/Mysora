// Basic client ID persisted per browser
function getClientId() {
  const key = "mysora_client_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = "web-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
    localStorage.setItem(key, id);
  }
  return id;
}

function getApiBase() {
  if (typeof window.MYSORA_API_BASE === "string" && window.MYSORA_API_BASE.length > 0) {
    return window.MYSORA_API_BASE.replace(/\/$/, "");
  }
  const meta = document.querySelector('meta[name="mysora-api-base"]');
  if (meta && meta.getAttribute("content")) {
    return meta.getAttribute("content").trim().replace(/\/$/, "");
  }
  return window.location.origin;
}

const API_BASE = getApiBase();

if (typeof window !== "undefined") {
  const host = window.location.hostname || "";
  const looksVercel = host.endsWith(".vercel.app") || host.endsWith(".netlify.app");
  if (looksVercel && API_BASE === window.location.origin) {
    console.warn(
      "[Mysora] Static host matches API base. Set <meta name=\"mysora-api-base\" content=\"https://YOUR-RAILWAY-URL\"> or window.MYSORA_API_BASE so /predict hits the backend."
    );
  }
}

async function fetchJson(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

// ============ Leaderboard rendering (home + fatiha) ============

async function loadLeaderboardInto(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return;
  try {
    const data = await fetchJson(`${API_BASE}/leaderboard/top?limit=5`);
    const entries = data.entries || [];
    if (!entries.length) {
      el.innerHTML = '<p class="hint">لم تُسجل أي نتائج بعد.</p>';
      return;
    }
    const items = entries
      .map(
        (e) =>
          `<li><span class="leaderboard-name">${e.name}</span><span class="leaderboard-score">${e.accuracy.toFixed(
            0
          )}% / ${e.duration_seconds.toFixed(1)}ث</span></li>`
      )
      .join("");
    el.innerHTML = `<ul>${items}</ul>`;
  } catch {
    el.innerHTML = '<p class="hint">تعذّر تحميل ترتيب الطلاب.</p>';
  }
}

// ============ Fatiha page logic ============

async function initFatihaPage() {
  const textEl = document.getElementById("fatiha-text");
  if (!textEl) return; // not on this page

  const btnStart = document.getElementById("btn-start");
  const btnReset = document.getElementById("btn-reset");
  const statusEl = document.getElementById("session-status");
  const nameInput = document.getElementById("user-name");
  const video = document.getElementById("video");
  const canvas = document.getElementById("capture-canvas");
  const detectedLetterEl = document.getElementById("detected-letter");
  const serverWordEl = document.getElementById("server-word");
  const accuracyEl = document.getElementById("accuracy-display");

  // Persist user name
  const nameKey = "mysora_username";
  const savedName = localStorage.getItem(nameKey);
  if (savedName && nameInput) nameInput.value = savedName;
  nameInput?.addEventListener("change", () =>
    localStorage.setItem(nameKey, nameInput.value || "")
  );

  const clientId = getClientId();

  const cfg = await fetchJson(`${API_BASE}/config/fatiha`);
  const target = cfg.target;
  const verses = cfg.verses || [target];

  // Build a joined target from verses (single space between verses) to align indices
  const joinedTarget = verses.join(" ");

  // Render verses as real ayat text (not per-letter boxes)
  const ayahEls = [];
  textEl.innerHTML = "";
  verses.forEach((verse, idx) => {
    const ayahDiv = document.createElement("div");
    ayahDiv.className = "quran-ayah";
    const textSpan = document.createElement("span");
    textSpan.className = "quran-ayah-text";
    textSpan.dataset.ayahIndex = String(idx);
    ayahDiv.appendChild(textSpan);

    const numSpan = document.createElement("span");
    numSpan.className = "quran-ayah-number";
    numSpan.textContent = `(${idx + 1})`;
    ayahDiv.appendChild(numSpan);

    ayahEls.push(textSpan);
    textEl.appendChild(ayahDiv);
  });

  let stream = null;
  let sending = false;
  let running = false;
  let inferIntervalId = null;
  let startTime = null;
  let totalPredictions = 0;
  let correctPredictions = 0;
  let lastWrongTimeout = null;
  let wrongFlash = null; // { index:number, char:string }

  function updateAccuracyDisplay() {
    const acc = totalPredictions ? (correctPredictions / totalPredictions) * 100 : 0;
    accuracyEl.textContent = `${acc.toFixed(0)}%`;
  }

  function renderAyat(progressIndex, wrong) {
    // Map global progressIndex into each verse slice
    let offset = 0;
    verses.forEach((verse, idx) => {
      const verseStart = offset;
      const verseEnd = offset + verse.length;

      const localProgress = Math.max(0, Math.min(verse.length, progressIndex - verseStart));
      const prefix = verse.slice(0, localProgress);
      const rest = verse.slice(localProgress);

      let html = "";

      // If wrong attempt falls inside this verse, insert it at the correct local position
      if (wrong && wrong.index >= verseStart && wrong.index < verseEnd) {
        const localWrong = wrong.index - verseStart;
        const p2 = verse.slice(0, localWrong);
        const r2 = verse.slice(localWrong);
        html =
          `<span class="quran-ayah-correct">${escapeHtml(p2)}</span>` +
          `<span class="quran-ayah-wrong">${escapeHtml(wrong.char)}</span>` +
          `<span class="quran-ayah-dim">${escapeHtml(r2)}</span>`;
      } else {
        html =
          `<span class="quran-ayah-correct">${escapeHtml(prefix)}</span>` +
          `<span class="quran-ayah-dim">${escapeHtml(rest)}</span>`;
      }

      ayahEls[idx].innerHTML = html;

      // add 1 for the inter-verse space in joinedTarget
      offset = verseEnd + (idx < verses.length - 1 ? 1 : 0);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function tickInference() {
    if (!running) return;
    if (sending) return;
    try {
      sending = true;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      // Enough resolution for MediaPipe hands + model (was 320×240; too small for reliable detection)
      const targetW = 640;
      const targetH = 480;
      canvas.width = targetW;
      canvas.height = targetH;
      ctx.drawImage(video, 0, 0, targetW, targetH);

      const blob = await new Promise((resolve) =>
        canvas.toBlob(resolve, "image/jpeg", 0.78)
      );
      if (!blob) return;

      const form = new FormData();
      form.append("client_id", clientId);
      form.append("image", blob, "frame.jpg");

      const res = await fetch(`${API_BASE}/predict?client_id=${encodeURIComponent(
        clientId
      )}`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        console.error("predict failed", res.status, detail.slice(0, 300));
        throw new Error(`predict failed: ${res.status}`);
      }
      const data = await res.json();

      const raw = data?.raw || {};
      const emitted = data?.stable?.emitted_arabic || "";
      const preview = (raw.arabic_preview || "").trim();
      const word = data?.word || "";
      const progressIndex = Number.isFinite(data?.progress_index) ? data.progress_index : word.length;
      const attempt = data?.attempt || null;
      const complete = !!data?.complete;

      // Show accepted Arabic first; else live preview from top-1 class (so user always sees feedback)
      if (emitted) {
        detectedLetterEl.textContent = emitted;
        detectedLetterEl.removeAttribute("title");
      } else if (preview) {
        detectedLetterEl.textContent = preview;
        const pct =
          raw.confidence != null ? (Number(raw.confidence) * 100).toFixed(0) : "";
        detectedLetterEl.title = raw.label ? `${raw.label}${pct ? " · " + pct + "%" : ""}` : "";
      } else if (raw.label) {
        detectedLetterEl.textContent = raw.label;
        detectedLetterEl.title =
          raw.confidence != null ? String(Math.round(Number(raw.confidence) * 100)) + "%" : "";
      }
      serverWordEl.textContent = word || "ـ";
      renderAyat(progressIndex, wrongFlash);

      // Track accuracy: treat each stable emission as one attempt
      if (data?.stable?.emitted_label) {
        totalPredictions += 1;
        if (attempt && attempt.accepted) {
          correctPredictions += 1;
        } else {
          // brief wrong flash at current expected index (does not advance)
          if (attempt && attempt.expected) {
            wrongFlash = { index: progressIndex, char: emitted || attempt.attempted || "" };
            renderAyat(progressIndex, wrongFlash);
            clearTimeout(lastWrongTimeout);
            lastWrongTimeout = setTimeout(() => {
              wrongFlash = null;
              renderAyat(progressIndex, null);
            }, 650);
          }
        }
        updateAccuracyDisplay();
      }

      if (complete && running) {
        running = false;
        if (inferIntervalId != null) {
          clearInterval(inferIntervalId);
          inferIntervalId = null;
        }
        const duration =
          (Date.now() - (startTime || Date.now())) / 1000;
        statusEl.textContent = `أحسنت! اكتملت السورة في ${duration.toFixed(1)} ثانية.`;
        btnStart.disabled = false;
        // send leaderboard entry
        const acc = totalPredictions
          ? (correctPredictions / totalPredictions) * 100
          : 0;
        const name = nameInput.value || "ضيف";
        fetch(
          `${API_BASE}/leaderboard/submit?name=${encodeURIComponent(
            name
          )}&duration_seconds=${encodeURIComponent(
            duration
          )}&accuracy=${encodeURIComponent(acc)}`,
          { method: "POST" }
        ).then(() => {
          loadLeaderboardInto("leaderboard-fatiha");
          loadLeaderboardInto("leaderboard-home");
        });
        return;
      }
    } catch (err) {
      console.error(err);
    } finally {
      sending = false;
    }
  }

  async function startSession() {
    if (running) return;
    try {
      // Reset backend session
      await fetchJson(
        `${API_BASE}/reset?client_id=${encodeURIComponent(clientId)}`,
        { method: "POST" }
      );
      // Reset local state
      totalPredictions = 0;
      correctPredictions = 0;
      updateAccuracyDisplay();
      detectedLetterEl.textContent = "ـ";
      serverWordEl.textContent = "ـ";
      wrongFlash = null;
      renderAyat(0, null);

      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
      }
      running = true;
      startTime = Date.now();
      statusEl.textContent = "جاري التعلم…";
      btnStart.disabled = true;
      if (inferIntervalId != null) clearInterval(inferIntervalId);
      inferIntervalId = setInterval(() => {
        void tickInference();
      }, 220);
    } catch (err) {
      console.error(err);
      statusEl.textContent = "تعذّر بدء الجلسة. تحقق من الكاميرا والاتصال.";
    }
  }

  async function resetSession() {
    running = false;
    if (inferIntervalId != null) {
      clearInterval(inferIntervalId);
      inferIntervalId = null;
    }
    btnStart.disabled = false;
    statusEl.textContent = "تمت إعادة التهيئة. يمكنك البدء من جديد.";
    await fetchJson(
      `${API_BASE}/reset?client_id=${encodeURIComponent(clientId)}`,
      { method: "POST" }
    );
    detectedLetterEl.textContent = "ـ";
    serverWordEl.textContent = "ـ";
    totalPredictions = 0;
    correctPredictions = 0;
    updateAccuracyDisplay();
    wrongFlash = null;
    renderAyat(0, null);
  }

  btnStart?.addEventListener("click", startSession);
  btnReset?.addEventListener("click", resetSession);

  loadLeaderboardInto("leaderboard-fatiha");
  renderAyat(0, null);
}

// ============ Home page init ============

function initHomePage() {
  loadLeaderboardInto("leaderboard-home");
}

document.addEventListener("DOMContentLoaded", () => {
  initHomePage();
  initFatihaPage();
});

