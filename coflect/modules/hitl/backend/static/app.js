(() => {
  const state = {
    connected: false,
    isPaused: false,
    metrics: {},
    xai: {},
    forecast: {},
    windowState: {},
    selectedSampleIdx: null,
    candidateImages: {},
    roi: null,
    dragStart: null,
    dragCurrent: null,
    lastBackend: "",
  };

  const els = {
    statusConnected: document.getElementById("status-connected"),
    statusTraining: document.getElementById("status-training"),
    statusReview: document.getElementById("status-review"),
    mStep: document.getElementById("m-step"),
    mBackend: document.getElementById("m-backend"),
    mLoss: document.getElementById("m-loss"),
    mAcc: document.getElementById("m-acc"),
    mFocus: document.getElementById("m-focus"),
    mSps: document.getElementById("m-sps"),
    mFStep: document.getElementById("m-fstep"),
    mFEpoch: document.getElementById("m-fepoch"),
    mFCount: document.getElementById("m-fcount"),
    xSample: document.getElementById("x-sample"),
    xTarget: document.getElementById("x-target"),
    xPred: document.getElementById("x-pred"),
    xKind: document.getElementById("x-kind"),
    xBackend: document.getElementById("x-backend"),
    xMethod: document.getElementById("x-method"),
    xModality: document.getElementById("x-modality"),
    xRisk: document.getElementById("x-risk"),
    xHorizon: document.getElementById("x-horizon"),
    xAgree: document.getElementById("x-agree"),
    xBox: document.getElementById("x-box"),
    xWarning: document.getElementById("x-warning"),
    xMs: document.getElementById("x-ms"),
    xModelStep: document.getElementById("x-model-step"),
    xTopProbs: document.getElementById("x-top-probs"),
    xImage: document.getElementById("x-image"),
    xPlaceholder: document.getElementById("x-placeholder"),
    imageBox: document.getElementById("image-box"),
    roiBox: document.getElementById("roi-box"),
    roiText: document.getElementById("roi-text"),
    candidateStrip: document.getElementById("candidate-strip"),
    instruction: document.getElementById("instruction"),
    strength: document.getElementById("strength"),
    btnSend: document.getElementById("btn-send"),
    btnPause: document.getElementById("btn-pause"),
    btnClearRoi: document.getElementById("btn-clear-roi"),
    btnUseLatest: document.getElementById("btn-use-latest"),
  };

  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  function toRoi(a, b) {
    return {
      x0: Math.min(a.x, b.x),
      y0: Math.min(a.y, b.y),
      x1: Math.max(a.x, b.x),
      y1: Math.max(a.y, b.y),
    };
  }

  function fixed(v, n) {
    return typeof v === "number" && Number.isFinite(v) ? v.toFixed(n) : "-";
  }

  function candidates() {
    return Array.isArray(state.forecast.candidates) ? state.forecast.candidates : [];
  }

  function selectedCandidate() {
    if (state.selectedSampleIdx === null) return null;
    return candidates().find((c) => c.sample_idx === state.selectedSampleIdx) || null;
  }

  function selectedCandidateXai() {
    if (state.selectedSampleIdx === null) return null;
    return state.candidateImages[state.selectedSampleIdx] || null;
  }

  function activeXai() {
    if (selectedCandidate()) {
      return selectedCandidateXai() || {};
    }
    return state.xai || {};
  }

  function imageUrl(payload) {
    if (!payload || !payload.png_b64) return "";
    return `data:image/png;base64,${payload.png_b64}`;
  }

  function displayRoi() {
    if (state.dragStart && state.dragCurrent) {
      return toRoi(state.dragStart, state.dragCurrent);
    }
    return state.roi;
  }

  function setText(el, value) {
    if (el) el.textContent = value;
  }

  function imageGeometry() {
    const box = els.imageBox.getBoundingClientRect();
    const img = els.xImage.getBoundingClientRect();
    if (img.width <= 0 || img.height <= 0 || els.xImage.hidden) {
      return {
        left: 0,
        top: 0,
        width: 0,
        height: 0,
      };
    }
    return {
      left: img.left - box.left,
      top: img.top - box.top,
      width: img.width,
      height: img.height,
    };
  }

  function renderRoi() {
    const roi = displayRoi();
    if (!roi) {
      els.roiBox.hidden = true;
      setText(els.roiText, "Not set");
      return;
    }
    const geom = imageGeometry();
    if (geom.width <= 0 || geom.height <= 0) {
      els.roiBox.hidden = true;
      setText(els.roiText, "Not set");
      return;
    }
    els.roiBox.hidden = false;
    els.roiBox.style.left = `${geom.left + roi.x0 * geom.width}px`;
    els.roiBox.style.top = `${geom.top + roi.y0 * geom.height}px`;
    els.roiBox.style.width = `${(roi.x1 - roi.x0) * geom.width}px`;
    els.roiBox.style.height = `${(roi.y1 - roi.y0) * geom.height}px`;
    setText(
      els.roiText,
      `[${roi.x0.toFixed(2)}, ${roi.y0.toFixed(2)}] → [${roi.x1.toFixed(2)}, ${roi.y1.toFixed(2)}]`,
    );
  }

  function renderCandidates() {
    const items = candidates();
    els.candidateStrip.innerHTML = "";
    for (const c of items) {
      const btn = document.createElement("button");
      btn.className = `candidate-btn ${state.selectedSampleIdx === c.sample_idx ? "active" : ""}`;
      btn.type = "button";
      btn.title = `sample ${c.sample_idx} risk ${fixed(c.risk_score, 3)}`;

      const thumb = state.candidateImages[c.sample_idx];
      if (thumb && thumb.png_b64) {
        const img = document.createElement("img");
        img.src = `data:image/png;base64,${thumb.png_b64}`;
        img.alt = "forecast";
        btn.appendChild(img);
      } else {
        const pending = document.createElement("div");
        pending.className = "thumb-pending";
        pending.textContent = "pending";
        btn.appendChild(pending);
      }

      const meta = document.createElement("div");
      meta.className = "thumb-meta";
      meta.textContent = `i${c.sample_idx} r${fixed(c.risk_score, 2)} h${c.horizon_epochs}`;
      btn.appendChild(meta);

      btn.addEventListener("click", () => {
        state.selectedSampleIdx = c.sample_idx;
        state.roi = null;
        state.dragStart = null;
        state.dragCurrent = null;
        render();
      });

      els.candidateStrip.appendChild(btn);
    }
  }

  function render() {
    const ax = activeXai();
    const sc = selectedCandidate();
    const reviewOpen =
      state.windowState.window_open ?? state.forecast.window_open ?? false;
    const reviewReason = state.windowState.reason ?? state.forecast.reason ?? "-";

    setText(els.statusConnected, `Status: ${state.connected ? "Connected" : "Disconnected"}`);
    setText(els.statusTraining, `Training: ${state.isPaused ? "Paused" : "Running"}`);
    setText(els.statusReview, `Review window: ${reviewOpen ? "Open" : "Closed"} (${reviewReason})`);

    setText(els.mStep, state.metrics.step ?? "-");
    setText(els.mBackend, state.metrics.backend ?? "-");
    setText(els.mLoss, fixed(state.metrics.loss, 4));
    setText(els.mAcc, fixed(state.metrics.acc, 4));
    setText(els.mFocus, fixed(state.metrics.focus_lambda, 3));
    setText(els.mSps, fixed(state.metrics.sps, 2));
    setText(els.mFStep, state.forecast.step ?? "-");
    setText(els.mFEpoch, fixed(state.forecast.epoch, 2));
    setText(els.mFCount, String(candidates().length));

    setText(els.xSample, sc?.sample_idx ?? ax.sample_idx ?? "-");
    setText(els.xTarget, sc?.target_class ?? ax.target_class ?? "-");
    setText(els.xPred, sc?.pred_class ?? ax.pred_class ?? "-");
    setText(els.xKind, ax.request_kind ?? "-");
    setText(els.xBackend, ax.backend ?? state.metrics.backend ?? "-");
    setText(els.xMethod, ax.xai_method ?? "-");

    if (ax.modality_focus && typeof ax.modality_focus === "object") {
      const modalityText = Object.entries(ax.modality_focus)
        .map(([k, v]) => `${k}:${fixed(v, 2)}`)
        .join(" | ");
      setText(els.xModality, modalityText || "-");
    } else {
      setText(els.xModality, "-");
    }

    const riskValue = sc?.risk_score;
    setText(els.xRisk, fixed(riskValue ?? ax.risk_score, 3));
    setText(els.xHorizon, String(sc?.horizon_epochs ?? ax.horizon_epochs ?? "-"));
    setText(els.xAgree, fixed(ax.xai_agreement, 3));
    if (Array.isArray(ax.focus_bbox) && ax.focus_bbox.length === 4) {
      const [x0, y0, x1, y1] = ax.focus_bbox;
      setText(els.xBox, `[${fixed(x0, 2)}, ${fixed(y0, 2)}] → [${fixed(x1, 2)}, ${fixed(y1, 2)}]`);
    } else {
      setText(els.xBox, "-");
    }
    els.xWarning.hidden = !(typeof ax.xai_agreement === "number" && ax.xai_agreement < 0.35);
    setText(els.xMs, fixed(ax.xai_ms, 1));
    setText(els.xModelStep, String(ax.model_step ?? "-"));

    if (Array.isArray(ax.top_classes) && Array.isArray(ax.top_probs) && ax.top_classes.length > 0) {
      const top = ax.top_classes
        .map((cls, i) => `c${cls}:${fixed(ax.top_probs[i], 2)}`)
        .join(" | ");
      setText(els.xTopProbs, top);
    } else {
      setText(els.xTopProbs, "-");
    }

    const url = imageUrl(ax);
    if (url) {
      els.xImage.src = url;
      els.xImage.hidden = false;
      els.xPlaceholder.hidden = true;
    } else {
      els.xImage.removeAttribute("src");
      els.xImage.hidden = true;
      els.xPlaceholder.hidden = false;
      els.xPlaceholder.textContent = sc
        ? `Waiting for XAI image for sample ${sc.sample_idx}…`
        : "Waiting for XAI image…";
    }

    els.btnPause.textContent = state.isPaused ? "Resume Training" : "Pause Training";

    renderRoi();
    renderCandidates();
  }

  function trimCandidateImages() {
    const keep = new Set(candidates().map((c) => c.sample_idx));
    if (keep.size === 0) {
      state.candidateImages = {};
      return;
    }
    const next = {};
    for (const [k, v] of Object.entries(state.candidateImages)) {
      const sid = Number(k);
      if (keep.has(sid)) {
        next[sid] = v;
      }
    }
    state.candidateImages = next;
  }

  function updateSelection() {
    const ids = candidates().map((c) => c.sample_idx);
    if (ids.length === 0) {
      state.selectedSampleIdx = null;
      return;
    }
    if (state.selectedSampleIdx !== null && ids.includes(state.selectedSampleIdx)) {
      return;
    }
    state.selectedSampleIdx = ids[0];
  }

  async function loadForecast(backend) {
    try {
      const params = backend ? `?backend=${encodeURIComponent(backend)}` : "";
      const res = await fetch(`/forecast/latest${params}`);
      if (!res.ok) return;
      const data = await res.json();
      if (!data.forecast) return;
      state.forecast = data.forecast;
      trimCandidateImages();
      updateSelection();
      render();
    } catch {
      // Ignore startup failures.
    }
  }

  function onMessage(msg) {
    const payload = msg.payload || {};
    if (msg.type === "metrics") {
      state.metrics = payload;
      if (payload.backend && payload.backend !== state.lastBackend) {
        state.lastBackend = payload.backend;
        void loadForecast(payload.backend);
      }
    } else if (msg.type === "feedback") {
      if (typeof payload.paused === "boolean") {
        state.isPaused = payload.paused;
      }
    } else if (msg.type === "trainer_paused") {
      state.isPaused = true;
    } else if (msg.type === "trainer_resumed") {
      state.isPaused = false;
    } else if (msg.type === "hitl_window") {
      state.windowState = payload;
    } else if (msg.type === "forecast_topk") {
      state.forecast = payload;
      state.windowState = {
        ...state.windowState,
        backend: payload.backend ?? state.windowState.backend,
        step: payload.step ?? state.windowState.step,
        epoch: payload.epoch ?? state.windowState.epoch,
        window_open: payload.window_open ?? state.windowState.window_open,
        reason: payload.reason ?? state.windowState.reason,
        candidate_count: Array.isArray(payload.candidates) ? payload.candidates.length : 0,
      };
      trimCandidateImages();
      updateSelection();
    } else if (msg.type === "xai_image") {
      state.xai = payload;
      if (payload.request_kind === "forecast" && typeof payload.sample_idx === "number" && payload.png_b64) {
        state.candidateImages[payload.sample_idx] = payload;
      }
    }
    render();
  }

  function connectWs() {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws`);

    ws.onopen = () => {
      state.connected = true;
      render();
    };

    ws.onclose = () => {
      state.connected = false;
      render();
      setTimeout(connectWs, 1000);
    };

    ws.onerror = () => {
      state.connected = false;
      render();
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        onMessage(msg);
      } catch {
        // Ignore malformed payloads.
      }
    };

    setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send("ping");
      }
    }, 3000);
  }

  function pointerToNorm(ev) {
    const rect = els.xImage.getBoundingClientRect();
    if (els.xImage.hidden || rect.width <= 0 || rect.height <= 0) return null;
    return {
      x: clamp01((ev.clientX - rect.left) / rect.width),
      y: clamp01((ev.clientY - rect.top) / rect.height),
    };
  }

  function commitDrag() {
    if (!state.dragStart || !state.dragCurrent) return;
    const roi = toRoi(state.dragStart, state.dragCurrent);
    const tooSmall = roi.x1 - roi.x0 < 0.01 || roi.y1 - roi.y0 < 0.01;
    state.roi = tooSmall ? null : roi;
    state.dragStart = null;
    state.dragCurrent = null;
    render();
  }

  async function postFeedback(payload) {
    await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async function sendFeedback() {
    const sc = selectedCandidate();
    const ax = activeXai();
    const sampleIdx = Number(sc?.sample_idx ?? ax?.sample_idx ?? state.xai?.sample_idx ?? 0);

    const payload = {
      step: Number(state.metrics.step ?? 0),
      sample_idx: Number.isFinite(sampleIdx) ? sampleIdx : 0,
      instruction: els.instruction.value || "",
    };

    const rawStrength = (els.strength.value || "").trim();
    if (rawStrength !== "") {
      const parsed = Number(rawStrength);
      if (Number.isFinite(parsed)) {
        const normalized = parsed > 1 ? parsed / 100 : parsed;
        payload.strength = clamp01(normalized);
      }
    }

    if (state.roi) {
      payload.roi = state.roi;
    }

    await postFeedback(payload);
    els.instruction.value = "";
  }

  async function togglePause() {
    const sc = selectedCandidate();
    const ax = activeXai();
    const sampleIdx = Number(sc?.sample_idx ?? ax?.sample_idx ?? state.xai?.sample_idx ?? 0);
    const paused = !state.isPaused;

    const payload = {
      step: Number(state.metrics.step ?? 0),
      sample_idx: Number.isFinite(sampleIdx) ? sampleIdx : 0,
      instruction: "",
      paused,
    };

    await postFeedback(payload);
    state.isPaused = paused;
    render();
  }

  els.btnSend.addEventListener("click", () => {
    void sendFeedback();
  });

  els.btnPause.addEventListener("click", () => {
    void togglePause();
  });

  els.btnClearRoi.addEventListener("click", () => {
    state.roi = null;
    state.dragStart = null;
    state.dragCurrent = null;
    render();
  });

  els.btnUseLatest.addEventListener("click", () => {
    state.selectedSampleIdx = null;
    state.roi = null;
    state.dragStart = null;
    state.dragCurrent = null;
    render();
  });

  els.imageBox.addEventListener("mousedown", (ev) => {
    if (!imageUrl(activeXai())) return;
    const p = pointerToNorm(ev);
    if (!p) return;
    state.dragStart = p;
    state.dragCurrent = p;
    renderRoi();
  });

  els.imageBox.addEventListener("mousemove", (ev) => {
    if (!state.dragStart) return;
    const p = pointerToNorm(ev);
    if (!p) return;
    state.dragCurrent = p;
    renderRoi();
  });

  els.imageBox.addEventListener("mouseup", () => {
    commitDrag();
  });

  els.imageBox.addEventListener("mouseleave", () => {
    if (state.dragStart) {
      commitDrag();
    }
  });

  void loadForecast();
  connectWs();
  render();
})();
