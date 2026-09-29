/* ============================================================
   AEGISVISION AI — INTERACTIVE FRONTEND APPLICATION LOGIC
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  // ------------------------------------------------------------
  // STATE MANAGEMENT
  // ------------------------------------------------------------
  const state = {
    webcamRunning: false,
    selectedModel: "h5",
    confidenceThresh: 0.5,
    audioAlertEnabled: true,
    stats: {
      scanned: 0,
      masked: 0,
      unmasked: 0,
      occluded: 0
    },
    logs: [],
    lastAlertTime: 0,
    lastFrameTime: performance.now(),
    latestFrameDetections: null
  };

  // MediaPipe Hand Skeletal Connections (21 Landmarks)
  const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index
    [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
    [9, 13], [13, 14], [14, 15], [15, 16],// Ring
    [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
    [0, 17]                               // Palm base
  ];

  // Offscreen canvas for capturing CLEAN video snapshots without green box overlay
  const offscreenCanvas = document.createElement("canvas");
  const offscreenCtx = offscreenCanvas.getContext("2d");

  // ------------------------------------------------------------
  // DOM ELEMENTS
  // ------------------------------------------------------------
  const videoElem = document.getElementById("webcam-video");
  const canvasElem = document.getElementById("webcam-canvas");
  const ctx = canvasElem.getContext("2d");

  const btnStartWebcam = document.getElementById("btn-start-webcam");
  const btnStopWebcam = document.getElementById("btn-stop-webcam");
  const modelSelect = document.getElementById("model-select");
  const confSlider = document.getElementById("conf-slider");
  const confValueDisplay = document.getElementById("conf-value");
  const toggleAudio = document.getElementById("toggle-audio");
  const liveLogsContainer = document.getElementById("live-logs");
  const fpsDisplay = document.getElementById("fps-display");

  // Stat Cards
  const statComplianceRate = document.getElementById("stat-compliance");
  const statTotalScanned = document.getElementById("stat-scanned");
  const statViolations = document.getElementById("stat-violations");
  const statOcclusions = document.getElementById("stat-occlusions");

  // Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  // Modal
  const infoModal = document.getElementById("info-modal");
  const btnOpenModal = document.getElementById("btn-open-modal");
  const btnCloseModal = document.getElementById("btn-close-modal");

  let audioCtx = null;

  function playAlertSound() {
    if (!state.audioAlertEnabled) return;
    const now = Date.now();
    if (now - state.lastAlertTime < 2000) return;
    state.lastAlertTime = now;

    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(880, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.3);

      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    } catch (e) {
      console.warn("Audio Context blocked or failed:", e);
    }
  }

  // ------------------------------------------------------------
  // TAB NAVIGATION & MODAL
  // ------------------------------------------------------------
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(targetTab).classList.add("active");
    });
  });

  btnOpenModal.addEventListener("click", () => infoModal.classList.add("active"));
  btnCloseModal.addEventListener("click", () => infoModal.classList.remove("active"));
  infoModal.addEventListener("click", (e) => {
    if (e.target === infoModal) infoModal.classList.remove("active");
  });

  modelSelect.addEventListener("change", (e) => {
    state.selectedModel = e.target.value;
    addLiveLog(`Model changed to: ${e.target.value.toUpperCase()}`, "info");
  });

  confSlider.addEventListener("input", (e) => {
    state.confidenceThresh = parseFloat(e.target.value);
    confValueDisplay.textContent = `${Math.round(state.confidenceThresh * 100)}%`;
  });

  toggleAudio.addEventListener("change", (e) => {
    state.audioAlertEnabled = e.target.checked;
  });

  // ------------------------------------------------------------
  // WEBCAM STREAMING & RENDER LOOP
  // ------------------------------------------------------------
  videoElem.addEventListener("loadedmetadata", () => {
    canvasElem.width = videoElem.videoWidth || 640;
    canvasElem.height = videoElem.videoHeight || 480;
    offscreenCanvas.width = videoElem.videoWidth || 640;
    offscreenCanvas.height = videoElem.videoHeight || 480;
  });

  btnStartWebcam.addEventListener("click", async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
      videoElem.srcObject = stream;
      await videoElem.play();

      const w = videoElem.videoWidth || 640;
      const h = videoElem.videoHeight || 480;
      canvasElem.width = w;
      canvasElem.height = h;
      offscreenCanvas.width = w;
      offscreenCanvas.height = h;

      state.webcamRunning = true;
      btnStartWebcam.style.display = "none";
      btnStopWebcam.style.display = "inline-flex";

      addLiveLog("Camera stream initiated", "info");
      requestAnimationFrame(processWebcamLoop);
    } catch (err) {
      alert("Could not access camera: " + err.message);
      console.error(err);
    }
  });

  btnStopWebcam.addEventListener("click", stopWebcam);

  function stopWebcam() {
    state.webcamRunning = false;
    state.latestFrameDetections = null;
    if (videoElem.srcObject) {
      videoElem.srcObject.getTracks().forEach(track => track.stop());
      videoElem.srcObject = null;
    }
    ctx.clearRect(0, 0, canvasElem.width, canvasElem.height);
    btnStartWebcam.style.display = "inline-flex";
    btnStopWebcam.style.display = "none";
    fpsDisplay.textContent = "0 FPS";
    addLiveLog("Camera stream stopped", "info");
  }

  let isProcessingFrame = false;

  async function processWebcamLoop() {
    if (!state.webcamRunning) return;

    // 1. Measure FPS
    const now = performance.now();
    const delta = now - state.lastFrameTime;
    state.lastFrameTime = now;
    const currentFps = Math.round(1000 / delta);
    fpsDisplay.textContent = `${currentFps} FPS`;

    // 2. Draw CLEAN video frame to OFFSCREEN canvas (No Bounding Boxes!)
    offscreenCtx.drawImage(videoElem, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

    // 3. Draw video frame + persistent detection overlays to VISIBLE canvas
    ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);
    if (state.latestFrameDetections) {
      drawDetectionsOnCanvas(state.latestFrameDetections);
    }

    // 4. Capture snapshot from CLEAN offscreen canvas and send to API
    if (!isProcessingFrame) {
      isProcessingFrame = true;
      try {
        const blob = await new Promise(resolve => offscreenCanvas.toBlob(resolve, 'image/jpeg', 0.7));
        if (blob && state.webcamRunning) {
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");
          formData.append("model", state.selectedModel);
          formData.append("conf", state.confidenceThresh);

          const response = await fetch("/api/predict/frame", {
            method: "POST",
            body: formData
          });

          if (response.ok) {
            const data = await response.json();
            state.latestFrameDetections = data;
            updateStatsAndLogs(data);
          }
        }
      } catch (e) {
        console.error("Frame inference error:", e);
      } finally {
        isProcessingFrame = false;
      }
    }

    if (state.webcamRunning) {
      requestAnimationFrame(processWebcamLoop);
    }
  }

  // ------------------------------------------------------------
  // CANVAS DRAWING (FACIAL BOXES + MEDIAPIPE HAND SKELETON)
  // ------------------------------------------------------------
  function drawDetectionsOnCanvas(data) {
    if (!data) return;

    // A. Draw MediaPipe Hand Skeletal Lines & Nodes
    if (data.hand_landmarks && data.hand_landmarks.length > 0) {
      data.hand_landmarks.forEach(landmarks => {
        // Draw Skeletal Lines connecting hand joints
        ctx.strokeStyle = "#00f2fe"; // Neon Cyan Lines
        ctx.lineWidth = 2.5;

        HAND_CONNECTIONS.forEach(([i, j]) => {
          const p1 = landmarks[i];
          const p2 = landmarks[j];
          if (p1 && p2) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        });

        // Draw Joint Nodes (Dots)
        landmarks.forEach(pt => {
          ctx.fillStyle = "#00e676"; // Emerald Green Nodes
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 4, 0, 2 * Math.PI);
          ctx.fill();
        });
      });
    }

    // B. Draw Hand Bounding Boxes
    if (data.hand_boxes && data.hand_boxes.length > 0) {
      ctx.strokeStyle = "#ffea00"; // Bright Yellow Box
      ctx.lineWidth = 2;
      data.hand_boxes.forEach(([hx1, hy1, hx2, hy2]) => {
        const hw = hx2 - hx1;
        const hh = hy2 - hy1;
        ctx.strokeRect(hx1, hy1, hw, hh);

        ctx.fillStyle = "#ffea00";
        ctx.font = "bold 11px sans-serif";
        ctx.fillRect(hx1, hy1 - 18, 110, 18);
        ctx.fillStyle = "#000000";
        ctx.fillText("HAND DETECTED", hx1 + 4, hy1 - 5);
      });
    }

    // C. Draw Face Bounding Boxes & Mask Labels
    if (data.faces && data.faces.length > 0) {
      data.faces.forEach(face => {
        const [sx, sy, ex, ey] = face.bbox;
        const w = ex - sx;
        const h = ey - sy;
        const rgbColor = `rgb(${face.color[0]}, ${face.color[1]}, ${face.color[2]})`;

        // Draw Rectangle
        ctx.strokeStyle = rgbColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(sx, sy, w, h);

        // Text & Background Badge
        let labelText = `${face.label}: ${face.confidence}%`;
        if (face.hand_occluded) {
          labelText += " ✋ [Occluded]";
        }

        ctx.font = "bold 13px 'Outfit', sans-serif";
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = rgbColor;
        ctx.fillRect(sx, sy - 24, textWidth + 12, 24);
        ctx.fillStyle = "#000000";
        ctx.fillText(labelText, sx + 6, sy - 7);
      });
    }
  }

  // ------------------------------------------------------------
  // STATS & AUDIT LOG UPDATES
  // ------------------------------------------------------------
  function updateStatsAndLogs(data) {
    if (!data.summary) return;
    const s = data.summary;

    if (s.total_faces > 0) {
      state.stats.scanned += s.total_faces;
      state.stats.masked += s.masked;
      state.stats.unmasked += s.no_mask;
      state.stats.occluded += s.hand_occluded;

      const totalCompliance = Math.round((state.stats.masked / state.stats.scanned) * 100);

      statComplianceRate.textContent = `${totalCompliance}%`;
      statTotalScanned.textContent = state.stats.scanned;
      statViolations.textContent = state.stats.unmasked;
      statOcclusions.textContent = state.stats.occluded;

      if (s.no_mask > 0 || s.hand_occluded > 0) {
        playAlertSound();
      }

      data.faces.forEach(face => {
        const timestamp = new Date().toLocaleTimeString();
        let logClass = "mask";
        if (face.label === "No Mask") logClass = "no-mask";
        if (face.hand_occluded) logClass = "occlusion";

        const logMsg = `${face.label} (${face.confidence}%) ${face.hand_occluded ? '[Hand Occluded]' : ''}`;
        addLiveLog(`[${timestamp}] Face Detected: ${logMsg}`, logClass);

        state.logs.unshift({
          time: timestamp,
          label: face.label,
          confidence: `${face.confidence}%`,
          occlusion: face.hand_occluded ? "YES" : "NO",
          model: s.model_used,
          latency: `${s.latency_ms} ms`
        });
      });

      renderAuditLogsTable();
    }
  }

  function addLiveLog(msg, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span>${msg}</span>`;
    liveLogsContainer.prepend(entry);

    if (liveLogsContainer.children.length > 20) {
      liveLogsContainer.removeChild(liveLogsContainer.lastChild);
    }
  }

  // ------------------------------------------------------------
  // IMAGE ANALYZER (TAB 2)
  // ------------------------------------------------------------
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("image-file-input");
  const imageResultsGrid = document.getElementById("image-results-grid");
  const imgAnnotatedPreview = document.getElementById("img-annotated-preview");
  const imgAnalysisMeta = document.getElementById("img-analysis-meta");

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleImageUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleImageUpload(e.target.files[0]);
    }
  });

  async function handleImageUpload(file) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", state.selectedModel);

    addLiveLog(`Uploading image: ${file.name}...`, "info");

    try {
      const response = await fetch("/api/predict/image", {
        method: "POST",
        body: formData
      });

      if (response.ok) {
        const data = await response.json();

        imgAnnotatedPreview.src = data.annotated_image;
        imageResultsGrid.style.display = "grid";

        const s = data.summary;
        imgAnalysisMeta.innerHTML = `
          <div class="panel-header">
            <h3>Image Analysis Report</h3>
            <p>Model: <strong>${s.model_used}</strong> | Latency: <strong>${s.latency_ms} ms</strong></p>
          </div>
          <div style="margin-top: 1rem; display: flex; flex-direction: column; gap: 0.5rem;">
            <div class="setting-item">
              <span>Faces Scanned:</span> <strong>${s.total_faces}</strong>
            </div>
            <div class="setting-item">
              <span>Masked:</span> <strong style="color: var(--accent-emerald)">${s.masked}</strong>
            </div>
            <div class="setting-item">
              <span>No Mask Violations:</span> <strong style="color: var(--accent-crimson)">${s.no_mask}</strong>
            </div>
            <div class="setting-item">
              <span>Hand Occlusions:</span> <strong style="color: var(--accent-amber)">${s.hand_occluded}</strong>
            </div>
            <div class="setting-item">
              <span>Compliance Score:</span> <strong>${s.compliance_rate}%</strong>
            </div>
          </div>
        `;

        updateStatsAndLogs(data);
      }
    } catch (e) {
      alert("Error analyzing image: " + e.message);
    }
  }

  // ------------------------------------------------------------
  // AUDIT LOGS TABLE & CSV EXPORT
  // ------------------------------------------------------------
  const logsTableBody = document.getElementById("logs-table-body");
  const btnExportCsv = document.getElementById("btn-export-csv");

  function renderAuditLogsTable() {
    if (!logsTableBody) return;
    logsTableBody.innerHTML = "";

    state.logs.slice(0, 30).forEach((item, idx) => {
      const tr = document.createElement("tr");
      let statusClass = item.label === "Mask" ? "mask" : "no-mask";

      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td>${item.time}</td>
        <td><span class="badge-status ${statusClass}">${item.label}</span></td>
        <td>${item.confidence}</td>
        <td>${item.occlusion}</td>
        <td>${item.model}</td>
        <td>${item.latency}</td>
      `;
      logsTableBody.appendChild(tr);
    });
  }

  btnExportCsv.addEventListener("click", () => {
    if (state.logs.length === 0) {
      alert("No audit logs available to export.");
      return;
    }

    let csvContent = "data:text/csv;charset=utf-8,Time,Label,Confidence,Hand Occluded,Model,Latency\n";
    state.logs.forEach(l => {
      csvContent += `${l.time},${l.label},${l.confidence},${l.occlusion},${l.model},${l.latency}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `AegisVision_Compliance_Report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });

  fetchSystemStatus();

  async function fetchSystemStatus() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        console.log("AegisVision Backend Telemetry Loaded:", data);
      }
    } catch (e) {
      console.warn("Backend API offline or starting up...", e);
    }
  }
});
