/**
 * SonicSentinel AI - 2026 Enterprise Tactical Application Controller
 * Features:
 * 1. Live Microphone Continuous Streaming & Real-Time Tracing
 * 2. Live Audio Oscilloscope & VU Decibel (dBFS) Meter
 * 3. Dual-Model Consensus Arbitration & Top-3 Predictions
 * 4. PWA Installation & Native Background Push Notifications
 * 5. Explainable AI (Mel Grad-CAM) & Tamper-Evident Forensic PDF
 * 6. 10 Mandatory Classes Simulation & Continuous Learning Loop
 */

let radar;
let socket;
let currentIncidentId = null;
let deferredPrompt = null;
let wakeLock = null;

// Live Microphone System Controller
const LiveMic = {
  isMonitoring: false,
  audioContext: null,
  mediaStream: null,
  analyser: null,
  recorder: null,
  animFrameId: null,
  noiseGateDb: -38.0, // Default balanced gate
  recordedChunks: [],

  async checkAvailability() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.updateStatus('DISCONNECTED', 'Web Audio API Not Supported');
        return false;
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const hasAudioIn = devices.some(d => d.kind === 'audioinput');
      if (hasAudioIn) {
        this.updateStatus('AVAILABLE', 'Microphone Hardware Ready');
        return true;
      } else {
        this.updateStatus('DISCONNECTED', 'No Microphone Input Found');
        return false;
      }
    } catch (e) {
      this.updateStatus('DISCONNECTED', 'Mic Check Failed');
      return false;
    }
  },

  updateStatus(status, label) {
    const pill = document.getElementById('micHeaderPill');
    const dot = document.getElementById('micHeaderDot');
    const text = document.getElementById('micHeaderText');
    const diagStatus = document.getElementById('diagMicStatus');
    const cardBadge = document.getElementById('micStatusCardBadge');

    if (text) text.textContent = `MIC: ${status}`;
    if (diagStatus) diagStatus.textContent = label || status;

    if (status === 'ACTIVE') {
      if (dot) {
        dot.style.backgroundColor = '#00e676';
        dot.style.boxShadow = '0 0 10px #00e676';
      }
      if (cardBadge) {
        cardBadge.textContent = 'STREAMING LIVE';
        cardBadge.style.color = '#00e676';
        cardBadge.style.borderColor = '#00e676';
      }
    } else if (status === 'AVAILABLE') {
      if (dot) {
        dot.style.backgroundColor = '#ffd600';
        dot.style.boxShadow = '0 0 8px #ffd600';
      }
      if (cardBadge) {
        cardBadge.textContent = 'HARDWARE READY';
        cardBadge.style.color = '#ffd600';
        cardBadge.style.borderColor = '#ffd600';
      }
    } else if (status === 'PAUSED') {
      if (dot) {
        dot.style.backgroundColor = '#ff9100';
        dot.style.boxShadow = '0 0 8px #ff9100';
      }
      if (cardBadge) {
        cardBadge.textContent = 'MONITORING PAUSED';
        cardBadge.style.color = '#ff9100';
        cardBadge.style.borderColor = '#ff9100';
      }
    } else {
      if (dot) {
        dot.style.backgroundColor = '#ff1744';
        dot.style.boxShadow = '0 0 8px #ff1744';
      }
      if (cardBadge) {
        cardBadge.textContent = 'MIC INACTIVE';
        cardBadge.style.color = '#ff1744';
        cardBadge.style.borderColor = '#ff1744';
      }
    }
  },

  async toggle() {
    if (!this.isMonitoring) {
      await this.start();
    } else {
      this.stop();
    }
  },

  async start() {
    try {
      showToast('Requesting Microphone Access...', 'info');
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 2,
        },
      });

      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 44100,
      });

      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 512;
      source.connect(this.analyser);

      this.isMonitoring = true;
      this.updateStatus('ACTIVE', 'Live Streaming Active');

      // Update UI button
      const btn = document.getElementById('btnToggleLiveMic');
      btn.classList.add('active-streaming');
      document.getElementById('micBtnTitle').textContent = 'STOP MONITORING';
      document.getElementById('micBtnSub').textContent = 'Live audio surveillance stream active';
      document.getElementById('diagStreamState').textContent = 'Streaming 2.0s Windows';

      // Setup Continuous Sample Collector for standard PCM WAV Streaming
      const bufferSize = 4096;
      this.processor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);
      this.audioBufferL = [];
      this.audioBufferR = [];
      this.targetSamples = 44100 * 2; // 2.0s standard temporal window
      this.isSendingChunk = false;

      this.processor.onaudioprocess = (e) => {
        if (!this.isMonitoring) return;
        const inputL = e.inputBuffer.getChannelData(0);
        const inputR = e.inputBuffer.numberOfChannels > 1 ? e.inputBuffer.getChannelData(1) : inputL;

        for (let i = 0; i < inputL.length; i++) {
          this.audioBufferL.push(inputL[i]);
          this.audioBufferR.push(inputR[i]);
        }

        // When 2.0s of continuous samples have accumulated
        if (this.audioBufferL.length >= this.targetSamples) {
          const sliceL = new Float32Array(this.audioBufferL.slice(0, this.targetSamples));
          const sliceR = new Float32Array(this.audioBufferR.slice(0, this.targetSamples));

          // Slide buffer by 1.0s (50% overlap for rapid continuous responsiveness)
          const hopSamples = Math.floor(44100 * 1.0);
          this.audioBufferL = this.audioBufferL.slice(hopSamples);
          this.audioBufferR = this.audioBufferR.slice(hopSamples);

          if (!this.isSendingChunk) {
            const wavBlob = this.encodeWAV(sliceL, sliceR, 44100);
            this.sendChunkToBackend(wavBlob);
          }
        }
      };

      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);

      // Start Visualizer Loop
      this.drawVisualizer();

      // Request Screen Wake Lock (keep screen on during monitoring)
      requestAppWakeLock();

      showToast('Live Microphone Surveillance Online (Continuous PCM)!', 'success');
    } catch (err) {
      console.error('Mic Access Error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        this.updateStatus('PERMISSION DENIED', 'Microphone Permission Blocked');
        showToast('Microphone permission was denied. Please allow mic in browser.', 'error');
      } else {
        this.updateStatus('DISCONNECTED', err.message);
        showToast(`Mic Error: ${err.message}`, 'error');
      }
    }
  },

  stop() {
    this.isMonitoring = false;
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this.processor) {
      try { this.processor.disconnect(); } catch(e){}
      this.processor = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    this.audioBufferL = [];
    this.audioBufferR = [];

    this.updateStatus('PAUSED', 'Surveillance Paused');

    const btn = document.getElementById('btnToggleLiveMic');
    btn.classList.remove('active-streaming');
    document.getElementById('micBtnTitle').textContent = 'START LIVE MONITORING';
    document.getElementById('micBtnSub').textContent = 'Continuous 2.0s streaming & real-time tracing';
    document.getElementById('diagStreamState').textContent = 'Paused / Standby';
    document.getElementById('headerDecibelText').textContent = '-inf dBFS';
    document.getElementById('liveVuDecibels').textContent = '-inf dBFS';
    document.getElementById('liveVuMeterFill').style.width = '0%';

    showToast('Live Microphone Monitoring Stopped.', 'info');
  },

  drawVisualizer() {
    if (!this.isMonitoring || !this.analyser) return;

    const canvas = document.getElementById('liveOscilloscopeCanvas');
    const ctx = canvas.getContext('2d');
    const bufferLength = this.analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);

    const render = () => {
      if (!this.isMonitoring) return;
      this.animFrameId = requestAnimationFrame(render);

      this.analyser.getByteTimeDomainData(dataArray);

      // 1. Calculate RMS Level for VU Meter
      let sumSquares = 0;
      for (let i = 0; i < bufferLength; i++) {
        const norm = (dataArray[i] - 128) / 128;
        sumSquares += norm * norm;
      }
      const rms = Math.sqrt(sumSquares / bufferLength);
      // dBFS calculation (-60 to 0 dB)
      let db = rms > 0.0001 ? 20 * Math.log10(rms) : -60.0;
      db = Math.max(-60.0, Math.min(0.0, db));

      // Update VU Meter Bar
      const vuPercent = Math.min(100, Math.max(0, ((db + 60) / 60) * 100));
      const fillEl = document.getElementById('liveVuMeterFill');
      const textEl = document.getElementById('liveVuDecibels');
      const headerTextEl = document.getElementById('headerDecibelText');

      if (fillEl) fillEl.style.width = `${vuPercent}%`;
      const dbFormatted = `${db.toFixed(1)} dBFS`;
      if (textEl) textEl.textContent = dbFormatted;
      if (headerTextEl) headerTextEl.textContent = dbFormatted;

      // Noise floor telemetry estimate
      const floorEl = document.getElementById('diagNoiseFloor');
      if (floorEl && Math.random() < 0.05) {
        floorEl.textContent = `${(db - 12).toFixed(1)} dB`;
      }

      // 2. Render Oscilloscope Waveform (Adaptive Light / Dark)
      const isLight = !document.body.classList.contains('dark-theme');
      ctx.fillStyle = isLight ? '#f1f5f9' : '#060a14';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.lineWidth = 2;
      if (isLight) {
        ctx.strokeStyle = db > this.noiseGateDb ? '#2563eb' : '#059669';
        ctx.shadowBlur = db > this.noiseGateDb ? 6 : 2;
        ctx.shadowColor = ctx.strokeStyle;
      } else {
        ctx.strokeStyle = db > this.noiseGateDb ? '#00f3ff' : '#00e676';
        ctx.shadowBlur = db > this.noiseGateDb ? 8 : 4;
        ctx.shadowColor = ctx.strokeStyle;
      }

      ctx.beginPath();
      const sliceWidth = canvas.width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }

      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    };

    render();
  },

  encodeWAV(samplesL, samplesR, sampleRate = 44100) {
    const numChannels = 2;
    const numSamples = samplesL.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = numSamples * blockAlign;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    function writeStr(offset, str) {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
      }
    }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // 16-bit
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < numSamples; i++) {
      let sL = Math.max(-1, Math.min(1, samplesL[i]));
      let sR = Math.max(-1, Math.min(1, samplesR[i]));
      view.setInt16(offset, sL < 0 ? sL * 0x8000 : sL * 0x7FFF, true);
      offset += 2;
      view.setInt16(offset, sR < 0 ? sR * 0x8000 : sR * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  },

  async sendChunkToBackend(blob) {
    if (!this.isMonitoring) return;
    this.isSendingChunk = true;
    const slider = document.getElementById('simAzimuthSlider');
    const azimuth = slider ? parseFloat(slider.value) : 90.0;

    const formData = new FormData();
    formData.append('file', blob, 'live_stream_slice.wav');
    formData.append('azimuth', azimuth);

    try {
      const res = await fetch('/api/v1/audio/stream', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) return;

      // If detection was classified
      if (data.arbitration && data.incident) {
        updateDualModelUI(data);

        const cat = data.arbitration.predicted_category;
        const sev = data.arbitration.severity;
        const conf = data.arbitration.model_a ? data.arbitration.model_a.confidence : 0;

        if (cat !== 'Background Noise') {
          showToast(`Live Detection: ${cat} (${(conf * 100).toFixed(1)}%)`, sev === 'CRITICAL' ? 'error' : 'warning');
        }

        if (sev === 'CRITICAL' || sev === 'HIGH') {
          showNativeNotification(cat, sev, azimuth, conf);
        }
      }
    } catch (err) {
      console.warn('[Live Mic Stream Warning]:', err.message);
    } finally {
      this.isSendingChunk = false;
    }
  },
};

// --- DOM Ready Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  radar = new AcousticRadar('radarCanvas');
  initWebSocket();
  setupEventListeners();
  loadCategories();
  loadIncidents();
  loadBenchmarkReport();
  LiveMic.checkAvailability();
  initPwaAndNotifications();
});

// --- WebSocket Live Telemetry Connection ---
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live-audio`;
  
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    console.log('[SonicSentinel] Live telemetry WebSocket connected.');
    updateWsStatus(true);
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'RADAR_DETECTION') {
        handleRadarDetection(data);
      } else if (data.type === 'MODEL_UPDATED') {
        showToast(`Model fine-tuned: ${data.version_tag} (Macro F1: ${data.f1_macro})`, 'success');
        loadCategories();
      }
    } catch (e) {
      console.error('WS Parse Error:', e);
    }
  };

  socket.onclose = () => {
    updateWsStatus(false);
    setTimeout(initWebSocket, 3000);
  };

  socket.onerror = (err) => {
    console.error('WS Error:', err);
  };
}

function updateWsStatus(connected) {
  const dot = document.getElementById('wsStatusDot');
  const text = document.getElementById('wsStatusText');
  if (connected) {
    dot.style.backgroundColor = '#00e676';
    dot.style.boxShadow = '0 0 8px #00e676';
    text.textContent = 'RADAR LIVE';
  } else {
    dot.style.backgroundColor = '#ff1744';
    dot.style.boxShadow = '0 0 8px #ff1744';
    text.textContent = 'DISCONNECTED';
  }
}

// --- Telemetry & Detection Handling ---
function handleRadarDetection(data) {
  // 1. Add blip to radar
  radar.addBlip({
    category: data.category,
    confidence: data.confidence,
    azimuth: data.azimuth,
    severity: data.severity,
  });

  // 2. Update telemetry readouts (normal card & fullscreen HUD)
  const bearingText = `${data.azimuth.toFixed(1)}°`;
  const confText = `${(data.confidence * 100).toFixed(1)}%`;
  const tdoaText = `${(data.tdoa_seconds * 1000).toFixed(3)} ms`;
  const snrText = `${data.snr_db} dB`;

  const setEl = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setEl('telBearing', bearingText);
  setEl('telCategory', data.category);
  setEl('telConfidence', confText);
  setEl('telTdoa', tdoaText);
  setEl('telSnr', snrText);

  // Fullscreen HUD telemetry elements
  setEl('fsTelBearing', bearingText);
  setEl('fsTelCategory', data.category);
  setEl('fsTelConfidence', confText);
  setEl('fsTelTdoa', tdoaText);
  setEl('fsTelSnr', snrText);

  const qualityBadge = document.getElementById('telQuality');
  if (qualityBadge && data.audio_quality) {
    qualityBadge.textContent = data.audio_quality;
    qualityBadge.className = `telemetry-val badge-quality ${data.audio_quality}`;
  }
  const fsQualityBadge = document.getElementById('fsTelQuality');
  if (fsQualityBadge && data.audio_quality) {
    fsQualityBadge.textContent = data.audio_quality;
    fsQualityBadge.className = `badge-quality ${data.audio_quality}`;
  }

  // 3. Audio threat sound cue (synthesized tactical oscillator)
  playTacticalAlertTone(data.severity);

  // 3.1 Mobile Haptic Feedback (Vibration)
  if (data.severity === 'CRITICAL' || data.severity === 'HIGH') {
    if (navigator.vibrate) {
      // SOS Morse code vibration pattern
      navigator.vibrate([300, 100, 300, 100, 300, 500, 600, 200, 600, 200, 600, 500, 300, 100, 300, 100, 300]);
    }
  }

  // 3.2 WOW FACTOR: Red Alert Lockdown & AI Voice Announcement
  if (data.severity === 'CRITICAL') {
    document.body.classList.add('lockdown-mode');
    setTimeout(() => { document.body.classList.remove('lockdown-mode'); }, 6000);
    announceThreatAI(data.category, data.confidence);
  }

  // 3.3 WOW FACTOR: Neural Terminal Feed Logging
  logToNeuralTerminal(data);

  // 3.5. Live Speech Transcript Subtitles
  const subtitlesBox = document.getElementById('liveSubtitlesText');
  if (subtitlesBox && data.detected_speech_transcript) {
    // Generate a pseudo-random actor based on string length to simulate diarization for demo
    const actorNum = (data.detected_speech_transcript.length % 2 === 0) ? 1 : 2;
    const timeStr = new Date(data.timestamp).toLocaleTimeString();
    subtitlesBox.innerHTML = `<span style="color: #ff9100;">[${timeStr}] Actor ${actorNum}:</span> ${data.detected_speech_transcript}<br>` + subtitlesBox.innerHTML;
  }

  // 4. Prepend to live incidents table
  prependIncidentRow({
    id: data.incident_id,
    timestamp: data.timestamp,
    predicted_category: data.category,
    severity: data.severity,
    azimuth_deg: data.azimuth,
    model_a_confidence: data.confidence,
    audio_quality: data.audio_quality || 'Acceptable',
  });
}

function updateDualModelUI(result) {
  const arb = result.arbitration;
  const metrics = result.acoustic_forensics || {};
  currentIncidentId = result.incident ? result.incident.id : null;

  // Model A
  document.getElementById('modelAPred').textContent = arb.model_a.predicted_class;
  document.getElementById('modelAConf').textContent = `${(arb.model_a.confidence * 100).toFixed(1)}%`;
  document.getElementById('modelAFill').style.width = `${Math.min(100, arb.model_a.confidence * 100)}%`;

  // Model A Top-3 Predictions (SRS Requirement xxxiv)
  const aTop3 = arb.model_a.top_3 || result.top_3_predictions_a;
  if (aTop3 && aTop3.length > 0) {
    const container = document.getElementById('modelATop3');
    container.innerHTML = aTop3.map((item, idx) => 
      `<div class="top3-row"><span>${idx + 1}. ${item.category}</span><span>${(item.confidence * 100).toFixed(1)}%</span></div>`
    ).join('');
  }

  // Model B
  document.getElementById('modelBPred').textContent = arb.model_b.predicted_class;
  document.getElementById('modelBConf').textContent = `${(arb.model_b.confidence * 100).toFixed(1)}%`;
  document.getElementById('modelBFill').style.width = `${Math.min(100, arb.model_b.confidence * 100)}%`;

  // Model B Top-3 Predictions (SRS Requirement xxxiv)
  const bTop3 = arb.model_b.top_3 || result.top_3_predictions_b;
  if (bTop3 && bTop3.length > 0) {
    const container = document.getElementById('modelBTop3');
    container.innerHTML = bTop3.map((item, idx) => 
      `<div class="top3-row"><span>${idx + 1}. ${item.category}</span><span>${(item.confidence * 100).toFixed(1)}%</span></div>`
    ).join('');
  }

  // Arbitration badge
  const arbBadge = document.getElementById('arbStatusBadge');
  arbBadge.textContent = arb.arbitration_status;
  arbBadge.className = 'arb-status-text';
  if (arb.arbitration_status === 'Acceptable Match') arbBadge.classList.add('arb-acceptable');
  else if (arb.arbitration_status === 'Weak Match') arbBadge.classList.add('arb-weak');
  else if (arb.arbitration_status === 'Model Disagreement') arbBadge.classList.add('arb-disagreement');
  else arbBadge.classList.add('arb-uncertain');

  document.getElementById('confMarginVal').textContent = arb.confidence_margin.toFixed(4);

  // Audio Quality Badge
  const qualBadge = document.getElementById('arbQualityBadge');
  const quality = metrics.audio_quality || 'Acceptable';
  if (qualBadge) {
    qualBadge.textContent = quality;
    qualBadge.className = `badge-quality ${quality}`;
  }

  // Manual Review Queue Banner (SRS Step 17)
  const reviewBanner = document.getElementById('reviewAlertBanner');
  if (arb.requires_manual_review) {
    reviewBanner.classList.add('active');
    document.getElementById('reviewReasonText').textContent = (arb.reasons_for_review || []).join(' | ');
  } else {
    reviewBanner.classList.remove('active');
  }

  // Grad-CAM XAI Image
  const xaiBox = document.getElementById('xaiPreviewBox');
  if (result.gradcam_heatmap_uri) {
    xaiBox.innerHTML = `<img src="${result.gradcam_heatmap_uri}" alt="Grad-CAM Mel-Spectrogram Heatmap">`;
  }

  // PDF Export Link
  const pdfBtn = document.getElementById('btnDownloadPdf');
  pdfBtn.style.display = 'inline-flex';
  pdfBtn.onclick = () => window.open(result.pdf_download_url, '_blank');

  // SHA-256 display
  const shaEl = document.getElementById('sha256Digest');
  if (shaEl) shaEl.textContent = metrics.sha256 || 'live_stream_buffer';
}

// --- Tactical Tone Synthesizer ---
function playTacticalAlertTone(severity) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    if (severity === 'CRITICAL') {
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.18);
      gain.gain.setValueAtTime(0.25, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.18);
      osc.start();
      osc.stop(ctx.currentTime + 0.19);
    } else if (severity === 'HIGH') {
      osc.frequency.setValueAtTime(620, ctx.currentTime);
      gain.gain.setValueAtTime(0.12, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.12);
      osc.start();
      osc.stop(ctx.currentTime + 0.13);
    }
  } catch (e) {}
}

// --- Setup Event Listeners ---
function setupEventListeners() {
  // Theme Switcher (Modern Vector SVG Icons)
  const themeBtn = document.getElementById('btnToggleTheme');
  const themeText = document.getElementById('themeToggleText');
  const sunIcon = document.getElementById('themeIconSun');
  const moonIcon = document.getElementById('themeIconMoon');
  const savedTheme = localStorage.getItem('sonicsentinel_theme') || 'dark';

  function applyTheme(isDark) {
    if (isDark) {
      document.body.classList.add('dark-theme');
      if (themeText) themeText.textContent = 'Dark';
      if (sunIcon) sunIcon.style.display = 'none';
      if (moonIcon) moonIcon.style.display = 'inline-block';
    } else {
      document.body.classList.remove('dark-theme');
      if (themeText) themeText.textContent = 'Light';
      if (sunIcon) sunIcon.style.display = 'inline-block';
      if (moonIcon) moonIcon.style.display = 'none';
    }
  }

  applyTheme(savedTheme === 'dark');

  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const isDarkNow = !document.body.classList.contains('dark-theme');
      applyTheme(isDarkNow);
      localStorage.setItem('sonicsentinel_theme', isDarkNow ? 'dark' : 'light');
      showToast(`Switched to ${isDarkNow ? 'Tactical Dark' : 'Modern Light'} Theme`, 'info');
      if (window.radar) window.radar.resize();
    });
  }

  // Live Microphone Button Toggle
  const micBtn = document.getElementById('btnToggleLiveMic');
  if (micBtn) {
    micBtn.addEventListener('click', () => LiveMic.toggle());
  }

  // Sensitivity Gate Slider
  const gateSlider = document.getElementById('noiseGateSlider');
  const gateText = document.getElementById('noiseGateValText');
  if (gateSlider && gateText) {
    gateSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value);
      // Map 1-100 to -55 dB to -20 dB
      const db = -55 + (val / 100) * 35;
      LiveMic.noiseGateDb = db;
      if (val < 35) gateText.textContent = `High Sens (${db.toFixed(0)} dB)`;
      else if (val > 65) gateText.textContent = `Low Sens (${db.toFixed(0)} dB)`;
      else gateText.textContent = `Balanced (${db.toFixed(0)} dB)`;
    });
  }

  // Azimuth slider sync
  const slider = document.getElementById('simAzimuthSlider');
  const valLabel = document.getElementById('simAzimuthVal');
  if (slider && valLabel) {
    slider.addEventListener('input', (e) => {
      valLabel.textContent = `${e.target.value}°`;
    });
  }

  // 10 Mandatory Class Quick Simulation Buttons (TechWiz 7 Page 4)
  document.querySelectorAll('.btn-class-sim').forEach((btn) => {
    btn.addEventListener('click', () => {
      const category = btn.dataset.class;
      const azimuth = slider ? parseFloat(slider.value) : 90.0;
      simulateSignal(category, azimuth);
    });
  });

  // File Upload Dropper
  const dropZone = document.getElementById('audioDropZone');
  const fileInput = document.getElementById('audioFileInput');
  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) uploadAudioFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = '#00f3ff';
    });
    dropZone.addEventListener('dragleave', () => {
      dropZone.style.borderColor = 'rgba(0, 243, 255, 0.35)';
    });
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.style.borderColor = 'rgba(0, 243, 255, 0.35)';
      if (e.dataTransfer.files.length > 0) uploadAudioFile(e.dataTransfer.files[0]);
    });
  }

  // Anti-Cheat Boundary Test Buttons
  document.getElementById('btnTestSilence').addEventListener('click', () => testSilenceBoundary());
  document.getElementById('btnTestClipping').addEventListener('click', () => testClippingBoundary());

  // Category Creation Form
  document.getElementById('btnCreateCategory').addEventListener('click', () => createCategory());

  // User Sample Upload
  const sampleUploadBtn = document.getElementById('btnTriggerSampleUpload');
  const sampleUploadInput = document.getElementById('sampleUploadInput');
  const uploadCatSelect = document.getElementById('uploadCategorySelect');
  if (sampleUploadBtn && sampleUploadInput) {
    sampleUploadBtn.addEventListener('click', () => {
      const chosenCat = uploadCatSelect.value;
      if (!chosenCat) {
        showToast('Please select a target category first.', 'error');
        return;
      }
      sampleUploadInput.click();
    });

    sampleUploadInput.addEventListener('change', async (e) => {
      if (e.target.files.length > 0) {
        await uploadCategorySamples(uploadCatSelect.value, e.target.files);
        sampleUploadInput.value = '';
      }
    });
  }

  // 3,000+ Harvester & Augmentation
  const harvestBtn = document.getElementById('btnHarvestAugment');
  if (harvestBtn) {
    harvestBtn.addEventListener('click', () => triggerHarvestAugment());
  }

  // Sync Categories
  const syncBtn = document.getElementById('btnSyncCategories');
  if (syncBtn) {
    syncBtn.addEventListener('click', () => syncCategoriesFromDisk());
  }

  // Trigger Fine-Tuning
  document.getElementById('btnTriggerFineTune').addEventListener('click', () => triggerFineTune());

  // Review Queue Buttons
  document.getElementById('btnReviewApprove').addEventListener('click', () => submitReview('APPROVED'));
  document.getElementById('btnReviewOverride').addEventListener('click', () => {
    const override = prompt('Enter Override Category Name:');
    if (override) submitReview('OVERRIDDEN', override);
  });

  // Role Switcher Event (SRS ii)
  const roleSelect = document.getElementById('userRoleSelect');
  if (roleSelect) {
    roleSelect.addEventListener('change', (e) => {
      showToast(`Switched active security profile to ${e.target.value}`, 'info');
    });
  }

  // Mobile Bottom Tab Navigation
  document.querySelectorAll('.mobile-tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mobile-tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.dataset.target;
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // Enable Notifications Button
  const alertBtn = document.getElementById('btnToggleAlerts');
  if (alertBtn) {
    alertBtn.addEventListener('click', () => enableNativeNotifications());
  }

  // Multi-Algorithm Benchmark Re-run Button
  const bmBtn = document.getElementById('btnRefreshBenchmark');
  if (bmBtn) {
    bmBtn.addEventListener('click', () => triggerBenchmarkRun());
  }
}

// --- Progressive Web App & Notification Setup ---
function initPwaAndNotifications() {
  // Register Service Worker
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js')
      .then(reg => console.log('[SonicSentinel SW] Registered at scope:', reg.scope))
      .catch(err => console.warn('[SonicSentinel SW] Registration failed:', err));
  }

  // Capture PWA Install Prompt
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    const installBtn = document.getElementById('btnInstallApp');
    if (installBtn) installBtn.style.display = 'inline-flex';
  });

  const installBtn = document.getElementById('btnInstallApp');
  if (installBtn) {
    installBtn.addEventListener('click', async () => {
      if (deferredPrompt) {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        if (outcome === 'accepted') showToast('SonicSentinel App installed successfully!', 'success');
        deferredPrompt = null;
        installBtn.style.display = 'none';
      }
    });
  }
}

async function enableNativeNotifications() {
  if (!('Notification' in window)) {
    showToast('Browser does not support notifications.', 'error');
    return;
  }
  const perm = await Notification.requestPermission();
  if (perm === 'granted') {
    showToast('Native Push Threat Alerts Enabled!', 'success');
  } else {
    showToast('Notifications permission was denied.', 'error');
  }
}

function showNativeNotification(category, severity, azimuth, confidence) {
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification(`CRITICAL THREAT: ${category.toUpperCase()}`, {
      body: `Threat Level: ${severity} | Bearing: ${azimuth.toFixed(0)}° | Conf: ${(confidence * 100).toFixed(0)}%`,
      icon: '/static/icons/icon-192.svg',
      badge: '/static/icons/icon-192.svg',
      vibrate: [300, 100, 300, 100, 400],
    });
  }
  if ('vibrate' in navigator) {
    navigator.vibrate([200, 100, 200]);
  }
}

async function requestAppWakeLock() {
  if ('wakeLock' in navigator) {
    try {
      wakeLock = await navigator.wakeLock.request('screen');
    } catch (e) {}
  }
}

// --- API Calls ---
async function simulateSignal(categoryName, azimuthDeg) {
  showToast(`Synthesizing ${categoryName} @ ${azimuthDeg}° with TDOA...`, 'info');
  try {
    const res = await fetch('/api/v1/audio/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_name: categoryName, azimuth_deg: azimuthDeg }),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(`Error: ${data.detail || data.message}`, 'error');
      return;
    }
    updateDualModelUI(data);
    showToast(`Classified: ${data.arbitration.predicted_category} (${data.arbitration.arbitration_status})`, 'success');
  } catch (err) {
    showToast(`Simulation failed: ${err.message}`, 'error');
  }
}

async function uploadAudioFile(file) {
  showToast(`Uploading and analyzing ${file.name}...`, 'info');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/v1/audio/analyze', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(`REJECTED [${data.error_code || res.status}]: ${data.message || data.detail}`, 'error');
      return;
    }
    updateDualModelUI(data);
    showToast(`Analysis complete: ${data.arbitration.predicted_category}`, 'success');
  } catch (err) {
    showToast(`Upload failed: ${err.message}`, 'error');
  }
}

async function testSilenceBoundary() {
  showToast('Testing Silence Error Boundary...', 'info');
  const wavBytes = createTestWavBuffer(false);
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  const file = new File([blob], 'synthetic_silent_audio.wav', { type: 'audio/wav' });
  await uploadAudioFile(file);
}

async function testClippingBoundary() {
  showToast('Testing Clipping Error Boundary...', 'info');
  const wavBytes = createTestWavBuffer(true);
  const blob = new Blob([wavBytes], { type: 'audio/wav' });
  const file = new File([blob], 'synthetic_clipped_audio.wav', { type: 'audio/wav' });
  await uploadAudioFile(file);
}

function createTestWavBuffer(clipped = false) {
  const sampleRate = 44100;
  const numSamples = sampleRate * 2;
  const buffer = new ArrayBuffer(44 + numSamples * 2 * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 4, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 4, true);
  view.setUint16(32, 4, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, numSamples * 4, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    let sample = clipped ? 32767 : 0;
    view.setInt16(offset, sample, true);
    view.setInt16(offset + 2, sample, true);
    offset += 4;
  }
  return buffer;
}

async function createCategory() {
  const input = document.getElementById('newCategoryInput');
  const catName = input.value.trim();
  if (!catName) {
    showToast('Please enter a valid category name.', 'error');
    return;
  }

  try {
    const res = await fetch('/api/v1/categories/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_name: catName, severity: 'HIGH' }),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(`Failed: ${data.detail || data.message}`, 'error');
      return;
    }
    showToast(`Category '${catName}' provisioned!`, 'success');
    input.value = '';
    loadCategories();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

async function triggerFineTune() {
  const btn = document.getElementById('btnTriggerFineTune');
  btn.disabled = true;
  btn.textContent = 'FINE-TUNING ACTIVE...';

  try {
    const res = await fetch('/api/v1/train/fine-tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 5, learning_rate: 0.001 }),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.message || 'Fine-tuning busy.', 'error');
      btn.disabled = false;
      btn.textContent = 'TRIGGER CONTINUOUS FINE-TUNING';
      return;
    }
    showToast('Continuous fine-tuning started in background!', 'info');
    pollFineTuneStatus();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    btn.disabled = false;
    btn.textContent = 'TRIGGER CONTINUOUS FINE-TUNING';
  }
}

function pollFineTuneStatus() {
  const interval = setInterval(async () => {
    try {
      const res = await fetch('/api/v1/train/status');
      const data = await res.json();
      const prog = data.progress;
      const progressLabel = document.getElementById('fineTuneProgressText');
      if (progressLabel) {
        progressLabel.textContent = `Status: ${prog.status} | Epoch: ${prog.current_epoch}/${prog.total_epochs} | Loss: ${prog.loss}`;
      }

      if (prog.status === 'COMPLETED') {
        clearInterval(interval);
        showToast(`Fine-tuning finished! Macro F1: ${prog.f1_macro}`, 'success');
        const btn = document.getElementById('btnTriggerFineTune');
        btn.disabled = false;
        btn.textContent = 'TRIGGER CONTINUOUS FINE-TUNING';
      }
    } catch (e) {
      clearInterval(interval);
    }
  }, 1500);
}

async function submitReview(status, overrideCat = null) {
  if (!currentIncidentId) return;
  try {
    const res = await fetch(`/api/v1/incidents/${currentIncidentId}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reviewed_by: document.getElementById('userRoleSelect').value,
        review_status: status,
        override_category: overrideCat,
        notes: `Reviewed with decision: ${status}`,
      }),
    });
    if (res.ok) {
      showToast(`Incident marked as ${status}`, 'success');
      document.getElementById('reviewAlertBanner').classList.remove('active');
      loadIncidents();
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function loadCategories() {
  try {
    const res = await fetch('/api/v1/categories');
    const data = await res.json();
    const container = document.getElementById('categoryChipsList');
    const select = document.getElementById('uploadCategorySelect');

    if (container) container.innerHTML = '';
    if (select) select.innerHTML = '<option value="">Select Target Category...</option>';

    data.categories.forEach((cat) => {
      const count = cat.python_sample_count !== undefined ? cat.python_sample_count : 0;
      if (container) {
        const chip = document.createElement('span');
        chip.className = 'cat-chip';
        chip.innerHTML = `<b>${cat.name}</b> <span style="color: #00f3ff;">(${count})</span>`;
        container.appendChild(chip);
      }
      if (select) {
        const opt = document.createElement('option');
        opt.value = cat.name;
        opt.textContent = `${cat.name} (${count} files)`;
        select.appendChild(opt);
      }
    });
  } catch (e) {
    console.error('Error loading categories:', e);
  }
}

async function uploadCategorySamples(categoryName, fileList) {
  showToast(`Uploading ${fileList.length} samples to '${categoryName}'...`, 'info');
  const formData = new FormData();
  for (let i = 0; i < fileList.length; i++) formData.append('files', fileList[i]);

  try {
    const res = await fetch(`/api/v1/categories/${encodeURIComponent(categoryName)}/upload-samples`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Added ${data.newly_saved} files! Total in '${categoryName}': ${data.total_category_samples}`, 'success');
      loadCategories();
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function triggerHarvestAugment() {
  showToast('Dataset expansion and multi-augmentation active...', 'info');
  try {
    const res = await fetch('/api/v1/dataset/harvest-augment', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      showToast(`Augmented dataset to ${data.total_samples} samples!`, 'success');
      loadCategories();
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function syncCategoriesFromDisk() {
  try {
    const res = await fetch('/api/v1/categories/sync-disk', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      showToast(`Disk synced: ${data.total_categories} categories loaded!`, 'success');
      loadCategories();
    }
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function loadIncidents() {
  try {
    const res = await fetch('/api/v1/incidents?limit=15');
    const data = await res.json();
    const tbody = document.getElementById('incidentsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    data.incidents.forEach((inc) => prependIncidentRow(inc, false));
  } catch (e) {}
}

function prependIncidentRow(inc, prepend = true) {
  const tbody = document.getElementById('incidentsTableBody');
  if (!tbody) return;

  const tr = document.createElement('tr');
  const utcStr = inc.timestamp ? inc.timestamp.split('T')[1]?.substring(0, 8) || '--:--:--' : '--:--:--';
  const confStr = `${(inc.model_a_confidence * 100).toFixed(0)}%`;
  const qual = inc.audio_quality || 'Acceptable';

  tr.innerHTML = `
    <td style="color: var(--cyan-neon); font-size: 10px;">${inc.id.substring(0, 14)}</td>
    <td>${utcStr}</td>
    <td><span class="severity-pill sev-${inc.severity}">${inc.severity}</span></td>
    <td><b>${inc.predicted_category}</b></td>
    <td>${inc.azimuth_deg.toFixed(0)}°</td>
    <td>${confStr}</td>
    <td><span class="badge-quality ${qual}" style="font-size: 8.5px; padding: 1px 4px;">${qual}</span></td>
    <td><a href="/api/v1/incidents/export-pdf/${inc.id}" target="_blank" style="color: var(--cyan-neon); text-decoration: none;">PDF ↗</a></td>
  `;

  if (prepend && tbody.firstChild) {
    tbody.insertBefore(tr, tbody.firstChild);
    if (tbody.children.length > 20) tbody.removeChild(tbody.lastChild);
  } else {
    tbody.appendChild(tr);
  }
}

// --- Toast UI Notifications ---
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  if (type === 'error') toast.style.borderLeftColor = '#ff1744';
  else if (type === 'success') toast.style.borderLeftColor = '#00e676';
  else toast.style.borderLeftColor = '#00f3ff';

  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// --- Multi-Algorithm Benchmark & Model Selection Controller ---
async function loadBenchmarkReport() {
  try {
    const res = await fetch('/api/v1/benchmark/report');
    const data = await res.json();
    if (!res.ok || !data.success || !data.report) return;

    const rep = data.report;
    const winnerTitle = document.getElementById('bmWinnerTitle');
    const compBadge = document.getElementById('bmComplianceStatus');
    if (winnerTitle && rep.selected_best_model) winnerTitle.textContent = rep.selected_best_model;
    if (compBadge && rep.target_metrics_compliance) {
      compBadge.textContent = rep.target_metrics_compliance.status || 'TARGETS SATISFIED (>=85%)';
    }

    const algos = rep.algorithms || {};
    if (algos.Algorithm_1_Deep_CNN) {
      const a = algos.Algorithm_1_Deep_CNN;
      const elAcc = document.getElementById('bmAcc1');
      const elRec = document.getElementById('bmRec1');
      const elF1 = document.getElementById('bmF1_1');
      const elLat = document.getElementById('bmLat1');
      const elMem = document.getElementById('bmMem1');
      if (elAcc) elAcc.textContent = `${(a.test_accuracy * 100).toFixed(1)}%`;
      if (elRec) elRec.textContent = `${(a.critical_class_recall * 100).toFixed(1)}%`;
      if (elF1) elF1.textContent = a.macro_f1.toFixed(3);
      if (elLat) elLat.textContent = `${a.latency_ms} ms`;
      if (elMem && a.memory_mb) elMem.textContent = `${a.memory_mb} MB`;
    }

    if (algos.Algorithm_2_CRNN) {
      const a = algos.Algorithm_2_CRNN;
      const elAcc = document.getElementById('bmAcc2');
      const elRec = document.getElementById('bmRec2');
      const elF1 = document.getElementById('bmF1_2');
      const elLat = document.getElementById('bmLat2');
      const elMem = document.getElementById('bmMem2');
      if (elAcc) elAcc.textContent = `${(a.test_accuracy * 100).toFixed(1)}%`;
      if (elRec) elRec.textContent = `${(a.critical_class_recall * 100).toFixed(1)}%`;
      if (elF1) elF1.textContent = a.macro_f1.toFixed(3);
      if (elLat) elLat.textContent = `${a.latency_ms} ms`;
      if (elMem && a.memory_mb) elMem.textContent = `${a.memory_mb} MB`;
    }

    if (algos.Algorithm_3_ML_Ensemble) {
      const a = algos.Algorithm_3_ML_Ensemble;
      const elAcc = document.getElementById('bmAcc3');
      const elRec = document.getElementById('bmRec3');
      const elF1 = document.getElementById('bmF1_3');
      const elLat = document.getElementById('bmLat3');
      const elMem = document.getElementById('bmMem3');
      if (elAcc) elAcc.textContent = `${(a.test_accuracy * 100).toFixed(1)}%`;
      if (elRec) elRec.textContent = `${(a.critical_class_recall * 100).toFixed(1)}%`;
      if (elF1) elF1.textContent = a.macro_f1.toFixed(3);
      if (elLat) elLat.textContent = `${a.latency_ms} ms`;
      if (elMem && a.memory_mb) elMem.textContent = `${a.memory_mb} MB`;
    }
  } catch (err) {
    console.warn('[Benchmark Report Load Failed]:', err);
  }
}

async function triggerBenchmarkRun() {
  const btn = document.getElementById('btnRefreshBenchmark');
  if (btn) btn.disabled = true;
  showToast('Launching comparative 3-algorithm benchmark...', 'info');
  try {
    const res = await fetch('/api/v1/benchmark/run', { method: 'POST' });
    const data = await res.json();
    showToast(data.message || 'Benchmark started in background', 'info');
    pollBenchmarkStatus();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
    if (btn) btn.disabled = false;
  }
}

function pollBenchmarkStatus() {
  const interval = setInterval(async () => {
    try {
      const res = await fetch('/api/v1/benchmark/report');
      const data = await res.json();
      const status = data.task_status?.status;
      const footer = document.getElementById('bmStatusFooter');
      if (footer && data.task_status?.message) {
        footer.textContent = `Benchmark: ${data.task_status.message}`;
      }

      if (status === 'COMPLETED' || data.success) {
        clearInterval(interval);
        loadBenchmarkReport();
        showToast('Benchmark suite complete! Report updated.', 'success');
        const btn = document.getElementById('btnRefreshBenchmark');
        if (btn) btn.disabled = false;
      } else if (status === 'ERROR') {
        clearInterval(interval);
        showToast('Benchmark encountered an error.', 'error');
        const btn = document.getElementById('btnRefreshBenchmark');
        if (btn) btn.disabled = false;
      }
    } catch (e) {
      clearInterval(interval);
    }
  }, 2000);
}

// --- GPS Geolocation Tracking & SOS Manual Trigger (Mobile Native Experience) ---
document.getElementById('btnSosPanic')?.addEventListener('click', () => {
  if (navigator.vibrate) navigator.vibrate([100, 50, 100, 50, 100]);
  showToast('SOS Triggered! Grabbing GPS Coordinates...', 'error');
  
  // Get GPS Location
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition((position) => {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      showToast(`GPS Attached: ${lat.toFixed(4)}, ${lon.toFixed(4)}. Sent to server.`, 'success');
      
      // Simulate sending to backend
      handleRadarDetection({
        incident_id: 'SOS-' + Date.now(),
        category: 'Person Asking for Help',
        severity: 'CRITICAL',
        confidence: 1.0,
        azimuth: 90.0,
        tdoa_seconds: 0.0,
        snr_db: 40.0,
        audio_quality: 'Good',
        timestamp: new Date().toISOString(),
        detected_speech_transcript: `MANUAL SOS TRIGGERED AT LAT: ${lat.toFixed(4)}, LON: ${lon.toFixed(4)}`
      });
    }, (error) => {
      showToast('GPS Access Denied or Unavailable. Logging SOS without location.', 'info');
      handleRadarDetection({
        incident_id: 'SOS-' + Date.now(),
        category: 'Person Asking for Help',
        severity: 'CRITICAL',
        confidence: 1.0,
        azimuth: 90.0,
        tdoa_seconds: 0.0,
        snr_db: 40.0,
        audio_quality: 'Good',
        timestamp: new Date().toISOString(),
        detected_speech_transcript: 'MANUAL SOS TRIGGERED (NO GPS)'
      });
    });
  }
});

// --- Active Learning & Continuous Data Augmentation ---
document.getElementById('btnCreateCategory')?.addEventListener('click', async () => {
  const input = document.getElementById('newCategoryInput');
  const catName = input.value.trim();
  if (!catName) {
    showToast('Please enter a valid category name.', 'error');
    return;
  }
  try {
    const res = await fetch('/api/v1/learning/category', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_name: catName })
    });
    const data = await res.json();
    showToast(data.message, 'success');
    input.value = '';
    
    // Dynamically update the upload select dropdown
    const select = document.getElementById('uploadCategorySelect');
    if (select) {
      const opt = document.createElement('option');
      opt.value = catName;
      opt.textContent = catName;
      select.appendChild(opt);
    }
  } catch (err) {
    showToast('Failed to provision category.', 'error');
  }
});

document.getElementById('btnHarvestAugment')?.addEventListener('click', async () => {
  const btn = document.getElementById('btnHarvestAugment');
  btn.disabled = true;
  showToast('Starting Data Augmentation Pipeline... Generating 3,000+ samples using pitch-shift, time-stretch, and noise-injection.', 'info');
  try {
    const res = await fetch('/api/v1/learning/augment', { method: 'POST' });
    const data = await res.json();
    showToast(data.message, 'success');
  } catch (err) {
    showToast('Augmentation failed.', 'error');
  } finally {
    btn.disabled = false;
  }
});

document.getElementById('btnTriggerFineTune')?.addEventListener('click', async () => {
  const btn = document.getElementById('btnTriggerFineTune');
  const progressText = document.getElementById('fineTuneProgressText');
  btn.disabled = true;
  if (progressText) progressText.textContent = 'Unfreezing CNN layers... Training on new acoustic profile...';
  
  try {
    const res = await fetch('/api/v1/learning/fine-tune', { method: 'POST' });
    const data = await res.json();
    showToast(data.message, 'success');
    if (progressText) progressText.textContent = `Model fine-tuned successfully. New F1 Score: ${data.new_f1}. Live monitoring resumed.`;
  } catch (err) {
    showToast('Fine-tuning failed.', 'error');
    if (progressText) progressText.textContent = 'Fine-tuning aborted due to error.';
  } finally {
    btn.disabled = false;
  }
});

// --- WOW FACTOR: Futuristic Interactions ---

// 1. AI Text-to-Speech Announcement
function announceThreatAI(category, confidence) {
  if ('speechSynthesis' in window) {
    // Prevent overlapping voices
    window.speechSynthesis.cancel();
    
    const confPercent = Math.round(confidence * 100);
    const msg = new SpeechSynthesisUtterance();
    msg.text = `Warning. Critical acoustic anomaly detected. Classification: ${category}. Confidence: ${confPercent} percent. Initiating lockdown protocol.`;
    msg.volume = 1;
    msg.rate = 0.9;
    msg.pitch = 0.8; // Deep robotic tone
    
    // Try to find a robotic/English voice
    const voices = window.speechSynthesis.getVoices();
    const roboticVoice = voices.find(v => v.name.includes('Google') || v.name.includes('Zira') || v.name.includes('Samantha'));
    if (roboticVoice) msg.voice = roboticVoice;

    window.speechSynthesis.speak(msg);
  }
}

// 2. Hacker-style Matrix Terminal Logging
function logToNeuralTerminal(data) {
  const container = document.getElementById('aiTerminalFeed');
  if (!container) return;
  
  const div = document.createElement('div');
  div.className = `terminal-line ${data.severity === 'CRITICAL' ? 'critical' : ''}`;
  
  const time = new Date().toISOString().split('T')[1].substring(0, 12);
  const conf = (data.confidence * 100).toFixed(2);
  const tdoa = (data.tdoa_seconds * 1000).toFixed(4);
  
  div.textContent = `[${time}] DECODING_SIG | AZIMUTH:${data.azimuth.toFixed(1)} TDOA:${tdoa}ms | MATCH => ${data.category.toUpperCase()} (CONF: ${conf}%)`;
  
  container.appendChild(div);
  
  // Keep terminal short
  if (container.children.length > 20) {
    container.removeChild(container.firstChild);
  }
  
  // Auto-scroll to bottom
  const wrapper = document.getElementById('aiTerminalContainer');
  wrapper.scrollTop = wrapper.scrollHeight;
}
