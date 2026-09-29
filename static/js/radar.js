/**
 * SonicSentinel AI - 360-Degree Acoustic Radar Visualizer
 * Dual-Theme High-DPI Aerospace Radar Engine (Pro Commercial Light Mode + Tactical Dark Mode)
 * Includes Dedicated Fullscreen Modal Support for Desktop & Mobile
 */

class AcousticRadar {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.sweepAngle = 0; // Current radar sweep angle in radians
    this.sweepSpeed = 0.024; // Radians per frame
    this.blips = []; // Active acoustic blip detections
    this.targetAzimuth = null; // Last detected azimuth needle in degrees
    this.needleAngle = null;
    this.isFullscreen = false;

    // Severity color mapping (Modern high-contrast palette)
    this.colors = {
      CRITICAL: '#f43f5e',
      HIGH: '#f97316',
      MEDIUM: '#eab308',
      LOW: '#06b6d4',
      NORMAL: '#10b981',
    };

    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', () => this.resize());
    window.addEventListener('orientationchange', () => setTimeout(() => this.resize(), 150));
    this.setupEvents();
    this.setupFullscreen();
    this.animate();
  }

  isLight() {
    return !document.body.classList.contains('dark-theme');
  }

  setupFullscreen() {
    // ESC key listener to exit fullscreen
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isFullscreen) {
        this.closeFullscreen();
      }
    });

    const fsBtn = document.getElementById('btnFullscreenRadar');
    if (fsBtn) {
      fsBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.toggleFullscreen();
      });
    }

    const closeBtn = document.getElementById('btnCloseFullscreenRadar');
    if (closeBtn) {
      closeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.closeFullscreen();
      });
    }

    const backdrop = document.getElementById('fsRadarBackdrop');
    if (backdrop) {
      backdrop.addEventListener('click', () => {
        this.closeFullscreen();
      });
    }
  }

  toggleFullscreen() {
    if (this.isFullscreen) {
      this.closeFullscreen();
    } else {
      this.openFullscreen();
    }
  }

  openFullscreen() {
    this.isFullscreen = true;
    const modal = document.getElementById('radarFullscreenModal');
    const fsWrap = document.getElementById('fsRadarCanvasWrap');

    if (modal && fsWrap && this.canvas) {
      fsWrap.appendChild(this.canvas);
      modal.style.display = 'flex';
      document.body.classList.add('radar-fs-active');
      this.syncFullscreenTelemetry();
      setTimeout(() => this.resize(), 30);
    }
  }

  closeFullscreen() {
    this.isFullscreen = false;
    const modal = document.getElementById('radarFullscreenModal');
    const normalWrap = document.getElementById('normalRadarWrap');

    if (modal && normalWrap && this.canvas) {
      normalWrap.appendChild(this.canvas);
      modal.style.display = 'none';
      document.body.classList.remove('radar-fs-active');
      setTimeout(() => this.resize(), 30);
    }
  }

  syncFullscreenTelemetry() {
    const copyVal = (fromId, toId) => {
      const fromEl = document.getElementById(fromId);
      const toEl = document.getElementById(toId);
      if (fromEl && toEl) toEl.textContent = fromEl.textContent;
    };
    copyVal('telBearing', 'fsTelBearing');
    copyVal('telCategory', 'fsTelCategory');
    copyVal('telConfidence', 'fsTelConfidence');
    copyVal('telTdoa', 'fsTelTdoa');
    copyVal('telSnr', 'fsTelSnr');
    const qFrom = document.getElementById('telQuality');
    const qTo = document.getElementById('fsTelQuality');
    if (qFrom && qTo) {
      qTo.textContent = qFrom.textContent;
      qTo.className = qFrom.className;
    }
  }

  resize() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    let size;

    if (this.isFullscreen) {
      const isMobile = window.innerWidth < 768;
      const maxW = window.innerWidth - (isMobile ? 24 : 80);
      const maxH = window.innerHeight - (isMobile ? 210 : 160);
      size = Math.max(260, Math.min(maxW, maxH, 780));
    } else {
      const parentWidth = this.canvas.parentElement ? this.canvas.parentElement.clientWidth : 340;
      size = Math.max(250, Math.min(parentWidth || 340, 380));
    }

    // Set canvas bitmap dimensions with DPR scaling for crisp lines
    this.canvas.width = Math.floor(size * dpr);
    this.canvas.height = Math.floor(size * dpr);
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;

    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.centerX = size / 2;
    this.centerY = size / 2;
    this.radius = (size / 2) - (this.isFullscreen ? 28 : 18);
  }

  setupEvents() {
    const handleCoordinateSelect = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = clientX - rect.left - this.centerX;
      const y = clientY - rect.top - this.centerY;
      
      let rad = Math.atan2(-y, x);
      if (rad < 0) rad += 2 * Math.PI;
      let deg = Math.round(rad * 180 / Math.PI);
      if (deg > 180) deg = 360 - deg;
      deg = Math.max(0, Math.min(180, deg));

      const slider = document.getElementById('simAzimuthSlider');
      const valLabel = document.getElementById('simAzimuthVal');
      if (slider && valLabel) {
        slider.value = deg;
        valLabel.textContent = `${deg}°`;
      }
      this.targetAzimuth = deg;
    };

    this.canvas.addEventListener('click', (e) => {
      handleCoordinateSelect(e.clientX, e.clientY);
    });

    // Touch support for mobile tap
    this.canvas.addEventListener('touchstart', (e) => {
      if (e.touches && e.touches[0]) {
        handleCoordinateSelect(e.touches[0].clientX, e.touches[0].clientY);
      }
    }, { passive: true });
  }

  addBlip(detection) {
    const azimuthDeg = detection.azimuth !== undefined ? detection.azimuth : 90;
    this.targetAzimuth = azimuthDeg;

    const thetaRad = (azimuthDeg * Math.PI) / 180.0;
    const canvasAngle = -thetaRad; 

    const distanceNorm = 0.38 + (Math.random() * 0.44);

    this.blips.push({
      angle: canvasAngle,
      azimuthDeg: azimuthDeg,
      radius: this.radius * distanceNorm,
      category: detection.category,
      confidence: detection.confidence,
      severity: detection.severity || 'MEDIUM',
      color: this.colors[detection.severity] || this.colors.LOW,
      alpha: 1.0,
      rippleRadius: 4,
      createdAt: Date.now(),
    });

    if (this.blips.length > 14) {
      this.blips.shift();
    }
  }

  drawGrid() {
    const ctx = this.ctx;
    const isLight = this.isLight();
    ctx.save();

    // Background circle with subtle modern gradient
    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
    if (isLight) {
      const bgGrad = ctx.createRadialGradient(
        this.centerX, this.centerY, 10,
        this.centerX, this.centerY, this.radius
      );
      bgGrad.addColorStop(0, '#ffffff');
      bgGrad.addColorStop(0.6, '#f8fafc');
      bgGrad.addColorStop(1, '#edf2f7');
      ctx.fillStyle = bgGrad;
    } else {
      const bgGrad = ctx.createRadialGradient(
        this.centerX, this.centerY, 10,
        this.centerX, this.centerY, this.radius
      );
      bgGrad.addColorStop(0, '#0a1020');
      bgGrad.addColorStop(0.8, '#060a14');
      bgGrad.addColorStop(1, '#03060d');
      ctx.fillStyle = bgGrad;
    }
    ctx.fill();

    // Concentric Range Rings (15m, 30m, 45m, 60m)
    const ringCounts = 4;
    ctx.strokeStyle = isLight ? 'rgba(148, 163, 184, 0.35)' : 'rgba(0, 240, 255, 0.16)';
    ctx.lineWidth = 1;

    for (let i = 1; i <= ringCounts; i++) {
      const r = (this.radius / ringCounts) * i;
      ctx.beginPath();
      ctx.arc(this.centerX, this.centerY, r, 0, Math.PI * 2);
      ctx.stroke();

      // Range distance labels
      ctx.fillStyle = isLight ? '#64748b' : 'rgba(0, 240, 255, 0.5)';
      ctx.font = `${this.isFullscreen ? 10 : 8}px "JetBrains Mono", monospace`;
      ctx.fillText(`${i * 15}m`, this.centerX + 5, this.centerY - r + 9);
    }

    // Crosshairs
    ctx.strokeStyle = isLight ? 'rgba(148, 163, 184, 0.45)' : 'rgba(0, 240, 255, 0.22)';
    ctx.beginPath();
    ctx.moveTo(this.centerX - this.radius, this.centerY);
    ctx.lineTo(this.centerX + this.radius, this.centerY);
    ctx.moveTo(this.centerX, this.centerY - this.radius);
    ctx.lineTo(this.centerX, this.centerY + this.radius);
    ctx.stroke();

    // Cardinal & Bearing Ticks
    const angles = [0, 45, 90, 135, 180, 225, 270, 315];
    ctx.font = `${this.isFullscreen ? 11 : 9}px "Inter", -apple-system, sans-serif`;
    ctx.fillStyle = isLight ? '#334155' : 'rgba(0, 240, 255, 0.75)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    angles.forEach(deg => {
      const rad = (deg * Math.PI) / 180;
      const x1 = this.centerX + Math.cos(rad) * (this.radius - (this.isFullscreen ? 9 : 7));
      const y1 = this.centerY + Math.sin(rad) * (this.radius - (this.isFullscreen ? 9 : 7));
      const x2 = this.centerX + Math.cos(rad) * this.radius;
      const y2 = this.centerY + Math.sin(rad) * this.radius;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      const labelDist = this.radius + (this.isFullscreen ? 14 : 11);
      const labelX = this.centerX + Math.cos(rad) * labelDist;
      const labelY = this.centerY + Math.sin(rad) * labelDist;
      
      let label = `${deg}°`;
      if (deg === 90) label = '090°';
      if (deg === 0) label = '000°';
      ctx.fillText(label, labelX, labelY);
    });

    // Outer border ring with subtle glow
    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
    ctx.strokeStyle = isLight ? '#0284c7' : 'rgba(0, 240, 255, 0.65)';
    ctx.lineWidth = this.isFullscreen ? 2.5 : 1.5;
    if (!isLight) {
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 8;
    }
    ctx.stroke();

    ctx.restore();
  }

  drawSweep() {
    const ctx = this.ctx;
    const isLight = this.isLight();
    ctx.save();

    const sweepTrail = Math.PI / 3.8;
    const gradient = ctx.createRadialGradient(
      this.centerX, this.centerY, 0,
      this.centerX, this.centerY, this.radius
    );
    if (isLight) {
      gradient.addColorStop(0, 'rgba(2, 132, 199, 0)');
      gradient.addColorStop(1, 'rgba(2, 132, 199, 0.18)');
    } else {
      gradient.addColorStop(0, 'rgba(0, 240, 255, 0)');
      gradient.addColorStop(1, 'rgba(0, 240, 255, 0.22)');
    }

    ctx.beginPath();
    ctx.moveTo(this.centerX, this.centerY);
    ctx.arc(
      this.centerX,
      this.centerY,
      this.radius,
      this.sweepAngle - sweepTrail,
      this.sweepAngle,
      false
    );
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Leading sweep line
    ctx.beginPath();
    ctx.moveTo(this.centerX, this.centerY);
    ctx.lineTo(
      this.centerX + Math.cos(this.sweepAngle) * this.radius,
      this.centerY + Math.sin(this.sweepAngle) * this.radius
    );
    ctx.strokeStyle = isLight ? 'rgba(2, 132, 199, 0.95)' : 'rgba(0, 240, 255, 0.95)';
    ctx.lineWidth = this.isFullscreen ? 2.5 : 2;
    ctx.shadowColor = isLight ? 'rgba(2, 132, 199, 0.4)' : '#00f0ff';
    ctx.shadowBlur = this.isFullscreen ? 12 : 8;
    ctx.stroke();

    ctx.restore();
  }

  drawTargetAzimuthNeedle() {
    if (this.targetAzimuth === null) return;

    const ctx = this.ctx;
    ctx.save();

    const targetRad = -(this.targetAzimuth * Math.PI) / 180.0;
    if (this.needleAngle === null) this.needleAngle = targetRad;
    this.needleAngle += (targetRad - this.needleAngle) * 0.14;

    ctx.beginPath();
    ctx.moveTo(this.centerX, this.centerY);
    ctx.lineTo(
      this.centerX + Math.cos(this.needleAngle) * this.radius,
      this.centerY + Math.sin(this.needleAngle) * this.radius
    );
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.9)';
    ctx.lineWidth = this.isFullscreen ? 2.2 : 1.6;
    ctx.setLineDash([5, 4]);
    ctx.stroke();

    // Target reticle pip
    const arrowX = this.centerX + Math.cos(this.needleAngle) * (this.radius - 4);
    const arrowY = this.centerY + Math.sin(this.needleAngle) * (this.radius - 4);
    ctx.beginPath();
    ctx.arc(arrowX, arrowY, this.isFullscreen ? 6 : 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#f43f5e';
    ctx.shadowColor = 'rgba(244, 63, 94, 0.65)';
    ctx.shadowBlur = 8;
    ctx.fill();

    ctx.restore();
  }

  drawBlips() {
    const ctx = this.ctx;
    const now = Date.now();

    for (let i = this.blips.length - 1; i >= 0; i--) {
      const blip = this.blips[i];
      const age = (now - blip.createdAt) / 1000;

      if (age > 22) {
        this.blips.splice(i, 1);
        continue;
      }

      blip.alpha = Math.max(0.18, 1.0 - (age / 22));
      blip.rippleRadius += 0.5;
      if (blip.rippleRadius > 28) blip.rippleRadius = 4;

      const x = this.centerX + Math.cos(blip.angle) * blip.radius;
      const y = this.centerY + Math.sin(blip.angle) * blip.radius;

      ctx.save();

      // Expanding sonar ripple
      ctx.beginPath();
      ctx.arc(x, y, blip.rippleRadius, 0, Math.PI * 2);
      ctx.strokeStyle = blip.color;
      ctx.lineWidth = 1.4;
      ctx.globalAlpha = Math.max(0, blip.alpha * (1 - blip.rippleRadius / 28));
      ctx.stroke();

      // Solid central blip marker with glow
      ctx.beginPath();
      ctx.arc(x, y, this.isFullscreen ? 6.5 : 5, 0, Math.PI * 2);
      ctx.fillStyle = blip.color;
      ctx.globalAlpha = blip.alpha;
      ctx.shadowColor = blip.color;
      ctx.shadowBlur = this.isFullscreen ? 10 : 7;
      ctx.fill();

      // Class text label
      if (blip.alpha > 0.35) {
        ctx.font = `600 ${this.isFullscreen ? 11 : 9.5}px "Inter", -apple-system, sans-serif`;
        ctx.fillStyle = this.isLight() ? '#0f172a' : '#f8fafc';
        ctx.globalAlpha = blip.alpha;
        ctx.shadowBlur = 0;
        ctx.textAlign = 'left';
        ctx.fillText(blip.category, x + 9, y + 3.5);
      }

      ctx.restore();
    }
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    this.drawGrid();
    this.drawSweep();
    this.drawTargetAzimuthNeedle();
    this.drawBlips();

    this.sweepAngle += this.sweepSpeed;
    if (this.sweepAngle >= Math.PI * 2) {
      this.sweepAngle = 0;
    }

    requestAnimationFrame(() => this.animate());
  }
}
