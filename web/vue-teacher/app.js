const { createApp, nextTick } = Vue;

createApp({
  data() {
    const sessionKey = "edge-teacher-session-id";
    let sessionId = sessionStorage.getItem(sessionKey);
    if (!sessionId) {
      sessionId = "user-" + Math.random().toString(36).slice(2, 8);
      sessionStorage.setItem(sessionKey, sessionId);
    }
    return {
      sessionId,
      palette: localStorage.getItem("teacher-palette") || "teal",
      lessons: [],
      studentVideos: [],
      selectedLessonId: "",
      selectedStudentVideo: "",
      source: "sample",
      analysisMode: localStorage.getItem("analysis-mode") || "standard",
      analysisModes: [
        { key: "smooth", label: "流畅模式", note: "每 5 帧姿态估计一次" },
        { key: "standard", label: "标准模式", note: "每 3 帧姿态估计一次" },
        { key: "accurate", label: "高精度模式", note: "每帧检测和姿态估计" },
      ],
      running: false,
      analysisStarting: false,
      socket: null,
      analysisRunId: 0,
      playbackTimer: null,
      suppressNextPauseStop: false,
      ignorePauseUntil: 0,
      manualStopRequested: false,
      connectionText: "未连接",
      report: {
        total: 0,
        focused: [],
        needs_attention: [],
        state_counts: {
          focused: 0,
          confused: 0,
          tired: 0,
          distracted: 0,
          absent: 0,
        },
        detections: 0,
        total_ms: 0,
        strategy: "unknown",
        pose_interval: 3,
        pose_refreshed: false,
      },
      action: {},
      lessonContext: {},
      coursePlaybackRate: 1,
      distractionAlertActive: false,
      paletteBeforeDistraction: null,
      reminderAudios: [
        "/assets/reminders/attention-1.m4a",
        "/assets/reminders/attention-2.m4a",
        "/assets/reminders/attention-3.m4a",
      ],
      analysisImage: "",
      sampleInfo: {},
      edge: { strategy: "unknown", connected: false },
      chatMessages: [],
      chatInput: "",
      chatLoading: false,
      chatLastUsage: null,
      chatTimeoutMs: 18000,
    };
  },
  computed: {
    selectedLesson() {
      return this.lessons.find((item) => item.id === this.selectedLessonId) || this.lessons[0] || {};
    },
    consoleUrl() {
      return `http://${window.location.hostname}:8008/console`;
    },
    edgeText() {
      const strategy = this.edge.strategy || this.edge.dao?.strategy || "unknown";
      if (this.edge.connected) return `边缘服务器已连接 · ${strategy}`;
      return "未连接边缘服务器 · 本地模式";
    },
    attentionCount() {
      return this.report.needs_attention?.length || 0;
    },
    focusedCount() {
      return this.report.focused?.length || 0;
    },
    stateItems() {
      const counts = this.report.state_counts || {};
      return [
        { key: "focused", label: "专注", value: counts.focused || 0 },
        { key: "confused", label: "困惑", value: counts.confused || 0 },
        { key: "tired", label: "疲劳", value: counts.tired || 0 },
        { key: "distracted", label: "分心", value: counts.distracted || 0 },
        { key: "absent", label: "缺勤", value: counts.absent || 0 },
      ];
    },
    selectedAnalysisMode() {
      return this.analysisModes.find((item) => item.key === this.analysisMode) || this.analysisModes[1];
    },
    analysisModeText() {
      return this.selectedAnalysisMode.note;
    },
    poseStatusText() {
      const interval = this.report.pose_interval || this.sampleInfo.pose_interval;
      if (!interval) return this.analysisModeText;
      const refreshed = this.sampleInfo.pose_refreshed || this.report.pose_refreshed;
      return `${this.selectedAnalysisMode.label} · 姿态间隔 ${interval} 帧${refreshed ? " · 本帧刷新" : ""}`;
    },
    confusedCount() {
      return this.report.state_counts?.confused || 0;
    },
    tiredCount() {
      return this.report.state_counts?.tired || 0;
    },
    distractedCount() {
      return this.report.state_counts?.distracted || 0;
    },
    absentCount() {
      if (this.studentCount === 0 && this.running) return 1;
      return this.report.state_counts?.absent || 0;
    },
    studentCount() {
      return this.report.total || this.report.detections || 0;
    },
    latencyText() {
      return `${Math.round(this.report.total_ms || 0)} ms`;
    },
    actionLevel() {
      const action = this.action.action || "";
      if (action.includes("暂停")) return "critical";
      if (action.includes("放慢") || action.includes("休息") || action.includes("提问")) return "warning";
      if (action.includes("继续")) return "good";
      return "idle";
    },
    actionStatusText() {
      const labels = {
        critical: "需要立即处理",
        warning: "建议调整教学",
        good: "状态稳定",
        idle: "等待分析",
      };
      return labels[this.actionLevel];
    },
    teacherMood() {
      const state = this.action.state || "";
      const actionText = `${this.action.action || ""} ${this.action.pace || ""} ${this.action.reason || ""}`;
      const counts = this.report.state_counts || {};

      if (state === "缺勤" || actionText.includes("缺勤") || counts.absent > 0) {
        return { className: "mood-absent", label: "等待" };
      }
      if (state === "分心" || actionText.includes("分心") || counts.distracted > 0) {
        return { className: "mood-distracted", label: "提醒" };
      }
      if (state === "困惑" || actionText.includes("困惑") || counts.confused > 0) {
        return { className: "mood-confused", label: "思考" };
      }
      if (state === "疲劳" || actionText.includes("疲劳") || actionText.includes("疲惫") || counts.tired > 0) {
        return { className: "mood-tired", label: "关切" };
      }
      if (state === "专注" || actionText.includes("专注") || counts.focused > 0) {
        return { className: "mood-focused", label: "鼓励" };
      }
      return { className: "mood-idle", label: "待机" };
    },
    desiredPlaybackRate() {
      const state = this.action.state || "";
      const actionText = `${this.action.action || ""} ${this.action.pace || ""}`;
      const distractedCount = this.report.state_counts?.distracted || 0;
      if (state === "分心" || actionText.includes("分心") || distractedCount > 0) {
        return 1;
      }
      if (actionText.includes("0.75")) {
        return 0.75;
      }
      return 1;
    },
    coursePlaybackRateText() {
      if (this.isDistracted) return "已暂停";
      return `${this.coursePlaybackRate.toFixed(2).replace(/\.00$/, "")}x`;
    },
    primaryButtonText() {
      return (this.running || this.analysisStarting) ? "暂停课程与检测" : "播放课程并开始检测";
    },
    daoDecision() {
      return this.edge.dao || {};
    },
    daoStageCount() {
      return this.daoDecision.stages_to_offload?.length || 0;
    },
    daoPriorityText() {
      const priority = Number(this.daoDecision.priority || 0);
      return priority.toFixed(2);
    },
    daoAdmissionText() {
      const labels = {
        pending: "等待决策",
        checking: "检查中",
        accept: "准入通过",
        window_available: "窗口可用",
        congested_high_priority_allowed: "高优先级放行",
        reject_no_neighbor: "无边缘节点",
        reject_congested_low_priority: "拥塞拒绝",
        reject_no_qualified_neighbor: "节点负载过高",
        local_no_edge: "无边缘服务器",
        local_status_failed: "状态查询失败",
        fallback: "规则兜底",
        hard_gate: "Hard Gate",
        load_balance_gate: "负载均衡 Gate",
        soft_decision: "Soft Decision",
      };
      return labels[this.daoDecision.admission] || this.daoDecision.admission || "等待决策";
    },
    daoWindowText() {
      const recent = this.daoDecision.recent_offloads ?? "-";
      const max = this.daoDecision.max_offloads_per_window ?? "-";
      return `${recent} / ${max}`;
    },
    daoTransferText() {
      const value = Number(this.daoDecision.estimated_transfer_ms || 0);
      return value ? `${value.toFixed(1)} ms` : "-";
    },
    daoTargetText() {
      return this.daoDecision.target_node_id || "本地执行";
    },
    daoNeighbor() {
      return this.daoDecision.neighbors?.[0] || {};
    },
    daoNeighborScoreText() {
      const score = Number(this.daoNeighbor.selection_score || 0);
      return score ? score.toFixed(2) : "-";
    },
    isDistracted() {
      const state = this.action.state || "";
      const actionText = `${this.action.action || ""} ${this.action.pace || ""}`;
      const distractedCount = this.report.state_counts?.distracted || 0;
      const absentCount = this.report.state_counts?.absent || 0;
      return state === "分心" || state === "缺勤" || actionText.includes("分心") || actionText.includes("缺勤") || distractedCount > 0 || absentCount > 0;
    },
    sampleText() {
      if (!this.sampleInfo.frame_index) return "等待采样帧";
      return `第 ${this.sampleInfo.frame_index} 帧 · ${this.sampleInfo.position || ""}`;
    },
    playbackTime() {
      const time = Number(this.lessonContext.playback_time || this.sampleInfo.playback_time || 0);
      const minutes = Math.floor(time / 60).toString().padStart(2, "0");
      const seconds = Math.floor(time % 60).toString().padStart(2, "0");
      return `${minutes}:${seconds}`;
    },
    shellClasses() {
      return ["app-shell", "day", `palette-${this.activePalette}`];
    },
    activePalette() {
      return this.isDistracted ? "red" : this.palette;
    },
  },
  watch: {
    shellClasses: {
      handler() {
        this.applyShellClasses();
      },
      immediate: true,
    },
    selectedLessonId() {
      nextTick(() => this.handleCourseLoadedMetadata());
      if (this.running) {
        this.restartAnalysis();
      }
    },
    source() {
      if (this.running) {
        this.restartAnalysis();
      }
    },
    selectedStudentVideo() {
      if (this.running && this.source === "sample") {
        this.restartAnalysis();
      }
    },
    analysisMode() {
      localStorage.setItem("analysis-mode", this.analysisMode);
      if (this.running) {
        this.restartAnalysis();
      }
    },
    desiredPlaybackRate() {
      this.applyCoursePlaybackRate();
    },
    isDistracted(isDistractedNow) {
      if (isDistractedNow) {
        this.triggerDistractionIntervention();
      } else {
        this.resetDistractionIntervention();
      }
    },
  },
  async mounted() {
    window.addEventListener("pagehide", () => {
      // 页面卸载兜底：关闭分析连接，防止后台上报残留
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try { this.socket.close(1000, "leave"); } catch (_e) {}
      }
    });
    await Promise.all([this.loadLessons(), this.loadStudentVideos(), this.loadStatus()]);
    this.applyShellClasses();
  },
  beforeUnmount() {
    this.stopAnalysis();
  },
  methods: {
    applyShellClasses() {
      const root = document.getElementById("app");
      if (!root) return;
      root.className = this.shellClasses.join(" ");
    },
    async loadLessons() {
      const response = await fetch("/api/lessons");
      const payload = await response.json();
      this.lessons = payload.lessons || [];
      this.selectedLessonId = this.lessons[0]?.id || "";
    },
    async loadStudentVideos() {
      const response = await fetch("/api/student-videos");
      const payload = await response.json();
      this.studentVideos = payload.videos || [];
      this.selectedStudentVideo = this.studentVideos[0]?.relative_path || "";
    },
    async loadStatus() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        if (!response.ok) return;
        const payload = await response.json();
        if (payload.edge) {
          this.edge = payload.edge;
        }
      } catch (_err) {
        // 状态同步失败不影响课程播放和实时分析。
      }
    },
    setPalette(palette) {
      this.palette = palette;
      localStorage.setItem("teacher-palette", palette);
    },
    setAnalysisMode(mode) {
      this.analysisMode = mode;
    },
    playRandomReminderAudio() {
      const index = Math.floor(Math.random() * this.reminderAudios.length);
      const audio = new Audio(this.reminderAudios[index]);
      audio.play().catch(() => {
        this.connectionText = "已暂停课程，请点击页面允许播放提醒录音";
      });
    },
    triggerDistractionIntervention() {
      if (this.distractionAlertActive) return;
      this.distractionAlertActive = true;
      this.paletteBeforeDistraction = this.palette;
      this.playRandomReminderAudio();
    },
    resetDistractionIntervention() {
      if (!this.distractionAlertActive) return;
      this.distractionAlertActive = false;
      this.paletteBeforeDistraction = null;
    },
    toggleAnalysis() {
      if (this.running || this.analysisStarting) {
        this.stopAnalysis({ manual: true });
      } else {
        this.startAnalysis();
      }
    },
    goClassroomMode() {
      // 切换到线下课堂模式前，先停止当前分析并中断上传，避免教师端出现重复记录
      if (this.running || this.analysisStarting) {
        this.stopAnalysis({ manual: true });
      }
      window.location.href = "/client";
    },
    restartAnalysis() {
      this.stopAnalysis();
      window.setTimeout(() => this.startAnalysis(), 150);
    },
    startAnalysis() {
      if (this.running || this.analysisStarting) return;
      if (this.socket && [WebSocket.CONNECTING, WebSocket.OPEN].includes(this.socket.readyState)) return;
      if (!this.selectedLessonId) {
        this.connectionText = "课程列表尚未加载完成，请稍后再试";
        return;
      }
      this.analysisStarting = true;
      this.manualStopRequested = false;
      const runId = this.analysisRunId + 1;
      this.analysisRunId = runId;
      this.connectionText = "正在启动决策";
      this.loadStatus();
      this.analysisImage = "";
      this.sampleInfo = {};
      this.lessonContext = {};
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${protocol}://${window.location.host}/ws/analysis`;
      this.socket = new WebSocket(wsUrl);
      const socket = this.socket;
      this.connectionText = "连接中";

      this.socket.onopen = () => {
        if (runId !== this.analysisRunId || socket !== this.socket || this.manualStopRequested) return;
        this.running = true;
        this.analysisStarting = false;
        this.connectionText = "实时分析中";
        this.socket.send(JSON.stringify({
          source: this.source,
          lesson_id: this.selectedLessonId,
          sample_video: this.selectedStudentVideo,
          analysis_mode: this.analysisMode,
          playback_time: this.getCoursePlaybackTime(),
          interval_ms: 300,
          max_frames: 1000,
        }));
        this.startPlaybackReporter();
      };

      this.socket.onmessage = (event) => {
        if (runId !== this.analysisRunId || socket !== this.socket || this.manualStopRequested) return;
        const payload = JSON.parse(event.data);
        if (payload.type === "analysis") {
          const isSamplingPreview = payload.report?.strategy === "sampling";
          if (!isSamplingPreview) {
            this.connectionText = "实时决策中";
            this.report = payload.report || this.report;
            this.action = payload.action || this.action;
            this.applyCoursePlaybackRate();
            if (this.isDistracted) {
              this.triggerDistractionIntervention();
            } else {
              this.resetDistractionIntervention();
            }
          }
          if (payload.image) {
            this.analysisImage = payload.image;
          }
          this.sampleInfo = payload.sample || this.sampleInfo;
          this.lessonContext = payload.lesson_context || this.lessonContext;
          this.edge = payload.edge || this.edge;
          if (isSamplingPreview) {
            this.connectionText = `模型加载/采样中 · ${payload.sample?.mode_label || this.selectedAnalysisMode.label}`;
          }
        } else if (payload.type === "started") {
          this.connectionText = `等待首帧采样 · ${payload.mode_label || this.selectedAnalysisMode.label}`;
        } else if (payload.type === "error") {
          this.connectionText = payload.message || "分析错误";
          this.running = false;
          this.analysisStarting = false;
        } else if (payload.type === "finished") {
          this.connectionText = "分析结束";
          this.running = false;
          this.analysisStarting = false;
        }
      };

      this.socket.onclose = () => {
        if (runId !== this.analysisRunId || socket !== this.socket) return;
        this.running = false;
        this.analysisStarting = false;
        if (this.connectionText === "实时分析中" || this.connectionText === "连接中") {
          this.connectionText = "已断开";
        }
      };

      this.socket.onerror = () => {
        if (runId !== this.analysisRunId || socket !== this.socket) return;
        this.connectionText = "连接错误";
        this.running = false;
        this.analysisStarting = false;
      };
    },
    stopAnalysis(options = {}) {
      if (options.manual) {
        this.manualStopRequested = true;
      }
      this.analysisRunId += 1;
      if (this.socket) {
        if (this.socket.readyState === WebSocket.OPEN) {
          try {
            this.socket.send(JSON.stringify({ type: "stop" }));
          } catch (_err) {
            // Closing the socket is enough; the server will also detect disconnect.
          }
        }
        if (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING) {
          this.socket.close(1000, "client-stop");
        }
        this.socket = null;
      }
      this.stopPlaybackReporter();
      this.running = false;
      this.analysisStarting = false;
      this.connectionText = "已停止";
      this.resetDistractionIntervention();
    },
    playCourseAndStartAnalysis() {
      this.manualStopRequested = false;
      this.startAnalysis();
      const video = this.$refs.courseVideo;
      if (!video) {
        return;
      }
      this.ignorePauseUntil = Date.now() + 2000;
      const playPromise = video.play();
      if (!playPromise || typeof playPromise.then !== "function") {
        return;
      }
      playPromise
        .then(() => {
          this.connectionText = this.running ? this.connectionText : "课程播放中";
        })
        .catch(() => {
          if (this.running || this.analysisStarting) return;
          this.connectionText = "请点击课程视频播放键后再开始检测";
        });
    },
    pauseCourseVideo() {
      const video = this.$refs.courseVideo;
      if (video && !video.paused) {
        this.suppressNextPauseStop = true;
        video.pause();
      }
    },
    getCoursePlaybackTime() {
      const video = this.$refs.courseVideo;
      return video ? Number(video.currentTime || 0) : 0;
    },
    startPlaybackReporter() {
      this.stopPlaybackReporter();
      this.playbackTimer = window.setInterval(() => {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        this.socket.send(JSON.stringify({
          type: "playback",
          lesson_id: this.selectedLessonId,
          playback_time: this.getCoursePlaybackTime(),
        }));
      }, 1000);
    },
    stopPlaybackReporter() {
      if (this.playbackTimer) {
        window.clearInterval(this.playbackTimer);
        this.playbackTimer = null;
      }
    },
    handleCourseLoadedMetadata() {
      this.applyCoursePlaybackRate();
    },
    handleCoursePause() {
      // 视频暂停与分析独立控制，不再自动停止分析
    },
    handleCoursePlay() {
      // 视频播放与分析独立控制，不再自动启动分析
    },
    applyCoursePlaybackRate() {
      const rate = this.desiredPlaybackRate;
      this.coursePlaybackRate = rate;
      nextTick(() => {
        const video = this.$refs.courseVideo;
        if (video) {
          video.playbackRate = rate;
        }
      });
    },
    async sendChatMessage() {
      const text = this.chatInput.trim();
      if (!text || this.chatLoading) return;
      this.chatInput = "";
      this.chatMessages.push({ role: "user", content: text });
      this.chatLoading = true;
      this.chatLastUsage = null;

      const history = this.chatMessages.slice(0, -1).slice(-10);
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), this.chatTimeoutMs);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            message: text,
            history: history,
            lesson_id: this.selectedLessonId,
          }),
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "请求失败 (" + response.status + ")");
        }

        const data = await response.json();
        this.chatMessages.push({ role: "assistant", content: data.response });
        this.chatLastUsage = data.usage;
      } catch (err) {
        const message = err.name === "AbortError"
          ? "AI 连接较慢，我先停止等待。可以稍后重试，或先使用规则反馈继续演示。"
          : err.message;
        this.chatMessages.push({
          role: "assistant",
          content: "\u274c " + message,
        });
      } finally {
        window.clearTimeout(timeoutId);
        this.chatLoading = false;
        nextTick(() => {
          const el = this.$refs.chatMessages;
          if (el) el.scrollTop = el.scrollHeight;
        });
      }
    },
    clearChat() {
      this.chatMessages = [];
      this.chatLastUsage = null;
    },
  },
}).mount("#app");
