/**
 * 线下课堂学生端 — Vue 3 轻量应用
 *
 * 连接 /ws/classroom-stream，实时显示学习状态分析结果。
 * 支持浏览器本机摄像头、服务器摄像头和服务器样例视频。
 */
const { createApp, ref, computed, onMounted, watch } = Vue;

function persistentBrowserClientId() {
  const key = "education-browser-client-id";
  let value = "";
  try {
    value = localStorage.getItem(key) || "";
  } catch (_error) {
    // 某些隐私模式会禁用持久化；本次页面会话仍使用唯一 ID。
  }
  if (!value) {
    const suffix = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    value = `client_web-${suffix}`;
    try { localStorage.setItem(key, value); } catch (_error) {}
  }
  return value;
}

createApp({
  setup() {
    // ---- 配置 ----
    const source = ref("browser_camera");
    const selectedVideo = ref("");
    const selectedCameraId = ref("");
    const cameraDevices = ref([]);
    const cameraError = ref("");
    const cameraReady = ref(false);
    const localVideo = ref(null);
    const analysisMode = ref("standard");
    const analysisModes = [
      { value: "smooth", label: "流畅" },
      { value: "standard", label: "标准" },
      { value: "detail", label: "精细" },
    ];

    // ---- 运行状态 ----
    const running = ref(false);
    const currentImage = ref("");
    const frameIndex = ref(0);
    const latencyMs = ref(0);
    const currentState = ref("unknown");
    const behaviorLabel = ref("");
    const daoStrategy = ref("unknown");
    const edgeConnected = ref(false);
    const reportActive = ref(false);

    // ---- 身份信息 ----
    const seatLabel = ref("");
    const seatInput = ref("");
    const nameInput = ref("");
    const dashboardUrl = ref("");
    const studentVideos = ref([]);
    const deviceId = ref(persistentBrowserClientId());

    // ---- 个人课堂学习报告面板（只展示教师下课后生成的最终报告） ----
    const myReport = ref(null);
    const myReportLoading = ref(false);
    const myReportError = ref("");

    let socket = null;
    let reportCheckTimer = null;
    let reportPanelTimer = null;
    let browserCameraStream = null;
    let frameTimer = null;
    let frameInFlight = false;
    const FRAME_INTERVAL_MS = 1500;
    const captureCanvas = document.createElement("canvas");

    // ---- 计算属性 ----
    const stateLabels = {
      focused: "专注",
      confused: "困惑",
      tired: "疲劳",
      distracted: "分心",
      absent: "缺勤",
      unknown: "等待检测",
    };
    const stateIcons = {
      focused: "✅",
      confused: "🤔",
      tired: "😴",
      distracted: "📱",
      absent: "❌",
      unknown: "⏳",
    };

    const stateLabel = computed(() => stateLabels[currentState.value] || currentState.value);
    const stateIcon = computed(() => stateIcons[currentState.value] || "⏳");
    const latencyText = computed(() => {
      const ms = latencyMs.value;
      if (!ms) return "";
      return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`;
    });
    const daoStrategyText = computed(() => {
      const labels = { offload: "卸载到边缘", local: "本地推理", unknown: "等待决策" };
      return labels[daoStrategy.value] || daoStrategy.value;
    });

    // ---- 个人报告辅助：状态顺序 / 专注等级 / Ovis 评语 ----
    const stateOrder = ["focused", "confused", "tired", "distracted", "absent"];

    function levelClass(level) {
      const map = { 优秀: "excellent", 良好: "good", 一般: "fair", 待提升: "needs" };
      return map[level] || "needs";
    }

    function ovisSections(source) {
      if (!source) return [];
      return [
        { label: "整体评价", text: source.overall_assessment },
        { label: "注意力模式", text: source.attention_pattern },
        { label: "疲劳模式", text: source.fatigue_pattern },
        { label: "学习亮点", text: (source.learning_strengths || []).join("；") },
        { label: "优先改进项", text: (source.priority_improvements || []).join("；") },
        { label: "教师建议", text: source.teacher_suggestion },
        { label: "学生建议", text: source.student_suggestion },
        { label: "下节课目标", text: source.next_class_goal },
      ].filter((item) => item.text);
    }

    const ovisItems = computed(() =>
      ovisSections(myReport.value && (myReport.value.ovis_comment || myReport.value.ovis_report))
    );
    const reportStateRows = computed(() => {
      const report = myReport.value || {};
      const durations = report.durations || {};
      const percentages = report.state_percentages || {};
      return stateOrder.map((key) => ({
        key,
        label: stateLabels[key] || key,
        seconds: durations[key] || 0,
        percent: percentages[key] || 0,
      }));
    });
    const reportStageRows = computed(() => (myReport.value?.stage_trend || []).map((item) => ({
      stage: item.stage || "课堂阶段",
      focus: Number(item.focus_rate || 0),
      distracted: Number(item.distracted_rate || 0),
      tired: Number(item.tired_rate || 0),
    })));

    // ---- 数据加载 ----
    async function loadStatus() {
      try {
        const res = await fetch("/api/status");
        const data = await res.json();
        const s = data.settings || {};
        let savedSeat = "";
        let savedName = "";
        try {
          savedSeat = localStorage.getItem("education-seat-label") || "";
          savedName = localStorage.getItem("education-student-name") || "";
        } catch (_error) {}
        seatLabel.value = savedSeat || s.seat_label || "";
        seatInput.value = seatLabel.value;
        nameInput.value = savedName || s.student_name || "";
        dashboardUrl.value = s.classroom_dashboard_url || "";
        if (s.analysis_mode && analysisModes.find((m) => m.value === s.analysis_mode)) {
          analysisMode.value = s.analysis_mode;
        }
      } catch (e) {
        console.error("loadStatus error:", e);
      }
    }

    async function loadStudentVideos() {
      try {
        const res = await fetch("/api/student-videos");
        const data = await res.json();
        // 后端 list_student_videos 返回 {name, relative_path}，
        // 这里统一映射为前端使用的 {path, label}
        studentVideos.value = (data.videos || []).map((v) => ({
          path: v.relative_path,
          label: v.name,
        }));
      } catch (e) {
        studentVideos.value = [];
      }
    }

    async function saveSeat() {
      seatLabel.value = seatInput.value;
      try {
        localStorage.setItem("education-seat-label", seatInput.value);
        localStorage.setItem("education-student-name", nameInput.value);
      } catch (_error) {}
      // 浏览器客户端身份只保存在本机，避免多个远程学生互相覆盖服务器全局设置。
      if (source.value !== "browser_camera") {
        try {
          const res = await fetch("/api/status");
          const data = await res.json();
          const s = data.settings || {};
          await fetch("/api/settings", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              edge_host: s.edge_host || "localhost",
              edge_port: s.edge_port || 50051,
              use_edge: s.use_edge !== undefined ? s.use_edge : true,
              camera_id: s.camera_id || 0,
              seat_label: seatInput.value,
              student_name: nameInput.value,
            }),
          });
        } catch (e) {
          console.error("saveSeat error:", e);
        }
      }
      if (running.value) {
        stopAnalysis();
        setTimeout(startAnalysis, 300);
      }
    }

    // ---- 浏览器本机摄像头 ----
    function browserCameraAvailable() {
      return window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
    }

    function stopBrowserCamera() {
      if (frameTimer) {
        clearTimeout(frameTimer);
        frameTimer = null;
      }
      frameInFlight = false;
      if (browserCameraStream) {
        browserCameraStream.getTracks().forEach((track) => track.stop());
        browserCameraStream = null;
      }
      if (localVideo.value) {
        localVideo.value.srcObject = null;
      }
      cameraReady.value = false;
    }

    async function refreshCameraDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const devices = await navigator.mediaDevices.enumerateDevices();
      cameraDevices.value = devices
        .filter((item) => item.kind === "videoinput")
        .map((item, index) => ({
          id: item.deviceId,
          label: item.label || `摄像头 ${index + 1}`,
        }));
      if (
        selectedCameraId.value &&
        !cameraDevices.value.some((item) => item.id === selectedCameraId.value)
      ) {
        selectedCameraId.value = "";
      }
    }

    async function startBrowserCamera() {
      stopBrowserCamera();
      cameraError.value = "";
      if (!browserCameraAvailable()) {
        cameraError.value = window.isSecureContext
          ? "当前浏览器不支持摄像头访问"
          : "远程浏览器摄像头需要 HTTPS 安全连接";
        throw new Error(cameraError.value);
      }
      const video = selectedCameraId.value
        ? { deviceId: { exact: selectedCameraId.value }, width: { ideal: 640 }, height: { ideal: 360 } }
        : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 360 } };
      try {
        browserCameraStream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
        if (!localVideo.value) throw new Error("摄像头预览组件尚未就绪");
        localVideo.value.srcObject = browserCameraStream;
        await localVideo.value.play();
        cameraReady.value = true;
        await refreshCameraDevices();
      } catch (error) {
        stopBrowserCamera();
        if (!cameraError.value) {
          if (error?.name === "NotAllowedError") {
            cameraError.value = "摄像头权限被拒绝，请在浏览器地址栏中允许访问";
          } else if (error?.name === "NotFoundError") {
            cameraError.value = "未检测到可用摄像头";
          } else {
            cameraError.value = `无法打开浏览器摄像头：${error?.message || error}`;
          }
        }
        throw error;
      }
    }

    function scheduleBrowserFrame(delay = FRAME_INTERVAL_MS) {
      if (frameTimer) clearTimeout(frameTimer);
      frameTimer = setTimeout(sendBrowserFrame, delay);
    }

    function sendBrowserFrame() {
      frameTimer = null;
      const video = localVideo.value;
      if (
        source.value !== "browser_camera" || !running.value || !cameraReady.value ||
        !video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        !socket || socket.readyState !== WebSocket.OPEN || frameInFlight
      ) {
        if (running.value && source.value === "browser_camera") scheduleBrowserFrame(300);
        return;
      }
      const sourceWidth = video.videoWidth || 640;
      const sourceHeight = video.videoHeight || 360;
      const scale = Math.min(640 / sourceWidth, 360 / sourceHeight, 1);
      captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
      captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
      const context = captureCanvas.getContext("2d", { alpha: false });
      context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      captureCanvas.toBlob((blob) => {
        if (!blob || !socket || socket.readyState !== WebSocket.OPEN || !running.value) {
          scheduleBrowserFrame(300);
          return;
        }
        frameInFlight = true;
        socket.send(blob);
      }, "image/jpeg", 0.78);
    }

    // ---- WebSocket 控制 ----
    let reconnectTimer = null;
    let manualStop = true;       // 初始为 true，只有 startAnalysis 才设 false
    let reconnectAttempts = 0;
    const MAX_RECONNECT = 20;    // 最多自动重连 20 次
    let joinedTeacherSession = false;

    function handleTeacherSession(session = {}) {
      if (session.active) {
        // 只有在学生确实运行分析时观察到上课，才把它视为本次课堂参与者。
        if (running.value) joinedTeacherSession = true;
        return;
      }
      if (joinedTeacherSession && session.ended_at && running.value) {
        joinedTeacherSession = false;
        stopAnalysis();
        console.log("[classroom] 教师已下课，已自动停止分析并释放推理资源");
      }
    }

    function toggleAnalysis() {
      if (running.value) {
        stopAnalysis();
      } else {
        startAnalysis();
      }
    }

    async function startAnalysis(options = {}) {
      if (socket) {
        socket.close();
        socket = null;
      }
      manualStop = false;
      if (!options.reconnecting) reconnectAttempts = 0;
      currentImage.value = "";
      frameIndex.value = 0;
      cameraError.value = "";

      if (source.value === "browser_camera") {
        try {
          await startBrowserCamera();
        } catch (error) {
          manualStop = true;
          running.value = false;
          console.error("browser camera error:", error);
          return;
        }
      } else {
        stopBrowserCamera();
      }
      running.value = true;

      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${protocol}//${location.host}/ws/classroom-stream`);

      socket.onopen = () => {
        reconnectAttempts = 0;   // 连接成功，重置计数
        socket.send(
          JSON.stringify({
            type: "start",
            source: source.value,
            sample_video: source.value === "sample" ? selectedVideo.value : undefined,
            device_id: deviceId.value || undefined,
            seat_label: seatInput.value,
            student_name: nameInput.value,
            analysis_mode: analysisMode.value,
            interval_ms: FRAME_INTERVAL_MS,
            max_frames: 0,
          })
        );
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleMessage(data);
        } catch (e) {
          console.error("parse error:", e);
        }
      };

      socket.onerror = (e) => {
        console.error("WebSocket error:", e);
      };

      socket.onclose = () => {
        running.value = false;
        socket = null;
        stopBrowserCamera();
        stopReportCheck();
        // —— 自动重连：非手动停止时，延迟后自动恢复采集 ——
        if (!manualStop && reconnectAttempts < MAX_RECONNECT) {
          reconnectAttempts++;
          const delay = Math.min(2000 * reconnectAttempts, 15000);
          console.log(`[auto-reconnect] 第 ${reconnectAttempts}/${MAX_RECONNECT} 次，${delay}ms 后重连...`);
          reconnectTimer = setTimeout(() => {
            if (!manualStop) startAnalysis({ reconnecting: true });
          }, delay);
        }
      };

      // 上报检测：每5秒检查一次是否有活跃上报
      startReportCheck();
    }

    function stopAnalysis() {
      manualStop = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "stop" }));
      }
      if (socket) {
        socket.close();
      }
      stopBrowserCamera();
      running.value = false;
      socket = null;
      stopReportCheck();
    }

    function goOnlineMode() {
      // 切换到线上网课模式前，先停止当前分析并中断上传，避免教师端出现重复记录
      stopAnalysis();
      window.location.href = "/";
    }

    function handleMessage(data) {
      if (data.type === "started") {
        if (source.value === "browser_camera") scheduleBrowserFrame(0);
        return;
      }
      if (data.type === "finished") {
        stopAnalysis();
        return;
      }
      if (data.type === "frame_error") {
        frameInFlight = false;
        cameraError.value = data.message || "浏览器摄像头帧上传失败";
        if (source.value === "browser_camera") scheduleBrowserFrame();
        return;
      }
      if (data.type === "error") {
        cameraError.value = data.message || "分析服务发生错误";
        frameInFlight = false;
        console.error("server error:", data.message);
        stopAnalysis();
        return;
      }
      if (data.type !== "analysis") return;

      frameIndex.value = data.frame_index || 0;
      latencyMs.value = data.elapsed_ms || 0;

      // 更新画面
      if (data.image) {
        currentImage.value = data.image;
      }

      // 更新学习状态
      const report = data.report || {};
      const stateCounts = report.state_counts || {};
      const dominantState = Object.entries(stateCounts).sort(
        (a, b) => b[1] - a[1]
      )[0];
      if (dominantState && dominantState[1] > 0) {
        currentState.value = dominantState[0];
      }

      // 更新行为标签
      const needsAttention = report.needs_attention || [];
      if (needsAttention.length > 0) {
        const reasons = needsAttention[0].reasons || [];
        behaviorLabel.value = reasons.join(", ") || "";
      } else {
        behaviorLabel.value = "";
      }

      // 更新 DAO 策略
      const edge = data.edge || {};
      const dao = edge.dao || {};
      daoStrategy.value = dao.strategy || edge.strategy || "unknown";
      edgeConnected.value = edge.connected || false;
      if (source.value === "browser_camera") {
        frameInFlight = false;
        scheduleBrowserFrame();
      }
    }

    // ---- 上报检测 ----
    function startReportCheck() {
      stopReportCheck();
      reportCheckTimer = setInterval(() => {
        reportActive.value = !!dashboardUrl.value;
      }, 5000);
      reportActive.value = !!dashboardUrl.value;
    }

    function stopReportCheck() {
      if (reportCheckTimer) {
        clearInterval(reportCheckTimer);
        reportCheckTimer = null;
      }
      reportActive.value = false;
    }

    // ---- 个人课堂学习报告（教师管理端 :8008 生成） ----
    // manual=true：面板「刷新」按钮触发，失败时展示错误；
    // manual=false：定时检查下课后的最终报告；上课期间服务端不会临时生成报告。
    async function loadMyReport(manual = false) {
      const base = dashboardUrl.value.replace(/\/+$/, "");
      if (!base || !deviceId.value) {
        if (manual) {
          myReportError.value = "未配置管理端地址或设备标识，无法生成报告";
        }
        return;
      }
      myReportLoading.value = true;
      if (manual) {
        myReportError.value = "";
      }
      try {
        const url = `${base}/api/students/${encodeURIComponent(deviceId.value)}/report`;
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          const session = body.session || {};
          handleTeacherSession(session);
          // 新课开始后立即清除上一节课的本地展示；上课和生成期间只显示状态提示。
          if (session.active) {
            myReport.value = null;
            myReportError.value = "";
          }
          if (manual) {
            myReportError.value = body.error || "报告尚未生成";
          }
        } else {
          myReport.value = await res.json();
          handleTeacherSession(myReport.value.session || {});
          myReportError.value = "";
        }
      } catch (e) {
        if (manual) {
          myReportError.value = "无法连接管理端服务器，请确认教师端已启动";
        }
        console.error("loadMyReport error:", e);
      } finally {
        myReportLoading.value = false;
      }
    }

    function formatDuration(seconds) {
      const value = Number(seconds);
      if (!Number.isFinite(value)) return "--";
      const m = Math.floor(value / 60);
      const s = Math.round(value % 60);
      return m > 0 ? `${m}分${s}秒` : `${s}秒`;
    }

    // ---- 模式切换时如果正在运行，重启 ----
    watch(analysisMode, () => {
      if (running.value) {
        stopAnalysis();
        setTimeout(startAnalysis, 300);
      }
    });
    watch(source, (newSource, oldSource) => {
      cameraError.value = "";
      if (oldSource === "browser_camera" && newSource !== "browser_camera") {
        stopBrowserCamera();
      }
      if (running.value) {
        stopAnalysis();
        setTimeout(startAnalysis, 300);
      }
    });
    watch(selectedCameraId, () => {
      if (running.value && source.value === "browser_camera") {
        stopAnalysis();
        setTimeout(startAnalysis, 300);
      }
    });

    // ---- 生命周期 ----
    // URL 参数 auto_start=1 时自动开始采集（树莓派 kiosk 场景）
    const autoStart = new URLSearchParams(location.search).get("auto_start") === "1";
    onMounted(async () => {
      window.addEventListener("pagehide", () => {
        // 页面卸载兜底：关闭分析连接，防止后台上报残留
        manualStop = true;
        stopBrowserCamera();
        if (socket && socket.readyState === WebSocket.OPEN) {
          try { socket.close(1000, "leave"); } catch (_e) {}
        }
        if (reportPanelTimer) {
          clearInterval(reportPanelTimer);
          reportPanelTimer = null;
        }
      });
      await Promise.all([loadStatus(), loadStudentVideos()]);
      // 只检查最终报告；接口在上课期间返回状态，不执行实时报告聚合。
      if (dashboardUrl.value && deviceId.value) {
        loadMyReport(false);
        reportPanelTimer = setInterval(() => loadMyReport(false), 5000);
      }
      // 自动启动：等待数据加载后延迟 500ms 启动采集
      if (autoStart) {
        setTimeout(() => startAnalysis(), 500);
      }
    });
    return {
      source,
      selectedVideo,
      selectedCameraId,
      cameraDevices,
      cameraError,
      cameraReady,
      localVideo,
      analysisMode,
      analysisModes,
      running,
      currentImage,
      frameIndex,
      latencyMs,
      currentState,
      behaviorLabel,
      daoStrategy,
      edgeConnected,
      reportActive,
      seatLabel,
      seatInput,
      nameInput,
      dashboardUrl,
      studentVideos,
      deviceId,
      myReport,
      myReportLoading,
      myReportError,
      stateLabels,
      loadMyReport,
      levelClass,
      ovisItems,
      reportStateRows,
      reportStageRows,
      formatDuration,
      stateLabel,
      stateIcon,
      latencyText,
      daoStrategyText,
      toggleAnalysis,
      saveSeat,
      goOnlineMode,
    };
  },
}).mount("#app");
