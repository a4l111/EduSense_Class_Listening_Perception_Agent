/**
 * 线下课堂教师管理端 — Vue 3 前端应用
 *
 * 每 2 秒轮询一次后端 API，展示所有学生的实时学习状态。
 */
const { createApp, ref, computed, onMounted, onUnmounted } = Vue;

createApp({
  setup() {
    // ---- 响应式状态 ----
    const students = ref([]);
    const summary = ref({});
    const alerts = ref([]);
    const loading = ref(true);
    const filter = ref("all"); // all | attention | offline
    const clockText = ref("");
    const currentHost = ref(window.location.origin);

    // ---- 课堂会话（开始上课 / 下课·生成报告） ----
    const teacherSession = ref({ active: false });

    // ---- 全班报告内展开查看个人报告 ----
    const expandedStudent = ref(null);
    const personalReport = ref(null);
    const personalLoading = ref(false);

    let pollTimer = null;
    let clockTimer = null;

    // ---- 计算属性 ----
    const totalStudents = computed(() => students.value.length);
    const onlineCount = computed(() => students.value.filter((s) => s.online).length);
    const attentionCount = computed(
      () =>
        students.value.filter(
          (s) => s.online && ["distracted", "absent", "tired"].includes(s.display_state)
        ).length
    );

    const filterOptions = computed(() => [
      { value: "all", label: "全部", count: students.value.length },
      {
        value: "attention",
        label: "需关注",
        count: students.value.filter(
          (s) => s.online && ["distracted", "absent", "tired"].includes(s.display_state)
        ).length,
      },
      {
        value: "offline",
        label: "离线",
        count: students.value.filter((s) => !s.online).length,
      },
    ]);

    const filteredStudents = computed(() => {
      if (filter.value === "attention") {
        return students.value.filter(
          (s) => s.online && ["distracted", "absent", "tired"].includes(s.display_state)
        );
      }
      if (filter.value === "offline") {
        return students.value.filter((s) => !s.online);
      }
      return students.value;
    });

    const stateBarItems = computed(() => {
      const ss = summary.value.state_summary || {};
      return [
        { key: "focused", label: "专注", count: ss.focused || 0 },
        { key: "confused", label: "困惑", count: ss.confused || 0 },
        { key: "tired", label: "疲劳", count: ss.tired || 0 },
        { key: "distracted", label: "分心", count: ss.distracted || 0 },
        { key: "absent", label: "缺勤", count: ss.absent || 0 },
        { key: "offline", label: "离线", count: ss.offline || 0 },
      ];
    });

    const teacherReport = computed(() => teacherSession.value.report || null);

    // ---- 状态 / 专注等级 / Ovis 评语辅助 ----
    const stateOrder = ["focused", "confused", "tired", "distracted", "absent"];
    const stateNameMap = {
      focused: "专注",
      confused: "困惑",
      tired: "疲劳",
      distracted: "分心",
      absent: "缺勤",
    };

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
        { label: "局限性说明", text: source.limitations },
      ].filter((item) => item.text);
    }

    const classOvisItems = computed(() => ovisSections(teacherReport.value && teacherReport.value.ovis_report));
    const personalOvisItems = computed(() =>
      ovisSections(personalReport.value && (personalReport.value.ovis_comment || personalReport.value.ovis_report))
    );
    const personalStateRows = computed(() => {
      const report = personalReport.value || {};
      const durations = report.durations || {};
      const percentages = report.state_percentages || {};
      return stateOrder.map((key) => ({
        key,
        label: stateNameMap[key] || key,
        seconds: durations[key] || 0,
        percent: percentages[key] || 0,
      }));
    });

    async function toggleStudentReport(deviceId) {
      if (expandedStudent.value === deviceId) {
        expandedStudent.value = null;
        return;
      }
      expandedStudent.value = deviceId;
      personalReport.value = null;
      personalLoading.value = true;
      try {
        const res = await fetch(`/api/students/${encodeURIComponent(deviceId)}/report`, { cache: "no-store" });
        if (res.ok) {
          personalReport.value = await res.json();
        } else {
          const body = await res.json().catch(() => ({}));
          personalReport.value = { error: body.error || "个人报告加载失败" };
        }
      } catch (e) {
        personalReport.value = { error: "个人报告加载失败，请稍后重试" };
        console.error("toggleStudentReport error:", e);
      } finally {
        personalLoading.value = false;
      }
    }

    // ---- 方法 ----
    function daoLabel(strategy) {
      const labels = {
        offload: "卸载",
        local: "本地",
        unknown: "等待",
      };
      return labels[strategy] || strategy;
    }

    function formatSeconds(sec) {
      const s = Math.round(sec || 0);
      if (s < 60) return s + "s";
      const m = Math.floor(s / 60);
      return m + "m" + (s % 60) + "s";
    }

    function updateClock() {
      const now = new Date();
      clockText.value =
        now.getHours().toString().padStart(2, "0") +
        ":" +
        now.getMinutes().toString().padStart(2, "0") +
        ":" +
        now.getSeconds().toString().padStart(2, "0");
    }

    async function loadStudents() {
      try {
        const res = await fetch("/api/students");
        const data = await res.json();
        students.value = data.students || [];
      } catch (e) {
        console.error("loadStudents error:", e);
      }
    }

    async function loadSummary() {
      try {
        const res = await fetch("/api/summary");
        summary.value = await res.json();
      } catch (e) {
        console.error("loadSummary error:", e);
      }
    }

    async function loadAlerts() {
      try {
        const res = await fetch("/api/alerts");
        const data = await res.json();
        alerts.value = data.alerts || [];
      } catch (e) {
        console.error("loadAlerts error:", e);
      }
    }

    async function loadAll() {
      loading.value = true;
      await Promise.all([loadStudents(), loadSummary(), loadAlerts(), loadTeacherSession()]);
      loading.value = false;
    }

    async function clearAll() {
      if (!confirm("确认清空所有学生数据？")) return;
      try {
        await fetch("/api/clear", { method: "POST" });
        await loadAll();
      } catch (e) {
        console.error("clearAll error:", e);
      }
    }

    // ---- 课堂会话控制 ----
    async function loadTeacherSession() {
      try {
        const res = await fetch("/api/teacher/session");
        teacherSession.value = await res.json();
        // 新一节课开始时收起上一节的个人报告展开
        if (teacherSession.value.active) {
          expandedStudent.value = null;
          personalReport.value = null;
        }
      } catch (e) {
        console.error("loadTeacherSession error:", e);
      }
    }

    async function startSession() {
      if (!confirm("从现在开始统计本次课堂的所有学生吗？")) return;
      try {
        await fetch("/api/teacher/session/start", { method: "POST" });
        await loadTeacherSession();
      } catch (e) {
        console.error("startSession error:", e);
        alert("开始上课失败，请检查管理端服务");
      }
    }

    async function finishSession() {
      if (!confirm("结束统计并异步生成全班与个人分析报告吗？")) return;
      try {
        const res = await fetch("/api/teacher/session/finish", { method: "POST" });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          alert(body.error || "下课操作失败");
          return;
        }
        await loadTeacherSession();
        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
      } catch (e) {
        console.error("finishSession error:", e);
        alert("下课操作失败，请检查管理端服务");
      }
    }

    // ---- 生命周期 ----
    onMounted(async () => {
      updateClock();
      clockTimer = setInterval(updateClock, 1000);
      await loadAll();
      pollTimer = setInterval(loadAll, 2000);
    });

    onUnmounted(() => {
      if (pollTimer) clearInterval(pollTimer);
      if (clockTimer) clearInterval(clockTimer);
    });

    return {
      students,
      summary,
      alerts,
      loading,
      filter,
      clockText,
      currentHost,
      teacherSession,
      teacherReport,
      expandedStudent,
      personalReport,
      personalLoading,
      toggleStudentReport,
      levelClass,
      classOvisItems,
      personalOvisItems,
      personalStateRows,
      totalStudents,
      onlineCount,
      attentionCount,
      filterOptions,
      filteredStudents,
      stateBarItems,
      daoLabel,
      formatSeconds,
      loadAll,
      clearAll,
      startSession,
      finishSession,
    };
  },
}).mount("#app");
