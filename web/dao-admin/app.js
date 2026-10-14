const { createApp, computed, onMounted, onUnmounted, reactive, ref } = Vue;

function numberLabel(value, suffix = "") {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric))
    return `0${suffix}`;
  return `${Math.round(numeric * 10) / 10}${suffix}`;
}

/* 通俗化文案：把「卸载 / 本地执行 / 准入」等专业词换成大众用户能看懂的说法 */
function strategyLabel(strategy) {
  if (strategy === "offload")
    return "交给服务器处理";
  if (strategy === "local")
    return "本机直接处理";
  if (strategy === "sampling")
    return "抽帧观察中";
  return "等待安排";
}

function admissionLabel(admission) {
  if (admission === "accept")
    return "已接收";
  if (String(admission || "").startsWith("local"))
    return "本机处理";
  if (admission === "reject")
    return "服务器繁忙，暂缓接收";
  if (admission === "fallback")
    return "按备用规则处理";
  if (admission === "waiting")
    return "等待任务";
  return admission || "排队中";
}

function strategyClass(strategy) {
  if (strategy === "offload")
    return "is-offload";
  if (strategy === "local")
    return "is-local";
  return "is-waiting";
}

function priorityText(percent) {
  if (percent >= 70)
    return "高 · 优先尽快处理";
  if (percent >= 40)
    return "中 · 正常排队";
  return "低 · 空闲时处理";
}

function userLabel(value) {
  const raw = String(value || "未知用户");
  const short = raw.replace(/^client_/, "").split("__", 1)[0];
  return short.startsWith("用户") ? short : `用户 ${short}`;
}

const StatCard = {
  props: ["title", "value", "hint", "tone"],
  template: `
    <article class="stat-card" :class="'tone-' + tone">
      <div class="stat-card__top">
        <span>{{ title }}</span>
        <i></i>
      </div>
      <strong>{{ value }}</strong>
      <small>{{ hint }}</small>
    </article>
  `,
};

const DeviceCard = {
  props: ["device"],
  methods: { numberLabel, strategyLabel, strategyClass },
  template: `
    <article class="device-card" :class="device.status">
      <div class="device-card__head">
        <div>
          <strong>{{ device.name }}</strong>
          <span>{{ device.role }}</span>
        </div>
        <em>{{ device.status === 'active' ? '使用中' : device.status === 'online' ? '在线' : '待接入' }}</em>
      </div>
      <div class="device-card__body">
        <span>已提交任务 <b>{{ device.tasks }}</b></span>
        <span>平均耗时 <b>{{ numberLabel(device.latency_ms, ' ms') }}</b></span>
      </div>
      <div class="device-card__foot">
        <span :class="['tag', strategyClass(device.strategy)]">{{ strategyLabel(device.strategy) }}</span>
        <span>{{ device.state }}</span>
      </div>
    </article>
  `,
};

const DaoPanel = {
  props: ["dao"],
  methods: { admissionLabel, numberLabel, strategyLabel, strategyClass },
  computed: {
    priorityPercent() {
      const raw = Number(this.dao?.priority || 0);
      const normalized = raw <= 1 ? raw * 100 : raw;
      return Math.max(0, Math.min(100, normalized));
    },
    stageText() {
      const stages = this.dao?.stages_to_offload || [];
      if (!Array.isArray(stages) || stages.length === 0)
        return "当前没有交给服务器的步骤";
      return stages.join("、");
    },
  },
  template: `
    <section class="panel dao-panel">
      <div class="section-heading">
        <span>决策</span>
        <strong>这一帧怎么安排</strong>
      </div>
      <div class="decision-hero" :class="strategyClass(dao.strategy)">
        <span>{{ strategyLabel(dao.strategy) }}</span>
        <strong>{{ admissionLabel(dao.admission) }}</strong>
      </div>
      <div class="meter-block">
        <div class="meter-label">
          <span>任务优先级</span>
          <b>{{ numberLabel(priorityPercent, '%') }}</b>
        </div>
        <div class="meter"><i :style="{ width: priorityPercent + '%' }"></i></div>
      </div>
      <div class="decision-grid">
        <span>交给哪台服务器</span><b>{{ dao.target_node_id || '云边协同服务器' }}</b>
        <span>处理步骤数</span><b>{{ dao.stage_count || 5 }}</b>
        <span>交给服务器的步骤</span><b>{{ stageText }}</b>
      </div>
    </section>
  `,
};

const PipelineFlow = {
  props: ["pipeline"],
  methods: {
    placementText(placement) {
      if (placement === "edge" || placement === "offload")
        return "服务器处理";
      if (placement === "local")
        return "本机处理";
      return placement || "--";
    },
  },
  template: `
    <section class="panel pipeline-panel">
      <div class="section-heading">
        <span>处理步骤</span>
        <strong>一帧画面的处理路径</strong>
      </div>
      <div class="pipeline-flow">
        <article v-for="stage in pipeline" :key="stage.id" class="stage-card" :class="{ offloaded: stage.offloaded }">
          <small>{{ stage.model }}</small>
          <strong>{{ stage.name }}</strong>
          <span>{{ placementText(stage.placement) }}</span>
        </article>
        <div v-if="pipeline.length === 0" class="empty-events">
          等待任务提交后，这里会展示每一帧经过的处理步骤。
        </div>
      </div>
    </section>
  `,
};

const ResourcePanel = {
  props: ["server", "comparison", "summary"],
  methods: { numberLabel },
  computed: {
    cpuWidth() {
      return `${Math.max(0, Math.min(100, Number(this.server?.cpu || 0)))}%`;
    },
    gpuWidth() {
      return `${Math.max(0, Math.min(100, Number(this.server?.gpu || 0)))}%`;
    },
    memoryWidth() {
      return `${Math.max(0, Math.min(100, Number(this.server?.memory || 0)))}%`;
    },
  },
  template: `
    <section class="panel resource-panel">
      <div class="section-heading">
        <span>资源</span>
        <strong>服务器当前状态</strong>
      </div>
      <div class="server-status">
        <div>
          <strong>{{ server.name }}</strong>
          <span>{{ server.host }}:{{ server.port }}</span>
        </div>
        <em :class="{ online: server.connected }">{{ server.connected ? '已连接' : '未连接' }}</em>
      </div>
      <div class="resource-list">
        <label>处理器 <b>{{ numberLabel(server.cpu, '%') }}</b></label>
        <div class="bar"><i class="cpu" :style="{ width: cpuWidth }"></i></div>
        <label>显卡 <b>{{ numberLabel(server.gpu, '%') }}</b></label>
        <div class="bar"><i class="gpu" :style="{ width: gpuWidth }"></i></div>
        <label>内存 <b>{{ numberLabel(server.memory, '%') }}</b></label>
        <div class="bar"><i class="memory" :style="{ width: memoryWidth }"></i></div>
      </div>
      <div class="comparison-box">
        <span>本机处理估计 {{ numberLabel(comparison.local_baseline_ms, ' ms') }}</span>
        <span>交给服务器估计 {{ numberLabel(comparison.offload_estimate_ms, ' ms') }}</span>
        <strong>能省 {{ numberLabel(comparison.saved_ms, ' ms') }}</strong>
      </div>
      <div class="slo-box" :class="{ pass: summary.slo_pass }">
        目标 {{ numberLabel(summary.slo_ms, ' ms') }} 内完成 · 平均 {{ numberLabel(summary.avg_latency_ms, ' ms') }}
      </div>
    </section>
  `,
};

const EventTable = {
  props: ["events"],
  methods: { admissionLabel, numberLabel, strategyLabel, strategyClass, userLabel },
  template: `
    <section class="panel event-panel">
      <div class="section-heading">
        <span>日志</span>
        <strong>处理记录</strong>
      </div>
      <div class="event-table">
        <div class="event-row event-head">
          <span>时间</span>
          <span>用户</span>
          <span>处理方式</span>
          <span>接收结果</span>
          <span>耗时</span>
          <span>等待耗时</span>
        </div>
        <div v-if="events.length === 0" class="empty-events">
          等待学生设备提交第一条任务。
        </div>
        <div v-for="event in events" :key="event.id" class="event-row">
          <span>{{ event.time_label }}</span>
          <span>{{ userLabel(event.device_id) }}</span>
          <span :class="['tag', strategyClass(event.strategy)]">{{ strategyLabel(event.strategy) }}</span>
          <span>{{ admissionLabel(event.admission) }}</span>
          <span>{{ numberLabel(event.latency_ms, ' ms') }}</span>
          <span>{{ numberLabel(event.queue || 0, ' ms') }}</span>
        </div>
      </div>
    </section>
  `,
};

/* ── 时间线图 1：任务分布热力图（谁的任务、何时处理、在哪里处理） ── */
const DeviceGantt = {
  props: ["events", "devices", "nowSeconds"],
  computed: {
    rows() {
      const now = Number(this.nowSeconds || Date.now() / 1000);
      const windowSec = 60;
      const start = now - windowSec;
      const bucketSec = 2;
      const colCount = 30;
      const bucketNow = Math.floor(now / bucketSec) * bucketSec;
      const firstBucket = bucketNow - (colCount - 1) * bucketSec;
      const devList = this.devices && this.devices.length
        ? this.devices
        : [{ id: "rpi-client-01", name: "树莓派" }];
      const buckets = {};
      for (const d of devList) {
        buckets[d.id] = Array.from({ length: colCount }, () => ({ offload: 0, local: 0 }));
      }
      for (const evt of (this.events || [])) {
        const ts = Number(evt.timestamp || 0);
        const idx = Math.floor((ts - firstBucket) / bucketSec);
        if (idx < 0 || idx >= colCount) continue;
        const did = evt.device_id || "rpi-client-01";
        if (!buckets[did]) continue;
        if (evt.strategy === "offload") buckets[did][idx].offload++;
        else if (evt.strategy === "local" || evt.strategy === "sampling") buckets[did][idx].local++;
      }
      return devList.map(d => ({
        id: d.id,
        name: d.name || d.id,
        cells: buckets[d.id].map((c, index) => {
          const total = c.offload + c.local;
          const dominant = c.offload >= c.local ? "offload" : "local";
          const count = dominant === "offload" ? c.offload : c.local;
          const timestamp = firstBucket + index * bucketSec;
          return {
            timestamp,
            cls: total === 0 ? "cell-idle" : dominant === "offload" ? "cell-offload" : "cell-local",
            style: {
              left: `${((timestamp - start) / windowSec) * 100}%`,
              width: `${(bucketSec / windowSec) * 100}%`,
              opacity: total === 0 ? "1" : String(Math.max(0.25, Math.min(1, count / 4))),
            },
            title: total === 0 ? "空闲" : `${total} 个任务 · 服务器处理 ${c.offload} · 本机处理 ${c.local}`,
          };
        }),
      }));
    },
    timeLabels() {
      return [
        { pos: "0%", text: "1分钟前" },
        { pos: "50%", text: "30秒前" },
        { pos: "100%", text: "现在" },
      ];
    },
  },
  template: `
    <section class="panel gantt-panel device-gantt">
      <div class="section-heading">
        <span>时间线</span>
        <strong>设备任务热力图 · 近 60 秒</strong>
      </div>
      <div class="gantt-wrapper">
        <div class="gantt-y-axis">
          <div class="gantt-axis-spacer"></div>
          <div v-for="row in rows" :key="row.id" class="gantt-y-label" :title="row.name">{{ row.name }}</div>
        </div>
        <div class="gantt-main">
          <div class="gantt-x-axis">
            <span v-for="tl in timeLabels" :key="tl.text" :style="{ left: tl.pos }">{{ tl.text }}</span>
          </div>
          <div class="gantt-rows">
            <div v-for="row in rows" :key="row.id" class="gantt-row-dev">
              <div v-for="cell in row.cells" :key="cell.timestamp"
                class="gantt-cell" :class="cell.cls" :style="cell.style"
                :title="cell.title"></div>
            </div>
          </div>
        </div>
      </div>
      <div class="gantt-legend">
        <i class="swatch swatch-offload"></i><span>交给服务器处理</span>
        <i class="swatch swatch-local"></i><span>本机直接处理</span>
        <i class="swatch swatch-idle"></i><span>空闲</span>
      </div>
    </section>
  `,
};

/* ── 分步处理时间线：用真实阶段起止时间展示多设备并发 ── */
const StageTimeline = {
  props: ["events", "devices", "nowSeconds"],
  methods: { numberLabel },
  computed: {
    rows() {
      const now = Number(this.nowSeconds || Date.now() / 1000);
      const windowSec = 20;
      const startWindow = now - windowSec;
      const deviceNames = Object.fromEntries((this.devices || []).map(d => [d.id, d.name || d.id]));
      const rows = {};
      const stageMeta = {
        "Frame Upload": ["上传画面", "stage-upload"],
        "Decode": ["读取画面", "stage-decode"],
        "YOLO Detect": ["人物检测", "stage-detect"],
        "Track": ["目标跟踪", "stage-track"],
        "ReID": ["身份关联", "stage-track"],
        "YOLO-pose": ["姿态分析", "stage-pose"],
        "State Fusion": ["状态识别", "stage-state"],
        "Response": ["返回结果", "stage-response"],
      };
      for (const evt of (this.events || [])) {
        const did = evt.device_id || "未知设备";
        if (!rows[did]) rows[did] = { id: did, name: deviceNames[did] || `用户 ${did}`, bars: [] };
        const source = (evt.actual_schedule || []).length ? evt.actual_schedule : (evt.planned_schedule || []);
        for (const item of source) {
          const meta = stageMeta[item.stage];
          if (!meta) continue;
          const started = Number(item.actual_start || item.planned_start || evt.timestamp || 0);
          let ended = Number(item.actual_end || item.planned_end || 0);
          if (!ended) ended = started + Number(item.duration_ms || 0) / 1000;
          if (!started || ended < startWindow || started > now + 1) continue;
          const left = (started - startWindow) / windowSec * 100;
          const width = Math.max(0.8, (ended - started) / windowSec * 100);
          rows[did].bars.push({
            key: `${evt.request_id || evt.id || evt.frame_index}-${item.stage}`,
            left: Math.max(-5, left),
            width: Math.min(100, width),
            label: meta[0],
            cls: meta[1],
            duration: Number(item.duration_ms || (ended - started) * 1000),
            frame: evt.frame_index ?? "—",
          });
        }
      }
      return Object.values(rows).filter(row => row.bars.length);
    },
  },
  template: `
    <section class="panel gantt-panel stage-timeline-panel">
      <div class="section-heading">
        <span>并发处理</span>
        <strong>各设备分步处理时间线 · 近 20 秒</strong>
      </div>
      <p class="chart-explain">不同设备位于不同横行；同一时刻出现多行彩色区段，表示服务器正在并发处理多个设备。</p>
      <div class="gantt-wrapper">
        <div class="gantt-y-axis">
          <div class="gantt-axis-spacer"></div>
          <div v-for="row in rows" :key="row.id" class="gantt-y-label" :title="row.name">{{ row.name }}</div>
          <div v-if="rows.length === 0" class="gantt-y-label gantt-muted">等待数据</div>
        </div>
        <div class="gantt-main">
          <div class="gantt-x-axis"><span style="left:0">20秒前</span><span style="left:50%">10秒前</span><span style="left:100%">现在</span></div>
          <div class="stage-timeline-area">
            <div v-for="row in rows" :key="row.id" class="stage-timeline-row">
              <div v-for="bar in row.bars" :key="bar.key" class="stage-timeline-bar" :class="bar.cls"
                :style="{ left: bar.left + '%', width: bar.width + '%' }"
                :title="row.name + ' · 第 ' + bar.frame + ' 帧 · ' + bar.label + ' · ' + numberLabel(bar.duration, ' ms')"></div>
            </div>
            <div v-if="rows.length === 0" class="stream-row stream-empty">等待学生设备上传画面</div>
          </div>
        </div>
      </div>
      <div class="gantt-legend stage-legend">
        <i class="swatch stage-upload"></i><span>上传/读取画面</span>
        <i class="swatch stage-detect"></i><span>人物检测</span>
        <i class="swatch stage-pose"></i><span>姿态分析</span>
        <i class="swatch stage-state"></i><span>状态识别</span>
        <i class="swatch stage-response"></i><span>返回结果</span>
      </div>
    </section>
  `,
};

/* ── 逐帧处理：每一帧在哪里处理、花了多久 ── */
const StreamGantt = {
  props: ["events", "devices", "nowSeconds"],
  methods: { numberLabel, admissionLabel },
  computed: {
    deviceStreams() {
      const now = Number(this.nowSeconds || Date.now() / 1000);
      const windowSec = 20;
      const start = now - windowSec;
      const devMap = {};
      for (const d of (this.devices || [])) devMap[d.id] = d.name || d.id;
      const byDevice = {};
      for (const evt of (this.events || [])) {
        const ts = Number(evt.timestamp || 0);
        if (ts < start) continue;
        const did = evt.device_id || "rpi-client-01";
        if (!byDevice[did]) byDevice[did] = { id: did, name: devMap[did] || did, bars: [] };
        const leftPct = (ts - start) / windowSec * 100;
        const barWidth = Math.min(12, Math.max(1.2, Number(evt.latency_ms || 0) / 40));
        byDevice[did].bars.push({
          key: evt.request_id || evt.id || `${did}-${evt.frame_index}-${ts}`,
          left: leftPct,
          width: barWidth,
          strategy: evt.strategy || "waiting",
          latency: Number(evt.latency_ms || 0),
          frame: evt.frame_index,
          admission: evt.admission ? admissionLabel(evt.admission) : "",
        });
      }
      return Object.values(byDevice);
    },
    timeLabels() {
      return [
        { pos: "0%", text: "20秒前" },
        { pos: "50%", text: "10秒前" },
        { pos: "100%", text: "现在" },
      ];
    },
  },
  template: `
    <section class="panel gantt-panel stream-gantt">
      <div class="section-heading">
        <span>逐帧处理</span>
        <strong>用户活跃与帧处理 · 近 20 秒</strong>
      </div>
      <div class="gantt-wrapper">
        <div class="gantt-y-axis">
          <div class="gantt-axis-spacer"></div>
          <div v-for="stream in deviceStreams" :key="stream.id" class="gantt-y-label" :title="stream.name">{{ stream.name }}</div>
          <div v-if="deviceStreams.length === 0" class="gantt-y-label gantt-muted">等待数据</div>
        </div>
        <div class="gantt-main">
          <div class="gantt-x-axis">
            <span v-for="tl in timeLabels" :key="tl.text" :style="{ left: tl.pos }">{{ tl.text }}</span>
          </div>
          <div class="stream-area">
            <div v-for="stream in deviceStreams" :key="stream.id" class="stream-row">
              <div v-for="bar in stream.bars" :key="bar.key"
                class="stream-bar"
                :class="'bar-' + bar.strategy"
                :style="{ left: bar.left + '%', width: bar.width + '%' }"
                :title="'第 ' + bar.frame + ' 帧 · 耗时 ' + numberLabel(bar.latency, ' ms') + ' · ' + bar.admission">
              </div>
            </div>
            <div v-if="deviceStreams.length === 0" class="stream-row stream-empty">
              等待视频画面处理数据
            </div>
          </div>
        </div>
      </div>
      <div class="gantt-legend">
        <i class="swatch swatch-offload"></i><span>交给服务器处理</span>
        <i class="swatch swatch-local"></i><span>本机直接处理</span>
        <i class="swatch swatch-waiting"></i><span>等待中</span>
      </div>
    </section>
  `,
};

createApp({
  components: {
    DaoPanel,
    DeviceCard,
    DeviceGantt,
    EventTable,
    StageTimeline,
    PipelineFlow,
    ResourcePanel,
    StatCard,
    StreamGantt,
  },
  setup() {
    const snapshot = ref(null);
    const loading = ref(true);
    const error = ref("");
    const lastUpdated = ref("--");
    const timelineNow = ref(Date.now() / 1000);
    let timer = null;
    let timelineFrame = null;

    const fallback = {
      summary: {
        total_requests: 0,
        offload_count: 0,
        local_count: 0,
        accepted_count: 0,
        offload_ratio: 0,
        avg_latency_ms: 0,
        slo_ms: 200,
        slo_pass: false,
        online_devices: 0,
      },
      server: {
        name: "云边协同服务器",
        host: "--",
        port: 50051,
        connected: false,
        cpu: 0,
        gpu: 0,
        memory: 0,
        queue: 0,
      },
      dao: { strategy: "waiting", admission: "waiting" },
      devices: [],
      pipeline: [],
      comparison: { local_baseline_ms: 0, offload_estimate_ms: 0, saved_ms: 0 },
      events: [],
    };

    const data = computed(() => snapshot.value || fallback);
    const summary = computed(() => data.value.summary || fallback.summary);
    const server = computed(() => data.value.server || fallback.server);
    const dao = computed(() => data.value.dao || fallback.dao);
    const devices = computed(() => data.value.devices || []);
    const pipeline = computed(() => data.value.pipeline || []);
    const comparison = computed(() => data.value.comparison || fallback.comparison);
    const events = computed(() => data.value.events || []);

    const priorityPercent = computed(() => {
      const raw = Number(dao.value?.priority || 0);
      const normalized = raw <= 1 ? raw * 100 : raw;
      return Math.max(0, Math.min(100, normalized));
    });

    /* ── 分区勾选：用户·队列·优先级 / 时间线 / 逐帧处理 / 日志 ── */
    const STORAGE_KEY = "edu-admin-section-toggles-v1";
    const sectionKeys = [
      { key: "users", label: "用户 · 队列 · 优先级" },
      { key: "detail", label: "设备 · 安排 · 资源 · 处理步骤" },
      { key: "timeline", label: "处理时间线" },
      { key: "stream", label: "设备活跃与帧处理" },
      { key: "logs", label: "日志" },
    ];
    const visible = reactive({ users: true, detail: true, timeline: true, stream: true, logs: true });
    (function restoreToggles() {
      try {
        const raw = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "null");
        if (raw && typeof raw === "object") {
          for (const key of Object.keys(visible)) {
            if (typeof raw[key] === "boolean")
              visible[key] = raw[key];
          }
        }
      }
      catch (err) {
        /* 本地缓存不可用时使用默认全部显示 */
      }
    })();

    function toggleSection(key) {
      visible[key] = !visible[key];
      try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(visible));
      }
      catch (err) {
        /* 忽略持久化失败 */
      }
    }

    const statCards = computed(() => [
      {
        title: "已处理任务",
        value: summary.value.total_requests,
        hint: "来自学生终端的视频分析请求",
        tone: "blue",
      },
      {
        title: "交给服务器比例",
        value: numberLabel(summary.value.offload_ratio, "%"),
        hint: `服务器处理 ${summary.value.offload_count} 次 / 本机处理 ${summary.value.local_count} 次`,
        tone: "green",
      },
      {
        title: "平均耗时",
        value: numberLabel(summary.value.avg_latency_ms, " ms"),
        hint: summary.value.slo_pass
          ? "当前低于目标时间，体验流畅"
          : `目标：${numberLabel(summary.value.slo_ms, ' ms')} 内完成`,
        tone: "orange",
      },
      {
        title: "在线用户",
        value: summary.value.online_devices,
        hint: `服务器当前排队 ${server.value.queue || 0} 项任务`,
        tone: "purple",
      },
    ]);

    async function loadSnapshot() {
      try {
        const response = await fetch("/api/admin/dao-snapshot", { cache: "no-store" });
        if (!response.ok)
          throw new Error(`HTTP ${response.status}`);
        snapshot.value = await response.json();
        lastUpdated.value = new Date().toLocaleTimeString();
        error.value = "";
      }
      catch (err) {
        error.value = `管理端数据加载失败：${err.message || err}`;
      }
      finally {
        loading.value = false;
      }
    }

    onMounted(() => {
      loadSnapshot();
      timer = window.setInterval(loadSnapshot, 2000);
      // 数据仍每 2 秒同步一次；时间轴独立平滑前进，避免图形在右半边反复跳动。
      const advanceTimeline = () => {
        timelineNow.value = Date.now() / 1000;
        timelineFrame = window.requestAnimationFrame(advanceTimeline);
      };
      timelineFrame = window.requestAnimationFrame(advanceTimeline);
    });

    onUnmounted(() => {
      if (timer)
        window.clearInterval(timer);
      if (timelineFrame)
        window.cancelAnimationFrame(timelineFrame);
    });

    return {
      comparison,
      dao,
      devices,
      error,
      events,
      lastUpdated,
      loading,
      numberLabel,
      pipeline,
      priorityPercent,
      priorityText,
      sectionKeys,
      server,
      statCards,
      strategyLabel,
      summary,
      toggleSection,
      timelineNow,
      visible,
    };
  },
  template: `
    <div class="admin-page">
      <header class="admin-topbar">
        <div>
          <h1>调度管理端</h1>
          <p>云边协同 · 一屏查看课堂设备、任务安排、处理时间线与运行记录</p>
        </div>
        <div class="top-actions">
          <a class="page-link home-link" href="/dashboard">课堂总览</a>
          <a class="page-link active" href="/admin">调度管理</a>
          <span class="status" :class="loading ? 'muted' : 'online'" :title="'更新于 ' + lastUpdated">{{ loading ? '同步中' : '已连接 ' + lastUpdated }}</span>
        </div>
      </header>

      <div class="admin-shell">
        <div v-if="error" class="error-banner">{{ error }}</div>

        <section class="stat-grid">
          <StatCard
            v-for="card in statCards"
            :key="card.title"
            :title="card.title"
            :value="card.value"
            :hint="card.hint"
            :tone="card.tone"
          />
        </section>

        <div class="toggle-bar">
          <span class="toggle-tip">显示内容</span>
          <button
            v-for="s in sectionKeys"
            :key="s.key"
            type="button"
            class="toggle-btn"
            :class="{ on: visible[s.key] }"
            @click="toggleSection(s.key)"
          >
            <i class="chk"></i>{{ s.label }}
          </button>
        </div>

        <section v-show="visible.users" class="panel user-panel">
          <div class="section-heading">
            <span>用户</span>
            <strong>用户 · 队列 · 优先级</strong>
          </div>
          <div class="queue-summary">
            <div class="queue-chip">
              <small>当前排队任务</small>
              <b>{{ server.queue || 0 }} 项</b>
            </div>
            <div class="queue-chip">
              <small>任务优先级</small>
              <b>{{ priorityText(priorityPercent) }}</b>
            </div>
            <div class="queue-chip">
              <small>当前处理方式</small>
              <b>{{ strategyLabel(dao.strategy) }}</b>
            </div>
          </div>
          <div class="meter-block">
            <div class="meter-label">
              <span>优先级水平（越高越先被处理）</span>
              <b>{{ numberLabel(priorityPercent, '%') }}</b>
            </div>
            <div class="meter"><i :style="{ width: priorityPercent + '%' }"></i></div>
          </div>
        </section>

        <main v-show="visible.detail" class="dashboard-layout">
          <section class="panel device-panel">
            <div class="section-heading">
              <span>设备</span>
              <strong>接入的课堂设备</strong>
            </div>
            <div class="device-list">
              <DeviceCard v-for="device in devices" :key="device.id" :device="device" />
              <div v-if="devices.length === 0" class="empty-events">
                暂无设备接入，等待学生设备上线。
              </div>
            </div>
          </section>
          <div class="center-stack">
            <DaoPanel :dao="dao" />
            <PipelineFlow :pipeline="pipeline" />
          </div>
          <ResourcePanel :server="server" :comparison="comparison" :summary="summary" />
        </main>

        <div v-show="visible.timeline">
          <StageTimeline :events="events" :devices="devices" :nowSeconds="timelineNow" />
        </div>

        <div v-show="visible.stream" class="gantt-row">
          <DeviceGantt :events="events" :devices="devices" :nowSeconds="timelineNow" />
          <StreamGantt :events="events" :devices="devices" :nowSeconds="timelineNow" />
        </div>

        <div v-show="visible.logs">
          <EventTable :events="events" />
        </div>
      </div>
    </div>
  `,
}).mount("#app");
