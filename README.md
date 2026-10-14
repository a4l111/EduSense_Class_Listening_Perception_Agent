# 智慧教育助手（education/ 子目录）

从毕业设计项目 `edge-video-scheduler-thesis` 迁移的独立智慧课堂教育栈，包含
DAO（DAG-Aware Offloading）卸载决策、边缘推理、教师端与学生端；不依赖挑战杯
工业检测模块。

## 目录结构

```
education/
├── web_api.py               # 教学端服务 (:8009)：教师工作台 + DAO管理台 + 学生采集端
├── classroom_dashboard.py   # 教师管理端 (:8008)：课堂大屏 + 管控台 + SQLite 历史库
├── smart_education.py       # DAO 卸载客户端（论文 Algorithm 4.1 内置实现）
├── inference_service.proto  # 教育 gRPC 服务契约
├── scripts/
│   └── grpc_inference_server.py   # gRPC 边缘推理服务 (:50051 + :8011 管理端)
├── src/                     # 依赖子集：grpc_gen / inference / dag / offload / runtime
├── web/                     # 四个前端页面（vue-teacher / dao-admin / classroom / classroom-client）
└── data/                    # 课程视频 / OpenVINO+MediaPipe 模型 / 学生视频样本
```

## 端口规划

| 端口  | 服务                | 页面                                      |
|-------|---------------------|-------------------------------------------|
| 50051 | gRPC 边缘推理       | （DAO 卸载目标，无页面）                   |
| 8011  | gRPC 自带管理端     | http://127.0.0.1:8011/admin （DAO 调度监控）|
| 8009  | 教学端 web_api      | `/` 教师工作台 · `/admin` 全课堂调度管理 · `/client` 学生采集端 |
| 8008  | 教师管理端          | `/dashboard` 课堂大屏 · `/admin` 调度管理兼容入口与全量数据汇总接口 |

## 首次安装（conda 环境 edu-server，或任何装齐依赖的环境）

```bat
C:\Users\13202\miniconda3\envs\edu-server\python.exe -m pip install -r education\requirements.txt
```

> 启动脚本会自动探测 `%USERPROFILE%\miniconda3\envs\edu-server\python.exe`；
> 也可用 `.\start_education.ps1 -Python <python.exe 路径>` 显式指定其他环境。
> 依赖清单与组员 `requirements\requirements-classroom-server.txt` 基本一致，二选一安装皆可。

> 首次启动时 Ultralytics 会下载 `yolov8n.pt` 和 `yolov8n-pose.pt`。当前提交仅包含
> OpenVINO `.xml`，未包含配套 `.bin` 权重；缺失时服务会降级为目标检测模式，补齐同名
> `.bin` 后自动启用课堂状态融合。

## 启动

```bat
cd education
start_education.bat            REM 或: powershell -ExecutionPolicy Bypass -File start_education.ps1
```

三个服务各占一个独立窗口（已运行的服务会自动跳过，可重复执行）：

1. gRPC 边缘推理（:50051 + :8011）—— 等待树莓派/教学端的卸载请求
2. 教学端 web_api（:8009）—— 教师工作台、学生采集端
3. 教师管理端（:8008）—— 课堂大屏、历史数据

也可单独启动某个服务：

```powershell
.\start_education.ps1 grpc        # 仅 gRPC 边缘推理
.\start_education.ps1 web         # 仅教学端
.\start_education.ps1 dashboard   # 仅教师管理端
.\start_education.ps1 -Device cpu # 无 GPU 环境用 CPU 推理
```

使用 VS Code Remote SSH 时，不要把树莓派的 8009 转发到电脑本机 8009；在 VS Code
“端口”面板取消该端口转发即可，SSH 连接无需关闭。

## 数据流

```
学生采集端(8009 /client) ─WebSocket→ web_api 分析
        │  (DAO 决策: 本地 OpenVINO 推理 ⇄ 卸载到 gRPC :50051)
        ▼
每 3s 课堂报告 POST → 教师管理端(8008) 内存态 + SQLite 持久化
        ▼
课堂大屏(8008 /dashboard) 轮询 /api/students /api/summary
调度任务流: 8008 后台线程每 3s 转发 :8011 /api/snapshot 落库
```

学生端提供三种采集源：

- **本机摄像头**：浏览器用 `getUserMedia()` 读取访问者设备摄像头，按处理背压将
  JPEG 帧经 WebSocket 上传；这是跨主机访问 `/client` 时的推荐模式。每个浏览器
  会在本地保存独立设备 ID、座位号和学生姓名，多名学生不会覆盖服务器全局身份设置。
- **服务器摄像头**：由 `web_api.py` 所在主机执行 `cv2.VideoCapture(camera_id)`，用于
  摄像头物理连接在服务器或树莓派上的部署。
- **样例视频**：读取服务器 `data/videos` 下的视频文件。

远程浏览器只有在 HTTPS 安全上下文中才能授权本机摄像头（`localhost` 除外）。FRP
TCP 映射可以保留，但公网入口需要由 Nginx/Caddy/Uvicorn TLS 提供 HTTPS，并同时转发
WebSocket；纯 `http://公网地址:端口/client` 只能使用服务器摄像头或样例视频。

## 主要接口（教师管理端 :8008，均已开 CORS）

- `GET /api/students` · `GET /api/summary` · `GET /api/alerts` —— 实时课堂态
- `GET /api/admin/snapshot` —— 转发 :8011 的 DAO 调度快照（降级时本地聚合）
- `GET /api/teacher/session` —— 本节课会话（active/started_at/ended_at）
- `GET /api/history/summary` —— 历史统计（状态分布/专注率/任务时延/SLO 达标率）
- `GET /api/history/events?page=&page_size=` —— 历史课堂帧事件分页
- `GET /api/history/tasks?page=&page_size=` —— 历史调度任务分页
- `GET /api/classroom/reports?page=&page_size=` —— 学生端课堂报告分页
- `POST /api/student/report` · `POST /api/clear` —— 学生上报 / 清空

历史库自动创建于 `education/data/education_history.db`（SQLite，三张表：
classroom_events / classroom_reports / classroom_tasks）。

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `EDGE_USE_EDGE` | `1`（启动脚本注入） | 教学端是否把推理卸载到边缘 gRPC |
| `EDGE_HOST` | `127.0.0.1`（启动脚本注入） | gRPC 边缘服务地址 |
| `CLASSROOM_DASHBOARD_URL` | `http://127.0.0.1:8008` | 课堂报告上报目标 |
| `EDUCATION_GRPC_ADMIN_URL` | `http://127.0.0.1:8011` | 教师管理端转发调度快照的来源 |
| `BROWSER_CAMERA_MAX_FRAME_BYTES` | `2097152` | 浏览器摄像头单帧上传上限（字节） |

## 停止

直接关闭对应的三个 PowerShell 窗口即可；SQLite 由 WAL/连接级自动落盘，
无需额外收尾操作。
