# ============================================================================
#  智慧教育助手（education/ 子目录）一键启动脚本
#
#  用法：
#    .\start_education.ps1                # 启动完整教育栈（三个独立窗口）
#    .\start_education.ps1 grpc           # 仅启动 gRPC 边缘推理 (:50051 + :8011 管理端)
#    .\start_education.ps1 web            # 仅启动教学端 web_api (:8009)
#    .\start_education.ps1 dashboard      # 仅启动教师管理端 (:8008)
#    .\start_education.ps1 -Device cpu    # 无 GPU 环境用 CPU 推理
#
#  端口规划：
#    :50051  gRPC 边缘推理（DAO 卸载目标）
#    :8011   gRPC 服务自带 DAO 调度监控管理端
#    :8009   教学端 web_api（教师工作台 / DAO 管理台 / 学生采集端）
#    :8008   教师管理端 classroom_dashboard（课堂大屏 + 管控台）
# ============================================================================
param(
    [Parameter(Position = 0)]
    [ValidateSet("all", "grpc", "web", "dashboard")]
    [string]$Service = "all",
    [ValidateSet("gpu", "cpu")]
    [string]$Device = "gpu",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$EduDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Python) {
    $candidates = if ($Device -eq "gpu") {
        @(
            (Join-Path $env:USERPROFILE "miniconda3\envs\challenge-cup-ovis\python.exe"),
            (Join-Path $env:USERPROFILE ".conda\envs\challenge_cup\python.exe")
        )
    } else {
        @(
            (Join-Path $env:USERPROFILE "miniconda3\envs\edu-server\python.exe"),
            (Join-Path $env:USERPROFILE ".conda\envs\challenge_cup\python.exe")
        )
    }
    $Python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Python -or -not (Test-Path $Python)) {
    throw "未找到教育栈 Python 环境，请用 -Python <python.exe 路径> 指定（推荐 conda env: edu-server）"
}
Write-Host "教育栈 Python: $Python"

if ($Device -eq "gpu") {
    $cudaReady = & $Python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
    if (($cudaReady | Select-Object -Last 1).Trim() -ne "1") {
        throw "选择了 GPU 模式，但当前 Python 环境没有 CUDA 版 PyTorch。请使用 challenge-cup-ovis 环境。"
    }
}

function Test-Http($Url) {
    try { Invoke-RestMethod $Url -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

if ($Service -eq "all") {
    $running = @()
    try {
        $grpcSnapshot = Invoke-RestMethod "http://127.0.0.1:8011/api/snapshot" -TimeoutSec 2
        if ($Device -eq "gpu" -and $grpcSnapshot.server.torch_device -notlike "cuda*") {
            throw "8011 已有旧的 CPU 教育推理服务运行（实际设备：$($grpcSnapshot.server.torch_device)）。请只关闭该电脑 PowerShell 窗口后重新运行本脚本。"
        }
        $running += "grpc"
    } catch {
        if ($_.Exception.Message -like "8011 已有旧的 CPU*") { throw }
    }
    try {
        $webStatus = Invoke-RestMethod "http://127.0.0.1:8009/api/status" -TimeoutSec 2
        if ($webStatus.runtime_os -and $webStatus.runtime_os -ne "Windows") {
            throw "电脑的 127.0.0.1:8009 当前被 VS Code 转发到 $($webStatus.runtime_os) 设备 $($webStatus.runtime_host)。请在树莓派远程窗口关闭 8009 自动转发。"
        }
        if (-not $webStatus.runtime_os -and "$($webStatus.settings.student_video_dir)" -like "/home/*") {
            throw "电脑的 127.0.0.1:8009 当前仍是树莓派服务。请在 VS Code 端口面板关闭 8009，并关闭远程自动转发。"
        }
        $running += "web"
    } catch {
        if ($_.Exception.Message -like "电脑的 127.0.0.1:8009*") { throw }
    }
    if (Test-Http "http://127.0.0.1:8008/api/summary")  { $running += "dashboard" }
    foreach ($t in @("grpc", "dashboard", "web")) {
        if ($running -contains $t) {
            Write-Host "[skip] $t 已在运行"
            continue
        }
        Start-Process powershell -ArgumentList @(
            "-NoExit", "-ExecutionPolicy", "Bypass",
            "-File", "`"$PSCommandPath`"",
            "-Service", $t, "-Device", $Device
        )
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  智慧教育栈已启动（等待各窗口打印就绪信息后再打开页面）"
    Write-Host "  教师管理端(课堂大屏): http://127.0.0.1:8008/"
    Write-Host "  教学端(教师工作台):   http://127.0.0.1:8009/   (DAO 管理台: /admin)"
    Write-Host "  DAO 调度监控:         http://127.0.0.1:8011/admin"
    Write-Host "============================================================"
    exit 0
}

Set-Location $EduDir
$env:YOLO_CONFIG_DIR = Join-Path $EduDir "data\ultralytics-config"
New-Item -ItemType Directory -Force (Join-Path $env:YOLO_CONFIG_DIR "Ultralytics") | Out-Null

if ($Service -eq "grpc") {
    if (Test-Http "http://127.0.0.1:8011/api/snapshot") {
        Write-Host "gRPC 推理服务已在运行 (:50051 / :8011)"
        exit 0
    }
    $torchDevice = if ($Device -eq "gpu") { "cuda:0" } else { "cpu" }
    & $Python scripts\grpc_inference_server.py --port 50051 --admin-port 8011 --device $Device --torch-device $torchDevice
}
elseif ($Service -eq "web") {
    try {
        $webStatus = Invoke-RestMethod "http://127.0.0.1:8009/api/status" -TimeoutSec 2
        if (($webStatus.runtime_os -and $webStatus.runtime_os -ne "Windows") -or
            (-not $webStatus.runtime_os -and "$($webStatus.settings.student_video_dir)" -like "/home/*")) {
            throw "127.0.0.1:8009 是树莓派转发端口，不是电脑教学端。请关闭 VS Code 的 8009 自动转发后重试。"
        }
        Write-Host "教学端已在运行 (:8009)"
        exit 0
    } catch {
        if ($_.Exception.Message -like "127.0.0.1:8009 是树莓派*") { throw }
    }
    # 教学端把推理请求卸载到本机 gRPC 边缘服务，并把课堂报告上报给教师管理端
    $env:EDGE_USE_EDGE = "1"
    $env:EDGE_HOST = "127.0.0.1"
    $env:EDGE_PORT = "50051"
    $env:CLASSROOM_DASHBOARD_URL = "http://127.0.0.1:8008"
    & $Python -m uvicorn web_api:app --host 0.0.0.0 --port 8009
}
elseif ($Service -eq "dashboard") {
    if (Test-Http "http://127.0.0.1:8008/api/summary") {
        Write-Host "教师管理端已在运行 (:8008)"
        exit 0
    }
    # 历史任务采集线程会转发 :8011 的调度快照，可用环境变量覆盖
    $env:EDUCATION_GRPC_ADMIN_URL = "http://127.0.0.1:8011"
    $env:CLASSROOM_OVIS_PYTHON = Join-Path $env:USERPROFILE "miniconda3\envs\challenge-cup-ovis\python.exe"
    $env:CLASSROOM_OVIS_TIMEOUT = "480"
    & $Python classroom_dashboard.py --host 0.0.0.0 --port 8008
}
