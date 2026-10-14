$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$b = 'http://127.0.0.1:8008'

Start-Sleep -Seconds 10

# 1) 会话初始状态（应含 generating / error 字段）
$s = Invoke-RestMethod "$b/api/teacher/session"
Write-Output ("INIT active={0} generating={1} error={2}" -f $s.active, $s.generating, $s.error)

# 2) 开始上课
$start = Invoke-RestMethod -Method Post "$b/api/teacher/session/start"
Write-Output ("START ok={0}" -f $start.ok)

# 3) 两个学生各上报 3 帧（时间戳用服务器端时间，写入 SQLite 事件流）
1..3 | ForEach-Object {
  $body1 = [System.Text.Encoding]::UTF8.GetBytes('{"device_id":"pi-test-01","seat_label":"A1","student_name":"ZhangSan","learning_state":"focused","latency_ms":88.5,"edge_connected":true}')
  $body2 = [System.Text.Encoding]::UTF8.GetBytes('{"device_id":"pi-test-02","seat_label":"A2","student_name":"LiSi","learning_state":"distracted","latency_ms":120.0,"edge_connected":true}')
  Invoke-RestMethod -Method Post "$b/api/student/report" -ContentType 'application/json; charset=utf-8' -Body $body1 | Out-Null
  Invoke-RestMethod -Method Post "$b/api/student/report" -ContentType 'application/json; charset=utf-8' -Body $body2 | Out-Null
  Start-Sleep -Milliseconds 1200
}

# 4) 上课中：只分析和记录状态，不生成或展示临时个人报告
try {
  Invoke-RestMethod "$b/api/students/pi-test-01/report" | Out-Null
  Write-Output "LIVE_REPORT unexpected=available"
} catch {
  $body = $_.ErrorDetails.Message | ConvertFrom-Json
  Write-Output ("LIVE_REPORT blocked=true session_active={0} message={1}" -f $body.session.active, $body.error)
}

# 5) 下课 → 立即返回 generating=true（异步）
$f = Invoke-RestMethod -Method Post "$b/api/teacher/session/finish"
Write-Output ("FINISH ok={0} generating={1}" -f $f.ok, $f.generating)

# 6) 轮询直到生成完成（Ovis 模型缺失时应快速降级 skip）
$deadline = (Get-Date).AddSeconds(40)
do {
  Start-Sleep -Seconds 2
  $s = Invoke-RestMethod "$b/api/teacher/session"
} while ($s.generating -and ((Get-Date) -lt $deadline))
Write-Output ("SESSION active={0} generating={1} error={2} hasReport={3}" -f $s.active, $s.generating, $s.error, ($null -ne $s.report))

$r = $s.report
if ($r) {
  Write-Output ("REPORT engine={0} students={1} focus={2} ranking={3} priority={4} stages={5} ovis_status={6}" -f $r.engine, $r.student_count, $r.focus_rate, @($r.focus_ranking).Count, @($r.priority_students).Count, @($r.stage_trend).Count, $r.ovis_status)
  if ($r.ovis_skip_reason) { Write-Output ("OVIS_SKIP {0}" -f $r.ovis_skip_reason) }
  if ($r.ovis_error) { Write-Output ("OVIS_ERROR {0}" -f $r.ovis_error) }
  @($r.focus_ranking) | ForEach-Object { Write-Output ("RANK {0} {1} {2}%" -f $_.student_name, $_.focus_level, $_.focus_rate) }
  Write-Output ("ACTIONS_COUNT {0}" -f @($r.teacher_actions).Count)
}

# 7) 个人最终报告（final=true，含 durations/stage_trend/evidence/session）
$p1 = Invoke-RestMethod "$b/api/students/pi-test-01/report"
Write-Output ("PERSONAL1 final={0} level={1} focus={2} durFocused={3} stages={4} evidence={5} sessGen={6}" -f $p1.final, $p1.focus_level, $p1.focus_rate, $p1.durations.focused, @($p1.stage_trend).Count, @($p1.evidence).Count, $p1.session.generating)
$p2 = Invoke-RestMethod "$b/api/students/pi-test-02/report"
Write-Output ("PERSONAL2 final={0} level={1} focus={2} durDistracted={3} evidence={4}" -f $p2.final, $p2.focus_level, $p2.focus_rate, $p2.durations.distracted, @($p2.evidence).Count)

# 8) 历史报告落库验证（classroom_reports 表：teacher-summary + 每学生）
$hist = Invoke-RestMethod "$b/api/classroom/reports?page=1&page_size=6"
Write-Output ("HISTORY rows={0}" -f @($hist.reports).Count)
@($hist.reports) | Select-Object -First 3 | ForEach-Object { Write-Output ("HIST {0} {1}" -f $_.device_id, $_.engine) }

# 9) 教师端页面资源版本 v=5
$h = Invoke-WebRequest "$b/dashboard" -UseBasicParsing
Write-Output ("TEACHER_V5={0}" -f ($h.Content -match 'v=5'))

# 10) 清空课堂恢复干净状态
Invoke-RestMethod -Method Post "$b/api/clear" | Out-Null
$after = Invoke-RestMethod "$b/api/teacher/session"
Write-Output ("CLEARED active={0} report={1}" -f $after.active, ($null -ne $after.report))
