@echo off
chcp 65001 >nul
REM ============================================================
REM  智慧教育栈一键启动（三个服务各占一个独立窗口）
REM  等价于: powershell -File start_education.ps1 -Service all
REM ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_education.ps1" -Service all
