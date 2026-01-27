@echo off
chcp 65001 >nul
title RVC 语音转换

echo ================================================
echo           RVC 语音转换系统
echo ================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查虚拟环境是否存在
if not exist "venv\Scripts\activate.bat" (
    echo 错误: 未找到虚拟环境
    echo 请先运行 setup.bat 创建虚拟环境
    echo.
    pause
    exit /b 1
)

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 启动程序
echo 启动 RVC 语音转换...
echo.
python run.py %*

pause
