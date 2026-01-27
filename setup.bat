@echo off
chcp 65001 >nul
title RVC 环境安装

echo ================================================
echo           RVC 环境安装
echo ================================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 显示 Python 版本
echo 检测到 Python:
python --version
echo.

REM 创建虚拟环境
if exist "venv" (
    echo 虚拟环境已存在，跳过创建
) else (
    echo 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo 错误: 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
)
echo.

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat

REM 升级 pip
echo 升级 pip...
python -m pip install --upgrade pip

echo.
echo ================================================
echo 请选择 PyTorch 版本:
echo ================================================
echo [1] CUDA 11.8 (推荐，适用于大多数显卡)
echo [2] CUDA 12.1 (较新显卡)
echo [3] CPU 版本 (无显卡或显存不足)
echo.

set /p choice="请输入选项 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo 安装 PyTorch (CUDA 11.8)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%choice%"=="2" (
    echo.
    echo 安装 PyTorch (CUDA 12.1)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%choice%"=="3" (
    echo.
    echo 安装 PyTorch (CPU)...
    pip install torch torchaudio
) else (
    echo 无效选项，默认安装 CUDA 11.8 版本
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo 安装其他依赖...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)

echo.
echo ================================================
echo 下载基础模型...
echo ================================================
python tools/download_models.py

echo.
echo ================================================
echo 安装完成！
echo ================================================
echo.
echo 运行 run.bat 启动程序
echo.

pause
