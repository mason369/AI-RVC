# -*- coding: utf-8 -*-
"""
日志工具模块 - 支持时间戳和颜色输出
"""
import sys
import logging
from datetime import datetime

try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)  # 初始化 colorama (Windows 兼容), autoreset确保每行重置
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    # 定义空的占位符
    class Fore:
        LIGHTBLACK_EX = GREEN = YELLOW = RED = CYAN = BLUE = MAGENTA = WHITE = LIGHTGREEN_EX = LIGHTCYAN_EX = LIGHTYELLOW_EX = LIGHTMAGENTA_EX = ""
    class Style:
        RESET_ALL = BRIGHT = DIM = ""
    class Back:
        pass


class Logger:
    """统一日志工具"""

    COLORS = {
        "DEBUG": Fore.LIGHTBLACK_EX,
        "INFO": Fore.GREEN,
        "SUCCESS": Fore.LIGHTGREEN_EX,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "STEP": Fore.CYAN,
        "DETAIL": Fore.LIGHTCYAN_EX,
        "PROGRESS": Fore.MAGENTA,
        "MODEL": Fore.LIGHTMAGENTA_EX,
        "AUDIO": Fore.BLUE,
        "CONFIG": Fore.LIGHTYELLOW_EX,
    }

    RESET = Style.RESET_ALL
    BRIGHT = Style.BRIGHT
    DIM = Style.DIM

    # 详细日志开关
    verbose = True

    @staticmethod
    def _log(level: str, msg: str, force_print: bool = True):
        """内部日志方法"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Logger.COLORS.get(level, "")
        reset = Logger.RESET

        # 根据级别决定前缀
        if level in ("INFO", "STEP", "SUCCESS"):
            prefix = ""
        elif level == "DETAIL":
            prefix = "  → "
        elif level == "PROGRESS":
            prefix = "  ◆ "
        elif level == "MODEL":
            prefix = "[模型] "
        elif level == "AUDIO":
            prefix = "[音频] "
        elif level == "CONFIG":
            prefix = "[配置] "
        else:
            prefix = f"[{level}] "

        output = f"{color}[{timestamp}]{prefix}{msg}{reset}"
        print(output, flush=True)

    @staticmethod
    def debug(msg: str):
        """调试日志 (灰色) - 仅在verbose模式下显示"""
        if Logger.verbose:
            Logger._log("DEBUG", msg)

    @staticmethod
    def info(msg: str):
        """信息日志 (绿色)"""
        Logger._log("INFO", msg)

    @staticmethod
    def success(msg: str):
        """成功日志 (亮绿色)"""
        Logger._log("SUCCESS", f"✓ {msg}")

    @staticmethod
    def warning(msg: str):
        """警告日志 (黄色)"""
        Logger._log("WARNING", msg)

    @staticmethod
    def error(msg: str):
        """错误日志 (红色)"""
        Logger._log("ERROR", msg)

    @staticmethod
    def step(current: int, total: int, msg: str):
        """步骤日志 (青色)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Logger.COLORS.get("STEP", "")
        reset = Logger.RESET
        print(f"{color}[{timestamp}][{current}/{total}] {msg}{reset}", flush=True)

    @staticmethod
    def detail(msg: str):
        """详细日志 (浅青色) - 用于显示处理细节"""
        if Logger.verbose:
            Logger._log("DETAIL", msg)

    @staticmethod
    def progress(msg: str):
        """进度日志 (紫色) - 用于显示处理进度"""
        Logger._log("PROGRESS", msg)

    @staticmethod
    def model(msg: str):
        """模型日志 (浅紫色) - 用于模型加载/卸载信息"""
        Logger._log("MODEL", msg)

    @staticmethod
    def audio(msg: str):
        """音频日志 (蓝色) - 用于音频处理信息"""
        Logger._log("AUDIO", msg)

    @staticmethod
    def config(msg: str):
        """配置日志 (浅黄色) - 用于配置信息"""
        if Logger.verbose:
            Logger._log("CONFIG", msg)

    @staticmethod
    def header(msg: str):
        """标题日志 (带分隔线)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Logger.COLORS.get("INFO", "")
        reset = Logger.RESET
        print(f"{color}[{timestamp}] {'=' * 50}{reset}", flush=True)
        print(f"{color}[{timestamp}] {msg}{reset}", flush=True)
        print(f"{color}[{timestamp}] {'=' * 50}{reset}", flush=True)

    @staticmethod
    def separator(char: str = "-", length: int = 40):
        """分隔线"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = Logger.COLORS.get("DEBUG", "")
        reset = Logger.RESET
        print(f"{color}[{timestamp}] {char * length}{reset}", flush=True)

    @staticmethod
    def set_verbose(enabled: bool):
        """设置详细日志模式"""
        Logger.verbose = enabled


# 便捷实例
log = Logger()


# ============ 配置标准 logging 模块使用颜色 ============

class ColoredFormatter(logging.Formatter):
    """为标准logging模块添加颜色支持"""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # 获取颜色
        color = self.LEVEL_COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL

        # 格式化时间
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建消息
        level_name = record.levelname
        module_name = record.name

        # 格式化输出
        formatted = f"{color}{timestamp} | {level_name} | {module_name} | {record.getMessage()}{reset}"
        return formatted


def setup_colored_logging(level=logging.INFO):
    """配置全局logging使用颜色输出"""
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # 移除现有的handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 添加带颜色的handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    return root_logger


# 自动配置logging颜色
setup_colored_logging(logging.INFO)

# 抑制第三方库的英文日志
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("audio_separator").setLevel(logging.WARNING)
