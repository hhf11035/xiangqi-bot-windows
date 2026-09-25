# Windows 迁移说明

## 平台兼容层

核心对局、CNN 棋盘识别、FEN 校验及 Pikafish UCI 逻辑没有重写。`platform_adapter.py` 负责隔离平台差异：

| 能力 | macOS 原实现 | Windows 实现 |
|---|---|---|
| 查找窗口 | Quartz `CGWindowListCopyWindowInfo` | Win32 `EnumWindows` / `GetWindowRect` |
| 窗口截图 | `screencapture -l` | `mss` 屏幕区域截图 |
| 激活窗口 | AppleScript | Win32 `SetForegroundWindow` |
| 鼠标点击 | Quartz `CGEvent` | `pyautogui` |
| Escape | Quartz 键盘事件 | `pyautogui.press` |
| Pikafish | macOS 无扩展名二进制 | Windows `pikafish.exe`，也可用 `PIKAFISH_PATH` 覆盖 |

## 仍为 macOS 专用的内容

- `app.py`：基于 AppKit/PyObjC 的原生 macOS 图形界面；Windows 端使用 CLI 和批处理启动入口。
- `test_click.py`、`test_identify.py`、`test_bot_debug.py`：原作者的 macOS 开发调试脚本，仍包含 Quartz、AppleScript、`screencapture` 和 `/tmp` 路径，不属于 Windows 运行入口。
- README 的 macOS App 构建章节和 PyInstaller 说明仍只适用于 macOS。

Windows 正式入口 `xiangqi_bot.py` 与 `continuous_play.py` 不再顶层导入 Quartz，也不直接调用 AppleScript 或 `screencapture`。
`xiangqi_cnn.py` 的采集与测试入口也已移除原先硬编码的 `/tmp` 导入路径，可直接从项目目录在 Windows 运行。

## 坐标与窗口限制

Windows 适配层启用 DPI 感知，截图与点击统一使用物理屏幕坐标。校准值继续按窗口宽高保存为相对比例，因此移动或调整窗口后可以复用。`mss` 捕获的是屏幕上的窗口区域，所以窗口必须可见、不能最小化，也不应被其他窗口遮挡。
窗口发现同时匹配标题和微信进程名，避免将标题中出现“天天象棋”的浏览器、编辑器或控制台误识别为棋盘。
