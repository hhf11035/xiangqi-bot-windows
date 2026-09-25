# Xiangqi Bot

Auto-play Chinese chess (天天象棋) on Windows or macOS using Pikafish engine + CNN vision + screen automation.

Windows adaptation of [yingwang/xiangqi-bot](https://github.com/yingwang/xiangqi-bot) by Ying Wang. The original MIT license and copyright notice are retained in [LICENSE](LICENSE). The Windows additions are described in [WINDOWS_PORT_NOTES.md](WINDOWS_PORT_NOTES.md).

[中文](#天天象棋机器人)

## How it works

1. **Pikafish** (~3000 Elo) calculates the best move
2. **CNN classifier** (PyTorch, 15-class, 100% val accuracy) identifies pieces from screenshots
3. **FEN validation** enforces piece count rules and auto-corrects misclassifications using confidence scores
4. **Double-shot parsing** takes a second screenshot when confidence is low, averages probabilities to handle animation artifacts
5. **Platform input adapter** clicks to execute moves

```
Screenshot → CNN parse board → FEN validation → Compare with tracked state → Detect opponent move
                  ↓                                                                ↓
          (low confidence?)                              Pikafish best move ← Current FEN
                  ↓                                                                ↓
          2nd screenshot →                               Platform click → Execute move
          average probs
```

## Platform

The CLI supports Windows 10/11 and macOS. Windows uses `mss`, Win32 window discovery, and `pyautogui`; macOS uses Quartz window discovery and `screencapture`. The AppKit GUI remains macOS-only.

## Windows

The CLI bot (`xiangqi_bot.py`) and multi-game supervisor (`continuous_play.py`) support Windows 10/11. CNN recognition, FEN validation, and the Pikafish UCI flow are unchanged. The macOS-only AppKit GUI (`app.py`) is not included in this port.

1. Install 64-bit Python 3.11 or 3.12 and open PowerShell in this project folder.
2. Run `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1` once. It creates an isolated `.venv` and installs dependencies.
3. A compatible Windows Pikafish 2026-01-02 executable (SSE4.1/POPCNT build) and its matching NNUE are included as `pikafish.exe` and `pikafish.nnue`. To use another build, set an absolute path: `$env:PIKAFISH_PATH = 'C:\\path\\to\\pikafish.exe'`.
4. Open WeChat Desktop and its 天天象棋 mini-program in a normal, visible window. Do not minimize it. If WeChat keeps the mini-program embedded, this port captures the matching WeChat window; place it alone and resize it consistently before first calibration.
5. Double-click `诊断.bat` first. If the window, screenshot, CNN, and Pikafish checks pass, start a game and double-click `启动单局.bat`. `启动连续对局.bat` enables the experimental automatic-rematch mode.

Windows screenshots use `mss`; clicks and Escape use `pyautogui`, while window discovery/activation uses Win32 APIs. Coordinates are screen pixels, while calibration is saved relative to the detected window. Move the mouse to the upper-left corner to trigger PyAutoGUI's emergency stop. If your game window has a different title, set `XIANGQI_WINDOW_TITLE` to a unique part of its title before starting.

`mss` captures the visible screen area occupied by the target window, so keep the board unminimized and unobstructed while the bot is running.

## Setup

### 1. Install Python dependencies

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

### 2. Install Pikafish engine

Pikafish is a Chinese chess engine (forked from Stockfish, rewritten for xiangqi rules). It's a C++ compiled binary that the bot communicates with via UCI protocol over stdin/stdout.

The Windows delivery includes `pikafish.exe`; the following commands are only for building a macOS binary from source:

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon   # for Apple Silicon
# or: make -j profile-build ARCH=x86-64   # for Intel Mac
cp pikafish ../../pikafish                  # copy binary to project root
```

The NNUE file (`pikafish.nnue`) is already included — it's the neural network weights that Pikafish uses to evaluate positions. Pikafish looks for it in the working directory automatically.

### 3. Run

```bash
# Open 天天象棋 in WeChat, start a game, then:
python3 xiangqi_bot.py
```

Calibration is automatic via CNN on first run — no manual input required. The result is saved to `calib.json` and reused until deleted.

Supports both red and black sides. Auto-detects orientation from king position.

## CNN Training

Training data is auto-collected during gameplay. Debug patches are saved to `debug/` organized by piece type for easy review.

### Data augmentation

The augmentation pipeline generates variants per original image:
- **Brightness** (0.8x, 1.2x)
- **Shift** (2px jitter in 4 directions)
- **Scale** (0.85x, 1.15x, 1.25x — simulates selected/enlarged pieces)
- **Low-resolution** (downscale to 20/24/30px then upscale — simulates small board blurriness)

### Improve accuracy

1. Play a game — debug patches auto-save to `debug/`
2. Review `debug/red_C/`, `debug/red_P/` etc. for misclassifications
3. Move wrong images to the correct `cnn_data/<piece_type>/` folder
4. Retrain:

```bash
python3 xiangqi_cnn.py augment   # regenerate augmented data
python3 xiangqi_cnn.py train --epochs 30
```

### Other CNN commands

```bash
python3 xiangqi_cnn.py collect   # Collect from initial position
python3 xiangqi_cnn.py augment   # Augment data (brightness, shift, scale, low-res simulation)
python3 xiangqi_cnn.py test      # Test on current board
```

## Continuous play (multi-game auto-restart)

> ⚠️ **Experimental — still unstable.** Occasional popup variants, animation
> timing, or layout changes may cause the supervisor to stall. Works for
> most cases but expect the occasional manual rescue.

`continuous_play.py` is a standalone CLI supervisor that wraps `Bot` in a
loop: when a game ends, it automatically clicks through the result popups
and starts the next game. Runs headless — no GUI dependency.

```bash
python3 continuous_play.py play                # unlimited games
python3 continuous_play.py play --max-games 5  # stop after 5 games
python3 continuous_play.py test-templates      # verify templates match current screen
python3 continuous_play.py detect-end          # one-shot end-of-game check
python3 continuous_play.py diag                # dump screenshot + window info
python3 continuous_play.py crop-templates      # print manual cropping workflow
```

**How it works:**

1. **Game-end detection** — strict (0.95) `cv2.matchTemplate` on a set of
   end-screen templates (`popup_end_banner`, `popup_level_up`,
   `popup_badge_earned`, `btn_play_again`, `btn_close_x_top`, `btn_confirm`,
   `btn_switch_opponent`). Also catches Pikafish `(none)` (mate/stalemate)
   and a 150s opponent-stuck safety net.
2. **Tiered recovery** — when a game ends, cycle through template clicks
   → blind relative-coord clicks → ESC×3 → refind-window until the board
   becomes visible again.
3. **Multi-scale template matching** — each template is matched at scales
   0.85–1.15x against a 0.35x downscaled screenshot for ~7x speedup over
   naive full-res matching. Stepped thresholds (0.92 → 0.88) with
   early-exit at 0.97+.
4. **Click verification** — every click takes before/after screenshots and
   uses `bot.images_changed` to confirm the screen reacted. Failed clicks
   go on a 30s cooldown to prevent death loops.
5. **Failure snapshots** — if recovery times out, a screenshot plus state
   log is dumped to `continuous_play_debug/<timestamp>/`. End-of-game
   screenshots are auto-saved to `end_snapshots/` for future template
   tuning.
6. **Ctrl+C** — graceful stop on first press; hard force-exit after 4s via
   a daemon-thread `os._exit(130)` watchdog, so a stuck subprocess can't
   block shutdown.

**Templates** live in `templates/`. Missing templates degrade gracefully
(tier 1 just has fewer candidates; tiers 2–4 still run).

**Does not modify** `xiangqi_bot.py` or `app.py` — both are still
available as the single-game entry points:

| Entry point | Multi-game? |
|---|---|
| `python3 xiangqi_bot.py` | no (single game) |
| `python3 app.py` / `天天象棋Bot.app` | no (GUI, single game) |
| `python3 continuous_play.py play` | **yes** (auto-restart) |

## macOS App

A native macOS GUI (`天天象棋Bot.app`) with board display, move history (四字记录法), and start/stop controls.

### Build

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

The app is output to `dist/天天象棋Bot.app`. It bundles Pikafish, the ONNX model, and all Python dependencies.

### Run directly (without building)

```bash
python3 app.py
```

## Files

| File | Description |
|------|-------------|
| `app.py` | macOS GUI app (AppKit/PyObjC, board view, move history) |
| `xiangqi_bot.py` | Bot engine (game loop, move execution, double-shot parsing) |
| `continuous_play.py` | Multi-game supervisor (auto-restart between games) |
| `xiangqi_cnn.py` | CNN model, training, inference, FEN validation |
| `platform_adapter.py` | Windows/macOS window, screenshot, and input adapter |
| `windows_diagnostics.py` | Read-only Windows readiness diagnostics |
| `requirements-windows.txt` | Windows Python dependencies |
| `setup_windows.ps1` | Creates/repairs the Windows virtual environment |
| `启动单局.bat` / `启动连续对局.bat` / `诊断.bat` | Windows launchers |
| `pikafish.exe` | Bundled Windows Pikafish engine |
| `pikafish.nnue` | Neural network evaluation file for Pikafish |
| `cnn_data/` | Training data by piece type (`red_R/`, `black_r/`, `empty/`, etc.) |
| `debug/` | Debug patches from latest game session |
| `calib.json` | Grid calibration data |
| `templates/` | Popup button templates used by `continuous_play.py` |
| `end_snapshots/` | Auto-saved end-of-game screenshots (for template tuning) |
| `continuous_play_debug/` | Failure dumps from `continuous_play.py` (screenshot + state) |

## License

This project's own code is released under the [MIT License](LICENSE).

The bundled Pikafish engine binary (`pikafish.exe`) and neural network weights (`pikafish.nnue`) are licensed under **GPL v3** (Pikafish is a fork of Stockfish). Source code: https://github.com/official-pikafish/Pikafish

---

# 天天象棋机器人

支持 Windows 10/11 与 macOS 的天天象棋自动对局程序。基于 Pikafish 引擎、CNN 视觉识别与屏幕自动化实现。

本项目基于 Ying Wang 的 [xiangqi-bot](https://github.com/yingwang/xiangqi-bot) 增加 Windows 支持，保留原项目的 MIT 许可证与版权声明。Windows 改动见 [WINDOWS_PORT_NOTES.md](WINDOWS_PORT_NOTES.md)。

## 工作原理

1. **Pikafish**（约 3000 Elo）计算最佳着法
2. **CNN 分类器**（PyTorch，15 类，验证集 100% 准确率）从截图识别棋子
3. **FEN 校验** 依据棋子数量规则，根据置信度自动修正误分类
4. **双次采样** 置信度较低时二次截图，对概率取平均以应对动画瞬态
5. **平台输入适配层** 模拟点击执行着法

## 运行平台

命令行版本支持 Windows 10/11 与 macOS。Windows 使用 `mss`、Win32 窗口接口与 `pyautogui`；macOS 使用 Quartz 与 `screencapture`。`app.py` 的 AppKit 图形界面仍仅支持 macOS。

### Windows 快速安装

1. 安装 64 位 Python 3.11 或更高版本。
2. 在项目目录运行 `powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1`。桌面交付版已经包含安装好的 `.venv`，通常无需再次安装。
3. 打开电脑版微信中的天天象棋并保持棋盘窗口可见。
4. 先双击 `诊断.bat`，通过后双击 `启动单局.bat`。实验性的连续对局使用 `启动连续对局.bat`。

Windows 版已包含相互匹配的 `pikafish.exe` 与 `pikafish.nnue`。如要使用其他引擎，可设置 `PIKAFISH_PATH` 为该引擎的绝对路径，并将其匹配的 `pikafish.nnue` 放在同一目录。

## 安装

**1. Python 依赖**

```bash
pip install opencv-python numpy pyautogui torch torchvision
```

**2. Pikafish 引擎**

Pikafish 为象棋引擎,由 Stockfish 衍生并针对象棋规则重写。下方命令仅用于自行编译 macOS 版本；Windows 交付版已包含可执行文件:

```bash
git clone https://github.com/official-pikafish/Pikafish.git
cd Pikafish/src
make -j profile-build ARCH=apple-silicon    # Apple Silicon
# make -j profile-build ARCH=x86-64         # Intel Mac
cp pikafish ../../pikafish
```

神经网络权重文件 `pikafish.nnue` 已包含,Pikafish 启动时自动加载。

**3. 运行**

```bash
python3 xiangqi_bot.py
```

首次运行时通过 CNN 自动完成棋盘网格校准,无需手动输入。校准结果写入 `calib.json` 并复用。红黑两方皆支持,依将/帅位置自动识别方向。

## CNN 训练

训练样本在对局过程中自动采集,按棋子类型归档至 `debug/`。

**数据增强**

- 亮度:0.8x、1.2x
- 位移:四方向 2 像素抖动
- 缩放:0.85x、1.15x、1.25x
- 低分辨率:降采样至 20 / 24 / 30 像素后升回

**提升识别精度**

1. 完成对局后,debug 样本自动保存至 `debug/`
2. 审查 `debug/red_C/`、`debug/red_P/` 等目录,识别误分类样本
3. 将错分样本移至 `cnn_data/<棋子类型>/`
4. 重新训练:

```bash
python3 xiangqi_cnn.py augment
python3 xiangqi_cnn.py train --epochs 30
```

**其他 CNN 命令**

```bash
python3 xiangqi_cnn.py collect   # 从初始局面采集
python3 xiangqi_cnn.py augment   # 生成增强数据
python3 xiangqi_cnn.py test      # 测试当前棋盘识别
```

## 连续对局

> ⚠️ **实验性功能,目前仍不稳定。** 遇到未见过的弹窗变体、动画时序或布局调整时,监督脚本可能失效。大多数情况下可用,偶尔需要人工介入。

`continuous_play.py` 为独立的 CLI 监督脚本,将 `Bot` 置于循环中运行:对局结束后自动关闭结算弹窗并开始下一局,无需图形界面。

```bash
python3 continuous_play.py play                # 无限对局
python3 continuous_play.py play --max-games 5  # 限定 5 局
python3 continuous_play.py test-templates      # 验证模板命中情况
python3 continuous_play.py detect-end          # 单次检测游戏结束
python3 continuous_play.py diag                # 输出截图与窗口信息
python3 continuous_play.py crop-templates      # 显示模板裁剪说明
```

**实现要点**

1. **结束检测** — 对 `popup_end_banner`、`popup_level_up`、`popup_badge_earned`、`btn_play_again`、`btn_close_x_top`、`btn_confirm`、`btn_switch_opponent` 等模板执行 `cv2.matchTemplate`,阈值 0.95。同时监听 Pikafish 返回 `(none)`(将死或困毙)以及 150 秒对方无应对的兜底信号
2. **分层恢复** — 依次尝试:模板点击 → 盲点相对坐标点击 → ESC × 3 → 重新查找窗口
3. **多尺度匹配** — 0.85–1.15 尺度区间,对 0.35x 降采样后的截图匹配,相比全分辨率约 7 倍加速。阶梯阈值 0.92 → 0.88,命中 0.97 以上时提前终止
4. **点击校验** — 点击前后截图对比,经 `bot.images_changed` 确认界面变化。失败按钮进入 30 秒冷却以避免死循环
5. **异常转储** — 恢复超时时将截图与状态日志写入 `continuous_play_debug/<时间戳>/`;结束画面自动保存至 `end_snapshots/` 以便后续模板调校
6. **Ctrl+C 退出** — 首次触发平缓停止;4 秒后由守护线程看门狗 `os._exit(130)` 强制退出

模板文件位于 `templates/`。缺失模板时自动降级,tier 1 候选减少而 tier 2–4 继续运行。

本脚本不修改 `xiangqi_bot.py` 与 `app.py`,两者仍作为单局入口可用:

| 入口 | 连续对局 |
|---|---|
| `python3 xiangqi_bot.py` | 单局 |
| `python3 app.py` / `天天象棋Bot.app` | 单局(图形界面) |
| `python3 continuous_play.py play` | 自动续局 |

## macOS 应用

原生 macOS 图形界面 `天天象棋Bot.app`,包含棋盘视图、四字记录法着法列表与起停控制。

**构建**

```bash
pip install pyinstaller
python3 -m PyInstaller xiangqi_bot.spec --noconfirm
```

产物位于 `dist/天天象棋Bot.app`,已打包 Pikafish、ONNX 模型与全部 Python 依赖。

**直接运行**

```bash
python3 app.py
```

## 文件说明

| 文件 | 说明 |
|------|-------------|
| `app.py` | macOS 图形界面(AppKit/PyObjC,棋盘视图,着法记录) |
| `xiangqi_bot.py` | Bot 引擎(主循环、着法执行、双次采样) |
| `continuous_play.py` | 连续对局监督脚本 |
| `xiangqi_cnn.py` | CNN 模型、训练、推理、FEN 校验 |
| `platform_adapter.py` | Windows/macOS 窗口、截图与输入适配层 |
| `windows_diagnostics.py` | 不执行点击的 Windows 就绪诊断 |
| `requirements-windows.txt` | Windows Python 依赖 |
| `setup_windows.ps1` | 创建或修复 Windows 隔离环境 |
| `启动单局.bat` / `启动连续对局.bat` / `诊断.bat` | Windows 启动入口 |
| `pikafish.exe` | 随附的 Windows Pikafish 引擎 |
| `pikafish.nnue` | Pikafish 神经网络权重 |
| `cnn_data/` | 按棋子类型组织的训练数据 |
| `debug/` | 最近对局的 debug 样本 |
| `calib.json` | 棋盘校准数据 |
| `templates/` | `continuous_play.py` 使用的模板图 |
| `end_snapshots/` | 自动保存的结束画面截图 |
| `continuous_play_debug/` | `continuous_play.py` 异常转储 |

## 许可证

本项目代码以 [MIT 许可证](LICENSE) 发布。

仓库内置的 Pikafish 引擎二进制（`pikafish`）与神经网络权重（`pikafish.nnue`）以 **GPL v3** 许可证发布（Pikafish 为 Stockfish 衍生项目）。Pikafish 源码：https://github.com/official-pikafish/Pikafish
