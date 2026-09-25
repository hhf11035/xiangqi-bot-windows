"""Read-only Windows readiness checks. This script never clicks the board."""
import os
import subprocess
import sys


def ok(label, detail=""):
    print(f"[通过] {label}{': ' + detail if detail else ''}")


def fail(label, detail):
    print(f"[失败] {label}: {detail}")


def main():
    failures = 0
    try:
        import cv2
        import mss  # noqa: F401
        import numpy  # noqa: F401
        import pyautogui  # noqa: F401
        import torch
        import torchvision  # noqa: F401
        ok("Python 依赖", f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}")
    except Exception as exc:
        fail("Python 依赖", str(exc))
        return 1

    root = os.path.dirname(os.path.abspath(__file__))
    engine = os.environ.get("PIKAFISH_PATH", os.path.join(root, "pikafish.exe"))
    try:
        proc = subprocess.run(
            [engine], input="uci\nisready\nposition startpos\ngo depth 1\nquit\n", text=True,
            capture_output=True, cwd=root, timeout=10)
        if (proc.returncode or "uciok" not in proc.stdout or
                "readyok" not in proc.stdout or "bestmove" not in proc.stdout):
            raise RuntimeError((proc.stderr or proc.stdout)[-300:])
        ok("Pikafish 引擎", os.path.basename(engine))
    except Exception as exc:
        fail("Pikafish 引擎", str(exc))
        failures += 1

    try:
        from platform_adapter import WindowAdapter
        adapter = WindowAdapter()
        adapter.find_wechat_xiangqi()
        image = adapter.screenshot()
        if image is None or image.size == 0:
            raise RuntimeError("截图为空")
        out = os.path.join(root, "windows_diagnostic.png")
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("无法保存诊断截图")
        encoded.tofile(out)
        ok("窗口与截图", f"{adapter.process_name} / {adapter.window_title} / "
           f"{adapter.width}x{adapter.height}，已保存 windows_diagnostic.png")
    except Exception as exc:
        fail("窗口与截图", str(exc))
        failures += 1

    try:
        from xiangqi_cnn import PieceClassifierCNN
        PieceClassifierCNN()
        ok("CNN 模型", "xiangqi_cnn.pt 已加载")
    except Exception as exc:
        fail("CNN 模型", str(exc))
        failures += 1

    print("\n诊断完成：" + ("所有检查通过。" if not failures else f"有 {failures} 项需要处理。"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
