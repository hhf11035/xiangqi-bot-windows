"""Platform-specific window, screenshot, and input operations."""
import ctypes
import ctypes.wintypes
import os
import platform
import time

IS_WINDOWS = platform.system() == "Windows"

# Enable DPI awareness before importing pyautogui. Otherwise Windows display
# scaling can make screenshots use physical pixels while clicks use virtual
# coordinates, causing a fixed 1.25x/1.5x offset.
if IS_WINDOWS:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

import cv2
import numpy as np
import pyautogui


class WindowAdapter:
    def __init__(self):
        self.window = None
        self.win_id = None
        self.window_title = ""
        self.process_name = ""
        self.x = self.y = 0
        self.width, self.height = 1628, 960

    def find_wechat_xiangqi(self):
        if IS_WINDOWS:
            candidates = self._windows_candidates()
            if not candidates:
                target = os.environ.get("XIANGQI_WINDOW_TITLE", "天天象棋")
                raise RuntimeError(
                    f"找不到标题包含“{target}”的窗口。请先在电脑版微信中打开天天象棋；"
                    "如窗口标题不同，请设置 XIANGQI_WINDOW_TITLE。")
            self.win_id, self.window_title, _, self.process_name = candidates[0]
            self._refresh_windows()
            return
        import Quartz
        for w in Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID):
            if "WeChat" in w.get("kCGWindowOwnerName", "") and "天天象棋" in w.get("kCGWindowName", ""):
                self.win_id = w["kCGWindowNumber"]
                b = w["kCGWindowBounds"]
                self.x, self.y, self.width, self.height = map(int, (b["X"], b["Y"], b["Width"], b["Height"]))
                return
        raise RuntimeError("天天象棋 window not found!")

    def _windows_candidates(self):
        """Enumerate visible top-level Windows windows, best match first."""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [
            ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD]
        kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD,
            ctypes.wintypes.LPWSTR, ctypes.POINTER(ctypes.wintypes.DWORD)]
        kernel32.QueryFullProcessImageNameW.restype = ctypes.wintypes.BOOL
        kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
        kernel32.CloseHandle.restype = ctypes.wintypes.BOOL
        target = os.environ.get("XIANGQI_WINDOW_TITLE", "天天象棋").casefold()
        process_names = {
            name.strip().casefold()
            for name in os.environ.get(
                "XIANGQI_WINDOW_PROCESS",
                "Weixin.exe,WeChat.exe,WeixinAppEx.exe,WeChatAppEx.exe",
            ).split(",")
            if name.strip()
        }
        matches = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        own_pid = os.getpid()

        def process_name(pid):
            handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:
                return ""
            try:
                size = ctypes.wintypes.DWORD(32768)
                path = ctypes.create_unicode_buffer(size.value)
                if not kernel32.QueryFullProcessImageNameW(
                        handle, 0, path, ctypes.byref(size)):
                    return ""
                return os.path.basename(path.value)
            finally:
                kernel32.CloseHandle(handle)

        def visit(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if not length:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if target in title.casefold():
                pid = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value == own_pid:
                    return True
                owner = process_name(pid.value)
                if owner.casefold() not in process_names:
                    return True
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                width = max(0, rect.right - rect.left)
                height = max(0, rect.bottom - rect.top)
                # Ignore tooltips, search suggestions, and hidden helper windows.
                if width >= 500 and height >= 400:
                    matches.append((int(hwnd), title, width * height, owner))
            return True

        user32.EnumWindows(callback_type(visit), 0)
        matches.sort(key=lambda item: item[2], reverse=True)
        return matches

    def _refresh_windows(self):
        if IS_WINDOWS and self.win_id:
            rect = ctypes.wintypes.RECT()
            if not ctypes.windll.user32.GetWindowRect(self.win_id, ctypes.byref(rect)):
                raise RuntimeError("天天象棋窗口已关闭。")
            self.x, self.y = rect.left, rect.top
            self.width, self.height = rect.right - rect.left, rect.bottom - rect.top
            if self.width <= 0 or self.height <= 0:
                raise RuntimeError("天天象棋窗口已最小化，请恢复窗口后重试。")

    def screenshot(self):
        self._refresh_windows()
        if IS_WINDOWS:
            # mss captures visible screen pixels. Always foreground the game so
            # the launch console or another window cannot contaminate the CNN.
            user32 = ctypes.windll.user32
            if user32.IsIconic(self.win_id):
                user32.ShowWindow(self.win_id, 9)  # SW_RESTORE
            user32.SetForegroundWindow(self.win_id)
            time.sleep(0.15)
            self._refresh_windows()
            import mss
            with mss.mss() as sct:
                shot = np.array(sct.grab({"left": self.x, "top": self.y, "width": self.width, "height": self.height}))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        import subprocess
        path = os.path.join(os.path.dirname(__file__), ".window_capture.png")
        subprocess.run(["screencapture", "-x", "-o", "-l", str(self.win_id), path], check=True, capture_output=True)
        image = cv2.imread(path)
        if image is None: raise RuntimeError("Screenshot failed!")
        return image

    def activate(self):
        if IS_WINDOWS:
            user32 = ctypes.windll.user32
            if user32.IsIconic(self.win_id):
                user32.ShowWindow(self.win_id, 9)  # SW_RESTORE
            user32.SetForegroundWindow(self.win_id)
            self._refresh_windows()
            time.sleep(.3)
        else:
            import subprocess
            subprocess.run(["osascript", "-e", 'tell application "WeChat" to activate'], capture_output=True, timeout=2)
            time.sleep(.3)
        # Focus the title bar without touching the game content.
        self.click(self.x + self.width // 2, self.y + 8)
        time.sleep(.2)

    def click(self, x, y):
        if IS_WINDOWS:
            pyautogui.moveTo(int(x), int(y), duration=0)
            pyautogui.click()
            return
        import Quartz
        point = Quartz.CGPointMake(int(x), int(y))
        move = Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, 0)
        down = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft)
        up = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, move)
        time.sleep(.08)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
        time.sleep(.08)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)

    def press_escape(self):
        pyautogui.press("esc")
