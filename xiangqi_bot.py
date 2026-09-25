#!/usr/bin/env python3
"""
Xiangqi Bot - Auto-play 天天象棋 using Pikafish engine.

Usage:
  1. Open 天天象棋 in WeChat, start a game (initial position)
  2. Run xiangqi_bot.py (or 启动单局.bat on Windows)
  3. Calibration is automatic; follow the prompts only if it falls back to manual mode
  4. Bot auto-plays. Press Ctrl+C to stop.
"""

import subprocess
import sys
import time
import os
import queue
import random
import threading
import ctypes
import numpy as np
import cv2
import pyautogui
from platform_adapter import WindowAdapter

pyautogui.FAILSAFE = True  # Move mouse to corner to abort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIKAFISH = os.environ.get("PIKAFISH_PATH", os.path.join(
    _SCRIPT_DIR, "pikafish.exe" if os.name == "nt" else "pikafish"))
PIKAFISH_DIR = os.path.dirname(os.path.abspath(PIKAFISH))
TEMPLATE_DIR = os.path.join(_SCRIPT_DIR, "templates")
SCREENSHOT_PATH = os.path.join(_SCRIPT_DIR, "screen.png")
CALIB_PATH = os.path.join(_SCRIPT_DIR, "calib.json")
DIAGNOSTIC_DIR = os.path.join(_SCRIPT_DIR, "故障诊断记录")
DIAGNOSTICS_ENABLED = os.environ.get("XIANGQI_DIAGNOSTICS") == "1"
MOVE_TIME_MS = 2500  # More reliable tactical search with a practical response time
ENDGAME_MOVE_TIME_MS = 6500
LATE_MIDDLEGAME_MOVE_TIME_MS = 4000
SEARCH_DEPTH = 0    # 0 = unlimited (use movetime), >0 = limit search depth
ENGINE_TIMEOUT_S = 10.0
ENGINE_THREADS = 4
ENGINE_HASH_MB = 512
ENGINE_MULTIPV = 1
TURN_POLL_INTERVAL_S = 0.5
TURN_WAIT_TIMEOUT_S = 150.0
MOVE_DELAY_MIN_S = 2.0
MOVE_DELAY_MAX_S = 10.0
CALIB_VERSION = 2

# Initial board layouts (screen space)
INIT_RED = [
    ['r','n','b','a','k','a','b','n','r'],
    [None]*9,
    [None,'c',None,None,None,None,None,'c',None],
    ['p',None,'p',None,'p',None,'p',None,'p'],
    [None]*9, [None]*9,
    ['P',None,'P',None,'P',None,'P',None,'P'],
    [None,'C',None,None,None,None,None,'C',None],
    [None]*9,
    ['R','N','B','A','K','A','B','N','R'],
]
INIT_BLACK = [
    ['R','N','B','A','K','A','B','N','R'],
    [None]*9,
    [None,'C',None,None,None,None,None,'C',None],
    ['P',None,'P',None,'P',None,'P',None,'P'],
    [None]*9, [None]*9,
    ['p',None,'p',None,'p',None,'p',None,'p'],
    [None,'c',None,None,None,None,None,'c',None],
    [None]*9,
    ['r','n','b','a','k','a','b','n','r'],
]


class Bot:
    def __init__(self):
        self.cols_logical = []  # 9 x-coords in logical screen space
        self.rows_logical = []  # 10 y-coords in logical screen space
        self.cell_w = 0
        self.cell_h = 0
        self.templates = {}
        self.patch_size = 0
        self.playing_red = True
        self.retina_scale = 2.0
        self.win_id = None
        self.win_x = 0
        self.win_y = 0
        self.cnn = None  # CNN classifier (loaded on demand)
        self.stop_flag = False  # Set to True to stop the bot
        self.move_delay_enabled = False
        self.observer_mode = os.environ.get('XIANGQI_OBSERVER') == '1'
        self.calib_ratios = None
        self.platform = WindowAdapter()
        self._engine_proc = None
        self._engine_lines = None
        self._engine_reader = None
        self._engine_lock = threading.Lock()
        self._last_stability_image = None

    def _save_failure_diagnostics(self, reason, images=None, metadata=None):
        """Best-effort failure bundle; diagnostics must never affect play."""
        if not DIAGNOSTICS_ENABLED:
            return None
        try:
            import json
            stamp = time.strftime("%Y%m%d_%H%M%S")
            millis = int((time.time() % 1) * 1000)
            safe_reason = ''.join(
                ch if ch.isalnum() or ch in '-_' else '_'
                for ch in str(reason))[:48]
            bundle_dir = os.path.join(
                DIAGNOSTIC_DIR, f"{stamp}_{millis:03d}_{safe_reason}")
            os.makedirs(bundle_dir, exist_ok=False)

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "reason": str(reason),
                "playing_red": self.playing_red,
                "window": {
                    "x": self.win_x, "y": self.win_y,
                    "width": getattr(self.platform, 'width', None),
                    "height": getattr(self.platform, 'height', None),
                    "retina_scale": self.retina_scale,
                },
                "calibration_ratios": list(self.calib_ratios)
                if self.calib_ratios else None,
                "metadata": metadata or {},
            }
            with open(os.path.join(bundle_dir, "record.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2,
                          default=str)
            for name, image in (images or {}).items():
                if image is None or not hasattr(image, 'shape'):
                    continue
                safe_name = ''.join(
                    ch if ch.isalnum() or ch in '-_' else '_'
                    for ch in str(name))[:48]
                image_path = os.path.join(bundle_dir, f"{safe_name}.png")
                encoded, payload = cv2.imencode(".png", image)
                if encoded:
                    payload.tofile(image_path)  # supports Chinese Windows paths
            print(f"  Diagnostic bundle saved: {bundle_dir}")
            return bundle_dir
        except Exception as exc:
            print(f"  Diagnostic save skipped: {exc}")
            return None

    # --- Window & Screenshot ---

    def find_window(self):
        self.platform.find_wechat_xiangqi()
        self.win_id = self.platform.win_id
        self.win_x, self.win_y = self.platform.x, self.platform.y
        print(f"  Window: id={self.win_id} pos=({self.win_x},{self.win_y}) size={self.platform.width}x{self.platform.height}")

    def screenshot_for_processing(self):
        """Capture window, return image. Uses unique filename each time."""
        full = self.platform.screenshot()
        self.win_x, self.win_y = self.platform.x, self.platform.y
        self.retina_scale = full.shape[1] / max(1, self.platform.width)
        self._refresh_grid_from_ratios()
        return full

    def _refresh_grid_from_ratios(self):
        """Keep calibrated grid coordinates aligned after a window move/resize."""
        if not self.calib_ratios:
            return
        rx1, ry1, rx2, ry2 = self.calib_ratios
        win_w, win_h = self.platform.width, self.platform.height
        if win_w <= 0 or win_h <= 0:
            return
        x1, y1 = self.win_x + rx1 * win_w, self.win_y + ry1 * win_h
        x2, y2 = self.win_x + rx2 * win_w, self.win_y + ry2 * win_h
        self.cell_w = (x2 - x1) / 8.0
        self.cell_h = (y2 - y1) / 9.0
        self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
        self.rows_logical = [y1 + j * self.cell_h for j in range(10)]
        self.calib_ratios = (rx1, ry1, rx2, ry2)

    def logical_to_pixel(self, lx, ly):
        """Convert logical screen coords to full-res pixel coords in window capture."""
        # Logical coords are absolute screen coords
        # Window capture starts at (win_x, win_y) in logical space
        px = (lx - self.win_x) * self.retina_scale
        py = (ly - self.win_y) * self.retina_scale
        return int(px), int(py)

    # --- Calibration ---

    def auto_calibrate(self, img):
        """Auto-detect board grid using CNN. Start from default ratios, fine-tune with grid search."""
        if not self.load_cnn():
            print("  Auto-calibrate: CNN not available")
            return False

        import json
        win_w = self._get_window_width()
        win_h = self._get_window_height()

        # Current Windows mini-program layout. The title bar makes the vertical
        # ratios different from the original macOS capture.
        if os.name == 'nt':
            DEFAULT_RX1, DEFAULT_RY1 = 0.3020, 0.1640
            DEFAULT_RX2, DEFAULT_RY2 = 0.6980, 0.8960
        else:
            DEFAULT_RX1, DEFAULT_RY1 = 0.2985, 0.1344
            DEFAULT_RX2, DEFAULT_RY2 = 0.7027, 0.9052

        def try_grid(rx1, ry1, rx2, ry2):
            """Try a grid configuration, return (total_confidence, non_empty_count)."""
            x1 = self.win_x + rx1 * win_w
            y1 = self.win_y + ry1 * win_h
            x2 = self.win_x + rx2 * win_w
            y2 = self.win_y + ry2 * win_h
            cw = (x2 - x1) / 8.0
            ch = (y2 - y1) / 9.0
            cols = [x1 + i * cw for i in range(9)]
            rows = [y1 + j * ch for j in range(10)]
            # Suppress FEN validation output during grid search
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                board = self.cnn.parse_board(
                    img, cols, rows, self.retina_scale,
                    self.win_x, self.win_y, cw, ch)
            finally:
                sys.stdout = old_stdout
            confidences = [
                float(self.cnn._cell_probs[r][c].max())
                for r in range(10) for c in range(9)
                if self.cnn._cell_probs[r][c] is not None
            ]
            n_pieces = sum(cell is not None for row in board for cell in row)
            has_kings = any('K' in row for row in board) and any('k' in row for row in board)
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            # Reject grids that invent pieces or lose either king. Among legal
            # candidates, prefer whole-board confidence instead of rewarding
            # extra non-empty predictions.
            legal_piece_count = 2 <= n_pieces <= 32
            score = avg_conf + (2.0 if has_kings else 0.0) + (1.0 if legal_piece_count else -3.0)
            return score, n_pieces

        # First try default ratios
        best_score, best_n = try_grid(DEFAULT_RX1, DEFAULT_RY1, DEFAULT_RX2, DEFAULT_RY2)
        best_params = (DEFAULT_RX1, DEFAULT_RY1, DEFAULT_RX2, DEFAULT_RY2)

        # Fine-tune: grid search around default with small offsets
        STEP = 0.004  # ~0.4% of window size
        for dx in [-STEP, 0, STEP]:
            for dy in [-STEP, 0, STEP]:
                for ds in [-STEP, 0, STEP]:  # scale adjustment
                    if dx == 0 and dy == 0 and ds == 0:
                        continue
                    rx1 = DEFAULT_RX1 + dx
                    ry1 = DEFAULT_RY1 + dy
                    rx2 = DEFAULT_RX2 + dx + ds
                    ry2 = DEFAULT_RY2 + dy + ds
                    score, n = try_grid(rx1, ry1, rx2, ry2)
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_params = (rx1, ry1, rx2, ry2)

        rx1, ry1, rx2, ry2 = best_params
        self.calib_ratios = (rx1, ry1, rx2, ry2)
        x1 = self.win_x + rx1 * win_w
        y1 = self.win_y + ry1 * win_h
        x2 = self.win_x + rx2 * win_w
        y2 = self.win_y + ry2 * win_h

        self.cell_w = (x2 - x1) / 8.0
        self.cell_h = (y2 - y1) / 9.0
        self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
        self.rows_logical = [y1 + j * self.cell_h for j in range(10)]

        print(f"  Auto-calibrate (CNN): {best_n} pieces, score={best_score:.2f}, cell={self.cell_w:.1f}x{self.cell_h:.1f}")

        if best_n < 2 or best_n > 32:
            print("  Auto-calibrate: invalid piece count")
            return False

        # Save only a grid that passed the basic recognition check.
        with open(CALIB_PATH, 'w') as f:
            json.dump({
                'version': CALIB_VERSION,
                'platform': os.name,
                'rx1': rx1, 'ry1': ry1, 'rx2': rx2, 'ry2': ry2,
            }, f)
        return True

    def calibrate(self):
        """Calibration: user confirms the two corner grid intersections."""
        print("\n=== CALIBRATION ===")
        input("Move mouse to the TOP-LEFT grid intersection, then press ENTER (do not click)...")
        x1, y1 = pyautogui.position()
        print(f"  Top-left: ({x1}, {y1})        ")

        input("Move mouse to the BOTTOM-RIGHT grid intersection, then press ENTER (do not click)...")
        x2, y2 = pyautogui.position()
        print(f"  Bottom-right: ({x2}, {y2})        ")

        win_w = self._get_window_width()
        win_h = self._get_window_height()
        inside_window = (
            self.win_x <= x1 <= self.win_x + win_w and
            self.win_x <= x2 <= self.win_x + win_w and
            self.win_y <= y1 <= self.win_y + win_h and
            self.win_y <= y2 <= self.win_y + win_h)
        cell_w = (x2 - x1) / 8.0
        cell_h = (y2 - y1) / 9.0
        if (not inside_window or x1 >= x2 or y1 >= y2 or
                cell_w < 20 or cell_h < 20):
            print("\n  ERROR: invalid calibration points; nothing was saved.")
            print("  Move the entire game window onto the screen, then run")
            print("  重新校准并启动.bat again and confirm the two corner grid intersections.")
            return False

        self.cell_w = cell_w
        self.cell_h = cell_h

        self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
        self.rows_logical = [y1 + j * self.cell_h for j in range(10)]
        self.calib_ratios = (
            (x1 - self.win_x) / win_w,
            (y1 - self.win_y) / win_h,
            (x2 - self.win_x) / win_w,
            (y2 - self.win_y) / win_h)

        print(f"\n  Cell size: {self.cell_w:.1f} x {self.cell_h:.1f} logical pixels")
        print(f"  Grid: x=[{x1:.0f}..{x2:.0f}] y=[{y1:.0f}..{y2:.0f}]")

        # Save calibration as relative to window (survives resize)
        import json
        with open(CALIB_PATH, 'w') as f:
            json.dump({
                'version': CALIB_VERSION,
                'platform': os.name,
                'rx1': self.calib_ratios[0],
                'ry1': self.calib_ratios[1],
                'rx2': self.calib_ratios[2],
                'ry2': self.calib_ratios[3],
            }, f)
        print(f"  Saved to {CALIB_PATH}")
        return True

    def _get_window_width(self):
        self.platform._refresh_windows()
        return self.platform.width

    def _get_window_height(self):
        self.platform._refresh_windows()
        return self.platform.height

    def load_calibration(self):
        """Load calibration, auto-adapt to current window size/position."""
        import json
        if not os.path.exists(CALIB_PATH):
            return False
        try:
            with open(CALIB_PATH) as f:
                d = json.load(f)

            # Legacy calibration shipped by the original macOS project. It was
            # measured on the author's window and must not be reused on Windows.
            if os.name == 'nt' and (
                    d.get('version') != CALIB_VERSION or
                    d.get('platform') != 'nt'):
                print("  Ignoring legacy calibration; recalibrating for this Windows window")
                return False

            if 'rx1' in d and not (
                    0 <= d['rx1'] < d['rx2'] <= 1 and
                    0 <= d['ry1'] < d['ry2'] <= 1):
                print("  Ignoring invalid calibration coordinates")
                return False

            # Support both relative (new) and absolute (old) formats
            if 'rx1' in d:
                win_w = self._get_window_width()
                win_h = self._get_window_height()
                self.calib_ratios = (d['rx1'], d['ry1'], d['rx2'], d['ry2'])
                x1 = self.win_x + d['rx1'] * win_w
                y1 = self.win_y + d['ry1'] * win_h
                x2 = self.win_x + d['rx2'] * win_w
                y2 = self.win_y + d['ry2'] * win_h
            else:
                x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
                if d.get('win_x') != self.win_x or d.get('win_y') != self.win_y:
                    dx = self.win_x - d.get('win_x', 0)
                    dy = self.win_y - d.get('win_y', 0)
                    x1 += dx; y1 += dy; x2 += dx; y2 += dy
                win_w = self._get_window_width()
                win_h = self._get_window_height()
                self.calib_ratios = (
                    (x1 - self.win_x) / win_w,
                    (y1 - self.win_y) / win_h,
                    (x2 - self.win_x) / win_w,
                    (y2 - self.win_y) / win_h)

            self.cell_w = (x2 - x1) / 8.0
            self.cell_h = (y2 - y1) / 9.0
            self.cols_logical = [x1 + i * self.cell_w for i in range(9)]
            self.rows_logical = [y1 + j * self.cell_h for j in range(10)]
            print(f"  Loaded calibration: cell={self.cell_w:.1f}x{self.cell_h:.1f}")
            return True
        except:
            return False

    def validate_calibration(self, img):
        """Reject a loaded grid when CNN evidence says it is visibly misaligned."""
        if not self.load_cnn():
            return True
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            board = self.cnn.parse_board(
                img, self.cols_logical, self.rows_logical,
                self.retina_scale, self.win_x, self.win_y,
                self.cell_w, self.cell_h)
        finally:
            sys.stdout = old_stdout
        piece_count = sum(cell is not None for row in board for cell in row)
        confidences = [
            float(self.cnn._cell_probs[r][c].max())
            for r in range(10) for c in range(9)
            if self.cnn._cell_probs[r][c] is not None
        ]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        valid = 10 <= piece_count <= 32 and avg_conf >= 0.70
        if not valid:
            print(f"  Loaded grid failed validation: pieces={piece_count}, confidence={avg_conf:.0%}")
        return valid

    # --- Orientation & Templates ---

    def detect_orientation(self, img):
        """Check if top-row pieces are red (→ user plays BLACK)."""
        ps = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.6)
        red_total = 0
        for ci in [0, 4, 8]:
            px, py = self.logical_to_pixel(self.cols_logical[ci], self.rows_logical[0])
            h, w = img.shape[:2]
            patch = img[max(0,py-ps):min(h,py+ps), max(0,px-ps):min(w,px+ps)]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
            m2 = cv2.inRange(hsv, (168, 60, 60), (180, 255, 255))
            red_total += (cv2.countNonZero(m1) + cv2.countNonZero(m2)) / max(1, patch.size//3)
        self.playing_red = red_total < 0.03
        print(f"  You play: {'RED' if self.playing_red else 'BLACK'}")

    def determine_orientation(self, board):
        """Require king placement and whole-board piece cases to agree."""
        red_kings = [(r, c) for r in range(10) for c in range(9)
                     if board[r][c] == 'K']
        black_kings = [(r, c) for r in range(10) for c in range(9)
                       if board[r][c] == 'k']
        votes = []
        if len(red_kings) == 1 and len(black_kings) == 1:
            if red_kings[0][0] == black_kings[0][0]:
                return None, "the two kings were detected on the same rank"
            votes.append(("king positions", red_kings[0][0] > black_kings[0][0]))

        red_bottom = sum(1 for r in range(5, 10) for p in board[r]
                         if p and p.isupper())
        red_top = sum(1 for r in range(5) for p in board[r]
                      if p and p.isupper())
        black_top = sum(1 for r in range(5) for p in board[r]
                        if p and p.islower())
        black_bottom = sum(1 for r in range(5, 10) for p in board[r]
                           if p and p.islower())
        normal = red_bottom + black_top
        reversed_view = red_top + black_bottom
        if abs(normal - reversed_view) >= 2:
            votes.append(("piece distribution", normal > reversed_view))

        if not votes:
            return None, "not enough reliable red/black evidence"
        if any(value != votes[0][1] for _, value in votes[1:]):
            details = ", ".join(f"{name}={'RED' if value else 'BLACK'}"
                                for name, value in votes)
            return None, f"orientation signals conflict ({details})"
        details = ", ".join(name for name, _ in votes)
        return votes[0][1], details

    def capture_templates(self, img):
        """Capture templates from ALL initial positions (multiple per piece)."""
        os.makedirs(TEMPLATE_DIR, exist_ok=True)
        init = INIT_RED if self.playing_red else INIT_BLACK
        ps = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.7)
        self.patch_size = ps
        self.templates = {}  # piece -> [tmpl1, tmpl2, ...]

        for r in range(10):
            for c in range(9):
                piece = init[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img, px, py, ps)
                if patch is not None:
                    if piece not in self.templates:
                        self.templates[piece] = []
                    self.templates[piece].append(patch)

        # Empty cells from many positions (including edges)
        self.templates['_'] = []
        for er in [1, 4, 5, 8]:  # rows that are fully empty
            for ec in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[ec], self.rows_logical[er])
                ep = self._extract(img, px, py, ps)
                if ep is not None:
                    self.templates['_'].append(ep)

        total = sum(len(v) for k, v in self.templates.items() if k != '_')
        print(f"  Templates: {len(self.templates)} types, {total} variants")

    def _extract(self, img, cx, cy, ps):
        h, w = img.shape[:2]
        x1, y1 = max(0, cx-ps), max(0, cy-ps)
        x2, y2 = min(w, cx+ps), min(h, cy+ps)
        p = img[y1:y2, x1:x2]
        return p if p.shape[0] >= ps and p.shape[1] >= ps else None

    # --- Color-based Piece Classifier (v2) ---

    def _extract_piece_center(self, img, px, py, radius=None):
        """Extract the circular piece region centered at (px, py)."""
        if radius is None:
            radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.28)
        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < radius or patch.shape[1] < radius:
            return None
        return patch

    def _find_piece_circle(self, img, px, py):
        """Detect the circular piece token using Hough circles.

        Returns (cx, cy, cr) in image coordinates if found, or None.
        The circle center is in ABSOLUTE image coordinates.
        """
        scale = min(self.cell_w, self.cell_h) * self.retina_scale
        search_r = int(scale * 0.55)
        piece_r_min = int(scale * 0.22)
        piece_r_max = int(scale * 0.42)
        center_tol = int(scale * 0.25)

        h, w = img.shape[:2]
        x1 = max(0, px - search_r)
        y1 = max(0, py - search_r)
        x2 = min(w, px + search_r)
        y2 = min(h, py + search_r)
        patch = img[y1:y2, x1:x2]

        if patch.shape[0] < 30 or patch.shape[1] < 30:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        ph, pw = gray.shape[:2]

        # Try multiple blur sizes and Hough sensitivity levels
        for blur_k in [7, 5, 9]:
            blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 2.0)
            for p1, p2 in [(70, 30), (60, 25), (50, 22), (80, 35)]:
                circles = cv2.HoughCircles(
                    blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=search_r,
                    param1=p1, param2=p2,
                    minRadius=piece_r_min,
                    maxRadius=piece_r_max
                )
                if circles is not None:
                    # Find circle closest to patch center
                    best_circ = None
                    best_dist = float('inf')
                    for circ in circles[0]:
                        cx, cy, cr = circ
                        dist = np.sqrt((cx - pw / 2) ** 2 + (cy - ph / 2) ** 2)
                        if dist < center_tol and dist < best_dist:
                            best_dist = dist
                            best_circ = circ
                    if best_circ is not None:
                        abs_cx = int(best_circ[0] + x1)
                        abs_cy = int(best_circ[1] + y1)
                        abs_cr = int(best_circ[2])
                        return (abs_cx, abs_cy, abs_cr)
        return None

    def _has_piece_v2(self, img, px, py):
        """Check if the cell at (px, py) has a piece.

        Uses Hough circle detection as primary method, with a color-based
        fallback that checks for the characteristic piece ring color.
        """
        # Primary: Hough circle detection
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            return True

        # Fallback: check for red/dark ring pixels in an annular region
        # This catches pieces that Hough misses (edge cases, partial visibility)
        scale = min(self.cell_w, self.cell_h) * self.retina_scale
        outer_r = int(scale * 0.35)
        inner_r = int(scale * 0.22)

        h, w = img.shape[:2]
        x1 = max(0, px - outer_r)
        y1 = max(0, py - outer_r)
        x2 = min(w, px + outer_r)
        y2 = min(h, py + outer_r)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < outer_r or patch.shape[1] < outer_r:
            return False

        ph, pw = patch.shape[:2]
        pcx, pcy = pw // 2, ph // 2

        # Create annular mask (ring where piece border would be)
        mask_outer = np.zeros((ph, pw), dtype=np.uint8)
        mask_inner = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(mask_outer, (pcx, pcy), outer_r, 255, -1)
        cv2.circle(mask_inner, (pcx, pcy), inner_r, 255, -1)
        ring_mask = cv2.subtract(mask_outer, mask_inner)

        # Check for red ring pixels (red piece borders)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, (0, 80, 100), (12, 255, 200))
        red_mask2 = cv2.inRange(hsv, (168, 80, 100), (180, 255, 200))
        red_ring = cv2.bitwise_and(cv2.bitwise_or(red_mask1, red_mask2), ring_mask)
        red_count = cv2.countNonZero(red_ring)

        # Check for dark ring pixels (black piece borders)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dark_ring_vals = gray[ring_mask > 0]
        dark_count = int(np.sum(dark_ring_vals < 100)) if len(dark_ring_vals) > 0 else 0

        ring_total = max(1, cv2.countNonZero(ring_mask))
        red_ratio = red_count / ring_total
        dark_ratio = dark_count / ring_total

        # A piece border needs both red/dark ring AND bright center (piece token is convex)
        # Be strict to avoid false positives from board borders and adjacent piece edges
        center_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(center_mask, (pcx, pcy), int(scale * 0.10), 255, -1)
        center_vals = gray[center_mask > 0]
        center_brightness = float(np.mean(center_vals)) if len(center_vals) > 0 else 0
        ring_brightness = float(np.mean(dark_ring_vals)) if len(dark_ring_vals) > 0 else 0

        has_ring = red_ratio > 0.15 or dark_ratio > 0.25
        has_bright_center = center_brightness > 140 and center_brightness > ring_brightness + 5

        return has_ring and has_bright_center

    def _classify_color_v2(self, img, px, py):
        """Classify a piece as RED or BLACK based on text/ring color.

        Red pieces have red-colored Chinese characters and ring on a tan background.
        Black pieces have dark/black characters and ring on a tan background.

        Returns 'red' or 'black'.
        """
        # Use the piece center for color analysis
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.22)

        # Try to use actual circle center if found
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            px, py = circle[0], circle[1]

        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Create circular mask
        ph, pw = patch.shape[:2]
        mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.circle(mask, (pw // 2, ph // 2), min(ph, pw) // 2, 255, -1)

        # Count red pixels (text/ring of red pieces)
        red_mask1 = cv2.inRange(hsv, (0, 70, 80), (12, 255, 255))
        red_mask2 = cv2.inRange(hsv, (168, 70, 80), (180, 255, 255))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(red_mask1, red_mask2), mask)
        red_count = cv2.countNonZero(red_mask)

        # Count dark pixels (text/ring of black pieces)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        dark_vals = gray[mask > 0]
        dark_count = int(np.sum(dark_vals < 80)) if len(dark_vals) > 0 else 0

        total = max(1, cv2.countNonZero(mask))
        red_ratio = red_count / total
        dark_ratio = dark_count / total

        if red_ratio > 0.03:
            return 'red'
        elif dark_ratio > 0.03:
            return 'black'
        else:
            return 'red' if red_ratio > dark_ratio else 'black'

    def _compute_feature_vector(self, img, px, py, grid_size=5):
        """Compute a structural feature vector from the character on a piece.

        Uses multiple complementary features:
        1. Stroke density in a grid (captures overall character shape)
        2. Horizontal and vertical edge density (captures stroke orientations)
        3. Multi-scale: coarse (4x4) and fine (6x6) grids

        Returns a normalized feature vector.
        """
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.18)

        # Use actual circle center if found
        circle = self._find_piece_circle(img, px, py)
        if circle is not None:
            px, py = circle[0], circle[1]

        h, w = img.shape[:2]
        x1 = max(0, px - radius)
        y1 = max(0, py - radius)
        x2 = min(w, px + radius)
        y2 = min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.shape[0] < radius or patch.shape[1] < radius:
            return None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ph, pw = gray.shape

        # Create circular mask
        mask = np.zeros((ph, pw), dtype=np.float32)
        cv2.circle(mask, (pw // 2, ph // 2), min(ph, pw) // 2, 1.0, -1)

        # Normalize brightness within mask
        masked_pixels = gray[mask > 0]
        if len(masked_pixels) == 0:
            return None
        local_mean = np.mean(masked_pixels)
        local_std = np.std(masked_pixels)
        if local_std < 5:
            return None

        # Normalized grayscale (mean-subtracted, std-normalized)
        norm_gray = (gray - local_mean) / local_std
        norm_gray = norm_gray * mask

        # Compute edge maps for orientation features
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * mask
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * mask

        features = []

        # Feature set 1: normalized brightness in grid cells (captures character structure)
        for gs in [4, 6]:  # multi-scale
            for gi in range(gs):
                for gj in range(gs):
                    y_start = int(gi * ph / gs)
                    y_end = int((gi + 1) * ph / gs)
                    x_start = int(gj * pw / gs)
                    x_end = int((gj + 1) * pw / gs)
                    region = norm_gray[y_start:y_end, x_start:x_end]
                    region_mask = mask[y_start:y_end, x_start:x_end]
                    mask_sum = np.sum(region_mask)
                    if mask_sum > 0:
                        features.append(float(np.sum(region) / mask_sum))
                    else:
                        features.append(0.0)

        # Feature set 2: horizontal vs vertical edge balance in grid cells
        for gi in range(4):
            for gj in range(4):
                y_start = int(gi * ph / 4)
                y_end = int((gi + 1) * ph / 4)
                x_start = int(gj * pw / 4)
                x_end = int((gj + 1) * pw / 4)

                sx = np.abs(sobel_x[y_start:y_end, x_start:x_end])
                sy = np.abs(sobel_y[y_start:y_end, x_start:x_end])
                region_mask = mask[y_start:y_end, x_start:x_end]
                mask_sum = np.sum(region_mask)
                if mask_sum > 0:
                    # Ratio of vertical to horizontal edges
                    sx_sum = float(np.sum(sx * region_mask))
                    sy_sum = float(np.sum(sy * region_mask))
                    total_edge = sx_sum + sy_sum + 1e-8
                    features.append(sx_sum / total_edge)  # horizontal dominance
                else:
                    features.append(0.5)

        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def _cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def capture_feature_vectors(self, img):
        """Capture feature vectors from known piece positions in the initial layout.

        This builds self.piece_features: dict mapping piece code -> list of feature vectors
        and self.piece_color_map: dict mapping piece code -> 'red' or 'black'
        """
        init = INIT_RED if self.playing_red else INIT_BLACK
        self.piece_features = {}  # piece -> [vec1, vec2, ...]
        self.piece_color_map = {}

        for r in range(10):
            for c in range(9):
                piece = init[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                vec = self._compute_feature_vector(img, px, py)
                if vec is not None:
                    if piece not in self.piece_features:
                        self.piece_features[piece] = []
                    self.piece_features[piece].append(vec)
                    # Map piece to color
                    if piece.isupper():
                        self.piece_color_map[piece] = 'red' if self.playing_red else 'black'
                    else:
                        self.piece_color_map[piece] = 'black' if self.playing_red else 'red'

        total = sum(len(v) for v in self.piece_features.values())
        print(f"  Feature vectors: {len(self.piece_features)} types, {total} vectors")

    def capture_feature_vectors_from_board(self, img, board):
        """Rebuild feature vectors from current known board state.

        Uses only cells where we confidently know what piece is there.
        """
        self.piece_features = {}
        for r in range(10):
            for c in range(9):
                piece = board[r][c]
                if piece is None:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                vec = self._compute_feature_vector(img, px, py)
                if vec is not None:
                    if piece not in self.piece_features:
                        self.piece_features[piece] = []
                    self.piece_features[piece].append(vec)

    def identify_v2(self, img, px, py):
        """Identify the piece at pixel position (px, py) using color-based classification.

        Returns piece code (e.g. 'R', 'n', 'K') or None if empty.

        Algorithm:
        1. Check if cell has a piece (Hough circle + edge density fallback)
        2. Classify red vs black via HSV analysis
        3. Match character structure via stroke-density feature vector similarity
        """
        # Step 1: Is there a piece here?
        if not self._has_piece_v2(img, px, py):
            return None

        # Step 2: Red or Black?
        color = self._classify_color_v2(img, px, py)

        # Step 3: Match type via feature vector
        vec = self._compute_feature_vector(img, px, py)
        if vec is None:
            return None

        if not hasattr(self, 'piece_features') or not self.piece_features:
            return None

        # Filter candidates by color
        candidates = {}
        for piece, vecs in self.piece_features.items():
            piece_color = self.piece_color_map.get(piece, 'unknown')
            if piece_color == color or color == 'unknown':
                candidates[piece] = vecs

        if not candidates:
            # Fall back to all pieces
            candidates = self.piece_features

        # Find best match by cosine similarity
        best_piece = None
        best_sim = -1.0

        for piece, vecs in candidates.items():
            for ref_vec in vecs:
                sim = self._cosine_similarity(vec, ref_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_piece = piece

        # Threshold check
        if best_sim < 0.60:
            return None

        return best_piece

    def parse_board_v2(self, img):
        """Parse the full board using identify_v2."""
        board = []
        for r in range(10):
            row = []
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                p = self.identify_v2(img, px, py)
                row.append(p)
            board.append(row)
        return board

    # --- Move validation helpers ---

    def _valid_destinations(self, piece, from_row, from_col, board):
        """Return set of (row, col) reachable positions for a piece.

        Uses basic Chinese chess movement rules. Not a full legal move generator
        (doesn't check for checks etc.) but validates piece-type movement patterns.
        """
        dests = set()
        is_upper = piece.isupper()
        p = piece.upper()

        # Determine which half of the board is "own side"
        # In screen coords: rows 0-4 are top, 5-9 are bottom
        # If playing red: red is bottom (5-9), black is top (0-4)
        # If playing black: red is top (0-4), black is bottom (5-9)
        if self.playing_red:
            own_top = 5 if is_upper else 0  # uppercase=red=bottom, lower=black=top
            own_bottom = 9 if is_upper else 4
            enemy_top = 0 if is_upper else 5
            enemy_bottom = 4 if is_upper else 9
        else:
            own_top = 0 if is_upper else 5
            own_bottom = 4 if is_upper else 9
            enemy_top = 5 if is_upper else 0
            enemy_bottom = 9 if is_upper else 4

        r, c = from_row, from_col

        if p == 'R':  # Rook: straight lines
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                while 0 <= nr <= 9 and 0 <= nc <= 8:
                    dests.add((nr, nc))
                    if board[nr][nc] is not None:
                        break
                    nr += dr
                    nc += dc

        elif p == 'N':  # Knight
            for dr, dc, br, bc in [
                (-2, -1, -1, 0), (-2, 1, -1, 0),
                (2, -1, 1, 0), (2, 1, 1, 0),
                (-1, -2, 0, -1), (-1, 2, 0, 1),
                (1, -2, 0, -1), (1, 2, 0, 1)
            ]:
                # Check blocking piece
                block_r, block_c = r + br, c + bc
                if 0 <= block_r <= 9 and 0 <= block_c <= 8:
                    if board[block_r][block_c] is not None:
                        continue
                nr, nc = r + dr, c + dc
                if 0 <= nr <= 9 and 0 <= nc <= 8:
                    dests.add((nr, nc))

        elif p == 'B':  # Bishop/Elephant: diagonal 2 steps, own half only
            for dr, dc in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
                # Check blocking piece at midpoint
                mr, mc = r + dr // 2, c + dc // 2
                if 0 <= mr <= 9 and 0 <= mc <= 8:
                    if board[mr][mc] is not None:
                        continue
                nr, nc = r + dr, c + dc
                if 0 <= nr <= 9 and 0 <= nc <= 8:
                    if own_top <= nr <= own_bottom:
                        dests.add((nr, nc))

        elif p == 'A':  # Advisor: diagonal 1 step within palace
            # Palace: top 3 rows if own side is top, bottom 3 if own side is bottom
            if own_top == 0:
                palace_r_min, palace_r_max = 0, 2
            else:
                palace_r_min, palace_r_max = 7, 9
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 3 <= nc <= 5 and palace_r_min <= nr <= palace_r_max:
                    dests.add((nr, nc))

        elif p == 'K':  # King: orthogonal 1 step within palace
            if own_top == 0:
                palace_r_min, palace_r_max = 0, 2
            else:
                palace_r_min, palace_r_max = 7, 9
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 3 <= nc <= 5 and palace_r_min <= nr <= palace_r_max:
                    dests.add((nr, nc))
            # Flying general (face-to-face kings) - can capture opposing king
            for dr in [1, -1]:
                nr = r + dr
                while 0 <= nr <= 9:
                    if board[nr][c] is not None:
                        if board[nr][c].upper() == 'K':
                            dests.add((nr, c))
                        break
                    nr += dr

        elif p == 'C':  # Cannon: straight lines, jump capture
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                jumped = False
                while 0 <= nr <= 9 and 0 <= nc <= 8:
                    if not jumped:
                        if board[nr][nc] is None:
                            dests.add((nr, nc))
                        else:
                            jumped = True
                    else:
                        if board[nr][nc] is not None:
                            dests.add((nr, nc))
                            break
                    nr += dr
                    nc += dc

        elif p == 'P':  # Pawn
            # Determine forward direction
            if self.playing_red:
                fwd = -1 if is_upper else 1
            else:
                fwd = 1 if is_upper else -1

            nr = r + fwd
            if 0 <= nr <= 9:
                dests.add((nr, c))

            # After crossing river, can move sideways
            crossed = (r <= own_top - 1) or (r >= own_bottom + 1)
            # More precise: pawn has crossed if it's in enemy half
            if enemy_top <= r <= enemy_bottom:
                crossed = True
            if crossed:
                for dc in [-1, 1]:
                    nc = c + dc
                    if 0 <= nc <= 8:
                        dests.add((r, nc))

        return dests

    def detect_move_v2(self, img_before, img_after, board_before):
        """Detect opponent's move using identify_v2 and move validation.

        Algorithm:
        1. Find cells that changed visually (pixel diff)
        2. Use identify_v2 to classify what's at each changed cell
        3. Determine source (cell that lost a piece) and dest (cell that gained/changed)
        4. Validate against piece movement rules
        5. Fall back to heuristics if needed
        """
        board_after = [row[:] for row in board_before]
        changed = []

        # Step 1: Find changed cells
        for r in range(10):
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                ps = self.patch_size
                h, w = img_before.shape[:2]
                x1, y1 = max(0, px - ps), max(0, py - ps)
                x2, y2 = min(w, px + ps), min(h, py + ps)
                p1 = img_before[y1:y2, x1:x2]
                p2 = img_after[y1:y2, x1:x2]
                if p1.shape != p2.shape:
                    continue
                diff = cv2.absdiff(p1, p2)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                change = np.count_nonzero(gray_diff > 30) / max(1, gray_diff.size)
                if change > 0.03:
                    changed.append((r, c, change))

        if not changed:
            print("    detect_move_v2: no changed cells")
            return board_before

        # Refresh feature vectors from unchanged cells in the new image
        self.capture_feature_vectors_from_board(img_after, board_before)

        # Step 2: Identify what's at each changed cell now
        cell_info = []
        for r, c, change_amt in changed:
            px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
            had_piece = board_before[r][c]
            has_piece_now = self._has_piece_v2(img_after, px, py)
            now_piece = self.identify_v2(img_after, px, py) if has_piece_now else None
            cell_info.append({
                'row': r, 'col': c,
                'change': change_amt,
                'had': had_piece,
                'has_now': has_piece_now,
                'now_piece': now_piece,
            })

        # Debug output
        for ci in cell_info:
            print(f"    Changed ({ci['row']},{ci['col']}): "
                  f"had={ci['had']} now={ci['now_piece']} "
                  f"has_piece={ci['has_now']} change={ci['change']:.3f}")

        # Step 3: Find source and destination
        # Source: cell that had an opponent piece and now is empty or has different piece
        # Destination: cell that now has a piece that wasn't there before, or has a different piece

        sources = []  # Cells that lost a piece
        dests = []    # Cells that gained/changed a piece

        for ci in cell_info:
            if ci['had'] is not None and not ci['has_now']:
                sources.append(ci)
            elif ci['had'] is None and ci['has_now']:
                dests.append(ci)
            elif ci['had'] is not None and ci['has_now']:
                # Piece was here and still a piece here - could be capture destination
                # or could be a highlight/selection artifact
                if ci['now_piece'] is not None and ci['now_piece'] != ci['had']:
                    # Different piece arrived - this is a capture destination
                    dests.append(ci)
                elif ci['change'] > 0.15:
                    # Large change with piece still there - might be dest of a capture
                    dests.append(ci)

        # Handle case: exactly one source, one or zero dests
        if len(sources) == 1 and len(dests) == 1:
            src = sources[0]
            dst = dests[0]
            moving_piece = src['had']

            # Step 4: Validate move
            valid_dests = self._valid_destinations(
                moving_piece, src['row'], src['col'], board_before)

            if (dst['row'], dst['col']) in valid_dests:
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [validated]")
                return board_after
            else:
                # Move doesn't match rules for this piece type - still apply it
                # but warn (could be wrong identification)
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [UNVALIDATED - may be wrong piece type]")
                return board_after

        elif len(sources) == 1 and len(dests) == 0:
            # Piece left but no dest found - might be a piece that moved to a cell
            # with similar appearance. Check all changed cells with piece present.
            src = sources[0]
            moving_piece = src['had']
            valid_dests = self._valid_destinations(
                moving_piece, src['row'], src['col'], board_before)

            # Check which changed cells with a piece are valid destinations
            for ci in cell_info:
                if ci is src:
                    continue
                if ci['has_now'] and (ci['row'], ci['col']) in valid_dests:
                    board_after[src['row']][src['col']] = None
                    board_after[ci['row']][ci['col']] = moving_piece
                    print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                          f"({ci['row']},{ci['col']}) [rule-validated dest]")
                    return board_after

            print(f"    Piece {moving_piece} left ({src['row']},{src['col']}) but no valid dest")

        elif len(sources) >= 2 and len(dests) >= 1:
            # Multiple changes - try to find valid move pairs
            # This can happen with move highlight effects
            best_move = None
            best_score = -1

            for src in sources:
                moving_piece = src['had']
                valid = self._valid_destinations(
                    moving_piece, src['row'], src['col'], board_before)
                for dst in dests:
                    if (dst['row'], dst['col']) in valid:
                        # Score by change amount (higher = more likely real change)
                        score = src['change'] + dst['change']
                        if score > best_score:
                            best_score = score
                            best_move = (src, dst, moving_piece)

            if best_move:
                src, dst, moving_piece = best_move
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = moving_piece
                print(f"    Move: {moving_piece} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [best valid from multiple]")
                return board_after

        # Step 5: Fallback - use the two most-changed cells
        changed_sorted = sorted(cell_info, key=lambda x: x['change'], reverse=True)
        if len(changed_sorted) >= 2:
            # Assume most changed are source and dest
            c1, c2 = changed_sorted[0], changed_sorted[1]
            # Source is the one that had a piece and lost it
            if c1['had'] is not None and not c1['has_now']:
                src, dst = c1, c2
            elif c2['had'] is not None and not c2['has_now']:
                src, dst = c2, c1
            elif c1['had'] is not None:
                src, dst = c1, c2
            else:
                src, dst = c2, c1

            if src['had'] is not None:
                board_after[src['row']][src['col']] = None
                board_after[dst['row']][dst['col']] = src['had']
                print(f"    Move: {src['had']} ({src['row']},{src['col']}) → "
                      f"({dst['row']},{dst['col']}) [fallback: most changed]")
                return board_after

        print("    detect_move_v2: could not determine move")
        return board_before

    # --- Board Parsing (v1 - template matching) ---

    def _masked_corr(self, patch, tmpl, mask):
        """Compute normalized correlation within circular mask."""
        p = patch[mask > 0].astype(np.float32)
        t = tmpl[mask > 0].astype(np.float32)
        if len(p) == 0 or p.std() < 1 or t.std() < 1:
            return 0.0
        pn = (p - p.mean()) / p.std()
        tn = (t - t.mean()) / t.std()
        return float(np.mean(pn * tn))

    def identify(self, img, px, py):
        patch = self._extract(img, px, py, self.patch_size)
        if patch is None:
            return None
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if gray.std() < 5:
            return None

        h, w = patch.shape[:2]
        # Circular mask — only compare inside the piece, ignore background
        mask = np.zeros((h, w), dtype=np.uint8)
        r = int(min(h, w) * 0.32)
        cv2.circle(mask, (w//2, h//2), r, 255, -1)
        # 3-channel mask for color images
        mask3 = cv2.merge([mask, mask, mask])

        best_piece, best_piece_sc = None, 0.0
        best_empty_sc = 0.0

        for pc, tmpls in self.templates.items():
            for tmpl in tmpls:
                t = cv2.resize(tmpl, (w, h))
                sc = self._masked_corr(patch, t, mask)
                if pc == '_':
                    best_empty_sc = max(best_empty_sc, sc)
                elif sc > best_piece_sc:
                    best_piece_sc, best_piece = sc, pc

        if best_empty_sc > best_piece_sc:
            return None
        if best_piece_sc - best_empty_sc < 0.08:
            return None
        return best_piece if best_piece_sc > 0.3 else None

    def parse_board(self, img):
        board = []
        for r in range(10):
            row = []
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                p = self.identify(img, px, py)
                row.append(p)
            board.append(row)
        return board

    def _cell_has_piece_now(self, img, col, row):
        """Check if a cell currently has a piece using brightness."""
        px, py = self.logical_to_pixel(self.cols_logical[col], self.rows_logical[row])
        ps = self.patch_size // 2  # smaller region for center check
        h, w = img.shape[:2]
        x1, y1 = max(0, px-ps), max(0, py-ps)
        x2, y2 = min(w, px+ps), min(h, py+ps)
        patch = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # Pieces are brighter circles with text = higher std
        return gray.std() > 25

    def get_legal_moves(self, fen):
        """Get all legal moves from pikafish for the given position."""
        try:
            result = subprocess.run(
                [PIKAFISH],
                input=f"uci\nisready\nposition fen {fen}\ngo perft 1\nquit\n",
                text=True, capture_output=True, cwd=PIKAFISH_DIR, timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            print("  Legal-move query failed or timed out")
            return []
        moves = []
        for raw in result.stdout.splitlines():
            line = raw.strip()
            head = line.split(':', 1)[0].strip()
            if ':' in line and len(head) == 4:
                moves.append(head)
        return moves

    def _find_move(self, old_fen, new_fen):
        """Find the UCI move that transforms old_fen → new_fen."""
        opp_turn = 'b' if self.playing_red else 'w'
        legal = self.get_legal_moves(f"{old_fen} {opp_turn} - - 0 1")
        for move in legal:
            # Simulate this move on a temp board
            src, dst = self.uci_to_screen_cells(move)
            # Quick check: just try to match FEN
            # (We already know the result board, so just find which legal move matches)
            fc, fr = ord(move[0]) - ord('a'), int(move[1])
            tc, tr = ord(move[2]) - ord('a'), int(move[3])
            if self.playing_red:
                sr1, sc1 = 9 - fr, fc
                sr2, sc2 = 9 - tr, tc
            else:
                sr1, sc1 = fr, 8 - fc
                sr2, sc2 = tr, 8 - tc
            # Parse old_fen to board, apply move, check if result matches new_fen
            # For speed, just return the first move that could explain the change
            temp = self._fen_to_board(old_fen)
            if temp and temp[sr1][sc1] is not None:
                temp[sr2][sc2] = temp[sr1][sc1]
                temp[sr1][sc1] = None
                if self.board_to_fen(temp) == new_fen:
                    return move
        return None

    def _fen_to_board(self, fen):
        """Convert FEN string to 10x9 board array."""
        rows = fen.split('/')
        if len(rows) != 10:
            return None
        board = []
        for row_str in rows:
            row = []
            for ch in row_str:
                if ch.isdigit():
                    row.extend([None] * int(ch))
                else:
                    row.append(ch)
            if len(row) != 9:
                return None
            board.append(row)
        if not self.playing_red:
            board = [row[::-1] for row in reversed(board)]
        return board

    def uci_to_screen_cells(self, move):
        """Convert UCI move to screen (row, col) pairs."""
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        if self.playing_red:
            return (9-fr, fc), (9-tr, tc)
        else:
            return (fr, 8-fc), (tr, 8-tc)

    def _apply_board_move(self, board, move):
        """Apply one UCI move to a screen-oriented board copy."""
        src, dst = self.uci_to_screen_cells(move)
        piece = board[src[0]][src[1]]
        if piece is None:
            return None
        result = [row[:] for row in board]
        result[src[0]][src[1]] = None
        result[dst[0]][dst[1]] = piece
        return result

    def _search_time_for_board(self, board):
        """Spend more time where sparse positions require deeper calculation."""
        piece_count = sum(cell is not None for row in board for cell in row)
        if piece_count <= 12:
            return ENDGAME_MOVE_TIME_MS
        if piece_count <= 20:
            return LATE_MIDDLEGAME_MOVE_TIME_MS
        return MOVE_TIME_MS

    def _match_unique_legal_transition(self, board_before, parsed_board,
                                       legal_moves):
        """Accept a visual board only if exactly one legal move produces it."""
        if not parsed_board:
            return None, None
        target = self.board_to_fen(parsed_board)
        matches = []
        for move in legal_moves:
            result = self._apply_board_move(board_before, move)
            if result is not None and self.board_to_fen(result) == target:
                matches.append((move, result))
        return matches[0] if len(matches) == 1 else (None, None)

    def _select_transition_consensus(self, observations, required=2):
        """Return a transition only after repeated independent agreement."""
        votes = {}
        for move, board in observations:
            if not move or board is None:
                continue
            key = (move, self.board_to_fen(board))
            votes[key] = votes.get(key, 0) + 1
        winners = [(count, key) for key, count in votes.items()
                   if count >= required]
        if len(winners) != 1:
            return None, None
        _, (move, _fen) = winners[0]
        for observed_move, board in observations:
            if (observed_move == move and board is not None and
                    self.board_to_fen(board) == _fen):
                return move, board
        return None, None

    def _select_transition_consensus_with_confidence(self, observations,
                                                     required=2):
        """Select repeated move/FEN agreement and assign A/B confidence."""
        grouped = {}
        for move, board, grade in observations:
            if not move or board is None or grade not in ('A', 'B'):
                continue
            key = (move, self.board_to_fen(board))
            grouped.setdefault(key, []).append((board, grade))
        winners = [(key, entries) for key, entries in grouped.items()
                   if len(entries) >= required]
        if len(winners) != 1:
            return None, None, None
        (move, _fen), entries = winners[0]
        strict_votes = sum(grade == 'A' for _, grade in entries)
        confidence = 'A' if strict_votes >= required else 'B'
        return move, entries[0][0], confidence

    def _state_is_safe_for_engine(self, board, confidence):
        """Final gate before engine search."""
        valid, reason = self.board_is_plausible(board)
        if not valid:
            return False, reason
        if confidence not in ('A', 'B'):
            return False, f"untrusted confidence {confidence!r}"
        return True, ""

    def _match_legal_transition_with_endpoint_changes(
            self, board_before, parsed_board, legal_moves,
            img_before, img_after):
        """Recover one opponent move despite CNN noise on unrelated cells."""
        move, result = self._match_unique_legal_transition(
            board_before, parsed_board, legal_moves)
        if move:
            return move, result
        if not parsed_board or img_before is None or img_after is None:
            return None, None

        candidates = []
        for legal_move in legal_moves:
            expected = self._apply_board_move(board_before, legal_move)
            if expected is None:
                continue
            src, dst = self.uci_to_screen_cells(legal_move)
            # Unrelated CNN mistakes are tolerated, but both endpoints must
            # describe the final position of this particular legal move.
            if (parsed_board[src[0]][src[1]] is not None or
                    parsed_board[dst[0]][dst[1]] != expected[dst[0]][dst[1]]):
                continue
            endpoint_cells = {src, dst}
            unrelated_mismatches = sum(
                parsed_board[r][c] != expected[r][c]
                for r in range(10) for c in range(9)
                if (r, c) not in endpoint_cells)
            # A few isolated CNN errors are expected. More than four means the
            # observed board is too unreliable to update the engine position.
            if unrelated_mismatches > 4:
                continue
            source_delta = self._piece_cell_change(
                img_before, img_after, src[0], src[1])
            destination_delta = self._piece_cell_change(
                img_before, img_after, dst[0], dst[1])
            if source_delta > 8.0 and destination_delta > 8.0:
                candidates.append(
                    (source_delta + destination_delta, legal_move, expected))

        candidates.sort(reverse=True, key=lambda item: item[0])
        if not candidates:
            return None, None
        if len(candidates) > 1:
            best, second = candidates[0][0], candidates[1][0]
            if best < second * 1.20 or best - second < 5.0:
                return None, None
        _, move, result = candidates[0]
        return move, result

    def _move_board_matches(self, parsed_board, expected_board, move,
                            source_changed=False, destination_changed=False):
        """Confirm our legal move without trusting unrelated noisy cells.

        An exact board match is preferred.  When CNN recognition is noisy away
        from the move, accept only if both move endpoints changed visually and
        the recognised source/destination have the expected final contents.
        """
        if not parsed_board or not expected_board:
            return False
        if self.board_to_fen(parsed_board) == self.board_to_fen(expected_board):
            return True
        src, dst = self.uci_to_screen_cells(move)
        return (source_changed and destination_changed and
                parsed_board[src[0]][src[1]] is None and
                parsed_board[dst[0]][dst[1]] == expected_board[dst[0]][dst[1]])

    def _move_visually_confirmed(self, before_img, previous_img, current_img,
                                 move):
        """Last-resort confirmation when CNN also misclassifies an endpoint."""
        if before_img is None or previous_img is None or current_img is None:
            return False
        if self._is_my_turn_image(current_img):
            return False
        if not self._is_opponent_turn_image(current_img):
            return False
        src, dst = self.uci_to_screen_cells(move)
        # Both endpoints must have changed substantially from before the click,
        # then remain stable across two consecutive observations.
        changed = (
            self._piece_cell_change(before_img, current_img, *src) > 8.0 and
            self._piece_cell_change(before_img, current_img, *dst) > 8.0)
        stable = (
            self._piece_cell_change(previous_img, current_img, *src) < 3.0 and
            self._piece_cell_change(previous_img, current_img, *dst) < 3.0)
        return changed and stable

    def _deselect_board(self):
        """Click between ranks in the river so no piece remains selected."""
        river_x = self.cols_logical[4]
        river_y = (self.rows_logical[4] + self.rows_logical[5]) / 2
        self.click(river_x, river_y)
        time.sleep(0.15)

    def _cell_change(self, img_before, img_after, r, c):
        """Compute pixel change at a specific cell (centered, no overlap)."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        # 35% of cell size as half-width — fits inside cell, no neighbor overlap
        # (old 0.7 caused 218px patch vs 156px cell spacing = massive overlap!)
        hs = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.35)
        h, w = img_before.shape[:2]
        x1, y1 = max(0, px-hs), max(0, py-hs)
        x2, y2 = min(w, px+hs), min(h, py+hs)
        p1 = img_before[y1:y2, x1:x2]
        p2 = img_after[y1:y2, x1:x2]
        if p1.shape != p2.shape or p1.size == 0:
            return 0
        return cv2.absdiff(p1, p2).mean()

    def _piece_cell_change(self, img_before, img_after, r, c):
        """Measure center-piece change while excluding move-highlight colors."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        hs = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.30)
        h, w = img_before.shape[:2]
        x1, y1 = max(0, px-hs), max(0, py-hs)
        x2, y2 = min(w, px+hs), min(h, py+hs)
        p1 = img_before[y1:y2, x1:x2]
        p2 = img_after[y1:y2, x1:x2]
        if p1.shape != p2.shape or p1.size == 0:
            return 0.0

        hsv1 = cv2.cvtColor(p1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(p2, cv2.COLOR_BGR2HSV)
        highlight = np.zeros(p1.shape[:2], dtype=np.uint8)
        for hsv in (hsv1, hsv2):
            green = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
            yellow = cv2.inRange(hsv, (20, 90, 100), (35, 255, 255))
            highlight = cv2.bitwise_or(highlight, green)
            highlight = cv2.bitwise_or(highlight, yellow)
        valid = highlight == 0
        if np.count_nonzero(valid) < valid.size * 0.40:
            return 0.0
        gray1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
        return float(cv2.absdiff(gray1, gray2)[valid].mean())

    def detect_move_perft(self, img_before, img_after, board_before, fen_before):
        """Detect opponent's move: score each legal move by pixel change at src+dst."""
        board_after = [row[:] for row in board_before]

        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_before} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        if not legal_moves:
            print(f"    No legal moves")
            return board_after

        # Score every legal move by how much its src+dst cells changed
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sc = self._piece_cell_change(img_before, img_after, src[0], src[1])
            dc = self._piece_cell_change(img_before, img_after, dst[0], dst[1])
            scored.append((move, src, dst, sc + dc))

        scored.sort(key=lambda x: -x[3])
        best_move, best_src, best_dst, best_score = scored[0]

        # Debug: show top 3
        top3 = [(m, f"{s:.1f}") for m, _, _, s in scored[:3]]
        print(f"    Top: {top3}")

        # Correct detections score 50-100+. Below 20 is always wrong.
        if best_score < 20:
            print(f"    ⚠ Score too low (Δ={best_score:.1f}), need > 20")
            return None
        if best_score < 40 and len(scored) > 1 and best_score < scored[1][3] * 2:
            print(f"    ⚠ Low confidence (Δ={best_score:.1f}), gap too small")
            return None

        piece = board_before[best_src[0]][best_src[1]]
        if piece:
            board_after[best_src[0]][best_src[1]] = None
            board_after[best_dst[0]][best_dst[1]] = piece
            print(f"    Move: {best_move} ({piece} {best_src}→{best_dst}) [Δ={best_score:.1f}]")
        else:
            print(f"    Move: {best_move} but no piece at {best_src}")
            return None

        return board_after

    # --- Highlight-based detection (most reliable) ---

    def _cell_highlight_score(self, img, r, c):
        """Detect green/yellow highlight at a cell. Returns highlight pixel ratio."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        hs = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.45)
        h, w = img.shape[:2]
        x1, y1 = max(0, px - hs), max(0, py - hs)
        x2, y2 = min(w, px + hs), min(h, py + hs)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        total = patch.shape[0] * patch.shape[1]
        # Green highlight: H 35-85
        green = cv2.inRange(hsv, (35, 40, 80), (85, 255, 255))
        # Yellow-green: H 20-35
        yellow = cv2.inRange(hsv, (20, 40, 80), (35, 255, 255))
        return (cv2.countNonZero(green) + cv2.countNonZero(yellow)) / total

    def detect_move_highlight(self, img, board_after_our_move, fen_after_our_move,
                              our_move=None):
        """Detect opponent's move by finding highlighted cells (green/yellow markers).

        天天象棋 highlights the source and destination of the last move.
        We find those cells and match against legal opponent moves.
        """
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_after_our_move} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)
        if not legal_moves:
            return None

        # Compute highlight score for every cell
        hl = {}
        for r in range(10):
            for c in range(9):
                hl[(r, c)] = self._cell_highlight_score(img, r, c)

        # Filter out our own move's highlight cells
        our_cells = set()
        if our_move:
            s, d = self.uci_to_screen_cells(our_move)
            our_cells = {(s[0], s[1]), (d[0], d[1])}

        # Score each legal move by highlight at src+dst
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sk, dk = (src[0], src[1]), (dst[0], dst[1])
            # Skip if this move's cells overlap with our move's highlight
            if sk in our_cells or dk in our_cells:
                scored.append((move, src, dst, 0.0))
                continue
            scored.append((move, src, dst, hl[sk] + hl[dk]))

        scored.sort(key=lambda x: -x[3])
        best = scored[0]

        # Debug: show top highlights and top moves
        top_cells = sorted(hl.items(), key=lambda x: -x[1])[:5]
        top_moves = [(m, f"{s:.3f}") for m, _, _, s in scored[:3]]
        print(f"    Hl cells: {[(f'{r},{c}', f'{s:.3f}') for (r,c),s in top_cells]}")
        print(f"    Hl moves: {top_moves}")

        if best[3] > 0.02:  # At least 2% highlight pixels across both cells
            move, src, dst, score = best
            piece = board_after_our_move[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_after_our_move]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    HlMove: {move} ({piece}) [score={score:.3f}]")
                return board_result

        return None

    # --- Single-image occupancy detection (no before/after needed) ---

    def _cell_feature(self, img, r, c):
        """Brightness std at cell center — high for pieces, low for empty cells."""
        px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
        radius = int(min(self.cell_w, self.cell_h) * self.retina_scale * 0.25)
        h, w = img.shape[:2]
        x1, y1 = max(0, px - radius), max(0, py - radius)
        x2, y2 = min(w, px + radius), min(h, py + radius)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def detect_move_occupancy(self, img, board_after_our_move, fen_after_our_move):
        """Detect opponent's move from a SINGLE image using occupancy analysis.

        Uses brightness std to distinguish pieces (high variance: bright center +
        dark text) from empty cells (low variance: uniform board color).
        Calibrates threshold on-the-fly from known cells.
        """
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_after_our_move} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        if not legal_moves:
            return None

        # Compute brightness std for every cell
        features = {}
        for r in range(10):
            for c in range(9):
                features[(r, c)] = self._cell_feature(img, r, c)

        # Collect cells that are src/dst of any legal move
        src_cells = set()
        dst_cells = set()
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            src_cells.add((src[0], src[1]))
            dst_cells.add((dst[0], dst[1]))

        # Safe references: cells not involved in any legal move
        occ_ref = [features[rc] for rc in
                   [(r, c) for r in range(10) for c in range(9)
                    if board_after_our_move[r][c] is not None and (r, c) not in src_cells]]
        emp_ref = [features[rc] for rc in
                   [(r, c) for r in range(10) for c in range(9)
                    if board_after_our_move[r][c] is None and (r, c) not in dst_cells]]

        if len(occ_ref) < 3 or len(emp_ref) < 3:
            print("    Occ: insufficient ref cells")
            return None

        occ_med = float(np.median(occ_ref))
        emp_med = float(np.median(emp_ref))
        threshold = (occ_med + emp_med) / 2

        if occ_med - emp_med < 5:
            print(f"    Occ: weak separation ({occ_med:.1f} vs {emp_med:.1f})")
            return None

        print(f"    Occ: occ={occ_med:.1f} emp={emp_med:.1f} thr={threshold:.1f}")

        # Score each legal move
        scored = []
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sf = features[(src[0], src[1])]
            df = features[(dst[0], dst[1])]
            # After move: src empty (low std), dst has piece (high std)
            src_empty = max(0, threshold - sf)
            dst_piece = max(0, df - threshold)
            score = src_empty * 2 + dst_piece  # src leaving is clearest signal
            scored.append((move, src, dst, score, sf, df))

        scored.sort(key=lambda x: -x[3])
        top3 = [(m, f"{s:.1f}") for m, _, _, s, _, _ in scored[:3]]
        print(f"    Occ top: {top3}")

        best = scored[0]
        margin_ok = len(scored) < 2 or best[3] > scored[1][3] * 1.3
        if best[3] > 3 and margin_ok:
            move, src, dst, _, sf, df = best
            piece = board_after_our_move[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_after_our_move]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    OccMove: {move} ({piece}) sf={sf:.1f} df={df:.1f}")
                return board_result

        print(f"    Occ: no confident move (best={best[3]:.1f})")
        return None

    def detect_move(self, img_before, img_after, board_before):
        """Detect opponent's move by finding which cells changed."""
        board_after = [row[:] for row in board_before]
        changed = []

        for r in range(10):
            for c in range(9):
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                ps = self.patch_size
                h, w = img_before.shape[:2]
                x1, y1 = max(0, px-ps), max(0, py-ps)
                x2, y2 = min(w, px+ps), min(h, py+ps)
                p1 = img_before[y1:y2, x1:x2]
                p2 = img_after[y1:y2, x1:x2]
                if p1.shape != p2.shape:
                    continue
                diff = cv2.absdiff(p1, p2)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                change = np.count_nonzero(gray_diff > 30) / max(1, gray_diff.size)
                if change > 0.03:
                    had_piece = board_before[r][c] is not None
                    has_piece_now = self._cell_has_piece_now(img_after, c, r)
                    changed.append((r, c, change, had_piece, has_piece_now))

        # Re-capture templates from KNOWN piece positions in the new image,
        # then re-parse the full board with fresh templates.
        fresh_templates = {}
        ps = self.patch_size
        for r in range(10):
            for c in range(9):
                piece = board_before[r][c]
                if piece is None:
                    continue
                # Only use pieces that DIDN'T change (still in original position)
                in_changed = any(cr == r and cc == c for cr, cc, *_ in changed)
                if in_changed:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img_after, px, py, ps)
                if patch is not None:
                    if piece not in fresh_templates:
                        fresh_templates[piece] = []
                    fresh_templates[piece].append(patch)

        # Also add empty templates from known empty positions
        fresh_templates['_'] = []
        for r in range(10):
            for c in range(9):
                if board_before[r][c] is not None:
                    continue
                in_changed = any(cr == r and cc == c for cr, cc, *_ in changed)
                if in_changed:
                    continue
                px, py = self.logical_to_pixel(self.cols_logical[c], self.rows_logical[r])
                patch = self._extract(img_after, px, py, ps)
                if patch is not None:
                    fresh_templates['_'].append(patch)
                    if len(fresh_templates['_']) >= 10:
                        break
            if len(fresh_templates.get('_', [])) >= 10:
                break

        # Save old templates, use fresh ones for re-parse
        old_templates = self.templates
        self.templates = fresh_templates

        # Re-parse all cells
        new_board = self.parse_board(img_after)

        # Restore original templates
        self.templates = old_templates

        # Find what changed
        src = dst = None
        for r in range(10):
            for c in range(9):
                old = board_before[r][c]
                new = new_board[r][c]
                if old != new:
                    if old is not None and new is None:
                        src = (r, c, old)
                    elif old is None and new is not None:
                        dst = (r, c, new)
                    elif old is not None and new is not None:
                        src = (r, c, old)
                        dst = (r, c, new)

        if src:
            board_after[src[0]][src[1]] = None
        if dst and src:
            board_after[dst[0]][dst[1]] = src[2]  # moving piece
            print(f"    Move: {src[2]} ({src[0]},{src[1]}) → ({dst[0]},{dst[1]})")
        elif src:
            print(f"    Piece left ({src[0]},{src[1]}) but no dest found")
        else:
            print(f"    No move detected via fresh re-parse")
            # Fallback: just return the fresh parse
            return new_board

        return board_after

    def board_to_fen(self, board):
        if self.playing_red:
            fb = board
        else:
            fb = [row[::-1] for row in reversed(board)]
        parts = []
        for row in fb:
            s, e = "", 0
            for p in row:
                if p is None:
                    e += 1
                else:
                    if e: s += str(e); e = 0
                    s += p
            if e: s += str(e)
            parts.append(s)
        return "/".join(parts)

    def board_is_plausible(self, board):
        """Safety gate before sending a visually parsed position to the engine."""
        if not board or len(board) != 10 or any(len(row) != 9 for row in board):
            return False, "board dimensions are invalid"
        flat = [cell for row in board for cell in row if cell is not None]
        if len(flat) < 2 or len(flat) > 32:
            return False, f"piece count is {len(flat)}"
        if flat.count('K') != 1 or flat.count('k') != 1:
            return False, f"king count is K={flat.count('K')}, k={flat.count('k')}"
        limits = {'K': 1, 'A': 2, 'B': 2, 'N': 2, 'R': 2, 'C': 2, 'P': 5,
                  'k': 1, 'a': 2, 'b': 2, 'n': 2, 'r': 2, 'c': 2, 'p': 5}
        for piece, limit in limits.items():
            if flat.count(piece) > limit:
                return False, f"too many {piece} pieces ({flat.count(piece)} > {limit})"
        canonical = board if self.playing_red else [row[::-1] for row in reversed(board)]
        red_king = next((r, c) for r in range(10) for c in range(9)
                        if canonical[r][c] == 'K')
        black_king = next((r, c) for r in range(10) for c in range(9)
                          if canonical[r][c] == 'k')
        if not (7 <= red_king[0] <= 9 and 3 <= red_king[1] <= 5):
            return False, f"red king is outside its palace at {red_king}"
        if not (0 <= black_king[0] <= 2 and 3 <= black_king[1] <= 5):
            return False, f"black king is outside its palace at {black_king}"
        return True, ""

    # --- Move Execution ---

    def uci_to_logical(self, move):
        fc, fr = ord(move[0]) - ord('a'), int(move[1])
        tc, tr = ord(move[2]) - ord('a'), int(move[3])
        if self.playing_red:
            s = [(fc, 9-fr), (tc, 9-tr)]
        else:
            s = [(8-fc, fr), (8-tc, tr)]
        return [(self.cols_logical[c], self.rows_logical[r]) for c, r in s]

    def load_cnn(self):
        """Load CNN piece classifier if available (prefers ONNX over PyTorch)."""
        if self.cnn:
            return True
        # Try ONNX first (faster startup, no PyTorch needed)
        try:
            from xiangqi_cnn_onnx import PieceClassifierCNN as OnnxClassifier
            onnx_path = os.path.join(_SCRIPT_DIR, 'xiangqi_cnn.onnx')
            if os.path.exists(onnx_path):
                self.cnn = OnnxClassifier(onnx_path)
                print("  CNN model loaded! (ONNX)")
                return True
        except Exception as e:
            print(f"  ONNX not available: {e}")
        # Fall back to PyTorch
        try:
            from xiangqi_cnn import PieceClassifierCNN, MODEL_PATH
            if os.path.exists(MODEL_PATH):
                self.cnn = PieceClassifierCNN(MODEL_PATH)
                print("  CNN model loaded! (PyTorch)")
                return True
        except Exception as e:
            print(f"  CNN not available: {e}")
        return False

    def parse_board_cnn(self, img):
        """Parse entire board using CNN with double-shot for low confidence cells.

        Takes a second screenshot after a short delay and averages probability
        distributions to reduce animation artifacts.
        """
        if not self.cnn:
            return None
        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug')
        step = getattr(self, '_debug_step', 0)
        self._debug_step = step + 1
        debug_prefix = f"s{step:03d}_"

        # First pass
        board = self.cnn.parse_board(
            img, self.cols_logical, self.rows_logical,
            self.retina_scale, self.win_x, self.win_y,
            self.cell_w, self.cell_h, debug_dir=debug_dir,
            debug_prefix=debug_prefix)

        # Check if any non-empty cell has low confidence
        LOW_CONF = 0.90
        needs_retry = False
        for r in range(10):
            for c in range(9):
                if board[r][c] is not None and self.cnn._cell_probs[r][c] is not None:
                    conf = float(self.cnn._cell_probs[r][c].max())
                    if conf < LOW_CONF:
                        needs_retry = True
                        break
            if needs_retry:
                break

        if needs_retry:
            import numpy as np
            probs1 = [[p.copy() if p is not None else None for p in row]
                      for row in self.cnn._cell_probs]
            time.sleep(0.3)
            img2 = self.screenshot_for_processing()
            board = self.cnn.parse_board(
                img2, self.cols_logical, self.rows_logical,
                self.retina_scale, self.win_x, self.win_y,
                self.cell_w, self.cell_h, debug_dir=debug_dir,
                debug_prefix=f"s{step:03d}b_")

            # Average probabilities for low-confidence cells
            try:
                from xiangqi_cnn_onnx import CLASSES
            except ImportError:
                from xiangqi_cnn import CLASSES
            changes = 0
            for r in range(10):
                for c in range(9):
                    p1 = probs1[r][c]
                    p2 = self.cnn._cell_probs[r][c]
                    if p1 is None or p2 is None:
                        continue
                    if float(p1.max()) < LOW_CONF or float(p2.max()) < LOW_CONF:
                        avg = (p1 + p2) / 2
                        pred = int(avg.argmax())
                        cls = CLASSES[pred]
                        new_piece = None if cls == '_' else cls
                        if new_piece != board[r][c]:
                            changes += 1
                        board[r][c] = new_piece
                        self.cnn._cell_probs[r][c] = avg
            if changes:
                print(f"    Double-shot: corrected {changes} cell(s)")
                board = self.cnn._validate_board(board)

        return board

    def detect_move_cnn(self, img, board_before, fen_before):
        """Detect opponent's move by CNN board parsing + diff with tracked state.

        Parses the entire board from a single screenshot, compares with
        the tracked board state, and matches differences against legal moves.
        """
        if not self.cnn:
            return None

        parsed = self.parse_board_cnn(img)
        if not parsed:
            return None

        # Find cells that differ between tracked and CNN-parsed board
        diffs = []
        for r in range(10):
            for c in range(9):
                old = board_before[r][c]
                new = parsed[r][c]
                if old != new:
                    diffs.append((r, c, old, new))

        if not diffs:
            return None  # No change detected

        # Match against legal opponent moves
        opp_turn = 'b' if self.playing_red else 'w'
        opp_fen = f"{fen_before} {opp_turn} - - 0 1"
        legal_moves = self.get_legal_moves(opp_fen)

        best_move = None
        best_match = 0
        for move in legal_moves:
            src, dst = self.uci_to_screen_cells(move)
            sr, sc = src[0], src[1]
            dr, dc = dst[0], dst[1]
            match = 0
            # Check if src cell changed from piece to empty/different
            for r, c, old, new in diffs:
                if r == sr and c == sc and old is not None:
                    match += 1
                if r == dr and c == dc:
                    match += 1
            if match > best_match:
                best_match = match
                best_move = move

        if best_move and best_match >= 1:
            src, dst = self.uci_to_screen_cells(best_move)
            piece = board_before[src[0]][src[1]]
            if piece:
                board_result = [row[:] for row in board_before]
                board_result[src[0]][src[1]] = None
                board_result[dst[0]][dst[1]] = piece
                print(f"    CNN: {best_move} ({piece}) [diffs={len(diffs)}, match={best_match}]")
                return board_result

        print(f"    CNN: {len(diffs)} diffs but no legal move match")
        return None

    def collect_cnn_data(self, img, board):
        """Save cell patches for CNN training (auto-labeled from tracked board)."""
        try:
            from xiangqi_cnn import collect_from_screenshot
            session = getattr(self, '_cnn_session', 0)
            n = collect_from_screenshot(
                img, self.cols_logical, self.rows_logical, board,
                self.retina_scale, self.win_x, self.win_y,
                self.cell_w, self.cell_h, session)
            self._cnn_session += 1
        except Exception:
            pass  # Don't let data collection break the game

    def activate_window(self):
        """Bring WeChat to front and focus the specific mini-program window."""
        self.platform.activate()

    def click(self, lx, ly):
        """Click through the active platform adapter."""
        self.platform.click(lx, ly)

    def _cgevent_click(self, x, y):
        """Compatibility alias retained for callers from older code."""
        self.platform.click(x, y)

    # --- Pikafish ---

    def _stop_pikafish(self, force=False):
        proc = self._engine_proc
        self._engine_proc = None
        self._engine_lines = None
        self._engine_reader = None
        if proc is None:
            return
        try:
            if force:
                proc.kill()
            else:
                proc.stdin.write("quit\n")
                proc.stdin.flush()
                proc.wait(timeout=0.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _start_pikafish(self):
        self._stop_pikafish(force=True)
        proc = subprocess.Popen(
            [PIKAFISH], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, cwd=PIKAFISH_DIR)
        lines = queue.Queue()

        def read_output():
            try:
                for raw in iter(proc.stdout.readline, ''):
                    lines.put(raw.rstrip())
            finally:
                lines.put(None)

        self._engine_proc = proc
        self._engine_lines = lines
        self._engine_reader = threading.Thread(target=read_output, daemon=True)
        self._engine_reader.start()
        try:
            proc.stdin.write("uci\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._stop_pikafish(force=True)
            return False

        deadline = time.monotonic() + ENGINE_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            if line == "uciok":
                break
        else:
            line = None
        if line != "uciok":
            self._stop_pikafish(force=True)
            return False

        try:
            proc.stdin.write(
                f"setoption name Threads value {ENGINE_THREADS}\n"
                f"setoption name Hash value {ENGINE_HASH_MB}\n"
                f"setoption name MultiPV value {ENGINE_MULTIPV}\n"
                "isready\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._stop_pikafish(force=True)
            return False

        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=0.05)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            if line == "readyok":
                return True
        self._stop_pikafish(force=True)
        return False

    def _ensure_pikafish(self):
        return (self._engine_proc is not None and
                self._engine_proc.poll() is None) or self._start_pikafish()

    def pikafish(self, fen, move_history=None, excluded=None,
                 movetime_ms=None):
        with self._engine_lock:
            return self._pikafish_locked(
                fen, move_history=move_history, excluded=excluded,
                movetime_ms=movetime_ms)

    def _pikafish_locked(self, fen, move_history=None, excluded=None,
                         movetime_ms=None):
        if not self._ensure_pikafish():
            return None, ""
        proc = self._engine_proc
        if move_history:
            pos_cmd = f"position fen {fen} moves {' '.join(move_history)}\n"
        else:
            pos_cmd = f"position fen {fen}\n"
        # Get all legal moves, exclude repetition moves
        if SEARCH_DEPTH > 0:
            go_cmd = f"go depth {SEARCH_DEPTH}"
        else:
            search_ms = MOVE_TIME_MS if movetime_ms is None else max(1, int(movetime_ms))
            go_cmd = f"go movetime {search_ms}"
        if excluded:
            legal = self.get_legal_moves(fen)
            allowed = [m for m in legal if m not in excluded]
            if allowed:
                go_cmd += f" searchmoves {' '.join(allowed)}"
        try:
            proc.stdin.write(f"{pos_cmd}{go_cmd}\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._stop_pikafish(force=True)
            return None, ""
        best, info = None, ""
        engine_timeout_s = ENGINE_TIMEOUT_S
        if SEARCH_DEPTH <= 0 and movetime_ms is not None:
            engine_timeout_s = max(
                ENGINE_TIMEOUT_S, search_ms / 1000.0 + 2.0)
        deadline = time.monotonic() + engine_timeout_s
        timed_out = False
        while time.monotonic() < deadline:
            try:
                line = self._engine_lines.get(
                    timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line is None:
                break
            if line.startswith('bestmove'):
                best = line.split()[1] if len(line.split()) > 1 else None
                break
            if 'score' in line:
                info = line
        else:
            timed_out = True
        if timed_out:
            print(f"  Pikafish timed out after {engine_timeout_s:.1f}s; restarting engine")
            self._stop_pikafish(force=True)
        return best, info

    def wait_for_our_turn(self, timeout_s=TURN_WAIT_TIMEOUT_S):
        """Poll at a controlled rate; return False when no turn signal arrives."""
        deadline = time.monotonic() + timeout_s
        dots = 0
        while not self.stop_flag and time.monotonic() < deadline:
            self.check_delay_hotkey()
            if self.is_my_turn():
                return True
            time.sleep(TURN_POLL_INTERVAL_S)
            dots += 1
            if dots % 20 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
        return False

    def check_delay_hotkey(self):
        """Toggle optional extra search time with the global F8 key."""
        if os.name != "nt":
            return False
        try:
            pressed_since_last_check = ctypes.windll.user32.GetAsyncKeyState(0x77) & 1
        except (AttributeError, OSError):
            return False
        if not pressed_since_last_check:
            return False
        self.move_delay_enabled = not self.move_delay_enabled
        state = "ON (add random 2-10s to search)" if self.move_delay_enabled else "OFF"
        print(f"\n  [F8] Extra search time: {state}", flush=True)
        return True

    def _search_time_for_turn(self, board):
        """Invest the optional human-like delay in engine search, not idling."""
        self.check_delay_hotkey()
        search_ms = self._search_time_for_board(board)
        if not self.move_delay_enabled:
            return search_ms
        extra_s = random.uniform(MOVE_DELAY_MIN_S, MOVE_DELAY_MAX_S)
        print(f"  Extra search time: {extra_s:.1f}s", flush=True)
        return search_ms + int(round(extra_s * 1000))

    def score_str(self, info):
        if 'score cp' in info:
            p = info.split()
            try: return f"{int(p[p.index('cp')+1])/100:+.1f}"
            except: pass
        if 'score mate' in info:
            p = info.split()
            try: return f"M{p[p.index('mate')+1]}"
            except: pass
        return "?"

    # --- Main ---

    def crop_board_region(self, img):
        """Crop the board grid area from the full capture for fast pixel comparison."""
        px0, py0 = self.logical_to_pixel(self.cols_logical[0], self.rows_logical[0])
        px8, py9 = self.logical_to_pixel(self.cols_logical[8], self.rows_logical[9])
        margin = 10
        return img[max(0,py0-margin):py9+margin, max(0,px0-margin):px8+margin].copy()

    def images_changed(self, img1, img2):
        """Compare two cropped board images. Returns True if significantly different."""
        if img1 is None or img2 is None:
            return True
        if img1.shape != img2.shape:
            return True
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        # Count pixels that changed by more than 30 intensity levels
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        change_ratio = changed_pixels / total_pixels
        # If more than 0.5% of pixels changed significantly, board changed
        return change_ratio > 0.005

    def _wait_for_board_stable(self, max_wait_s=3.0, interval_s=0.25,
                               required_stable=2):
        """Wait for consecutive stable board frames, bounded and fail-closed."""
        previous = self.screenshot_for_processing()
        self._last_stability_image = previous
        stable_count = 0
        deadline = time.monotonic() + max_wait_s
        while not self.stop_flag and time.monotonic() < deadline:
            time.sleep(interval_s)
            current = self.screenshot_for_processing()
            self._last_stability_image = current
            previous_crop = self.crop_board_region(previous)
            current_crop = self.crop_board_region(current)
            if self.images_changed(previous_crop, current_crop):
                stable_count = 0
            else:
                stable_count += 1
                if stable_count >= required_stable:
                    return current
            previous = current
        return None

    def crop_avatar_region(self, img):
        """Crop our avatar frame (bottom-right), excluding the clock below it."""
        h, w = img.shape[:2]
        x1 = int(w * 0.775)
        y1 = int(h * 0.705)
        x2 = int(w * 0.838)
        y2 = int(h * 0.810)
        return img[y1:y2, x1:x2].copy()

    def crop_opponent_avatar_region(self, img):
        """Crop opponent avatar frame (upper-left), excluding the clock below it."""
        h, w = img.shape[:2]
        x1 = int(w * 0.163)
        y1 = int(h * 0.165)
        x2 = int(w * 0.228)
        y2 = int(h * 0.272)
        return img[y1:y2, x1:x2].copy()

    def _check_green_border(self, avatar):
        """Check green pixel ratio in avatar border region."""
        h, w = avatar.shape[:2]
        border_mask = np.ones((h, w), dtype=bool)
        mx, my = int(w * 0.15), int(h * 0.15)
        border_mask[my:h-my, mx:w-mx] = False
        hsv = cv2.cvtColor(avatar, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_in_border = np.count_nonzero(green_mask[border_mask])
        border_pixels = border_mask.sum()
        return (green_in_border / border_pixels)

    def is_my_turn(self):
        """Check if it's our turn by detecting green in our avatar border."""
        img = self.screenshot_for_processing()
        return self._is_my_turn_image(img)

    def _is_my_turn_image(self, img):
        return self._check_green_border(self.crop_avatar_region(img)) > 0.005

    def _is_opponent_turn_image(self, img):
        return self._check_green_border(
            self.crop_opponent_avatar_region(img)) > 0.005

    def run(self):
        try:
            self._run()
        finally:
            self._stop_pikafish()

    def _run(self):
        print("=== Xiangqi Bot (Pikafish) ===\n")
        if not os.path.exists(PIKAFISH):
            print(f"ERROR: {PIKAFISH} not found"); sys.exit(1)

        print("[1] Finding window...")
        self.find_window()

        print("[2] Taking screenshot...")
        img = self.screenshot_for_processing()
        print(f"  Image: {img.shape[1]}x{img.shape[0]}, retina={self.retina_scale:.2f}x")

        print("[3] Calibration...")
        force_manual = os.environ.get('XIANGQI_MANUAL_CALIBRATION') == '1'
        force_recalibrate = os.environ.get('XIANGQI_RECALIBRATE') == '1'
        loaded_calibration = (False if force_manual or force_recalibrate
                              else self.load_calibration())
        if loaded_calibration and not self.validate_calibration(img):
            loaded_calibration = False
        if not loaded_calibration:
            if force_manual:
                print("  Manual calibration requested")
                if not self.calibrate():
                    return
                img = self.screenshot_for_processing()
                if not self.validate_calibration(img):
                    print("  ERROR: manual calibration did not match a visible board; stopping.")
                    return
            elif not self.auto_calibrate(img):
                print("  Auto-calibrate failed, falling back to manual...")
                if not self.calibrate():
                    return
                img = self.screenshot_for_processing()  # retake after manual calibration
                if not self.validate_calibration(img):
                    print("  ERROR: manual calibration did not match a visible board; stopping.")
                    return

        print("[4] Waiting for board to stabilize...")
        # Keep re-parsing until two consecutive reads give the same FEN
        if self.load_cnn():
            import io
            # Suppress output during stabilization
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                board = self.parse_board_cnn(img)
            finally:
                sys.stdout = old_stdout
            prev_fen = self.board_to_fen(board)
            for stabilize_try in range(20):
                time.sleep(1.0)
                img = self.screenshot_for_processing()
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    board = self.parse_board_cnn(img)
                finally:
                    sys.stdout = old_stdout
                curr_fen = self.board_to_fen(board)
                if curr_fen == prev_fen:
                    print(f"  Board stable after {stabilize_try + 1} check(s)")
                    break
                prev_fen = curr_fen
            else:
                print("  Warning: board did not stabilize, proceeding anyway")

        print("[5] Detecting orientation...")
        # Clean up old debug data at game start
        import shutil
        dbg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug')
        if os.path.exists(dbg_root):
            shutil.rmtree(dbg_root, ignore_errors=True)
        self._debug_step = 0
        print(f"  Debug patches dir: {dbg_root}")

        if self.cnn:
            board = self.parse_board_cnn(img)
            orientation, evidence = self.determine_orientation(board)
            if orientation is None:
                print(f"  ERROR: cannot safely determine our side: {evidence}")
                return
            self.playing_red = orientation
            print(f"  You play: {'RED' if self.playing_red else 'BLACK'} "
                  f"({evidence})")
        else:
            self.detect_orientation(img)
            self.capture_templates(img)
            board = self.parse_board(img)
            print("  Board parsed by template matching (no CNN)")

        fen = self.board_to_fen(board)
        plausible, reason = self.board_is_plausible(board)
        if not plausible:
            print(f"\nERROR: board recognition is invalid ({reason}).")
            print("Run 重新校准并启动.bat again with the complete board visible.")
            return
        print(f"\n  FEN: {fen}")

        # Print board
        for r in range(10):
            line = " "
            for c in range(9):
                p = board[r][c]
                line += f" {p}" if p else " ."
            print(line)

        # === CNN-driven game loop ===
        # Simple: poll board with CNN, detect changes, react
        turn = "w" if self.playing_red else "b"
        n = 0
        last_fen = fen
        last_board = [row[:] for row in board]  # deep copy for diff
        state_confidence = 'A'
        self._cnn_session = int(time.time())

        print(f"\n--- Game loop (playing {'RED' if self.playing_red else 'BLACK'}) ---\n")
        print("  Press F8 to add a random 2-10s to engine search.\n")

        # Wait until it's our turn before starting (handles mid-game start too)
        if not self.is_my_turn():
            print("  Waiting for our turn...", end="", flush=True)
            if not self.wait_for_our_turn():
                print(" timed out; no turn indicator detected")
                return
            self._deselect_board()
            stable_img = self._wait_for_board_stable()
            if stable_img is None:
                self._save_failure_diagnostics(
                    "initial_board_never_stabilized",
                    {"latest_observation": self._last_stability_image},
                    {"tracked_fen": fen})
                print(" ERROR: board animation did not stabilize; stopping.")
                return
            print(" done")
            opponent_side = 'b' if self.playing_red else 'w'
            legal_moves = self.get_legal_moves(
                f"{fen} {opponent_side} - - 0 1")
            observations = []
            for observation_index in range(3):
                img = (stable_img if observation_index == 0
                       else self.screenshot_for_processing())
                parsed = self.parse_board_cnn(img) if self.cnn else self.parse_board(img)
                observed_move, matched = self._match_unique_legal_transition(
                    last_board, parsed, legal_moves)
                observations.append((observed_move, matched, 'A'))
                time.sleep(0.25)
            opponent_move, matched, state_confidence = \
                self._select_transition_consensus_with_confidence(
                observations)
            if not opponent_move:
                self._save_failure_diagnostics(
                    "initial_opponent_consensus_failed",
                    {"latest_observation": img},
                    {"tracked_fen": fen,
                     "observations": [
                         (move, self.board_to_fen(state) if state else None)
                         for move, state, _grade in observations]})
                print("  ERROR: opponent move lacked multi-frame legal consensus; stopping.")
                return
            board = matched
            fen = self.board_to_fen(board)
            last_board = [row[:] for row in board]
            print(f"  Verified opponent move: {opponent_move} "
                  f"[confidence {state_confidence}]")
            print(f"  Board → {fen}")
        else:
            print("  It's our turn, starting immediately")

        while not self.stop_flag:
            try:
                # Step 1: Ask pikafish for best move
                state_valid, state_reason = self._state_is_safe_for_engine(
                    board, state_confidence)
                if not state_valid:
                    self._save_failure_diagnostics(
                        "tracked_position_unsafe", metadata={
                            "reason": state_reason,
                            "tracked_fen": fen,
                        })
                    print(f"  ERROR: tracked position is unsafe ({state_reason}); "
                          "not sending it to Pikafish.")
                    return
                full_fen = f"{fen} {turn} - - 0 1"

                print(f"  FEN → Pikafish: {full_fen}")
                search_time_ms = self._search_time_for_turn(board)
                if search_time_ms != MOVE_TIME_MS:
                    print(f"  Engine search budget: {search_time_ms / 1000:.1f}s")
                best, info = self.pikafish(
                    full_fen, movetime_ms=search_time_ms)

                if not best or best == '(none)':
                    print("  No legal engine move; stopping without clicking.")
                    return

                n += 1
                sc = self.score_str(info)
                print(f"[{n}] {best} ({sc})")

                expected_board = self._apply_board_move(board, best)
                if expected_board is None:
                    print(f"  ERROR: engine move {best} starts from an empty tracked cell.")
                    return
                if self.observer_mode:
                    print(f"  Observer mode: suggested move {best}; no click was made.")
                    return

                if self.stop_flag:
                    return

                # Step 2: execute one move transactionally. Selection highlights
                # never count as success; the exact expected board and turn switch do.
                click_ok = False
                for click_try in range(2):
                    self.activate_window()
                    self._deselect_board()
                    before_move_img = self.screenshot_for_processing()
                    pts = self.uci_to_logical(best)
                    self.click(pts[0][0], pts[0][1])
                    time.sleep(0.25)
                    self.click(pts[1][0], pts[1][1])
                    # Animations and the network response can outlive the old
                    # fixed 0.45s delay. Poll briefly for a settled, confirmed
                    # result instead of treating the first frame as failure.
                    src_cell, dst_cell = self.uci_to_screen_cells(best)
                    confirm_deadline = time.monotonic() + 2.0
                    visual_candidate_img = None
                    while time.monotonic() < confirm_deadline:
                        time.sleep(0.25)
                        after_img = self.screenshot_for_processing()
                        parsed_after = (self.parse_board_cnn(after_img)
                                        if self.cnn else self.parse_board(after_img))
                        source_changed = self._piece_cell_change(
                            before_move_img, after_img, *src_cell) > 3.0
                        destination_changed = self._piece_cell_change(
                            before_move_img, after_img, *dst_cell) > 3.0
                        board_matches = self._move_board_matches(
                            parsed_after, expected_board, best,
                            source_changed, destination_changed)
                        turn_switched = not self._is_my_turn_image(after_img)
                        if board_matches and turn_switched:
                            click_ok = True
                            opponent_before_img = after_img.copy()
                            break
                        endpoint_changed = (
                            source_changed and destination_changed)
                        dual_turn_switched = (
                            turn_switched and
                            self._is_opponent_turn_image(after_img))
                        if endpoint_changed and dual_turn_switched:
                            if self._move_visually_confirmed(
                                    before_move_img, visual_candidate_img,
                                    after_img, best):
                                print("  Move confirmed by stable endpoints and turn switch.")
                                click_ok = True
                                opponent_before_img = after_img.copy()
                                break
                            visual_candidate_img = after_img.copy()
                        else:
                            visual_candidate_img = None
                    if click_ok:
                        break
                    if click_try == 0:
                        print("  Move was not fully confirmed; deselecting and retrying once...")

                if not click_ok:
                    self._deselect_board()
                    self._save_failure_diagnostics(
                        "our_move_not_confirmed",
                        {"before_move": before_move_img,
                         "latest_observation": after_img},
                        {"move": best, "tracked_fen": fen,
                         "expected_fen": self.board_to_fen(expected_board)})
                    print(f"  ERROR: move {best} was not confirmed; stopping safely.")
                    return

                board = expected_board
                fen = self.board_to_fen(board)
                last_board = [row[:] for row in board]

                # Step 3: Wait for our turn (poll green border detection)
                print("  Waiting...", end="", flush=True)
                if not self.wait_for_our_turn():
                    print(" timed out; no turn indicator detected")
                    return
                self._deselect_board()
                stable_img = self._wait_for_board_stable()
                if stable_img is None:
                    self._save_failure_diagnostics(
                        "board_never_stabilized",
                        {"before_opponent_move": opponent_before_img,
                         "latest_observation": self._last_stability_image},
                        {"tracked_fen": fen})
                    print(" ERROR: board animation did not stabilize; stopping.")
                    return
                print(" done")

                # Step 4: prefer an exact legal transition. If unrelated CNN
                # cells are noisy, confirm the unique legal move from its two
                # endpoints and their actual pixel changes.
                opponent_side = 'b' if self.playing_red else 'w'
                legal_moves = self.get_legal_moves(
                    f"{fen} {opponent_side} - - 0 1")
                observations = []
                for observation_index in range(3):
                    img = (stable_img if observation_index == 0
                           else self.screenshot_for_processing())
                    parsed = (self.parse_board_cnn(img)
                              if self.cnn else self.parse_board(img))
                    observed_move, matched = self._match_unique_legal_transition(
                        last_board, parsed, legal_moves)
                    grade = 'A'
                    if not observed_move:
                        observed_move, matched = \
                            self._match_legal_transition_with_endpoint_changes(
                                last_board, parsed, legal_moves,
                                opponent_before_img, img)
                        grade = 'B'
                    observations.append((observed_move, matched, grade))
                    time.sleep(0.25)
                opponent_move, matched, state_confidence = \
                    self._select_transition_consensus_with_confidence(
                    observations)
                if not opponent_move:
                    self._save_failure_diagnostics(
                        "opponent_consensus_failed",
                        {"before_opponent_move": opponent_before_img,
                         "latest_observation": img},
                        {"tracked_fen": fen,
                         "observations": [
                             (move, self.board_to_fen(state) if state else None)
                             for move, state, _grade in observations]})
                    print("  ERROR: opponent move lacked multi-frame legal consensus; stopping.")
                    return
                board = matched
                fen = self.board_to_fen(board)
                print(f"  Verified opponent move: {opponent_move} "
                      f"[confidence {state_confidence}]")

                # Print board
                print(f"  Board → {fen}")
                for r in range(10):
                    line = "  "
                    for c in range(9):
                        p = board[r][c]
                        line += f" {p}" if p else " ."
                    print(line)

                # Show FEN diff from previous board
                if last_board:
                    col_names = "abcdefghi"
                    diffs = []
                    for r in range(10):
                        for c in range(9):
                            old_p = last_board[r][c]
                            new_p = board[r][c]
                            if old_p != new_p:
                                old_s = old_p if old_p else '.'
                                new_s = new_p if new_p else '.'
                                diffs.append(f"({r},{c}){col_names[c]}{9-r}: {old_s}→{new_s}")
                    if diffs:
                        print(f"  Δ {', '.join(diffs)}")
                last_board = [row[:] for row in board]

            except KeyboardInterrupt:
                print("\nStopped."); break
            except Exception as e:
                print(f"\nErr: {e}")
                import traceback; traceback.print_exc()
                time.sleep(2)


if __name__ == '__main__':
    Bot().run()
