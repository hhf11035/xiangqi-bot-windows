"""Offline smoke tests for the Windows port; no window activation or clicks."""
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from platform_adapter import WindowAdapter
import xiangqi_bot
import offline_replay
from xiangqi_bot import Bot, PIKAFISH, PIKAFISH_DIR


ROOT = os.path.dirname(os.path.abspath(__file__))


class WindowsPortTests(unittest.TestCase):
    def test_offline_replay_helpers_find_bundle_and_compare_boards(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            older = os.path.join(temp_dir, "older")
            newer = os.path.join(temp_dir, "newer")
            os.makedirs(older)
            os.makedirs(newer)
            for path in (older, newer):
                with open(os.path.join(path, "record.json"), "w",
                          encoding="utf-8") as handle:
                    handle.write("{}")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))
            self.assertEqual(offline_replay.latest_bundle(temp_dir), newer)

        tracked = [[None] * 9 for _ in range(10)]
        observed = [row[:] for row in tracked]
        observed[2][3] = 'p'
        self.assertEqual(offline_replay.board_differences(tracked, observed), [{
            "row": 2, "col": 3, "tracked": None, "observed": 'p'}])

    def test_offline_replay_launcher_uses_project_environment(self):
        launcher = os.path.join(ROOT, "离线回放最近故障.bat")
        with open(launcher, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn('.venv\\Scripts\\python.exe', content)
        self.assertIn('offline_replay.py', content)

    def test_offline_transition_analysis_reports_legal_candidates(self):
        bot = Bot()
        bot.playing_red = True
        tracked = [[None] * 9 for _ in range(10)]
        tracked[0][0] = 'r'
        observed = bot._apply_board_move(tracked, 'a9a8')
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with mock.patch.object(bot, 'get_legal_moves',
                               return_value=['a9a8', 'a9b9']), \
             mock.patch.object(bot, '_piece_cell_change', return_value=20.0):
            analysis = offline_replay.analyze_transition(
                bot, bot.board_to_fen(tracked), tracked, observed,
                image, image)
        self.assertEqual(analysis['legal_move_count'], 2)
        self.assertEqual(analysis['exact_match'], 'a9a8')
        self.assertEqual(analysis['relaxed_match'], 'a9a8')
        self.assertEqual(len(analysis['top_candidates']), 2)

    def test_failure_diagnostics_are_isolated_and_best_effort(self):
        bot = Bot()
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(xiangqi_bot, "DIAGNOSTIC_DIR", temp_dir), \
             mock.patch.object(xiangqi_bot, "DIAGNOSTICS_ENABLED", True):
            bundle = bot._save_failure_diagnostics(
                "test_failure", {"board": image}, {"fen": "test"})
            self.assertTrue(os.path.isfile(os.path.join(bundle, "record.json")))
            self.assertTrue(os.path.isfile(os.path.join(bundle, "board.png")))

        with mock.patch.object(xiangqi_bot.os, "makedirs",
                               side_effect=OSError("disk unavailable")):
            self.assertIsNone(bot._save_failure_diagnostics("test_failure"))

    def test_board_stability_requires_two_consecutive_stable_frames(self):
        bot = Bot()
        frames = [np.full((10, 10, 3), value, dtype=np.uint8)
                  for value in range(4)]
        bot.screenshot_for_processing = mock.MagicMock(side_effect=frames)
        bot.crop_board_region = mock.MagicMock(side_effect=lambda image: image)
        bot.images_changed = mock.MagicMock(
            side_effect=[True, False, False])
        with mock.patch.object(xiangqi_bot.time, "sleep"):
            result = bot._wait_for_board_stable(max_wait_s=1.0)
        self.assertIs(result, frames[-1])
        self.assertEqual(bot.screenshot_for_processing.call_count, 4)

    def test_board_stability_timeout_returns_none(self):
        bot = Bot()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        bot.screenshot_for_processing = mock.MagicMock(return_value=frame)
        self.assertIsNone(bot._wait_for_board_stable(max_wait_s=0))
        self.assertIs(bot._last_stability_image, frame)

    def test_piece_change_ignores_colored_move_highlight(self):
        bot = Bot()
        bot.win_x = bot.win_y = 0
        bot.retina_scale = 1.0
        bot.cell_w = bot.cell_h = 40
        bot.cols_logical = [50] * 9
        bot.rows_logical = [50] * 10
        before = np.full((100, 100, 3), 150, dtype=np.uint8)
        highlight_only = before.copy()
        cv2.circle(highlight_only, (50, 50), 8, (0, 255, 0), -1)
        piece_change = before.copy()
        cv2.circle(piece_change, (50, 50), 8, (20, 20, 20), -1)

        self.assertLess(bot._piece_cell_change(
            before, highlight_only, 0, 0), 1.0)
        self.assertGreater(bot._piece_cell_change(
            before, piece_change, 0, 0), 8.0)

    def test_engine_and_working_directory_exist(self):
        self.assertTrue(os.path.isfile(PIKAFISH))
        self.assertEqual(PIKAFISH_DIR, os.path.dirname(os.path.abspath(PIKAFISH)))

    def test_pikafish_can_calculate_a_move(self):
        result = subprocess.run(
            [PIKAFISH],
            input="uci\nisready\nposition startpos\ngo depth 1\nquit\n",
            text=True,
            capture_output=True,
            cwd=PIKAFISH_DIR,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("uciok", result.stdout)
        self.assertIn("readyok", result.stdout)
        self.assertIn("bestmove", result.stdout)

    def test_window_adapter_screenshot_contract(self):
        adapter = WindowAdapter()
        adapter.win_id = 123
        adapter.x, adapter.y, adapter.width, adapter.height = 10, 20, 800, 600
        fake_shot = np.zeros((600, 800, 4), dtype=np.uint8)

        fake_grabber = mock.MagicMock()
        fake_grabber.grab.return_value = fake_shot
        fake_context = mock.MagicMock()
        fake_context.__enter__.return_value = fake_grabber

        with mock.patch.object(adapter, "_refresh_windows"), \
             mock.patch.object(ctypes.windll.user32, "SetForegroundWindow"), \
             mock.patch.object(ctypes.windll.user32, "IsIconic", return_value=False), \
             mock.patch.object(xiangqi_bot.time, "sleep"), \
             mock.patch("mss.mss", return_value=fake_context):
            image = adapter.screenshot()

        fake_grabber.grab.assert_called_once_with(
            {"left": 10, "top": 20, "width": 800, "height": 600})
        self.assertEqual(image.shape, (600, 800, 3))

    @unittest.skipUnless(sys.platform == "win32", "Windows-only window enumeration")
    def test_window_discovery_does_not_match_non_wechat_titles(self):
        adapter = WindowAdapter()
        candidates = adapter._windows_candidates()
        allowed = {"weixin.exe", "wechat.exe", "weixinappex.exe", "wechatappex.exe"}
        self.assertTrue(all(item[3].casefold() in allowed for item in candidates))

    def test_bot_delegates_platform_operations(self):
        bot = Bot()
        bot.platform = mock.MagicMock()
        bot.platform.win_id = 456
        bot.platform.x, bot.platform.y = 30, 40
        bot.platform.width, bot.platform.height = 1000, 700
        bot.platform.screenshot.return_value = np.zeros((700, 1000, 3), dtype=np.uint8)

        bot.find_window()
        image = bot.screenshot_for_processing()
        bot.activate_window()
        bot.click(100, 200)

        self.assertEqual((bot.win_id, bot.win_x, bot.win_y), (456, 30, 40))
        self.assertEqual(image.shape, (700, 1000, 3))
        self.assertEqual(bot.retina_scale, 1.0)
        bot.platform.activate.assert_called_once_with()
        bot.platform.click.assert_called_once_with(100, 200)

    def test_calibrated_grid_follows_window_move_and_resize(self):
        bot = Bot()
        bot.calib_ratios = (.30, .20, .70, .90)
        bot.win_x, bot.win_y = 100, 200
        bot.platform.width, bot.platform.height = 1000, 700
        bot._refresh_grid_from_ratios()
        self.assertAlmostEqual(bot.cols_logical[0], 400)
        self.assertAlmostEqual(bot.rows_logical[0], 340)

        bot.win_x, bot.win_y = 300, 400
        bot.platform.width, bot.platform.height = 1200, 800
        bot._refresh_grid_from_ratios()
        self.assertAlmostEqual(bot.cols_logical[0], 660)
        self.assertAlmostEqual(bot.rows_logical[0], 560)
        self.assertAlmostEqual(bot.cols_logical[-1], 1140)
        self.assertAlmostEqual(bot.rows_logical[-1], 1120)

    def test_legacy_calibration_is_rejected_on_windows(self):
        bot = Bot()
        with mock.patch.object(xiangqi_bot, "CALIB_PATH", os.path.join(ROOT, "_legacy_calib_test.json")), \
             mock.patch.object(xiangqi_bot.os, "name", "nt"):
            try:
                with open(xiangqi_bot.CALIB_PATH, "w", encoding="utf-8") as handle:
                    json.dump({"rx1": .2, "ry1": .2, "rx2": .8, "ry2": .8}, handle)
                self.assertFalse(bot.load_calibration())
            finally:
                if os.path.exists(xiangqi_bot.CALIB_PATH):
                    os.remove(xiangqi_bot.CALIB_PATH)

    def test_inverted_relative_calibration_is_rejected(self):
        bot = Bot()
        path = os.path.join(ROOT, "_inverted_calib_test.json")
        with mock.patch.object(xiangqi_bot, "CALIB_PATH", path):
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({
                        "version": xiangqi_bot.CALIB_VERSION,
                        "platform": os.name,
                        "rx1": .8, "ry1": .7, "rx2": .2, "ry2": .3,
                    }, handle)
                self.assertFalse(bot.load_calibration())
            finally:
                if os.path.exists(path):
                    os.remove(path)

    def test_manual_calibration_rejects_reversed_points(self):
        bot = Bot()
        bot.win_x, bot.win_y = 800, 500
        path = os.path.join(ROOT, "_manual_calib_test.json")
        with mock.patch.object(xiangqi_bot, "CALIB_PATH", path), \
             mock.patch("builtins.input", side_effect=["", ""]), \
             mock.patch.object(xiangqi_bot.pyautogui, "position",
                               side_effect=[(1851, 1023), (849, 864)]), \
             mock.patch.object(bot, "_get_window_width", return_value=1221), \
             mock.patch.object(bot, "_get_window_height", return_value=773):
            self.assertFalse(bot.calibrate())
        self.assertFalse(os.path.exists(path))

    def test_recalibration_launcher_prefers_automatic_calibration(self):
        launcher = os.path.join(ROOT, "重新校准并启动.bat")
        with open(launcher, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("XIANGQI_RECALIBRATE=1", content)
        self.assertNotIn("XIANGQI_MANUAL_CALIBRATION=1", content)

    def test_turn_wait_polls_at_controlled_interval(self):
        bot = Bot()
        bot.is_my_turn = mock.MagicMock(side_effect=[False, False, True])
        with mock.patch.object(xiangqi_bot.time, "sleep") as sleep:
            self.assertTrue(bot.wait_for_our_turn(timeout_s=5))
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(xiangqi_bot.TURN_POLL_INTERVAL_S)

    def test_turn_indicator_uses_the_avatar_frame(self):
        bot = Bot()
        image = np.zeros((773, 1221, 3), dtype=np.uint8)
        x1, x2 = int(1221 * .775), int(1221 * .838)
        y1, y2 = int(773 * .705), int(773 * .810)
        image[y1:y1 + 5, x1:x2] = (0, 255, 0)
        image[y2 - 5:y2, x1:x2] = (0, 255, 0)
        image[y1:y2, x1:x1 + 5] = (0, 255, 0)
        image[y1:y2, x2 - 5:x2] = (0, 255, 0)
        bot.screenshot_for_processing = mock.MagicMock(return_value=image)

        self.assertTrue(bot.is_my_turn())

    def test_orientation_requires_consistent_board_evidence(self):
        bot = Bot()
        red, evidence = bot.determine_orientation(xiangqi_bot.INIT_RED)
        black, _ = bot.determine_orientation(xiangqi_bot.INIT_BLACK)
        self.assertTrue(red, evidence)
        self.assertFalse(black)

        conflicting = [row[:] for row in xiangqi_bot.INIT_RED]
        conflicting[0][4], conflicting[9][4] = 'K', 'k'
        orientation, reason = bot.determine_orientation(conflicting)
        self.assertIsNone(orientation)
        self.assertIn("conflict", reason)

    def test_visual_board_must_match_one_unique_legal_transition(self):
        bot = Bot()
        bot.playing_red = True
        before = [[None] * 9 for _ in range(10)]
        before[0][0] = 'r'
        expected = bot._apply_board_move(before, 'a9a8')
        move, result = bot._match_unique_legal_transition(
            before, expected, ['a9a8', 'a9b9'])
        self.assertEqual(move, 'a9a8')
        self.assertEqual(result, expected)

        inaccurate = [row[:] for row in expected]
        inaccurate[5][5] = 'p'
        move, result = bot._match_unique_legal_transition(
            before, inaccurate, ['a9a8', 'a9b9'])
        self.assertIsNone(move)
        self.assertIsNone(result)

    def test_opponent_transition_requires_two_matching_observations(self):
        bot = Bot()
        board_a = [[None] * 9 for _ in range(10)]
        board_a[0][0] = 'r'
        board_b = [row[:] for row in board_a]
        board_b[1][0] = board_b[0][0]
        board_b[0][0] = None
        move, result = bot._select_transition_consensus([
            ('a9a8', board_b), (None, None), ('a9a8', board_b)])
        self.assertEqual(move, 'a9a8')
        self.assertEqual(result, board_b)

        move, result = bot._select_transition_consensus([
            ('a9a8', board_b), ('a9b9', board_a), (None, None)])
        self.assertIsNone(move)
        self.assertIsNone(result)

    def test_transition_confidence_requires_repeated_strict_votes_for_a(self):
        bot = Bot()
        board = [[None] * 9 for _ in range(10)]
        board[0][0] = 'r'
        move, result, grade = \
            bot._select_transition_consensus_with_confidence([
                ('a9a8', board, 'A'), ('a9a8', board, 'A'),
                ('a9a8', board, 'B')])
        self.assertEqual((move, grade), ('a9a8', 'A'))
        self.assertEqual(result, board)

        move, result, grade = \
            bot._select_transition_consensus_with_confidence([
                ('a9a8', board, 'A'), ('a9a8', board, 'B'),
                (None, None, 'B')])
        self.assertEqual((move, grade), ('a9a8', 'B'))
        self.assertEqual(result, board)

    def test_engine_gate_rejects_untrusted_confidence(self):
        bot = Bot()
        board = [[None] * 9 for _ in range(10)]
        board[0][4] = 'k'
        board[9][4] = 'K'
        self.assertEqual(bot._state_is_safe_for_engine(board, 'A'), (True, ''))
        safe, reason = bot._state_is_safe_for_engine(board, 'C')
        self.assertFalse(safe)
        self.assertIn('untrusted confidence', reason)

    def test_our_move_confirmation_tolerates_only_unrelated_cnn_noise(self):
        bot = Bot()
        bot.playing_red = True
        before = [[None] * 9 for _ in range(10)]
        before[0][0] = 'r'
        expected = bot._apply_board_move(before, 'a9a8')

        noisy = [row[:] for row in expected]
        noisy[5][5] = 'p'
        self.assertTrue(bot._move_board_matches(
            noisy, expected, 'a9a8', True, True))
        self.assertFalse(bot._move_board_matches(
            noisy, expected, 'a9a8', True, False))

        wrong_destination = [row[:] for row in noisy]
        wrong_destination[1][0] = None
        self.assertFalse(bot._move_board_matches(
            wrong_destination, expected, 'a9a8', True, True))

    def test_opponent_transition_tolerates_unrelated_cnn_noise(self):
        bot = Bot()
        bot.playing_red = True
        before = [[None] * 9 for _ in range(10)]
        before[0][0] = 'r'
        before[0][1] = 'n'
        expected = bot._apply_board_move(before, 'a9a8')
        parsed = [row[:] for row in expected]
        parsed[5][5] = 'p'  # unrelated CNN error

        changes = {(0, 0): 30.0, (1, 0): 35.0,
                   (0, 1): 2.0, (1, 1): 2.0}
        with mock.patch.object(
                bot, '_piece_cell_change',
                side_effect=lambda _a, _b, r, c: changes.get((r, c), 0.0)):
            move, result = bot._match_legal_transition_with_endpoint_changes(
                before, parsed, ['a9a8', 'b9b8'], object(), object())
        self.assertEqual(move, 'a9a8')
        self.assertEqual(result, expected)

    def test_opponent_transition_rejects_too_many_cnn_errors(self):
        bot = Bot()
        bot.playing_red = True
        before = [[None] * 9 for _ in range(10)]
        before[0][0] = 'r'
        expected = bot._apply_board_move(before, 'a9a8')
        parsed = [row[:] for row in expected]
        for c in range(1, 6):
            parsed[5][c] = 'p'
        with mock.patch.object(bot, '_piece_cell_change', return_value=30.0):
            move, result = bot._match_legal_transition_with_endpoint_changes(
                before, parsed, ['a9a8'], object(), object())
        self.assertIsNone(move)
        self.assertIsNone(result)

    def test_opponent_transition_rejects_ambiguous_endpoint_changes(self):
        bot = Bot()
        bot.playing_red = True
        before = [[None] * 9 for _ in range(10)]
        before[0][0] = 'r'
        before[0][1] = 'n'
        parsed = [[None] * 9 for _ in range(10)]
        parsed[1][0] = 'r'
        parsed[1][1] = 'n'
        with mock.patch.object(bot, '_piece_cell_change', return_value=20.0):
            move, result = bot._match_legal_transition_with_endpoint_changes(
                before, parsed, ['a9a8', 'b9b8'], object(), object())
        self.assertIsNone(move)
        self.assertIsNone(result)

    def test_our_move_can_be_confirmed_visually_when_endpoint_cnn_is_wrong(self):
        bot = Bot()
        bot.playing_red = True
        before = object()
        previous = object()
        current = object()
        # Calls: before→current at src/dst, previous→current at src/dst.
        with mock.patch.object(
                bot, '_piece_cell_change', side_effect=[24.0, 28.0, 1.0, 1.5]), \
             mock.patch.object(bot, '_is_my_turn_image', return_value=False), \
             mock.patch.object(bot, '_is_opponent_turn_image', return_value=True):
            self.assertTrue(bot._move_visually_confirmed(
                before, previous, current, 'a0c1'))

    def test_visual_move_confirmation_requires_stability_and_both_turn_frames(self):
        bot = Bot()
        bot.playing_red = True
        image = object()
        with mock.patch.object(
                bot, '_piece_cell_change', side_effect=[24.0, 28.0, 5.0, 1.0]), \
             mock.patch.object(bot, '_is_my_turn_image', return_value=False), \
             mock.patch.object(bot, '_is_opponent_turn_image', return_value=True):
            self.assertFalse(bot._move_visually_confirmed(
                image, image, image, 'a0c1'))
        with mock.patch.object(bot, '_is_my_turn_image', return_value=False), \
             mock.patch.object(bot, '_is_opponent_turn_image', return_value=False):
            self.assertFalse(bot._move_visually_confirmed(
                image, image, image, 'a0c1'))

    def test_move_delay_is_invested_in_search_time(self):
        bot = Bot()
        bot.move_delay_enabled = True
        with mock.patch.object(bot, "check_delay_hotkey", return_value=False), \
             mock.patch.object(xiangqi_bot.random, "uniform", return_value=4.2) as uniform:
            self.assertEqual(bot._search_time_for_turn([[None] * 9 for _ in range(10)]),
                             10700)

        uniform.assert_called_once_with(
            xiangqi_bot.MOVE_DELAY_MIN_S, xiangqi_bot.MOVE_DELAY_MAX_S)

    def test_f8_toggles_move_delay(self):
        bot = Bot()
        with mock.patch.object(xiangqi_bot.os, "name", "nt"), \
             mock.patch.object(ctypes.windll.user32, "GetAsyncKeyState", side_effect=[1, 1]):
            self.assertTrue(bot.check_delay_hotkey())
            self.assertTrue(bot.move_delay_enabled)
            self.assertTrue(bot.check_delay_hotkey())
            self.assertFalse(bot.move_delay_enabled)

    def test_engine_strength_settings_and_default_delay(self):
        bot = Bot()
        self.assertEqual(xiangqi_bot.MOVE_TIME_MS, 2500)
        self.assertEqual(xiangqi_bot.ENGINE_THREADS, 4)
        self.assertEqual(xiangqi_bot.ENGINE_HASH_MB, 512)
        self.assertFalse(bot.move_delay_enabled)

    def test_endgame_search_time_increases_as_material_decreases(self):
        bot = Bot()
        opening = [[None] * 9 for _ in range(10)]
        for index in range(21):
            opening[index // 9][index % 9] = 'P'
        late = [[None] * 9 for _ in range(10)]
        for index in range(13):
            late[index // 9][index % 9] = 'P'
        ending = [[None] * 9 for _ in range(10)]
        for index in range(12):
            ending[index // 9][index % 9] = 'P'

        self.assertEqual(bot._search_time_for_board(opening), 2500)
        self.assertEqual(bot._search_time_for_board(late), 4000)
        self.assertEqual(bot._search_time_for_board(ending), 6500)

    def test_persistent_engine_receives_configured_search(self):
        bot = Bot()
        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None
        bot._engine_proc = fake_proc
        bot._engine_lines = mock.MagicMock()
        bot._engine_lines.get.return_value = "bestmove a0a1"
        with mock.patch.object(bot, "_ensure_pikafish", return_value=True):
            best, _ = bot.pikafish("startpos")
        self.assertEqual(best, "a0a1")
        fake_proc.stdin.write.assert_called_once_with(
            "position fen startpos\ngo movetime 2500\n")

    def test_failed_move_can_be_excluded_from_engine_search(self):
        bot = Bot()
        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None
        bot._engine_proc = fake_proc
        bot._engine_lines = mock.MagicMock()
        bot._engine_lines.get.return_value = "bestmove b0b1"
        with mock.patch.object(bot, "_ensure_pikafish", return_value=True), \
             mock.patch.object(bot, "get_legal_moves",
                               return_value=["a0a1", "b0b1"]):
            best, _ = bot.pikafish("startpos", excluded={"a0a1"})
        self.assertEqual(best, "b0b1")
        fake_proc.stdin.write.assert_called_once_with(
            "position fen startpos\ngo movetime 2500 searchmoves b0b1\n")

    def test_board_plausibility_requires_both_kings(self):
        bot = Bot()
        board = [[None] * 9 for _ in range(10)]
        board[0][4] = 'k'
        board[9][4] = 'K'
        self.assertEqual(bot.board_is_plausible(board), (True, ""))
        board[9][4] = None
        valid, reason = bot.board_is_plausible(board)
        self.assertFalse(valid)
        self.assertTrue("king count" in reason or "piece count" in reason)

    def test_board_plausibility_rejects_piece_limits_and_palace_errors(self):
        bot = Bot()
        board = [[None] * 9 for _ in range(10)]
        board[0][4] = 'k'
        board[9][4] = 'K'
        board[6][0] = board[6][2] = board[6][4] = 'P'
        board[6][6] = board[6][8] = board[5][1] = 'P'
        valid, reason = bot.board_is_plausible(board)
        self.assertFalse(valid)
        self.assertIn("too many P", reason)

        board[5][1] = None
        board[9][4] = None
        board[6][4] = 'K'
        valid, reason = bot.board_is_plausible(board)
        self.assertFalse(valid)
        self.assertIn("outside its palace", reason)

    def test_legal_move_query_has_a_hard_timeout(self):
        bot = Bot()
        with mock.patch.object(
                xiangqi_bot.subprocess, "run",
                side_effect=subprocess.TimeoutExpired("pikafish", 5)):
            self.assertEqual(bot.get_legal_moves("startpos"), [])

    def test_observer_launcher_never_enables_clicking_mode(self):
        launcher = os.path.join(ROOT, "启动观察模式.bat")
        with open(launcher, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("XIANGQI_OBSERVER=1", content)

    def test_engine_read_has_real_timeout(self):
        bot = Bot()
        fake_proc = mock.MagicMock()
        fake_proc.poll.return_value = None

        def blocked_readline():
            time.sleep(1)
            return ""

        fake_proc.stdout.readline.side_effect = blocked_readline
        with mock.patch.object(xiangqi_bot.subprocess, "Popen", return_value=fake_proc), \
             mock.patch.object(xiangqi_bot, "ENGINE_TIMEOUT_S", 0.1):
            started = time.monotonic()
            best, _ = bot.pikafish("startpos")
            elapsed = time.monotonic() - started
        self.assertIsNone(best)
        self.assertLess(elapsed, 0.5)
        fake_proc.kill.assert_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
