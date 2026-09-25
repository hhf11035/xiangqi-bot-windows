#!/usr/bin/env python3
"""Offline replay for a saved Xiangqi failure bundle. Never clicks a window."""

import json
import os
import sys

import cv2
import numpy as np

from xiangqi_bot import Bot, DIAGNOSTIC_DIR


def latest_bundle(root=DIAGNOSTIC_DIR):
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "record.json")):
            candidates.append(path)
    return max(candidates, key=os.path.getmtime) if candidates else None


def read_image(path):
    try:
        payload = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(payload, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def board_differences(tracked, observed):
    if not tracked or not observed:
        return []
    differences = []
    for row in range(10):
        for col in range(9):
            if tracked[row][col] != observed[row][col]:
                differences.append({
                    "row": row,
                    "col": col,
                    "tracked": tracked[row][col],
                    "observed": observed[row][col],
                })
    return differences


def analyze_transition(bot, tracked_fen, tracked, observed,
                       before_image=None, current_image=None):
    """Explain exact/relaxed legal matching for an opponent transition."""
    opponent_side = 'b' if bot.playing_red else 'w'
    legal_moves = bot.get_legal_moves(
        f"{tracked_fen} {opponent_side} - - 0 1")
    exact_move, _ = bot._match_unique_legal_transition(
        tracked, observed, legal_moves)
    relaxed_move = None
    candidates = []
    if before_image is not None and current_image is not None:
        relaxed_move, _ = bot._match_legal_transition_with_endpoint_changes(
            tracked, observed, legal_moves, before_image, current_image)
        for move in legal_moves:
            expected = bot._apply_board_move(tracked, move)
            if expected is None:
                continue
            src, dst = bot.uci_to_screen_cells(move)
            source_delta = bot._piece_cell_change(
                before_image, current_image, *src)
            destination_delta = bot._piece_cell_change(
                before_image, current_image, *dst)
            endpoint_match = (
                observed[src[0]][src[1]] is None and
                observed[dst[0]][dst[1]] == expected[dst[0]][dst[1]])
            endpoints = {src, dst}
            unrelated_mismatches = sum(
                observed[row][col] != expected[row][col]
                for row in range(10) for col in range(9)
                if (row, col) not in endpoints)
            candidates.append({
                "move": move,
                "source": list(src),
                "destination": list(dst),
                "source_delta": round(source_delta, 3),
                "destination_delta": round(destination_delta, 3),
                "combined_delta": round(source_delta + destination_delta, 3),
                "endpoint_match": endpoint_match,
                "unrelated_mismatches": unrelated_mismatches,
            })
        candidates.sort(key=lambda item: item["combined_delta"], reverse=True)
    return {
        "legal_move_count": len(legal_moves),
        "exact_match": exact_move,
        "relaxed_match": relaxed_move,
        "top_candidates": candidates[:10],
    }


def replay(bundle):
    record_path = os.path.join(bundle, "record.json")
    with open(record_path, encoding="utf-8") as handle:
        record = json.load(handle)

    ratios = record.get("calibration_ratios")
    window = record.get("window") or {}
    tracked_fen = (record.get("metadata") or {}).get("tracked_fen")
    image_names = ("latest_observation.png", "before_opponent_move.png",
                   "before_move.png")
    image_path = next((os.path.join(bundle, name) for name in image_names
                       if os.path.isfile(os.path.join(bundle, name))), None)
    if not ratios or len(ratios) != 4:
        raise RuntimeError("诊断记录缺少 calibration_ratios；请使用新版程序生成记录。")
    if not tracked_fen:
        raise RuntimeError("诊断记录没有 tracked_fen，无法比较局面。")
    if not image_path:
        raise RuntimeError("诊断记录中没有可回放截图。")

    image = read_image(image_path)
    if image is None:
        raise RuntimeError("无法读取诊断截图。")

    bot = Bot()
    bot.playing_red = bool(record.get("playing_red", True))
    bot.win_x = int(window.get("x") or 0)
    bot.win_y = int(window.get("y") or 0)
    bot.retina_scale = float(window.get("retina_scale") or 1.0)
    bot.platform.width = int(window.get("width") or image.shape[1])
    bot.platform.height = int(window.get("height") or image.shape[0])
    bot.calib_ratios = tuple(float(value) for value in ratios)
    bot._refresh_grid_from_ratios()
    if not bot.load_cnn():
        raise RuntimeError("CNN模型无法加载。")

    observed = bot.cnn.parse_board(
        image, bot.cols_logical, bot.rows_logical,
        bot.retina_scale, bot.win_x, bot.win_y,
        bot.cell_w, bot.cell_h)
    tracked = bot._fen_to_board(tracked_fen)
    before_path = os.path.join(bundle, "before_opponent_move.png")
    before_image = read_image(before_path) if os.path.isfile(before_path) else None
    report = {
        "bundle": bundle,
        "source_image": os.path.basename(image_path),
        "reason": record.get("reason"),
        "tracked_fen": tracked_fen,
        "observed_fen": bot.board_to_fen(observed),
        "differences": board_differences(tracked, observed),
    }
    if "opponent" in str(record.get("reason", "")):
        report["transition_analysis"] = analyze_transition(
            bot, tracked_fen, tracked, observed, before_image, image)
    report_path = os.path.join(bundle, "replay_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report_path, report


def main():
    bundle = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else latest_bundle()
    if not bundle:
        print("没有找到可回放的故障诊断记录。")
        return 1
    try:
        report_path, report = replay(bundle)
    except Exception as exc:
        print(f"离线回放失败：{exc}")
        return 1
    print(f"回放完成：{report_path}")
    print(f"跟踪FEN：{report['tracked_fen']}")
    print(f"识别FEN：{report['observed_fen']}")
    print(f"差异棋格：{len(report['differences'])}")
    for item in report["differences"]:
        print(f"  ({item['row']},{item['col']}): "
              f"{item['tracked']} -> {item['observed']}")
    transition = report.get("transition_analysis")
    if transition:
        print(f"严格匹配：{transition['exact_match']}")
        print(f"容错匹配：{transition['relaxed_match']}")
        print("像素变化最高的合法候选：")
        for item in transition["top_candidates"][:5]:
            print(f"  {item['move']}: Δ={item['combined_delta']:.1f}, "
                  f"端点={item['endpoint_match']}, "
                  f"无关差异={item['unrelated_mismatches']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
