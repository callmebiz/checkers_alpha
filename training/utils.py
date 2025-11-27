import re
import json
import os
from datetime import datetime


def parse_iteration_line(line: str):
    """Parse a single iteration summary log line into a dict.

    Expected format (from TrainingMetrics.log_iteration_summary):
    Iter {i}: Games {total_games} | P1:{p1_wins} P-1:{p_neg1_wins} D:{draws} |
    Avg moves {avg_game_length} | Avg MCTS/move {avg_mcts_per_move} | Samples {total_samples} |
    Policy {policy_loss} Value {value_loss} | LR {learning_rate} | Time {total_time}s | ETA ...

    Returns None if line doesn't match.
    """
    if not re.search(r"\bIter\b|\b\d+:", line):
        return None

    # use regex to extract common fields
    try:
        it_match = re.search(r"(?:Iter\s+)?(\d+):", line)
        iter_idx = int(it_match.group(1)) if it_match else None

        games_m = re.search(r"Games\s+(\d+)(?:/(\d+))?", line)
        total_games = int(games_m.group(1)) if games_m else None
        requested_games = int(games_m.group(2)) if games_m and games_m.group(2) else None

        p1_m = re.search(r"P1:(\d+)", line)
        p1_wins = int(p1_m.group(1)) if p1_m else None

        pneg1_m = re.search(r"P-1:(\d+)", line)
        p_neg1_wins = int(pneg1_m.group(1)) if pneg1_m else None

        draw_m = re.search(r"D:(\d+)", line)
        draws = int(draw_m.group(1)) if draw_m else None

        avg_moves_m = re.search(r"Avg moves\s+([0-9.]+)", line)
        avg_game_length = float(avg_moves_m.group(1)) if avg_moves_m else None

        avg_mcts_m = re.search(r"Avg MCTS/move\s+([0-9.]+)", line)
        avg_mcts_per_move = float(avg_mcts_m.group(1)) if avg_mcts_m else None

        samples_m = re.search(r"Samples\s+(\d+)", line)
        total_samples = int(samples_m.group(1)) if samples_m else None

        policy_m = re.search(r"Policy(?:\s+Loss)?\s+([0-9.eE+-]+)", line)
        policy_loss = float(policy_m.group(1)) if policy_m else None

        value_m = re.search(r"Value(?:\s+Loss)?\s+([0-9.eE+-]+)", line)
        value_loss = float(value_m.group(1)) if value_m else None

        lr_m = re.search(r"LR\s+([0-9.eE+-]+)", line)
        learning_rate = float(lr_m.group(1)) if lr_m else None

        time_m = re.search(r"Time\s+([0-9.]+)s", line)
        total_time = float(time_m.group(1)) if time_m else None

        total_m = re.search(r"Total\s+([0-9.eE+-]+)\s*\|", line)
        total_loss = float(total_m.group(1)) if total_m else None

        selfplay_m = re.search(r"Selfplay\s+([0-9.]+)s", line)
        selfplay_time = float(selfplay_m.group(1)) if selfplay_m else None

        train_m = re.search(r"Train\s+([0-9.]+)s", line)
        training_time = float(train_m.group(1)) if train_m else None
        # optional event counts
        cap_m = re.search(r"Captures\s+(\d+)", line)
        captures = int(cap_m.group(1)) if cap_m else None
        prom_m = re.search(r"Promotions\s+(\d+)", line)
        promotions = int(prom_m.group(1)) if prom_m else None
        mj_m = re.search(r"MultiJumps\s+(\d+)", line)
        multi_jumps = int(mj_m.group(1)) if mj_m else None
        # optional win streak
        streak_m = re.search(r"WinStreak\s+(\d+)", line)
        win_streak = int(streak_m.group(1)) if streak_m else None

    except Exception:
        return None

    # attach requested_games if present
    parsed = {
        'iteration': iter_idx,
        'game_stats': {
            'total_games': total_games,
            'requested_games': requested_games if 'requested_games' in locals() else None,
            'p1_wins': p1_wins,
            'p_neg1_wins': p_neg1_wins,
            'draws': draws,
            'avg_game_length': avg_game_length,
            'avg_mcts_per_move': avg_mcts_per_move,
            'total_samples': total_samples,
            'captures': captures,
            'promotions': promotions,
            'multi_jumps': multi_jumps,
            'win_streak': win_streak,
        },
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'total_loss': total_loss,
        'learning_rate': learning_rate,
        'timing': {
            'selfplay_time': selfplay_time,
            'training_time': training_time,
            'total_time': total_time,
        }
    }
    return parsed


def parse_log_file_to_history(log_path: str):
    """Parse a training log file and return a training_history dict compatible
    with `TrainingMetrics.training_history` structure.
    """
    history = {
        'iterations': [],
        'policy_losses': [],
        'value_losses': [],
        'total_losses': [],
        'learning_rates': [],
        'game_stats': [],
        'timing': []
    }

    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed = parse_iteration_line(line)
            if not parsed:
                continue
            history['iterations'].append(parsed['iteration'])
            history['policy_losses'].append(parsed['policy_loss'])
            history['value_losses'].append(parsed['value_loss'])
            history['total_losses'].append(parsed['total_loss'])
            history['learning_rates'].append(parsed['learning_rate'])
            history['game_stats'].append(parsed['game_stats'])
            history['timing'].append(parsed['timing'])

    return history


def write_history_json_from_log(log_path: str, out_path: str = None):
    """Parse `log_path` and write a JSON training history file. If `out_path`
    is None, writes `training_history_from_log_<timestamp>.json` next to the log.
    Returns path to the written file.
    """
    history = parse_log_file_to_history(log_path)

    if out_path is None:
        log_dir = os.path.dirname(log_path)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f"training_history_from_log_{ts}.json"
        out_path = os.path.join(log_dir, out_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    return out_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse training log into JSON history')
    parser.add_argument('logfile', help='Path to training .log file')
    parser.add_argument('--out', help='Output JSON path (optional)')
    args = parser.parse_args()

    out = write_history_json_from_log(args.logfile, args.out)
    print(f"Wrote training history JSON to: {out}")
