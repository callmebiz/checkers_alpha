# training/metrics.py

import json
import time
import os


class TrainingMetrics:
    def __init__(self, log_dir, timestamp, logger=None):
        self.log_dir = log_dir
        self.timestamp = timestamp
        self.logger = logger
        self.start_time = time.time()
        
        self.training_history = {
            'iterations': [],
            'policy_losses': [],
            'value_losses': [],
            'total_losses': [],
            'learning_rates': [],
            'game_stats': [],
            'timing': []
        }
    
    def update_history(self, iteration, policy_loss, value_loss, total_loss, learning_rate, 
                      game_stats, selfplay_time, training_time, total_time, inference_time=0.0,
                      epoch_losses=None, epoch_timing=None, grad_norm=None, max_grad=None,
                      gpu_stats=None, checkpoints=None):
        """Update training history with metrics from current iteration"""
        self.training_history['iterations'].append(iteration)
        self.training_history['policy_losses'].append(policy_loss)
        self.training_history['value_losses'].append(value_loss)
        self.training_history['total_losses'].append(total_loss)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['game_stats'].append(game_stats)
        self.training_history['timing'].append({
            'selfplay_time': selfplay_time,
            'inference_time': inference_time,
            'training_time': training_time,
            'total_time': total_time
        })
        # Optional detailed fields
        if epoch_losses is not None:
            self.training_history.setdefault('epoch_losses', []).append(epoch_losses)
        if epoch_timing is not None:
            self.training_history.setdefault('epoch_timing', []).append(epoch_timing)
        if grad_norm is not None or max_grad is not None:
            self.training_history.setdefault('gradients', []).append({'grad_norm': grad_norm, 'max_grad': max_grad})
        if gpu_stats is not None:
            self.training_history.setdefault('gpu', []).append(gpu_stats)
        if checkpoints is not None:
            self.training_history.setdefault('checkpoints', []).extend(checkpoints)
    
    def save_training_history(self, args):
        """Save training history to JSON file"""
        history_filename = (
            f"training_history_"
            f"iter{args['num_iterations']}_"
            f"games{args['num_self_play_iterations']}_"
            f"mcts{args['num_searches']}_"
            f"{self.timestamp}.json"
        )
        history_file = os.path.join(self.log_dir, history_filename)
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        if self.logger:
            self.logger.info(f"Training history saved to {history_file}")
        return history_file
    
    def estimate_time_remaining(self, current_iter, total_iterations):
        """Estimate remaining training time"""
        if current_iter == 0:
            return "Unknown"
        
        elapsed = time.time() - self.start_time
        remaining_iters = total_iterations - current_iter
        time_per_iter = elapsed / current_iter
        remaining_seconds = remaining_iters * time_per_iter
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
    
    def log_iteration_summary(self, iteration, iteration_stats):
        """Log clean iteration summary with time estimates"""
        if not self.logger:
            return
            
        balance_status = ""
        if iteration_stats['p1_win_rate'] > 0.7:
            balance_status = f"[P1 bias {iteration_stats['p1_win_rate']:.1%}]"
        elif iteration_stats['p_neg1_win_rate'] > 0.7:
            balance_status = f"[P-1 bias {iteration_stats['p_neg1_win_rate']:.1%}]"
        else:
            balance_status = "[Balanced]"
        
        time_remaining = self.estimate_time_remaining(iteration + 1, iteration_stats.get('total_iterations', 1))

        # Compute total loss if available
        try:
            total_loss = iteration_stats.get('total_loss', None)
            if total_loss is None:
                total_loss = iteration_stats.get('policy_loss', 0.0) + iteration_stats.get('value_loss', 0.0)
        except Exception:
            total_loss = None

        # Show only total completed games
        games_str = f"{iteration_stats['total_games']}"

        # Build log
        cap = int(iteration_stats.get('captures_total', 0) or 0)
        prom = int(iteration_stats.get('promotions_total', 0) or 0)
        mj = int(iteration_stats.get('multi_jumps_total', 0) or 0)
        max_streak = int(iteration_stats.get('max_win_streak', 0) or 0)
        gn = iteration_stats.get('grad_norm')
        mg = iteration_stats.get('max_grad')

        msg = (
            f"{iteration}: Games {games_str} | "
            f"P1:{iteration_stats['p1_wins']} P-1:{iteration_stats['p_neg1_wins']} D:{iteration_stats['draws']} {balance_status} | "
            f"Avg moves {iteration_stats.get('avg_game_length', 0):.1f} | "
            f"Avg MCTS/move {iteration_stats.get('avg_mcts_per_move', 0):.2f} | "
            f"Samples {iteration_stats.get('total_samples', 0)} | "
            f"Policy Loss {iteration_stats['policy_loss']:.3f} Value Loss {iteration_stats['value_loss']:.3f} Total Loss {total_loss:.3f} | "
            f"LR {iteration_stats['learning_rate']:.4f} | "
            f"Time {iteration_stats['total_time']:.1f}s | "
            f"Selfplay {iteration_stats.get('selfplay_time', 0):.2f}s Inference {iteration_stats.get('inference_time', 0):.2f}s Train {iteration_stats.get('training_time', 0):.2f}s | "
            f"ETA {time_remaining} | "
            f"Captures {cap} Promotions {prom} MultiJumps {mj} | "
            f"Max Win Streak {max_streak} | "
            f"GradNorm {gn or 0:.3f} MaxGrad {mg or 0:.3f}"
        )

        # Log via logger and also append to the per-run log file to ensure a single-line-per-iteration
        # summary is always present even if logging configuration changes.
        try:
            self.logger.info(msg)
        except Exception:
            pass
