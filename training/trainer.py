# training/trainer.py

import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import logging
import os
import sys
import signal
from datetime import datetime
from tqdm import tqdm
import json

from .checkpoints import CheckpointManager
from .metrics import TrainingMetrics
from .self_play import SelfPlayManager
from . import mlflow_logger as mlfl


class Trainer:
    def __init__(self, model, optimizer, game, mcts, state_encoder, args, 
                 initial_state=None, log_dir="logs", use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.mcts = mcts
        self.state_encoder = state_encoder
        self.args = args
        self.initial_state = initial_state
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize mixed precision scaler
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            print("Using mixed precision training")
        else:
            self.scaler = None
            print("Using standard precision training")
        
        # Initialize logging
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{self.timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.checkpoint_manager = CheckpointManager(model, optimizer, game, self.scaler, self.logger)
        self.metrics = TrainingMetrics(log_dir, self.timestamp, self.logger)
        self.self_play_manager = SelfPlayManager(game, mcts, state_encoder, self.logger)
        # MLflow integration (optional)
        self.mlflow_run = None
        self.use_mlflow = bool(self.args.get('use_mlflow', False))
        self.mlflow_log_interval = int(self.args.get('mlflow_log_interval', 1))
        self.mlflow_run_id = None
        if self.use_mlflow:
            try:
                self.mlflow_run = mlfl.init_mlflow(
                    experiment_name=self.args.get('mlflow_experiment_name', 'checkers'),
                    run_name=self.timestamp,
                    tracking_uri=self.args.get('mlflow_tracking_uri')
                )
                params = {
                    'num_iterations': self.args.get('num_iterations'),
                    'num_self_play_iterations': self.args.get('num_self_play_iterations'),
                    'num_searches': self.args.get('num_searches'),
                    'batch_size': self.args.get('batch_size'),
                    'num_epochs': self.args.get('num_epochs')
                }
                mlfl.log_params(params)
                # Log run id to help find the run in the UI
                try:
                    run_id = None
                    if self.mlflow_run and hasattr(self.mlflow_run, 'info'):
                        run_id = self.mlflow_run.info.run_id
                    elif hasattr(self.mlflow_run, 'run_id'):
                        run_id = self.mlflow_run.run_id
                    if run_id:
                        self.mlflow_run_id = run_id
                        self.logger.info(f"MLflow run id: {run_id}")
                except Exception:
                    pass
            except Exception:
                # Non-fatal: continue without mlflow
                self.logger.warning('MLflow init failed; continuing without MLflow')
        
        # Training state
        self.current_iteration = 0
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)
        
        self.logger.info("AlphaZero Training Started")
        # Reproducibility: optional seeding and deterministic mode
        seed = self.args.get('seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if self.args.get('deterministic', False):
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.logger.info('CUDNN set to deterministic mode')
            except Exception:
                pass
        
    def graceful_shutdown(self, signum, frame):
        """Handle interruption gracefully"""
        self.logger.info(f"Received signal {signum}, saving checkpoint and shutting down...")
        self.checkpoint_manager.save_checkpoint(
            self.current_iteration, "interrupted_checkpoint", 
            self.metrics.training_history, self.args, self.timestamp,
            mlflow_run_id=self.mlflow_run_id
        )
        try:
            # Update manifest to point to latest checkpoint
            manifest_path = os.path.join("interrupted_checkpoint", "manifest.json")
            # best-effort: write a minimal manifest in the interrupted folder
            m = {
                'run_name': os.path.basename(os.path.abspath('interrupted_checkpoint')),
                'timestamp': self.timestamp,
                'mlflow_run_id': self.mlflow_run_id,
            }
            with open(manifest_path, 'w') as mf:
                json.dump(m, mf, indent=2)
        except Exception:
            pass
        sys.exit(0)

    def _write_manifest(self, model_dir, config_path=None, update_fields=None):
        """Write or update a manifest.json file in the model_dir with run metadata."""
        manifest_path = os.path.join(model_dir, 'manifest.json')
        manifest = {}
        manifest['run_name'] = os.path.basename(os.path.abspath(model_dir))
        manifest['timestamp'] = self.timestamp
        try:
            manifest['start_time'] = datetime.fromtimestamp(self.metrics.start_time).isoformat()
        except Exception:
            manifest['start_time'] = datetime.now().isoformat()
        if config_path:
            manifest['config_path'] = config_path
        manifest['mlflow_run_id'] = self.mlflow_run_id

        # checkpoint_latest: prefer checkpoints subfolder
        ck_latest_candidates = [
            os.path.join(model_dir, 'checkpoints', 'checkpoint_latest.pt'),
            os.path.join(model_dir, 'checkpoint_latest.pt')
        ]
        ck_latest = None
        for c in ck_latest_candidates:
            if os.path.exists(c):
                ck_latest = c
                break
        manifest['checkpoint_latest'] = ck_latest

        # training history: look for the most recent training_history_*.json in log_dir
        try:
            import glob
            hist_files = glob.glob(os.path.join(self.log_dir, 'training_history_*.json'))
            manifest['training_history'] = hist_files[-1] if hist_files else None
        except Exception:
            manifest['training_history'] = None

        if update_fields:
            manifest.update(update_fields)

        try:
            os.makedirs(model_dir, exist_ok=True)
            with open(manifest_path, 'w') as mf:
                json.dump(manifest, mf, indent=2)
        except Exception:
            pass
        return manifest_path
    
    def train_on_batch(self, memory, iteration):
        """Train the model on a batch of experiences"""
        if len(memory) == 0:
            return 0, 0
            
        random.shuffle(memory)
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        num_batches = (len(memory) + self.args['batch_size'] - 1) // self.args['batch_size']
        batch_pbar = tqdm(
            range(0, len(memory), self.args['batch_size']),
            desc=f"Iteration {iteration} Training",
            total=num_batches,
            leave=False,
            unit="batch"
        )
        
        for batch_idx in batch_pbar:
            batch_end = min(len(memory), batch_idx + self.args['batch_size'])
            sample = memory[batch_idx:batch_end]
            
            if len(sample) == 0:
                continue
                
            state, policy_targets, value_targets = zip(*sample)
            
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    out_policy, out_value = self.model(state)
                    log_probs = F.log_softmax(out_policy, dim=1)
                    policy_loss = -torch.sum(policy_targets * log_probs) / policy_targets.size(0)
                    value_loss = F.mse_loss(out_value, value_targets)
                    total_loss = policy_loss + value_loss
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out_policy, out_value = self.model(state)
                log_probs = F.log_softmax(out_policy, dim=1)
                policy_loss = -torch.sum(policy_targets * log_probs) / policy_targets.size(0)
                value_loss = F.mse_loss(out_value, value_targets)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batch_count += 1
            
            batch_pbar.set_postfix({
                'Policy Loss': f'{policy_loss.item():.4f}',
                'Value Loss': f'{value_loss.item():.4f}',
                'Total Loss': f'{total_loss.item():.4f}'
            })
        
        batch_pbar.close()
        return total_policy_loss / batch_count, total_value_loss / batch_count
    
    def train(self, model_dir, resume_from=None, use_parallel=False, num_processes=None, config_path=None):
        """Main training loop"""
        self.logger.info(f"Starting training for {self.args['num_iterations']} iterations")
        self.logger.info(f"Model directory: {model_dir}")
        
        # Resume from checkpoint if specified
        start_iteration = 0
        if resume_from:
            checkpoint = self.checkpoint_manager.load_checkpoint(resume_from)
            if checkpoint:
                start_iteration = checkpoint['iteration'] + 1
                self.metrics.training_history = checkpoint['training_history']
                self.args.update(checkpoint['args'])
                self.logger.info(f"Resuming training from iteration {start_iteration}")
            else:
                self.logger.error("Failed to load checkpoint, starting from scratch")
        
        overall_start_time = time.time()
        self.metrics.start_time = overall_start_time

        # Prepare per-run subfolders
        models_dir = os.path.join(model_dir, 'models')
        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        logs_dir = os.path.join(model_dir, 'logs')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # Write initial manifest for this run (will be updated during training)
        try:
            self._write_manifest(model_dir, config_path=config_path)
        except Exception:
            self.logger.debug('Failed to write initial manifest')
        # Save a snapshot of config and args into the model_dir/models folder for easy inspection
        try:
            import json
            models_dir = os.path.join(model_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            cfg = {
                'args': self.args,
                'game_config': {
                    'row_count': self.game.row_count,
                    'col_count': self.game.col_count,
                    'buffer_count': self.game.buffer_count,
                    'draw_move_limit': self.game.draw_move_limit,
                    'repetition_limit': self.game.repetition_limit,
                    'history_timesteps': self.game.history_timesteps
                }
            }
            # include replay buffer size hint if present in args
            if 'replay_buffer_size' in self.args:
                cfg['replay_buffer_size'] = self.args.get('replay_buffer_size')
            cfg_path = os.path.join(models_dir, 'config_snapshot.json')
            with open(cfg_path, 'w') as cf:
                json.dump(cfg, cf, indent=2)
        except Exception:
            pass
        
        if use_parallel:
            self.logger.info(f"Using parallel self-play with {num_processes or 'auto'} processes")
        
        main_pbar = tqdm(
            range(start_iteration, self.args['num_iterations']),
            desc="Training Progress",
            unit="iteration",
            position=0,
            initial=start_iteration,
            total=self.args['num_iterations']
        )
        
        for i in main_pbar:
            self.current_iteration = i
            # checkpoint events collected during this iteration
            iteration_checkpoints = []
            iteration_start_time = time.time()
            memory = []
            self.model.eval()
            
            # Alternate starting player each iteration for balance
            starting_player = 1 if i % 2 == 0 else -1
            
            game_outcomes = {'p1_wins': 0, 'p_neg1_wins': 0, 'draws': 0}
            game_stats_list = []
            outcome_sequence = []
            successful_games = 0
            
            # Self-play phase
            selfplay_start_time = time.time()
            
            if use_parallel:
                # Parallel self-play
                results = self.self_play_manager.play_games_parallel(
                    self.args['num_self_play_iterations'],
                    starting_player,
                    i,
                    self.args,
                    self.initial_state,
                    num_processes
                )

                # Process results
                for game_memory, final_outcome_p1, _, game_stats in results:
                    if len(game_memory) > 0:
                        memory += game_memory
                        successful_games += 1
                        game_stats_list.append(game_stats)
                        outcome_sequence.append(final_outcome_p1)

                        # Track game outcomes
                        if final_outcome_p1 == 1:
                            game_outcomes['p1_wins'] += 1
                        elif final_outcome_p1 == -1:
                            game_outcomes['p_neg1_wins'] += 1
                        else:
                            game_outcomes['draws'] += 1

                print(f"Parallel self-play completed: {successful_games}/{self.args['num_self_play_iterations']} games")
            
            if not use_parallel:
                # Sequential self-play (original implementation)
                selfplay_pbar = tqdm(
                    range(self.args['num_self_play_iterations']),
                    desc=f"Self-play (P{starting_player} starts)",
                    leave=False,
                    position=1,
                    unit="game"
                )
                
                for game_idx in selfplay_pbar:
                    try:
                        game_memory, final_outcome_p1, _, game_stats = self.self_play_manager.play_game(
                            starting_player, game_idx, iteration=i, args=self.args, initial_state=self.initial_state
                        )
                        
                        if len(game_memory) > 0:
                            memory += game_memory
                            successful_games += 1
                            game_stats_list.append(game_stats)
                            outcome_sequence.append(final_outcome_p1)
                            
                            # Track game outcomes
                            if final_outcome_p1 == 1:
                                game_outcomes['p1_wins'] += 1
                            elif final_outcome_p1 == -1:
                                game_outcomes['p_neg1_wins'] += 1
                            else:
                                game_outcomes['draws'] += 1
                            
                            selfplay_pbar.set_postfix({
                                'P1 Wins': game_outcomes['p1_wins'],
                                'P-1 Wins': game_outcomes['p_neg1_wins'],
                                'Draws': game_outcomes['draws'],
                                'Avg Moves': f"{np.mean([gs['moves'] for gs in game_stats_list]):.1f}",
                                'Samples': len(memory)
                            })
                                
                    except Exception as e:
                        self.logger.error(f"Error in self-play game {game_idx}: {e}")
                        continue
                
                selfplay_pbar.close()
            
            selfplay_time = time.time() - selfplay_start_time
            
            # Calculate game statistics
            total_games = sum(game_outcomes.values())
            if total_games == 0:
                self.logger.error("No successful games completed, skipping iteration")
                continue
                
            p1_win_rate = game_outcomes['p1_wins'] / total_games
            p_neg1_win_rate = game_outcomes['p_neg1_wins'] / total_games
            draw_rate = game_outcomes['draws'] / total_games
            
            avg_game_length = np.mean([gs['moves'] for gs in game_stats_list])
            avg_game_time = np.mean([gs['game_time'] for gs in game_stats_list])
            avg_mcts_per_move = np.mean([gs['avg_mcts_per_move'] for gs in game_stats_list])
            # aggregate event counts
            total_captures = int(sum(gs.get('captures', 0) for gs in game_stats_list))
            total_promotions = int(sum(gs.get('promotions', 0) for gs in game_stats_list))
            total_multi_jumps = int(sum(gs.get('multi_jumps', 0) for gs in game_stats_list))
            total_inference_time = float(sum(gs.get('inference_time', 0.0) for gs in game_stats_list))

            # compute max win streaks in the sequence of game outcomes
            max_streak_p1 = 0
            max_streak_pneg1 = 0
            cur_p1 = cur_pneg1 = 0
            for o in outcome_sequence:
                if o == 1:
                    cur_p1 += 1
                    cur_pneg1 = 0
                elif o == -1:
                    cur_pneg1 += 1
                    cur_p1 = 0
                else:
                    cur_p1 = cur_pneg1 = 0
                max_streak_p1 = max(max_streak_p1, cur_p1)
                max_streak_pneg1 = max(max_streak_pneg1, cur_pneg1)
            max_win_streak = max(max_streak_p1, max_streak_pneg1)
            
            if len(memory) == 0:
                self.logger.warning("No training data collected, skipping training")
                continue
            
            # Training phase
            self.model.train()
            training_start_time = time.time()
            
            total_policy_loss = 0
            total_value_loss = 0
            
            epoch_pbar = tqdm(
                range(self.args['num_epochs']),
                desc="Training Epochs",
                leave=False,
                position=1,
                unit="epoch"
            )
            epoch_losses = []
            epoch_timing = []
            for epoch in epoch_pbar:
                e_start = time.time()
                policy_loss, value_loss = self.train_on_batch(memory, i)
                e_time = time.time() - e_start
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                epoch_losses.append({'policy_loss': policy_loss, 'value_loss': value_loss, 'total_loss': policy_loss + value_loss})
                epoch_timing.append(e_time)
                
                epoch_pbar.set_postfix({
                    'Policy Loss': f'{policy_loss:.4f}',
                    'Value Loss': f'{value_loss:.4f}',
                    'Total Loss': f'{policy_loss + value_loss:.4f}'
                })
            
            epoch_pbar.close()
            training_time = time.time() - training_start_time
            
            avg_policy_loss = total_policy_loss / self.args['num_epochs']
            avg_value_loss = total_value_loss / self.args['num_epochs']
            avg_total_loss = avg_policy_loss + avg_value_loss
            
            # Learning rate decay
            current_lr = self.optimizer.param_groups[0]['lr']
            if i > 0 and i % self.args.get('lr_decay_steps', 15) == 0:
                old_lr = current_lr
                decay_factor = self.args.get('lr_decay_factor', 0.8)
                new_lr = old_lr * decay_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
                self.logger.info(f"LR decay: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Save model and checkpoint into run subfolders
            torch.save(self.model.state_dict(), os.path.join(models_dir, f"model_{i}.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(models_dir, f"optimizer_{i}.pt"))
            
            if i % 5 == 0 or i == self.args['num_iterations'] - 1:
                self.checkpoint_manager.save_checkpoint(
                    i, checkpoints_dir, self.metrics.training_history, self.args, self.timestamp,
                    mlflow_run_id=self.mlflow_run_id
                )
                # record checkpoint event
                ck_path = os.path.join(checkpoints_dir, f"checkpoint_{i}.pt")
                latest_path = os.path.join(checkpoints_dir, 'checkpoint_latest.pt')
                iteration_checkpoints.append({'iteration': i, 'path': ck_path, 'latest_path': latest_path})
                try:
                    self._write_manifest(model_dir, config_path=config_path)
                except Exception:
                    pass
                # Optionally log checkpoint artifacts to MLflow
                if self.use_mlflow:
                    try:
                        ck_path = os.path.join(checkpoints_dir, f"checkpoint_{i}.pt")
                        latest_path = os.path.join(checkpoints_dir, 'checkpoint_latest.pt')
                        if os.path.exists(ck_path):
                            try:
                                mlfl.log_artifact(ck_path, artifact_path=f'checkpoints/iter_{i}')
                            except Exception:
                                pass
                        if os.path.exists(latest_path):
                            try:
                                mlfl.log_artifact(latest_path, artifact_path='checkpoints/latest')
                            except Exception:
                                pass
                    except Exception:
                        self.logger.warning('Failed to log checkpoint to MLflow')
            
            iteration_time = time.time() - iteration_start_time
            
            # Compile iteration statistics
            iteration_stats = {
                'iteration': i,
                'total_games': total_games,
                'requested_games': self.args.get('num_self_play_iterations', None),
                'p1_wins': game_outcomes['p1_wins'],
                'p_neg1_wins': game_outcomes['p_neg1_wins'],
                'draws': game_outcomes['draws'],
                'p1_win_rate': p1_win_rate,
                'p_neg1_win_rate': p_neg1_win_rate,
                'draw_rate': draw_rate,
                'avg_game_length': avg_game_length,
                'avg_game_time': avg_game_time,
                'avg_mcts_per_move': avg_mcts_per_move,
                'total_samples': len(memory),
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'total_loss': avg_total_loss,
                'learning_rate': current_lr,
                'selfplay_time': selfplay_time,
                'inference_time': total_inference_time,
                'training_time': training_time,
                'total_time': iteration_time,
                'captures_total': total_captures,
                'promotions_total': total_promotions,
                'multi_jumps_total': total_multi_jumps,
                'max_win_streak': max_win_streak,
                'max_win_streak_p1': max_streak_p1,
                'max_win_streak_pneg1': max_streak_pneg1,
                'total_iterations': self.args['num_iterations']
            }
            
            # Update training history
            # Compute gradient stats after training step(s)
            grad_norm = None
            max_grad = None
            try:
                total_norm_sq = 0.0
                max_g = 0.0
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    # L2 norm per-parameter (Euclidean)
                    param_norm = float(p.grad.data.norm(2).item())
                    total_norm_sq += param_norm ** 2
                    m = float(p.grad.data.abs().max())
                    if m > max_g:
                        max_g = m
                grad_norm = float(total_norm_sq ** 0.5)
                max_grad = float(max_g)
            except Exception:
                grad_norm = None
                max_grad = None

            # checkpoint events list (if any saved this iteration)
            checkpoints = iteration_checkpoints

            # Attach gradient stats into iteration_stats so the summary includes them
            iteration_stats['grad_norm'] = grad_norm
            iteration_stats['max_grad'] = max_grad

            self.metrics.update_history(
                i, avg_policy_loss, avg_value_loss, avg_total_loss, current_lr,
                iteration_stats, selfplay_time, training_time, iteration_time,
                inference_time=total_inference_time,
                epoch_losses=epoch_losses, epoch_timing=epoch_timing,
                grad_norm=grad_norm, max_grad=max_grad,
                checkpoints=checkpoints
            )
            
            # Log summary
            self.metrics.log_iteration_summary(i, iteration_stats)
            
            # Update progress bar
            main_pbar.set_postfix({
                'P1 Win%': f'{p1_win_rate:.1%}',
                'Draws%': f'{draw_rate:.1%}',
                'Policy Loss': f'{avg_policy_loss:.4f}',
                'Value Loss': f'{avg_value_loss:.4f}',
                'LR': f'{current_lr:.6f}',
                'Time': f'{iteration_time:.1f}s'
            })
            # Log selected metrics to MLflow at configured interval
            if self.use_mlflow and (i % self.mlflow_log_interval == 0):
                try:
                    metrics_to_log = {
                        'policy_loss': avg_policy_loss,
                        'value_loss': avg_value_loss,
                        'total_loss': avg_total_loss,
                        'p1_win_rate': p1_win_rate,
                        'p_neg1_win_rate': p_neg1_win_rate,
                        'draw_rate': draw_rate,
                        'avg_game_length': avg_game_length,
                        'total_samples': len(memory),
                        'learning_rate': current_lr,
                        # Timing breakdowns: useful to track performance over iterations
                        'selfplay_time': selfplay_time,
                        'inference_time': total_inference_time,
                        'training_time': training_time,
                        'total_time': iteration_time
                    }
                    try:
                        mlfl.log_metrics(metrics_to_log, step=i)
                    except Exception:
                        pass
                except Exception:
                    self.logger.warning('Failed to log metrics to MLflow')
            
            
        
        main_pbar.close()
        
        # Final training summary
        total_training_time = time.time() - overall_start_time
        self.logger.info(f"Training Complete! Total time: {total_training_time:.1f}s ({total_training_time/3600:.2f}h)")
        self.logger.info(f"Models saved to: {model_dir}")
        
        # Save final training history and checkpoint
        try:
            final_history = self.metrics.save_training_history(self.args)
            if self.use_mlflow and final_history and os.path.exists(final_history):
                try:
                    mlfl.log_artifact(final_history, artifact_path='training_history')
                except Exception:
                    self.logger.warning('Failed to log final training history to MLflow')
        except Exception:
            self.logger.exception('Failed to save final training history')
            self.checkpoint_manager.save_checkpoint(
                self.args['num_iterations'] - 1, checkpoints_dir, 
            self.metrics.training_history, self.args, self.timestamp, is_best=True,
            mlflow_run_id=self.mlflow_run_id
        )
        try:
            self._write_manifest(model_dir, update_fields={'checkpoint_latest': os.path.join(model_dir, 'checkpoint_latest.pt')})
        except Exception:
            pass
        try:
            # update manifest to point to checkpoint in checkpoints subfolder
            self._write_manifest(model_dir, update_fields={'checkpoint_latest': os.path.join(checkpoints_dir, 'checkpoint_latest.pt')})
        except Exception:
            pass
        # End MLflow run if active
        if self.use_mlflow:
            try:
                mlfl.end_run()
            except Exception:
                pass