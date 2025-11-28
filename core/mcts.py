# core/mcts.py

import numpy as np
import torch
import time


class Node:
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, depth=0):
        self.game = game
        self.args = args
        self.state = state
        self.player = player
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.depth = depth
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            return float('inf')
        
        q_value = child.value_sum / child.visit_count
        exploration = self.args['C'] * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
        ucb = q_value + exploration
        return ucb
        
    def expand(self, policy):
        children_added = []
        
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.game.get_next_state(self.state, action, self.player)
                
                if child_state.get('jump_again') is not None:
                    next_player = self.player
                else:
                    next_player = self.game.get_opponent(self.player)
                    
                child = Node(self.game, self.args, child_state, next_player,
                             parent=self, action_taken=action, prior=prob,
                             depth=self.depth + 1)
                self.children.append(child)
                children_added.append(child)
        
        return children_added[0] if children_added else None
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            if self.parent.player != self.player:
                parent_value = self.game.get_opponent_value(value)
            else:
                parent_value = value
            self.parent.backpropagate(parent_value)


class MCTS:
    def __init__(self, game, args, model, state_encoder):
        self.game = game
        self.args = args
        self.model = model
        self.state_encoder = state_encoder
    
    @torch.no_grad
    def search(self, state, player, temperature=1.0, add_noise=True, return_visit_counts=False):
        _, is_terminal = self.game.get_value_and_terminated(state, player)
        if is_terminal:
            return np.zeros(self.game.action_size)
        
        root = Node(self.game, self.args, state, player)
        root.visit_count = 1  # Set visit count after creation
        
        # Get initial policy and value from neural network (measure inference time)
        model_time = 0.0
        start = time.time()
        policy, value = self.model(
            torch.tensor(self.state_encoder.encode_state(state, player), device=self.model.device).unsqueeze(0)
        )
        model_time += time.time() - start
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        # Add Dirichlet noise if requested
        if add_noise:
            de = self.args['dirichlet_epsilon']
            da = self.args['dirichlet_alpha']
            policy = (1 - de) * policy + de * np.random.dirichlet([da] * self.game.action_size)
        
        # Mask invalid moves
        valid_moves = self.game.get_valid_moves(state, player)
        policy *= valid_moves
        
        if np.sum(policy) == 0:
            return np.zeros(self.game.action_size)
        
        policy /= np.sum(policy)
        root.expand(policy)
        
        # Perform MCTS simulations
        for _ in range(self.args['num_searches']):
            node = root
            
            # Selection
            while node.is_fully_expanded() and len(node.children) > 0:
                node = node.select()
                
            # Check if leaf is terminal
            _, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
            
            if not is_terminal:
                # Evaluation (measure inference time)
                start = time.time()
                policy, value = self.model(
                    torch.tensor(self.state_encoder.encode_state(node.state, node.player), 
                               device=self.model.device).unsqueeze(0)
                )
                model_time += time.time() - start
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state, node.player)
                policy *= valid_moves
                
                if np.sum(policy) > 0:
                    policy /= np.sum(policy)
                    value = value.item()
                    
                    if not node.is_fully_expanded():
                        node.expand(policy)
                else:
                    value = 0
                    
            # Backpropagation
            node.backpropagate(value)
        
        # Generate action probabilities with temperature
        action_probs = np.zeros(self.game.action_size)
        visit_counts = np.zeros(self.game.action_size)
        
        for child in root.children:
            visit_counts[child.action_taken] = child.visit_count
        
        if np.sum(visit_counts) > 0:
            if temperature == 0:
                best_action = np.argmax(visit_counts)
                action_probs[best_action] = 1.0
            else:
                visit_counts = visit_counts.astype(np.float64)
                if temperature != 1.0:
                    visit_counts = np.power(visit_counts, 1.0 / temperature)
                
                if np.sum(visit_counts) > 0:
                    action_probs = visit_counts / np.sum(visit_counts)
        
        # Store inference time for external inspection
        self._last_search_inference_time = float(model_time)

        if return_visit_counts:
            return action_probs, visit_counts
        return action_probs

    def get_last_search_inference_time(self):
        """Return the wall-clock seconds spent in model forward calls during the last search."""
        return getattr(self, '_last_search_inference_time', 0.0)