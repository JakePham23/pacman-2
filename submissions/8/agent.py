"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path
from collections import deque
import random
import math
import time

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
        
        # Example: Simple greedy approach (replace with your algorithm)
        row_diff = target[0] - my_position[0]
        col_diff = target[1] - my_position[1]
        
        # Try to move towards ghost
        if abs(row_diff) > abs(col_diff):
            primary_move = Move.DOWN if row_diff > 0 else Move.UP
            desired_steps = abs(row_diff)
        else:
            primary_move = Move.RIGHT if col_diff > 0 else Move.LEFT
            desired_steps = abs(col_diff)

        action = self._choose_action(
            my_position,
            [primary_move],
            map_state,
            desired_steps
        )
        if action:
            return action

        # If the primary direction is blocked, try other moves
        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
        if action:
            return action
        
        return (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


class GhostAgent(BaseGhostAgent):
    """
    Ghost Ninja Pro (Updated):
    - Hỗ trợ tính toán Pacman Speed 2 chính xác.
    - Dùng Iterative Deepening để kiểm soát thời gian.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.survival_target = None
        self.time_start = 0 
        self.time_limit = 0.9 

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self.time_start = time.time()
        
        # 1. Cập nhật thông tin
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
            self.survival_target = None
        else:
            self.steps_since_seen += 1

        target_enemy = enemy_position or self.last_known_enemy_pos

        # --- CHIẾN THUẬT A: ĐI TUẦN ---
        if target_enemy is None or self.steps_since_seen > 5:
            if not self.survival_target or my_position == self.survival_target:
                 self.survival_target = self.find_nearest_intersection(my_position, map_state)
            path = self.bfs_find_path(my_position, self.survival_target, map_state)
            if path: return path[0]
            return self.get_random_valid_move(my_position, map_state)

        # --- CHIẾN THUẬT B: ITERATIVE DEEPENING MINIMAX ---
        else:
            best_move = Move.STAY
            # Pacman Speed 2 tính toán rất nặng, nên depth có thể chỉ lên được 4-6
            for depth in range(1, 20): 
                try:
                    if time.time() - self.time_start > self.time_limit: break
                    
                    _, move = self.minimax(
                        my_pos=my_position, 
                        enemy_pos=target_enemy, 
                        depth=depth, 
                        is_maximizing=True, 
                        map_state=map_state
                    )
                    
                    if time.time() - self.time_start < self.time_limit:
                        best_move = move
                    else:
                        break 
                except TimeoutError:
                    break
            
            return best_move

    # =========================================================================
    # MINIMAX VỚI MÔ PHỎNG PACMAN SPEED 2
    # =========================================================================

    def minimax(self, my_pos, enemy_pos, depth, is_maximizing, map_state):
        if time.time() - self.time_start > self.time_limit: raise TimeoutError()

        # 1. Điều kiện dừng
        if depth == 0 or my_pos == enemy_pos:
            return self.evaluate_ninja_state(my_pos, enemy_pos, map_state), Move.STAY

        valid_moves = self.get_valid_moves_with_pos(my_pos, map_state)
        if not valid_moves: return -1000, Move.STAY

        best_move = valid_moves[0][1]

        if is_maximizing: # Lượt GHOST
            max_eval = -math.inf
            for next_pos, move in valid_moves:
                # Pacman cũng sẽ di chuyển dựa trên vị trí mới của Ghost
                eval_score, _ = self.minimax(next_pos, enemy_pos, depth - 1, False, map_state)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move

        else: # Lượt PACMAN (Mô phỏng 2 bước)
            min_eval = math.inf
            
            # --- QUAN TRỌNG: Lấy tất cả các ô Pacman tới được trong 2 bước ---
            pacman_reachable = self.get_pacman_reachable_positions(enemy_pos, 2, map_state)
            
            # Nếu Pacman có thể bắt được Ghost trong lượt này -> Game Over ngay
            if my_pos in pacman_reachable:
                return -10000, None

            for next_enemy_pos in pacman_reachable:
                # Ghost đi tiếp dựa trên vị trí mới của Pacman
                eval_score, _ = self.minimax(my_pos, next_enemy_pos, depth - 1, True, map_state)
                if eval_score < min_eval:
                    min_eval = eval_score
            return min_eval, None

    # =========================================================================
    # HÀM MỚI: TÌM CÁC Ô PACMAN TỚI ĐƯỢC (SPEED N)
    # =========================================================================
    
    def get_pacman_reachable_positions(self, start_pos, speed, map_state):
        """
        Trả về tập hợp tất cả các vị trí mà Pacman có thể đứng sau 'speed' bước đi.
        Dùng BFS giới hạn độ sâu.
        """
        reachable = set()
        queue = deque([(start_pos, 0)]) # (vị trí, số bước đã đi)
        
        # Để tối ưu, ta có thể dùng set visited cho từng tầng, nhưng ở đây
        # với speed=2 thì số lượng ô không nhiều (tối đa 13 ô), nên ta duyệt hết.
        
        while queue:
            curr, steps = queue.popleft()
            
            # Nếu đã đi hết số bước cho phép -> Thêm vào danh sách đích
            if steps == speed:
                reachable.add(curr)
                continue
            
            # Nếu chưa hết bước, vẫn thêm vào danh sách (vì Pacman có thể dừng lại hoặc đi chưa hết tốc lực)
            # Tuy nhiên, giả định Pacman luôn đi max tốc độ để bắt mình thì:
            # reachable.add(curr) <--- Có thể bỏ dòng này nếu Pacman buộc phải đi đủ 2 bước
            
            # Tìm các ô tiếp theo
            moves = self.get_valid_moves_with_pos(curr, map_state)
            
            # Nếu bị kẹt (không đi được nữa) dù chưa hết speed -> Coi như dừng tại đây
            if not moves:
                reachable.add(curr)
            
            for next_pos, _ in moves:
                # Lưu ý: Không dùng visited toàn cục để cho phép Pacman đi qua đi lại (nếu cần thiết)
                # Nhưng để tránh vòng lặp vô tận thì speed giới hạn đã lo rồi.
                queue.append((next_pos, steps + 1))
                
        # Lọc lại: Với Minimax, ta chỉ quan tâm các vị trí CUỐI CÙNG mà Pacman có thể đứng.
        return reachable

    # --- CÁC HÀM CŨ GIỮ NGUYÊN ---
    def evaluate_ninja_state(self, my_pos, enemy_pos, map_state):
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        if dist == 0: return -10000
        score = dist * 10 
        is_visible = self.check_line_of_sight(my_pos, enemy_pos, map_state)
        if not is_visible:
            score += 500
            if my_pos[0] != enemy_pos[0] and my_pos[1] != enemy_pos[1]: score += 50
        else:
            score -= 200
        escapes = len(self.get_valid_moves_with_pos(my_pos, map_state))
        if escapes <= 1: score -= 300
        return score

    def check_line_of_sight(self, pos1, pos2, map_state):
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 != r2 and c1 != c2: return False
        if r1 == r2:
            start, end = min(c1, c2), max(c1, c2)
            for c in range(start + 1, end):
                if map_state[r1, c] == 1: return False
            return True
        if c1 == c2:
            start, end = min(r1, r2), max(r1, r2)
            for r in range(start + 1, end):
                if map_state[r, c1] == 1: return False
            return True
        return False
        
    def find_nearest_intersection(self, start_pos, map_state):
        queue = deque([start_pos])
        visited = {start_pos}
        while queue:
            curr = queue.popleft()
            moves = self.get_valid_moves_with_pos(curr, map_state)
            if len(moves) >= 3 and curr != start_pos: return curr
            for next_pos, _ in moves:
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return start_pos

    def bfs_find_path(self, start, end, map_state):
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            curr, path = queue.popleft()
            if curr == end: return path
            for next_pos, move in self.get_valid_moves_with_pos(curr, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return []

    def get_valid_moves_with_pos(self, pos, map_state):
        valid = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                valid.append(((nr, nc), move))
        return valid

    def get_random_valid_move(self, pos, map_state):
        moves = self.get_valid_moves_with_pos(pos, map_state)
        if moves: return random.choice(moves)[1]
        return Move.STAY