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
from heapq import heappush, heappop
from model import PacmanNet
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
        self.last_move = None
        self.internal_map = None # Will store 0 for empty, 1 for wall, -1 for unknown
        self.map_initialized = False

    def _update_map_memory(self, map_state):
        """Merge current observation into internal memory map"""
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.map_initialized = True
        
        # Update visible cells: where map_state is not -1 (unseen)
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # 1. Update Memory
        self._update_map_memory(map_state)
        
        # 2. Track Enemy
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        
        # 3. Decision Making
        if target:
            # Hunter Mode: Path to enemy
            path = self.astar(my_position, target, self.internal_map)
        else:
            # Explore Mode: Tìm ô -1 (sương mù) gần nhất
            unknowns = np.argwhere(self.internal_map == -1)
            if len(unknowns) > 0:
                # Tìm ô -1 có khoảng cách Manhattan nhỏ nhất
                distances = np.sum(np.abs(unknowns - np.array(my_position)), axis=1)
                nearest_unknown = tuple(unknowns[np.argmin(distances)])
                path = self.astar(my_position, nearest_unknown, self.internal_map)
            else:
                # Nếu đã khám phá hết map mà ko thấy địch -> đi random hoặc tuần tra
                # Tạm thời đi về giữa map hoặc đi random valid move
                path = self.astar(my_position, (10, 10), self.internal_map)

        chosen_move = Move.STAY
        
        if path:
            chosen_move = path[0]
        else:
            # Fallback if A* fails (blocked or at target)
            valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                           if self._is_valid_move(my_position, m, self.internal_map)]
            if valid_moves:
                # Prefer keeping same direction if valid
                if self.last_move in valid_moves:
                    chosen_move = self.last_move
                else:
                    chosen_move = valid_moves[0]

        # [cite_start]4. Momentum Logic (Quan trọng: Xử lý bước đi 2 ô) [cite: 37, 39, 41]
        steps = 1
        
        # Chỉ kích hoạt đi 2 bước nếu thỏa mãn ĐỦ 3 điều kiện:
        # 1. Bot muốn đi (không đứng im)
        # 2. Hướng đi mới TRÙNG với hướng cũ (đang lấy đà)
        # 3. QUAN TRỌNG: Arena cho phép đi >= 2 bước (self.pacman_speed >= 2)
        if (chosen_move != Move.STAY and 
            chosen_move == self.last_move and 
            self.pacman_speed >= 2):
            
            # Kiểm tra xem bước thứ 2 có bị đâm đầu vào tường không
            if self._can_move_steps(my_position, chosen_move, self.internal_map, 2):
                steps = 2
            else:
                steps = 1
        else:
            steps = 1 # Trường hợp rẽ, hoặc Arena chỉ cho speed=1
        
        self.last_move = chosen_move
        return (chosen_move, steps)
    
    # Helper methods (you can add more)
    def _can_move_steps(self, pos, move, map_data, steps_to_check):
        """Check if we can move N steps in a direction without hitting wall"""
        r, c = pos
        dr, dc = move.value
        for i in range(1, steps_to_check + 1):
            nr, nc = r + dr * i, c + dc * i
            # Check bounds and walls
            if not (0 <= nr < map_data.shape[0] and 0 <= nc < map_data.shape[1]):
                return False
            if map_data[nr, nc] == 1: # Wall
                return False
        return True

    def _is_valid_move(self, pos, move, map_data):
        return self._can_move_steps(pos, move, map_data, 1)
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
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
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        # SỬA: Đi được nếu không phải là tường (1). Nghĩa là 0 và -1 đều đi được.
        return map_state[row, col] != 1
    
    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    
    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        """Get all valid neighboring positions and their moves."""
        neighbors = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def astar(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Find optimal path from start to goal using A* search.
        
        Returns:
            List of Move enums representing the path, or [] if no path
        """
        def heuristic(pos):
            """Manhattan distance heuristic."""
            return self._manhattan_distance(pos, goal)
        
        # Priority queue stores (f_cost, counter, position, path)
        # counter is for tiebreaking
        frontier = [(0, 0, start, [])]
        visited = set()
        counter = 0
        
        while frontier:
            f_cost, _, current_pos, path = heappop(frontier)
            
            # Found the goal!
            if current_pos == goal:
                return path
            
            # Skip if already visited
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            
            # Explore neighbors
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    g_cost = len(new_path)  # Cost so far
                    h_cost = heuristic(next_pos)  # Estimated cost to goal
                    f_cost = g_cost + h_cost  # Total estimated cost
                    counter += 1
                    heappush(frontier, (f_cost, counter, next_pos, new_path))
        
        # No path found
        return []


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY
        
        # Example: Simple evasive approach (replace with your algorithm)
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        
        # Try to move away from Pacman
        if abs(row_diff) > abs(col_diff):
            move = Move.DOWN if row_diff > 0 else Move.UP
        else:
            move = Move.RIGHT if col_diff > 0 else Move.LEFT
        
        # Check if move is valid
        if self._is_valid_move(my_position, move, map_state):
            return move
        
        # If not valid, try other moves
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return move
        
        return Move.STAY
    
    # Helper methods (you can add more)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
