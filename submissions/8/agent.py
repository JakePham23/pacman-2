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
import random
import torch
import math
import time
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except ImportError:
    # Fallback for local testing if src is not reachable
    class BasePacmanAgent: 
        def __init__(self, **kwargs): pass
    class BaseGhostAgent: 
        def __init__(self, **kwargs): pass
    class Move:
        UP = ( -1, 0)
        DOWN = ( 1, 0)
        LEFT = ( 0, -1)
        RIGHT = ( 0, 1)
        STAY = ( 0, 0)
        @property
        def value(self): return self

try:
    # Try to import from the same directory
    from model import PacmanNet
except ImportError:
    # Fallback if model.py is not found
    PacmanNet = None
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
        self.last_seen_map = None #Lưu step cuối cùng nhìn thấy ô đó
        # --- LOAD MODEL ML ---
        self.device = torch.device("cpu") # Nộp bài bắt buộc dùng CPU
        self.model = None
        if PacmanNet:
            try:
                self.model = PacmanNet()
                # Tự động tìm file .pt trong cùng thư mục
                current_dir = Path(__file__).parent
                # Ưu tiên file smart ghost, nếu không có thì tìm file dqn thường
                model_path = current_dir / "pacman_smart_ghost.pt"
                if not model_path.exists():
                    model_path = current_dir / "pacman_dqn.pt"
                
                if model_path.exists():
                    # Load weights (map_location='cpu' để an toàn)
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    # print(f"Model loaded: {model_path.name}")
                else:
                    print("Warning: No .pt file found! Pacman will use A* only.")
                    self.model = None
            except Exception as e:
                print(f"Model load error: {e}")
                self.model = None

    def _update_map_memory(self, map_state, step_number):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.map_initialized = True
            self.last_seen_map = np.full_like(map_state, -1) # Khởi tạo bảng thời gian
        
        # Update visible cells: where map_state is not -1 (unseen)
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]

        # Ghi số bước (step_number) hiện tại vào những ô đang nhìn thấy
        self.last_seen_map[visible_mask] = step_number

    def find_frontier(self, my_pos):
        """
        Tìm Frontier (Đường biên giới):
        Là các ô đã biết là ĐƯỜNG ĐI (0) nhưng nằm cạnh vùng CHƯA BIẾT (-1).
        Mục tiêu: Đi đến đây để mở rộng tầm nhìn an toàn.
        """
        # Nếu chưa có bản đồ thì chịu, không tìm được
        if self.internal_map is None: 
            return None
        
        rows, cols = self.internal_map.shape
        
        # 1. Lọc ra tất cả các ô đang là ĐƯỜNG ĐI (0) trong bộ nhớ
        empty_cells = np.argwhere(self.internal_map == 0) #danh sách toạ độ tất cả các ô đường đi (0) mà Pacman đã biết.
        
        frontiers = []

        # 2. Duyệt qua từng ô đường đi để xem nó có phải là "Biên giới" không
        for r, c in empty_cells:
            is_frontier = False
            # Kiểm tra 4 ô xung quanh (Lên, Xuống, Trái, Phải)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                
                # Nếu hàng xóm nằm trong phạm vi bản đồ
                if 0 <= nr < rows and 0 <= nc < cols:
                    # VÀ hàng xóm là SƯƠNG MÙ (-1)
                    if self.internal_map[nr, nc] == -1:
                        is_frontier = True
                        break # Chỉ cần 1 hướng có sương mù là đủ điều kiện
            
            if is_frontier:
                frontiers.append((r, c))
        
        # Nếu không tìm thấy biên giới nào (tức là map đã sáng hết 100%)
        if not frontiers:
            return None

        # 3. Sắp xếp các điểm biên giới theo khoảng cách gần Pacman nhất
        # Để Pacman ưu tiên khám phá vùng gần trước, đỡ chạy lòng vòng
        frontiers.sort(key=lambda pos: self._manhattan_distance(my_pos, tuple(pos)))
        
        # Trả về tọa độ của biên giới gần nhất (dạng tuple)
        return tuple(frontiers[0])
        
    def get_ml_action(self, map_data, my_pos, enemy_pos):
        """Chạy Model để lấy nước đi tốt nhất"""
        try:
            # 1. Preprocess Map (Giống lúc train)
            # Map lúc train là 0=Empty, 1=Wall, 2=Pacman, 3=Ghost. Không có -1.
            # Nên ta thay -1 (unseen) thành 0 (coi như đi được) hoặc 1 (tường) để model không bị loạn.
            # An toàn nhất: Coi unseen là tường để tránh đâm bậy, hoặc empty để dũng cảm.
            # Chọn: Replace -1 -> 0 (Optimistic)
            input_map = map_data.copy()
            input_map[input_map == -1] = 0 
            
            # Đánh dấu vị trí
            if my_pos: input_map[my_pos] = 2
            if enemy_pos: input_map[enemy_pos] = 3
            
            # 2. Tensor conversion
            state_tensor = torch.FloatTensor(input_map).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 3. Last move vector
            move_idx = -1
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            if self.last_move in all_moves:
                move_idx = all_moves.index(self.last_move)
            
            last_move_vec = torch.zeros(1, 4).to(self.device)
            if move_idx >= 0:
                last_move_vec[0, move_idx] = 1.0
                
            # 4. Predict
            with torch.no_grad():
                q_values = self.model(state_tensor, last_move_vec)
                action_idx = torch.argmax(q_values).item()
                
            predicted_move = all_moves[action_idx]
            
            reverse_map = {Move.UP: Move.DOWN, Move.DOWN: Move.UP, 
                           Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}
            
            # Nếu nước đi mới là quay ngược lại hướng cũ -> Bỏ qua, để cho A* xử lý
            if predicted_move == reverse_map.get(self.last_move, None):
                return Move.STAY # Trả về STAY để kích hoạt fallback A* (A* khôn hơn trong việc gỡ rối)
            # --------------------------------------------------

            # 5. Safety Check
            if self._can_move_steps(my_pos, predicted_move, map_data, 1):
                return predicted_move
            else:
                return Move.STAY
                
        except Exception:
            return Move.STAY
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self.step_count = step_number
        self._update_map_memory(map_state,step_number)
        
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
        
        target = enemy_position or self.last_known_enemy_pos
        
        chosen_move = Move.STAY
        use_ml = False

        # --- CHIẾN THUẬT QUYẾT ĐỊNH ---
        # 1. Nếu thấy địch (hoặc biết vị trí gần đây) -> Dùng ML để săn (Aggressive)
        if target and self.model is not None:
            # Chỉ dùng ML nếu địch đang hiển thị trên bản đồ (để input chính xác)
            # Hoặc nếu ta giả lập vị trí địch vào bản đồ memory
            chosen_move = self.get_ml_action(self.internal_map, my_position, target)
            use_ml = True
            
        # 2. Nếu không thấy địch hoặc ML fail -> Dùng A* để đi tuần tra/khám phá
        if not use_ml or chosen_move == Move.STAY:
            # Dùng hàm find_frontier thay vì tìm sương mù random
            frontier = self.find_frontier(my_position)
            
            if frontier:
                # Nếu tìm thấy biên giới -> Đi đến đó để mở map
                path = self.astar(my_position, frontier, self.internal_map)
                if path: chosen_move = path[0]
            else:
                # CHIẾN THUẬT TUẦN TRA (PATROL)
                # Map sáng hết mà ko thấy địch -> Đi đến nơi "cũ nhất" (step bé nhất trong last_seen_map)
                
                walkable_mask = (self.internal_map == 0)
                if np.any(walkable_mask):
                    # Tìm timestamp nhỏ nhất
                    min_step = np.min(self.last_seen_map[walkable_mask])
                    # Lấy danh sách các ô có timestamp đó
                    oldest_places = np.argwhere((self.last_seen_map == min_step) & walkable_mask)
                    
                    if len(oldest_places) > 0:
                        # Chọn ngẫu nhiên 1 điểm cũ nhất
                        import random
                        idx = random.randint(0, len(oldest_places) - 1)
                        patrol_target = tuple(oldest_places[idx])
                        path = self.astar(my_position, patrol_target, self.internal_map)
                        if path: chosen_move = path[0]
                
                # Fallback Random (Phòng khi kẹt)
                if chosen_move == Move.STAY:
                    valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                                   if self._can_move_steps(my_position, m, self.internal_map, 1)]
                    if valid_moves:
                        if self.last_move in valid_moves:
                            chosen_move = self.last_move
                        else:
                            import random
                            chosen_move = random.choice(valid_moves)

        # --- MOMENTUM LOGIC (SPEED 2) ---
        steps = 1
        if chosen_move != Move.STAY:
            if (chosen_move == self.last_move and self.pacman_speed >= 2):
                if self._can_move_steps(my_position, chosen_move, self.internal_map, 2):
                    steps = 2
                else:
                    steps = 1
            else:
                steps = 1
        
        self.last_move = chosen_move
        return (chosen_move, steps)
    
    # Helper methods (you can add more)
    def _can_move_steps(self, pos, move, map_data, steps_to_check):
        r, c = pos
        dr, dc = move.value
        for i in range(1, steps_to_check + 1):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < map_data.shape[0] and 0 <= nc < map_data.shape[1]):
                return False
            if map_data[nr, nc] == 1: 
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
    
    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_pos = (pos[0] + dr, pos[1] + dc)
            # A* đi được vào ô 0 và -1
            if (0 <= next_pos[0] < map_state.shape[0] and 
                0 <= next_pos[1] < map_state.shape[1] and 
                map_state[next_pos] != 1):
                neighbors.append((next_pos, move))
        return neighbors
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def astar(self, start, goal, map_state):
        if start == goal: return []
        frontier = [(0, 0, start, [])]
        visited = set()
        counter = 0
        while frontier:
            _, _, current, path = heappop(frontier)
            if current == goal: return path
            if current in visited: continue
            visited.add(current)
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    new_path = path + [move]
                    g = len(new_path)
                    h = self._manhattan_distance(next_pos, goal)
                    f = g + h
                    counter += 1
                    heappush(frontier, (f, counter, next_pos, new_path))
        return []


class GhostAgent(BaseGhostAgent):
    """
    Ghost Ninja Pro Max (Hardcore):
    - Hỗ trợ --capture-distance 1 (Bị bắt khi đứng cạnh).
    - Hỗ trợ --pacman-speed 2.
    - Alpha-Beta Pruning & Dead-end Avoidance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_known_enemy_pos = None
        self.steps_since_seen = 0
        self.survival_target = None
        
        # --- CẤU HÌNH QUAN TRỌNG TỪ ARENA ---
        self.capture_dist = 1  # Khoảng cách bị bắt (sửa thành 0 nếu luật đổi)
        self.time_limit = 0.85 # Giới hạn thời gian an toàn
        self.time_start = 0 

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self.time_start = time.time()
        
        # 1. Cập nhật thông tin kẻ địch
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
            self.survival_target = None
        else:
            self.steps_since_seen += 1

        target_enemy = enemy_position or self.last_known_enemy_pos

        # --- CHIẾN THUẬT A: ĐI TUẦN (Mất dấu > 7 bước hoặc chưa từng thấy) ---
        if target_enemy is None or self.steps_since_seen > 7:
            if not self.survival_target or my_position == self.survival_target:
                 self.survival_target = self.find_nearest_intersection(my_position, map_state)
            
            path = self.bfs_find_path(my_position, self.survival_target, map_state)
            if path: return path[0]
            return self.get_random_valid_move(my_position, map_state)

        # --- CHIẾN THUẬT B: ALPHA-BETA MINIMAX ---
        else:
            best_move = Move.STAY
            # Iterative Deepening: Tăng dần độ sâu tìm kiếm
            for depth in range(1, 20): 
                try:
                    if time.time() - self.time_start > self.time_limit: break
                    
                    _, move = self.minimax(
                        my_pos=my_position, 
                        enemy_pos=target_enemy, 
                        depth=depth, 
                        alpha=-math.inf,
                        beta=math.inf,
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
    # MINIMAX (SỬA ĐỔI CHO CAPTURE DISTANCE 1)
    # =========================================================================

    def minimax(self, my_pos, enemy_pos, depth, alpha, beta, is_maximizing, map_state):
        if time.time() - self.time_start > self.time_limit: raise TimeoutError()

        # 1. Kiểm tra điều kiện thua ngay lập tức (Capture Distance Logic)
        curr_dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        if curr_dist <= self.capture_dist:
            return -100000, None # Bị bắt (kể cả đứng cạnh cũng chết)

        # 2. Hết độ sâu tìm kiếm
        if depth == 0:
            return self.evaluate_ninja_state(my_pos, enemy_pos, map_state), Move.STAY

        # 3. Lượt GHOST (Maximizing)
        if is_maximizing:
            valid_moves = self.get_valid_moves_with_pos(my_pos, map_state)
            # Heuristic sort: Ưu tiên ô xa địch
            valid_moves.sort(key=lambda x: abs(x[0][0]-enemy_pos[0]) + abs(x[0][1]-enemy_pos[1]), reverse=True)

            if not valid_moves: return -100000, Move.STAY # Kẹt -> Chết

            max_eval = -math.inf
            best_move = valid_moves[0][1]

            for next_pos, move in valid_moves:
                # Nếu đi vào ô mà khoảng cách tới địch <= capture_dist -> Tự sát -> Bỏ qua hoặc điểm thấp
                if abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1]) <= self.capture_dist:
                     eval_score = -90000 # Điểm phạt tự sát
                else:
                    eval_score, _ = self.minimax(next_pos, enemy_pos, depth - 1, alpha, beta, False, map_state)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha: break 

            return max_eval, best_move

        # 4. Lượt PACMAN (Minimizing - Speed 2)
        else:
            min_eval = math.inf
            
            # Pacman đi 2 bước
            pacman_reachable = self.get_pacman_reachable_positions(enemy_pos, 2, map_state)
            
            # Kiểm tra xem Pacman có thể nhảy tới vùng nguy hiểm không
            # Vùng nguy hiểm: Bất kỳ ô nào cách Ghost <= capture_dist
            for p_pos in pacman_reachable:
                dist_check = abs(my_pos[0] - p_pos[0]) + abs(my_pos[1] - p_pos[1])
                if dist_check <= self.capture_dist:
                    return -100000, None # Pacman bắt được Ghost

            for next_enemy_pos in pacman_reachable:
                eval_score, _ = self.minimax(my_pos, next_enemy_pos, depth - 1, alpha, beta, True, map_state)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                
                beta = min(beta, eval_score)
                if beta <= alpha: break 

            return min_eval, None

    # =========================================================================
    # HÀM ĐÁNH GIÁ (UPDATED FOR CAPTURE DIST 1)
    # =========================================================================

    def evaluate_ninja_state(self, my_pos, enemy_pos, map_state):
        dist = abs(my_pos[0] - enemy_pos[0]) + abs(my_pos[1] - enemy_pos[1])
        
        # Nếu khoảng cách <= 1, coi như đã chết (đã check ở trên nhưng thêm cho chắc)
        if dist <= self.capture_dist: return -100000
        
        # Nếu khoảng cách <= 3 (Pacman đi 2 bước + 1 bước bắt), cực kỳ nguy hiểm
        # Pacman Speed 2 có tầm với thực tế là: 2 bước di chuyển + 1 bước bắt = 3 ô
        if dist <= self.capture_dist + 2: 
            return -5000 + (dist * 100) # Vẫn ưu tiên xa hơn một chút trong vùng tử thần

        score = dist * 10 

        # Line of Sight
        is_visible = self.check_line_of_sight(my_pos, enemy_pos, map_state)
        if not is_visible:
            score += 600 # Tăng thưởng ẩn nấp vì game khó hơn
            if my_pos[0] != enemy_pos[0] and my_pos[1] != enemy_pos[1]: 
                score += 100
        else:
            score -= 300

        # Dead-end Check
        if self.is_dead_end(my_pos, map_state):
            score -= 3000 # Phạt nặng hơn
        
        return score

    # =========================================================================
    # CÁC HÀM BỔ TRỢ (GIỮ NGUYÊN)
    # =========================================================================

    def is_dead_end(self, pos, map_state):
        moves = self.get_valid_moves_with_pos(pos, map_state)
        return len(moves) <= 1

    def get_pacman_reachable_positions(self, start_pos, speed, map_state):
        reachable = set()
        queue = deque([(start_pos, 0)]) 
        while queue:
            curr, steps = queue.popleft()
            if steps == speed:
                reachable.add(curr)
                continue
            
            # Giả định Pacman thông minh, hắn sẽ không đứng lại nếu chưa bắt được
            # Trừ khi việc đứng lại giúp hắn bắt được (đã check ở logic dist <= 1)
            # reachable.add(curr) 
            
            moves = self.get_valid_moves_with_pos(curr, map_state)
            if not moves: reachable.add(curr)
            for next_pos, _ in moves:
                queue.append((next_pos, steps + 1))
        return list(reachable)

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
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                if map_state[nr, nc] != 1:
                    valid.append(((nr, nc), move))
        return valid

    def get_random_valid_move(self, pos, map_state):
        moves = self.get_valid_moves_with_pos(pos, map_state)
        if moves: return random.choice(moves)[1]
        return Move.STAY
        
