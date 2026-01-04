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
    Pacman Hybrid Agent:
    - Deep Learning (DQN) để ra quyết định chiến lược.
    - A* Interception (Đón đầu) làm phương án dự phòng (Fallback).
    - Hỗ trợ Speed 2 Momentum.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Pacman AI + A*"
        
        # --- 1. SETUP MEMORY ---
        self.internal_map = None 
        self.map_initialized = False
        self.last_known_enemy_pos = None
        self.last_move = Move.STAY
        
        # --- 2. SETUP MACHINE LEARNING (PYTORCH) ---
        self.device = torch.device("cpu") # Bắt buộc dùng CPU cho bài nộp
        self.model = None
        
        # Thử load model nếu class PacmanNet tồn tại (bạn cần đảm bảo đã import class này)
        # Nếu chưa import PacmanNet thì đoạn này sẽ bị try/except bỏ qua
        try:
            # from your_model_file import PacmanNet # <-- Bỏ comment nếu cần import
            if 'PacmanNet' in globals():
                self.model = PacmanNet()
                
                # Tìm file weights
                current_dir = Path(__file__).parent
                model_path = current_dir / "pacman_smart_ghost.pt" # Ưu tiên
                if not model_path.exists():
                    model_path = current_dir / "pacman_dqn.pt"     # Dự phòng
                
                if model_path.exists():
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    print(f"✅ Đã load AI Model: {model_path.name}")
                else:
                    print("⚠️ Không tìm thấy file .pt! Sẽ chạy bằng A* thuần.")
                    self.model = None
            else:
                print("⚠️ Class PacmanNet chưa được định nghĩa/import. Chạy A*.")
        except Exception as e:
            print(f"❌ Lỗi load Model: {e}")
            self.model = None

    def _update_map_memory(self, map_state):
        if not self.map_initialized:
            self.internal_map = np.full_like(map_state, -1)
            self.map_initialized = True
        visible_mask = map_state != -1
        self.internal_map[visible_mask] = map_state[visible_mask]

    # --- PHẦN AI (MACHINE LEARNING) ---
    def get_ml_action(self, map_data, my_pos, enemy_pos):
        if self.model is None: return Move.STAY
        try:
            # 1. Tiền xử lý Map: Thay -1 (mù) bằng 0 (trống) hoặc 1 (tường) tùy cách bạn train
            input_map = map_data.copy()
            input_map[input_map == -1] = 0 
            
            # Đánh dấu vị trí trên map (nếu model của bạn cần feature này)
            # Lưu ý: Logic này phụ thuộc vào cách bạn design mạng DQN
            if my_pos: input_map[my_pos] = 2
            if enemy_pos: input_map[enemy_pos] = 3
            
            # 2. Chuyển sang Tensor
            state_tensor = torch.FloatTensor(input_map).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 3. Vector nước đi trước (Last move)
            move_idx = -1
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            if self.last_move in all_moves:
                move_idx = all_moves.index(self.last_move)
            
            last_move_vec = torch.zeros(1, 4).to(self.device)
            if move_idx >= 0: last_move_vec[0, move_idx] = 1.0
                
            # 4. AI Suy nghĩ
            with torch.no_grad():
                # Giả định hàm forward nhận (map, last_move)
                q_values = self.model(state_tensor, last_move_vec)
                action_idx = torch.argmax(q_values).item()
            
            predicted_move = all_moves[action_idx]
            
            # 5. Kiểm tra an toàn cơ bản (Safety Check)
            if self._can_move_steps(my_pos, predicted_move, map_data, 1):
                return predicted_move
            return Move.STAY
            
        except Exception as e:
            # print(f"ML Error: {e}")
            return Move.STAY

    # --- GAME LOOP CHÍNH ---
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        self._update_map_memory(map_state)
        
        # Xác định mục tiêu
        target = None
        if enemy_position:
            self.last_known_enemy_pos = enemy_position
            target = enemy_position
        else:
            target = self.last_known_enemy_pos

        chosen_move = Move.STAY
        
        # --- BƯỚC 1: THỬ DÙNG AI (MACHINE LEARNING) ---
        if self.model and target:
            # Dùng AI để đoán nước đi
            chosen_move = self.get_ml_action(self.internal_map, my_position, target)

        # --- BƯỚC 2: FALLBACK SANG A* (NẾU AI FAIL HOẶC MUỐN ĐỨNG IM) ---
        # Nếu AI trả về STAY hoặc không có AI, ta dùng thuật toán A* Interception
        if chosen_move == Move.STAY:
            # A. Tính vị trí đón đầu (Interception)
            aim_pos = target
            if enemy_position and self.last_known_enemy_pos and enemy_position != self.last_known_enemy_pos:
                 # Logic đơn giản: Nếu Ghost vừa di chuyển, dự đoán nó đi tiếp hướng đó
                 d_r = enemy_position[0] - self.last_known_enemy_pos[0]
                 d_c = enemy_position[1] - self.last_known_enemy_pos[1]
                 pred_pos = (enemy_position[0] + d_r, enemy_position[1] + d_c)
                 if self._is_valid_position(pred_pos, self.internal_map):
                     aim_pos = pred_pos

            # B. Tìm đường A*
            path = []
            if aim_pos:
                path = self.astar(my_position, aim_pos, self.internal_map)
            
            # C. Nếu không có đường tới Ghost, đi khám phá (Explore)
            if not path:
                path = self.explore_strategy(my_position)
            
            if path:
                chosen_move = path[0]

        # --- BƯỚC 3: MOMENTUM (SPEED 2 CHECK) ---
        # Kiểm tra xem có thể phóng 2 bước với nước đi đã chọn không
        steps_to_take = 1
        if chosen_move != Move.STAY and self.pacman_speed >= 2:
            # Nếu AI chọn đi lên, và ô tiếp theo nữa cũng trống -> Đi 2 bước
            if self._can_move_steps(my_position, chosen_move, self.internal_map, 2):
                steps_to_take = 2
                
        self.last_move = chosen_move
        return chosen_move, steps_to_take

    # --- CÁC HÀM THUẬT TOÁN (A*, HELPER) ---
    def explore_strategy(self, my_pos):
        unknowns = np.argwhere(self.internal_map == -1)
        if len(unknowns) > 0:
            dists = np.sum(np.abs(unknowns - np.array(my_pos)), axis=1)
            nearest_unknown = tuple(unknowns[np.argmin(dists)])
            return self.astar(my_pos, nearest_unknown, self.internal_map)
        
        # Fallback Random nếu map sáng hết
        valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                       if self._can_move_steps(my_pos, m, self.internal_map, 1)]
        if valid_moves: return [random.choice(valid_moves)]
        return []

    def astar(self, start, goal, map_state):
        if start == goal: return []
        frontier = []
        heappush(frontier, (0, 0, start, []))
        visited = {start}
        while frontier:
            _, _, current, path = heappop(frontier)
            if current == goal: return path
            if len(path) > 40: continue # Cắt ngắn nếu quá xa
            
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    # Heuristic Manhattan
                    priority = len(path) + 1 + abs(next_pos[0]-goal[0]) + abs(next_pos[1]-goal[1])
                    heappush(frontier, (priority, len(path)+1, next_pos, path + [move]))
        return []

    def _get_neighbors(self, pos, map_state):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                if map_state[nr, nc] != 1: 
                    neighbors.append(((nr, nc), move))
        return neighbors

    def _can_move_steps(self, pos, move, map_data, steps_to_check):
        r, c = pos
        dr, dc = move.value
        for i in range(1, steps_to_check + 1):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < map_data.shape[0] and 0 <= nc < map_data.shape[1]): return False
            if map_data[nr, nc] == 1: return False
        return True

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return (0 <= r < map_state.shape[0] and 0 <= c < map_state.shape[1] and map_state[r, c] != 1)

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
        
