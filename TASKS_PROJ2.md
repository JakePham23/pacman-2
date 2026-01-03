# Group 8
**Mục tiêu:** Đạt thứ hạng cao trong Tournament với cấu hình: Limited Vision (Tầm nhìn 5 ô - theo mục 1.1), Pacman Speed 2, Capture Distance 1 (Bắt được Ghost khi ở cùng một vị trí).

---

## 🏃 1. Lead Architect (Memory & Exploration)
- [ ] **Hoàn thiện bản đồ vĩnh viễn (Fixed Map Memory):** 
    - Sửa hàm `_update_map_memory` để lưu vết tường (wall) vĩnh viễn. 
    - *Logic:* `if old_val in [0, 1] and new_val == -1: continue`. Chỉ ghi đè `-1` bằng thông tin mới, không làm ngược lại.
- [ ] **Thuật toán Khám phá (Frontier Search):** 
    - Viết hàm `find_frontier()`: Tìm các ô trống (`0`) nhưng có ít nhất một hướng hàng xóm là sương mù (`-1`).
    - Khi không thấy Ghost, Pacman sẽ dùng A* đi đến Frontier gần nhất để mở bản đồ thay vì đi random.
- [ ] **Quản lý vùng mù (Timestamping):** 
    - Lưu thêm thông tin `last_seen_step` cho từng ô. Nếu một vùng quá lâu chưa được quét lại, hãy đặt ưu tiên cao để Pacman quay lại kiểm tra (vì Ghost có thể đang nấp ở đó).

---

## 🏹 2. Pacman Specialist (Pursuit & Interception)
- [ ] **Chiến thuật Capture Dist 1:** 
    - Đích đến của Pacman là chính xác vị trí của Ghost (`target = enemy_position`).
    - *Chú ý:* Phải dẫm lên cùng một ô với Ghost mới được tính là bắt thành công.
- [ ] **Dự đoán đón đầu (Interception):** 
    - Tính toán vector vận tốc Ghost: `velocity = (curr_pos[0] - last_pos[0], curr_pos[1] - last_pos[1])`.
    - Pacman sẽ A* tới vị trí dự đoán: `predict_pos = current_ghost_pos + velocity`.
- [ ] **Tối ưu Tốc độ 2 (Speed 2 Runner):** 
    - Viết logic kiểm tra đường thẳng: Nếu đang đi thẳng và phía trước có ít nhất 2 ô trống, trả về `(Move, 2)` để duy trì đà tấn công.

---

## 👻 3. Ghost Specialist (Stealth & Survival)
- [ ] **Alpha-Beta Pruning (Bắt buộc):** 
    - Thêm tham số `alpha`, `beta` vào hàm `minimax`. 
    - Giúp Ghost có thể nhìn sâu tới 6-8 bước (depth) thay vì 4 bước như hiện tại trong cùng một khoảng thời gian 0.9s.
- [ ] **Cập nhật Win Condition (Dist == 0):** 
    - Trong Minimax, nếu khoảng cách Manhattan giữa Ghost và Pacman == 0, trả về điểm phạt cực nặng (Bị bắt).
- [ ] **Tránh Ngõ Cụt (Dead-end Avoidance):** 
    - Thêm hàm `is_dead_end(pos)`: Kiểm tra xem ô đó có phải đường cụt không.
    - Trong hàm Evaluate, trừ điểm nặng nếu Ghost di chuyển vào các hành lang cụt chỉ có 1 lối thoát.

---

## 🤖 4. ML & QA Engineer (Optimization & Testing)
- [ ] **Tối ưu Model DQN:** 
    - Điều chỉnh tiền xử lý dữ liệu trước khi đưa vào Model. Truyền `internal_map` (không sương mù ở vùng đã đi) thay vì chỉ truyền `obs` (có sương mù) của Arena.
- [ ] **Hệ thống Benchmark tự động:** 
    - Viết script `benchmark.py` để chạy Arena 50-100 trận không có đồ họa (`--no-viz`).
    - Xuất báo cáo CSV: Tỉ lệ thắng của Pacman, số bước sống sót trung bình của Ghost.
- [ ] **Timeout Guard:** 
    - Sử dụng `time.time()` để ngắt hàm `step` ở mốc 0.85s. Luôn có một nước đi dự phòng (fallback move) nhanh chóng (ví dụ: A* đơn giản hoặc đi thẳng) nếu Minimax quá tải.

## 🛠 Lưu ý chung
- **Định nghĩa Capture Distance 1:** Pacman thắng khi dẫm lên cùng vị trí với Ghost (Manhattan distance = 0).
- **Lệnh chạy kiểm thử chuẩn:**
  ```bash
  python3 arena.py --seek 8 --hide 8 --capture-distance 1 --pacman-speed 2 --pacman-obs-radius 5 --ghost-obs-radius 5
  ```
- Tất cả các file bổ sung (model, script) phải nằm trong thư mục của team (`submissions/8/`) để không bị lỗi khi nộp bài.
