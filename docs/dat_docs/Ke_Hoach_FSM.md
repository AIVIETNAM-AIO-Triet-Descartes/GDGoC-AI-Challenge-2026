# Kế hoạch Triển khai Bot bằng Mô hình FSM kết hợp Utility Scoring

Bản kế hoạch này mô tả chi tiết cách xây dựng một con bot mạnh mẽ, kết hợp giữa **Máy trạng thái hữu hạn (FSM)** để định hướng chiến lược và **Chấm điểm Tiện ích (Utility Scoring)** để đưa ra quyết định cuối cùng. 

---

## 1. Cấu trúc Cốt lõi (Vòng lặp mỗi Step)

Mỗi khi hàm `act(self, obs)` được gọi, bot sẽ tuân theo luồng xử lý (pipeline) tối ưu CPU nhất:

1.  **Cập nhật Danger Map (Time-layered):** Quét toàn bộ bom và tính toán chính xác tick nào ô nào sẽ nổ (tính cả bom nổ dây chuyền).
2.  **KHẨN CẤP MỨC ĐỘ 1 (Escape Override):** Nếu vị trí hiện tại có `danger_map == 1` (sẽ nổ ngay lượt tới), **BỎ QUA MỌI TÍNH TOÁN BÊN DƯỚI** và lập tức chạy thuật toán tìm đường (BFS) để chạy trốn. 
3.  **Bộ Lọc Hành Động (Action Pre-filtering):** Trước khi chấm điểm, loại bỏ ngay lập tức các hành động không hợp lệ. Đặc biệt với lệnh Đặt Bom: Hết bom, đang đứng trùng với bom khác, hoặc đặt xong không có đường thoát $\rightarrow$ Loại bỏ. (Sử dụng Cache cho việc check đường thoát để tối ưu CPU).
4.  **Đánh giá Phase (FSM):** Dùng `power_score` để xác định Phase và lấy bộ `weights` (trọng số).
5.  **Chấm điểm Hành động (Utility Scoring):** Chấm điểm cho các hành động hợp lệ dựa trên trọng số của Phase. Các hành động nguy hiểm được tính toán phạt mềm phân tầng (Stratified Penalty). Trả về hành động điểm cao nhất.

---

## 2. Chi tiết các Bước Triển khai

### 2.1. Nâng cấp Bản đồ Nguy hiểm (Time-layered Danger Map)
Khởi tạo `danger_map` là một mảng 2D lưu giá trị `tick` (số lượt) ô đó sẽ phát nổ. Ô an toàn là $\infty$.
Để tính đúng `tick`, bắt buộc phải xử lý **Nổ dây chuyền (Chain Explosion)**:
*   Sắp xếp bom theo `timer` từ thấp đến cao.
*   Phóng tia (Raycast) cho bom nổ sớm nhất. Nếu tia nổ chạm vào quả bom B, `timer` của B lập tức bị ép giảm xuống bằng `timer` của quả bom A. Lặp lại cho đến khi tính xong.
*   **Reachable Safe Tiles (Cache BFS):** Tính MỘT LẦN danh sách các ô an toàn có thể đến được bằng BFS từ đầu hiệp và lưu cache. Các bước tính Utility Scoring phía sau chỉ việc tra cứu.

### 2.2. Điều kiện Chuyển Phase (FSM Triggers)
Số lượng bom quan trọng hơn nhiều so với bán kính nổ. Bạn cần ít nhất 2 quả bom để gài bẫy (trapping).
```python
power_score = (bomb_radius * 1.5) + (bombs_left * 3)

if power_score < 8:
    current_phase = "FARMER"
elif step < 350:
    current_phase = "ZONER"
else:
    current_phase = "ASSASSIN"
```

### 2.3. Công thức Utility Scoring toàn diện (Chuẩn hóa)
Mỗi hành động $A$ sẽ được tính điểm $Score(A) = \sum (Weight_i \times \text{Norm}(Value_i))$.
**CỰC KỲ QUAN TRỌNG:** Mọi giá trị thành phần phải được chuẩn hóa (Normalize) về khoảng `[0, 1]`. 

Các thành phần được tính điểm:
1.  **Item & Box:** Khoảng cách tới vật phẩm gần nhất nghịch đảo. Item cho sức mạnh vĩnh viễn nên điểm `Item_Reward` phải luôn > `Box_Reward` (Rương chỉ là phương tiện).
2.  **Survivability Score (Sống sót):** Đo lường tính an toàn của ô đích sau khi giả lập hành động. Ví dụ: `distance_to_nearest_safe_tile`. Rất hiệu quả để bắt các trường hợp bot sắp bị kẹp bởi 2 quả bom.
3.  **Kill Reward (Tối ưu CPU):** Điểm thưởng khi lệnh Đặt Bom dồn địch vào ngõ cụt.
    *   *Tối ưu CPU:* Chỉ chạy thuật toán giả lập BFS cho địch **NẾU** địch nằm cách bot $\le \text{radius} + 2$. Tránh tốn CPU vô ích.
4.  **Enemy Pressure (Áp lực lên địch):** `Norm(LOS_Bonus + 1 / (distance_to_enemy + 1))`. Check tia nhìn (Line of Sight) để tránh tạo áp lực ảo qua tường.
5.  **Mobility Reward (Điểm Cơ Động):** Số ô an toàn có thể tiếp cận được trong bán kính 3 bước từ ô đích. 
6.  **Territory Score (Kiểm soát Lãnh thổ):** `reachable_area_size` (Số lượng ô trống kiểm soát được trong 5 bước).
    *   *Lưu ý (Double-dipping):* Vì Mobility (3 bước) và Territory (5 bước) khá tương đồng, chúng ta giảm bớt trọng số của Territory để không "thưởng 2 lần" cho cùng một đặc tính không gian mở.
7.  **Danger Penalty (Phạt phân tầng):** Đừng phạt chung chung 1 mức. Hãy chia nhỏ để bot biết cân nhắc rủi ro:
    *   `danger == 1` $\rightarrow$ $-\infty$ (Loại bỏ)
    *   `danger == 2` $\rightarrow$ Penalty rất nặng (vd: $-5.0$)
    *   `danger == 3` $\rightarrow$ Penalty vừa (vd: $-2.0$)
    *   `danger >= 4` $\rightarrow$ $0$ (Bỏ qua)

### 2.4. Trọng số theo Phase

#### A. Phase: FARMER (Nông Dân)
*   **Trọng tâm:** Tối đa hóa `Item` và dọn `Box`. 
*   **Trọng số:** `{"item": 1.0, "box": 0.7, "kill": 0.0, "pressure": 0.1, "survive": 1.0, "mobility": 0.4, "territory": 0.2}`

#### B. Phase: ZONER (Kiểm soát)
*   **Trọng tâm:** Giữ áp lực giao tranh và nhặt đồ.
*   **Trọng số:** `{"item": 0.6, "box": 0.4, "kill": 0.8, "pressure": 0.5, "survive": 1.0, "mobility": 0.5, "territory": 0.3}`

#### C. Phase: ASSASSIN & LATE GAME (Sát Thủ / Xử lý Hòa)
*   **Trọng tâm:** Giết địch, chiếm lãnh thổ và Tối ưu chỉ số phụ.
*   **Trọng số:** `{"item": 0.4, "box": 0.3, "kill": 1.0, "pressure": 0.8, "survive": 1.0, "mobility": 0.6, "territory": 0.5}`

---

## 3. Tóm tắt Code Logic (Pseudo-code)

```python
def act(self, obs):
    danger_map = self.build_time_layered_danger_map(obs)
    
    # 1. KHẨN CẤP MỨC ĐỘ 1: Sắp nổ ngay lượt sau
    if danger_map[my_pos.x][my_pos.y] == 1:
        return self.find_best_escape(my_pos, danger_map)
        
    # [Cache 1] Lưu các ô an toàn để đánh giá hành động
    reachable_safe_tiles = self.bfs_safe_tiles(my_pos, danger_map)
    
    # Lấy weights theo Phase
    weights = self.get_weights_from_phase(obs)
        
    best_action = 0
    max_score = -infinity
    
    # [Cache 2] Tối ưu: Chỉ tính hàm can_escape() cho bomb ĐÚNG 1 LẦN để tránh quá tải CPU
    bomb_escape_valid = None 
    
    for action in [0, 1, 2, 3, 4, 5]:
        next_pos = self.simulate_move(my_pos, action)
        
        # BỘ LỌC HÀNH ĐỘNG (PRE-FILTERING)
        if danger_map[next_pos.x][next_pos.y] == 1:
            continue # Tự sát
            
        if action == 5: # PLACE_BOMB
            if bombs_left == 0 or my_pos in current_bombs:
                continue # Lệnh bom lỗi
                
            if bomb_escape_valid is None:
                # Chỉ tính 1 lần nếu cần
                bomb_escape_valid = self.can_escape(my_pos, virtual_danger_map)
            
            if not bomb_escape_valid:
                continue # Đặt xong hết đường chạy
                
        # CHẤM ĐIỂM UTILITY (ĐÃ CHUẨN HÓA)
        score = self.evaluate_action_normalized(action, obs, danger_map, reachable_safe_tiles, weights)
        
        if score > max_score:
            max_score = score
            best_action = action
            
    return best_action
```
