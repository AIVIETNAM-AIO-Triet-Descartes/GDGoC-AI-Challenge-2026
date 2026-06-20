# Tổng quan về GDGoC AI Challenge 2026 - Bomberland

## 1. Source code này làm gì?

Đây là kho mã nguồn (repository) toàn diện cho cuộc thi lập trình trí tuệ nhân tạo **"GDGoC AI Challenge 2026"**, với chủ đề là game **Bomberland** - một trò chơi chiến thuật đa tác tử (multi-agent) lấy cảm hứng từ tựa game Bomb IT (hay Bomberman) kinh điển.

Mã nguồn này chứa:
*   **Hệ thống tổ chức giải đấu (Infrastructure):** Server đăng ký, hệ thống chấm điểm và đánh giá tự động ở chế độ nền (background evaluation engine), tính điểm xếp hạng theo thuật toán TrueSkill.
*   **Game Engine:** Lõi xử lý logic của trò chơi Bomberland, môi trường lưới (grid) 13x13 nơi các tác tử (agents) di chuyển, đặt bom, phá rương, nhặt vật phẩm và tiêu diệt lẫn nhau.
*   **Starter Kit (Mẫu cho người chơi):** Nằm trong thư mục `agent/`, bao gồm các con bot mẫu (baselines) từ ngẫu nhiên (random), dùng quy tắc (rule-based) cho đến bot sử dụng Học tăng cường (Reinforcement Learning - DQN) để bạn tham khảo.
*   **Công cụ kiểm thử (Scripts):** Các script giúp người chơi tự cho bot của mình đấu tập (local match) và đánh giá sức mạnh trước khi nộp bài.

---

## 2. Bạn nên làm gì với source code này?

Nếu bạn là **người tham gia (Participant)**, mục tiêu của bạn là lập trình ra một AI Agent thông minh nhất để đánh bại các đối thủ khác. Hãy làm theo các bước sau:

### Bước 1: Cài đặt môi trường
1. Yêu cầu **Python 3.11+**.
2. Cài đặt các thư viện cần thiết bằng lệnh:
   ```bash
   pip install -r requirements.txt
   ```

### Bước 2: Hiểu luật chơi
*   Trận đấu gồm 4 người chơi trên bản đồ 13x13.
*   **Input (Quan sát):** Mỗi lượt, bot của bạn nhận được dữ liệu (observation) bao gồm: bản đồ (vị trí cỏ, tường, rương, vật phẩm), thông tin người chơi (vị trí, máu, số bom), và danh sách bom đang có trên sân.
*   **Output (Hành động):** Bạn có **tối đa 100ms** để trả về 1 trong 6 hành động: `0` (Đứng im), `1` (Trái), `2` (Phải), `3` (Lên), `4` (Xuống), `5` (Đặt bom).

### Bước 3: Lập trình Agent của bạn
*   Bạn cần viết logic cho bot trong file tên là `agent.py`. Trong file này **bắt buộc phải có** một class tên là `Agent` và phương thức `act(self, obs: dict) -> int`.
*   Hãy tham khảo các bot mẫu trong thư mục `agent/` để biết cách viết. Bạn có thể dùng thuật toán tìm đường (BFS, A*), các tập luật If/Else (Heuristics) hoặc train mô hình AI (Deep Reinforcement Learning).
*   **Lưu ý:** Bot không được kết nối mạng (không được xài API ChatGPT/Gemini), không được chạy quá 100ms/lượt.

### Bước 4: Chạy thử nghiệm trên máy của bạn (Local Testing)
Trước khi nộp bài, hãy tự cho bot của mình đấu với các bot mẫu của BTC để xem sức mạnh:
*   **Chạy trận đấu có giao diện hình ảnh:**
    ```bash
    python -m scripts.participant.run_local_match --agent_paths path/to/your/agent/ None None None --visualize true
    ```
*   **Ước lượng thứ hạng tự động (chạy 100 trận để đo tỉ lệ thắng):**
    ```bash
    python -m scripts.participant.estimate_rankings --agent_path path/to/your/agent/ --num_matches 100
    ```

### Bước 5: Đóng gói và Nộp bài (Submission)
1.  **Đóng gói:** Nén file `agent.py` của bạn (cùng với các file trọng số AI `.pth` nếu có) vào một file `.zip`. (Lưu ý: `agent.py` phải nằm ở thư mục gốc của file nén, không nằm trong thư mục con).
2.  **Đăng ký:** Điền form đăng ký đội thi của BTC để nhận `Team ID` và `Submission Token` qua email.
3.  **Nộp bài:** Dùng form nộp bài để upload file `.zip`. Hệ thống sẽ tự động chạy code của bạn đấu với các đối thủ khác và cập nhật điểm TrueSkill lên bảng xếp hạng (Leaderboard). Bạn có thể nộp tối đa **3 lần/ngày**.

> **Mục tiêu tối thượng:** Tối ưu hóa thuật toán, giành điểm TrueSkill cao nhất để lọt vào Top 8 tiến tới Vòng Chung Kết (Grand Finals)!

---

## 3. Chi tiết hoạt động của các Bot mẫu (Baselines) trong thư mục `agent/`

Ban tổ chức đã cung cấp một loạt các bot từ dễ đến khó để bạn làm quen. Dưới đây là logic hoạt động chi tiết của từng bot:

### 3.1. `random_agent.py` (Bot Ngẫu nhiên)
*   **Thuật toán:** Không có thuật toán gì cả. Dùng hàm `random.randint(0, 5)`.
*   **Cách hoạt động:** Bot sẽ di chuyển ngẫu nhiên hoặc đặt bom một cách mù quáng ở mọi lượt đi. Không quan tâm đến bản đồ, bom hay kẻ địch. Nó rất dễ tự sát bằng chính bom của mình.

### 3.2. `simple_rule_agent.py` (Bot Luật Đơn giản)
*   **Thuật toán:** Dựa trên tập luật tĩnh (If-Else) kết hợp với thuật toán tìm đường ngắn nhất BFS (Breadth-First Search).
*   **Cách hoạt động (Thứ tự ưu tiên):**
    1.  **Né bom:** Nếu vị trí hiện tại nằm trong vùng nổ của bom (danger), dùng BFS tìm đường chạy ra ô an toàn gần nhất.
    2.  **Ăn vật phẩm:** Nếu an toàn, dùng BFS đi nhặt các vật phẩm (rương nổ rớt ra). Ưu tiên nhặt vật phẩm mình đang thiếu (bán kính hoặc số lượng).
    3.  **Đặt bom:** Nếu đang đứng cạnh một kẻ địch hoặc đứng cạnh rương, VÀ kiểm tra thấy đường lui (sau khi đặt bom) là an toàn, bot sẽ đặt bom (hành động `5`).
    4.  **Phá rương:** Tìm đường đi đến các vị trí trống bên cạnh rương để chuẩn bị đặt bom.
    5.  **Đi dạo:** Nếu không có gì làm, đi ngẫu nhiên vào các ô an toàn.

### 3.3. `box_farmer_agent.py` (Bot Nông dân Phá rương)
*   **Thuật toán:** Tương tự `simple_rule_agent` nhưng loại bỏ hoàn toàn ý định tấn công kẻ địch.
*   **Cách hoạt động:** Chỉ tập trung vào 3 việc: (1) Né bom -> (2) Đi nhặt vật phẩm -> (3) Đi đến cạnh rương và đặt bom phá rương. Bot này rất thụ động trong giao tranh nhưng "cày" vật phẩm lại rất tốt.

### 3.4. `smarter_rule_agent.py` (Bot Thông minh hơn)
*   **Thuật toán:** Tập luật If-Else phức tạp hơn, kiểm tra thêm tầm nhìn trực tiếp (line-of-sight).
*   **Cách hoạt động:**
    1.  Né bom (ưu tiên số 1).
    2.  Nhặt vật phẩm.
    3.  **Tấn công:** Kiểm tra xem có kẻ địch nào nằm trên **cùng một đường thẳng** (hàng ngang hoặc hàng dọc) trong phạm vi nổ của bom hay không (dùng hàm `_line_clear`). Nếu có và đường lui an toàn, nó sẽ đặt bom ngay lập tức để tiêu diệt đối thủ.
    4.  Phá rương (nếu không tấn công được).
    5.  **Săn mồi:** Nếu không có rương hay vật phẩm để phá, nó sẽ dùng BFS tìm đường tiếp cận (chasing) kẻ địch gần nhất.

### 3.5. `tactical_rule_agent.py` (Bot Chiến thuật)
*   **Thuật toán:** Đánh giá điểm số (scoring) cho các ô an toàn để né bom tốt hơn.
*   **Cách hoạt động:** Điểm khác biệt lớn nhất là khi né bom, thay vì chỉ chạy ra ô an toàn đầu tiên tìm thấy, nó sẽ tính toán xem ô đó có dễ bị "kẹt" (dead end) hay không bằng cách đếm số ô trống xung quanh. Điều này giúp Tactical Agent né bom mượt mà hơn, khó bị dồn vào góc chết. Các hành động tấn công và phá rương tương tự `smarter_rule_agent`.

### 3.6. `genius_rule_agent.py` (Bot Thiên tài - Baseline mạnh nhất)
*   **Thuật toán:** Quản lý trạng thái (State Machine) với chế độ `escape_mode`.
*   **Cách hoạt động:**
    *   Nó duy trì một biến trạng thái `self.escape_mode`.
    *   Mỗi khi nó quyết định đặt bom để phá rương hoặc giết địch, nó lập tức bật `escape_mode = True`.
    *   Ở các lượt đi tiếp theo, chừng nào `escape_mode` còn bật, nó sẽ dùng BFS với **độ sâu lên đến 10 bước** (`search_depth=10`) để tập trung 100% vào việc chạy trốn thật xa khỏi vụ nổ. Nó chỉ tắt chế độ này khi thực sự đã đến nơi an toàn tuyệt đối.
    *   Cách tiếp cận này giúp bot Genius cực kỳ cẩn trọng, gần như không bao giờ tự sát hoặc chết do bom của đối phương.

### 3.7. `dqn_agent/` (Bot Học Tăng Cường - Deep Reinforcement Learning)
*   **Thuật toán:** Deep Q-Network (Mạng Nơ-ron Học Sâu).
*   **Cách hoạt động:** Thay vì viết luật If-Else thủ công, bot này sử dụng mạng Nơ-ron (PyTorch) nhận đầu vào là trạng thái ma trận của bản đồ (`map_state`) và các chỉ số máu/bom của người chơi (`aux_state`). Nó sẽ tính toán và dự đoán "giá trị Q" (Q-value) cho cả 6 hành động có thể thực hiện. Hành động nào có Q-value cao nhất sẽ được chọn. Đi kèm là file trọng số `.pth` - thành quả sau khi cho AI tự chơi thử-sai hàng triệu lượt trong môi trường để tự đúc kết ra chiến thuật.

---

## 4. Cách tính điểm & Xếp hạng (Scoring System)

Hệ thống đánh giá của giải đấu sử dụng thuật toán **TrueSkill** (của Microsoft) để xếp hạng các bot, thay vì chỉ cộng/trừ điểm đơn thuần.

### 4.1. Thuật toán TrueSkill
Mỗi bot sẽ có 2 chỉ số:
*   **μ (mu):** Kỹ năng trung bình ước tính (khởi điểm là 100.0).
*   **σ (sigma):** Độ hoài nghi/sai số của hệ thống về bot (khởi điểm là 33.333). Chỉ số này sẽ giảm dần khi bot thi đấu càng nhiều trận.
*   **Điểm xếp hạng (Score):** Bằng `μ - 3σ`. Việc trừ đi `3σ` giúp đảm bảo bảng xếp hạng ưu tiên những bot thi đấu ổn định nhiều trận (sigma thấp) hơn là những bot ăn may một vài trận đầu.

### 4.2. Xếp hạng trong 1 trận đấu
Trong một trận đấu 4 người, thứ hạng được quyết định bằng **thứ tự bị tiêu diệt**:
*   Sống sót cuối cùng = Hạng 1 (Thắng).
*   Chết đầu tiên = Hạng 4 (Thua).
*   Nếu 2 bot chết cùng lúc (cùng 1 step) -> Hòa (chia sẻ thứ hạng).
*   **Tie-breaker (Xử lý hòa khi hết 500 steps):** Nếu trận đấu kéo dài quá 500 lượt mà vẫn còn nhiều bot sống sót, hệ thống sẽ xếp hạng dựa trên sự "đóng góp/tích cực" trong game theo thứ tự ưu tiên sau:
    1. Giết nhiều mạng nhất (Most Kills).
    2. Phá nhiều rương nhất (Most Boxes Destroyed).
    3. Nhặt nhiều vật phẩm nhất (Most Items Collected).
    4. Đặt nhiều bom nhất (Most Bombs Placed).
    Nếu tất cả chỉ số này đều bằng nhau thì mới tính là hòa nhau.

### 4.3. Tiêu chí xếp hạng trên Bảng xếp hạng (Leaderboard)
Nếu có nhiều bot trùng điểm, hệ thống sẽ ưu tiên theo thứ tự:
1. **Điểm Score cao hơn** (`μ - 3σ`).
2. Nếu bằng nhau: Chọn bot có **μ cao hơn**.
3. Nếu vẫn bằng: Chọn bot có **σ thấp hơn** (đã đánh nhiều trận hơn/ổn định hơn).
4. Nếu vẫn bằng: Bot nào **nộp (submit) gần đây nhất** sẽ xếp trên.

---

## 5. Hướng dẫn Chiến lược Tối ưu Điểm số (Optimization Strategy)

Dựa trên cách tính điểm TrueSkill ở trên, để đạt được top 1 trên bảng xếp hạng, bạn cần tối ưu bot của mình theo các nguyên tắc cốt lõi sau:

### 5.1. Sống sót là ưu tiên số 1 (Survival is King)
Bởi vì thứ hạng trong trận được quyết định hoàn toàn vào **thứ tự bị tiêu diệt**, việc bạn giết được bao nhiêu địch không quan trọng bằng việc bạn sống được bao lâu. 
*   **Không tự sát:** Đa số các bot yếu (như Random hay Simple) thường tự chết do bom của chính mình. Hãy tham khảo cơ chế `escape_mode` của `genius_rule_agent` để code logic tính toán đường lui thật xa (độ sâu thuật toán BFS từ 8-10 bước) trước khi quyết định đặt bom.
*   **Tránh giao tranh sớm:** Đừng vội vã lao vào giữa bản đồ tìm người để giết ở những lượt đầu. Hãy ở góc an toàn để farm vật phẩm và chờ các bot khác tự tàn sát nhau. Chỉ cần bạn sống sót đến top 2, điểm số của bạn đã là trận Thắng hoặc Hòa.

### 5.2. Tối ưu hàm Reward nếu dùng Học Tăng Cường (Reinforcement Learning)
Nếu bạn định phát triển mạng DQN (như thư mục `dqn_agent` bạn đang xem), file `reward.py` hiện tại đang cho các mức thưởng phạt chưa hoàn toàn tối ưu cho TrueSkill:
*   Đang có quá nhiều phần thưởng lẻ tẻ (`item_collection: 0.1`, `plant_near_box: 0.05`). Điều này làm AI dễ bị phân tâm và "tham lam" đi nhặt đồ dẫn đến chết.
*   **Cách sửa tốt nhất:** Tăng cực mạnh điểm phạt cho việc tự chết (`agent_death: -2.0` có thể tăng lên `-10.0` hoặc `-20.0`) và tăng điểm thưởng khi thắng (`win`). Mục tiêu ép mô hình AI phải học cách "Né bom tuyệt đối" trước, sau đó mới học cách đi phá rương và cuối cùng là tiêu diệt địch.

### 5.3. Tận dụng "Tie-breaker" (Chỉ số phụ) ở bước 500
Nếu bot của bạn code quá cẩn thận và liên tục sống sót đến hết 500 lượt, bạn sẽ hòa điểm với các bot phòng thủ khác. Để vượt lên giành Hạng 1 trong những pha hòa này, hệ thống sẽ xét chỉ số phụ.
*   **Thứ tự cày chỉ số phụ an toàn:** Đi nhặt vật phẩm và phá rương là cách an toàn nhất để tích lũy chỉ số phụ. 
*   Hãy code bot của bạn sao cho: Ở những nhịp không có bom đe dọa, tranh thủ đặt bom phá tường lấy rương để tăng `Most Boxes Destroyed` và nhặt đồ để tăng `Most Items Collected`.

### 5.4. Tận dụng công thức Score = μ - 3σ
Điểm Leaderboard của bạn sẽ bị trừ đi `3σ` (độ hoài nghi của hệ thống). 
*   Bot của bạn càng đánh ít trận, `σ` càng lớn, điểm số bị trừ càng nặng.
*   **Chiến thuật:** Ngay khi bot có thể né bom cơ bản và không lỗi, hãy **Nộp bài (Submit) ngay lập tức!** Càng nộp sớm, bot của bạn càng được hệ thống bốc vào đánh nhiều trận liên tục trong background. Đánh được hàng trăm trận sẽ giúp `σ` nhỏ đi, kéo điểm Score của bạn tăng vọt lên sát với mức thực lực `μ`.

### 5.5. Tối ưu thuật toán trong 100ms
Cuối cùng, nếu code chạy quá 100ms/lượt, server sẽ ép bot của bạn "Đứng im" (Action `0`), điều này sẽ khiến bạn chết oan uổng trong đống bom. Hãy tối ưu hàm BFS bằng cách dùng `collections.deque` thay vì `list`, và giới hạn độ sâu thuật toán tìm đường hợp lý (không quét toàn bộ ma trận ở mỗi bước).
