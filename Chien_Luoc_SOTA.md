# Các Chiến lược Thuật toán Đỉnh cao (SOTA) cho Bomberland

Để chiến thắng giải đấu này, bạn cần vượt qua các Baseline của Ban Tổ Chức (vốn đã dùng BFS độ sâu cao rất tốt). Vì giới hạn phần cứng là **CPU Only và 100ms/step**, các thuật toán cần phải vừa đủ thông minh, vừa được tối ưu hóa cực kỳ gắt gao về tốc độ. 

Dưới đây là các hướng tiếp cận SOTA (State-of-the-Art) khả thi nhất cho cuộc thi:

---

## 1. Học Tăng Cường Lai (Hierarchical RL & Action Masking)
DQN hiện tại trong baseline khá cổ điển và rất khó train để có tỷ lệ thắng cao. Các bài báo SOTA hiện nay cho thể loại Grid-world game khuyên dùng **PPO (Proximal Policy Optimization)** kết hợp với **Masking**.

*   **Ý tưởng:** 
    *   Bạn không dùng mạng Neural Network để quyết định mọi thứ một cách mù quáng. Thay vào đó, bạn chạy một bộ lọc quy tắc cứng (Hard-coded BFS) ở vòng ngoài để xác định các hành động *chắc chắn gây chết* (ví dụ đi vào ô bom chuẩn bị nổ).
    *   Mạng RL sẽ nhận bản đồ làm đầu vào, tính toán chiến thuật và đưa ra xác suất cho các hành động. Tuy nhiên, bạn sẽ **Che (Mask)** các hành động tử thần kia đi bằng cách gán xác suất của chúng bằng 0.
*   **Lợi ích:** Mạng RL không bao giờ mắc lỗi ngớ ngẩn (như tự sát) vì đã có bộ luật BFS bảo vệ. Khắc phục được nhược điểm lớn nhất của AI là thỉnh thoảng đi "ngáo".
*   **Cách làm thực tế:** Có thể dùng thuật toán Hierarchical RL (RL Phân tầng). Tầng cao là mạng Nơ-ron đưa ra tọa độ đích đến chiến lược (VD: Đi tới góc trái trên). Tầng thấp là thuật toán A* truyền thống chịu trách nhiệm tìm đường an toàn tới đó.

## 2. Tìm kiếm Cây Monte Carlo (Monte Carlo Tree Search - MCTS)
Đây là thuật toán đằng sau AlphaGo. Với game Perfect Information (nhìn thấy toàn bộ bản đồ), MCTS là vũ khí hủy diệt nếu bạn tối ưu được code Python.

*   **Ý tưởng:** Tại mỗi bước đi, tạo ra một bản sao ẩn của bản đồ, chơi mô phỏng (playout) ngẫu nhiên nhiều kịch bản tương lai có thể xảy ra trong 5-10 lượt tới. Hành động nào dẫn đến kịch bản sống/thắng nhiều nhất sẽ được chọn.
*   **Khó khăn:** Có 4 người chơi, mỗi người 6 hành động -> $6^4 = 1296$ trường hợp rẽ nhánh mỗi lượt. Mô phỏng bằng Python thuần sẽ gây Time Out (quá 100ms).
*   **Giải pháp (Decoupled MCTS):** 
    *   Giả định 3 đối thủ sẽ chơi theo một luật cố định (ví dụ coi chúng như Tactical Agent).
    *   Chỉ rẽ nhánh cây cho bản thân mình (6 nhánh). Giới hạn độ sâu ở mức 4-6 bước.
    *   Thay vì mô phỏng tới cuối game, bạn tính điểm trạng thái (Heuristic Evaluation) ở cuối lá cây: Tính khoảng cách tới rương, mức độ an toàn so với bom, vị trí ép góc địch.

## 3. Bản đồ Ảnh hưởng (Influence Maps) - Tốc độ cực cao
Đây là thuật toán "quốc dân" được sử dụng cực rộng rãi trong các tựa game chiến thuật thời gian thực (RTS) vì tốc độ tính toán bằng ma trận Numpy siêu nhanh (dưới 5ms/step).

*   **Ý tưởng:** Bạn phủ lên ma trận 13x13 một bản đồ "điểm số tiềm năng".
    *   **Bom:** Tỏa ra "sức ảnh hưởng Âm" (trừ điểm rất nặng) dọc theo hình chữ thập. Timer của bom càng thấp, điểm âm càng đậm.
    *   **Rương/Vật phẩm:** Tỏa ra "sức ảnh hưởng Dương" (cộng điểm) lan ra các ô xung quanh.
    *   **Địch:** Tỏa ra ảnh hưởng Âm (để tránh xa) hoặc Dương (để áp sát tấn công).
*   **Hành động:** Nhiệm vụ của bot lúc này siêu đơn giản: Nó chỉ cần leo dốc (Gradient Ascent) theo thuật toán BFS để tìm đường an toàn ngắn nhất đi tới những ô có "Tổng điểm Influence Dương cao nhất".
*   **Lợi ích:** Tính toán bằng ma trận siêu lẹ, không bao giờ lo 100ms timeout, code đơn giản dễ debug hơn RL.

## 4. Kỹ thuật ép góc (Trapping/Minimax)
Bản chất để thắng ở cuối game, bạn cần dồn đối thủ vào đường cùng.
*   **Chiến thuật:** Lưu lịch sử di chuyển của địch. Dự đoán quỹ đạo chạy của địch.
*   Bạn có thể đặt bom một cách có chủ đích không phải để nổ trúng địch ngay, mà để **chặn đường lùi** của địch, ép địch phải chui vào một góc cụt hoặc vùng bom của người khác. Áp dụng cây Minimax cục bộ (khoảng cách hẹp 5x5) khi trên bản đồ chỉ còn bạn và 1 đối thủ (1vs1) sẽ mang lại độ chính xác chết người.

---

## 🏆 ĐỀ XUẤT LỘ TRÌNH VÔ ĐỊCH (Khuyên Dùng)

Bạn nên xây dựng một con bot **Hybrid (Kết hợp thuật toán truyền thống và MCTS)** thay vì dùng Machine Learning thuần túy. Machine Learning cần quá nhiều thời gian training và khó debug khi thi đấu trong thời gian ngắn.

1.  **Cốt lõi Sinh tồn (BFS + Influence Map):** Xóa bỏ hoàn toàn DQN. Code một bộ luật mới bằng Numpy kết hợp BFS. Mục tiêu: Nhìn vào là biết ô nào an toàn 100%, ô nào có nguy cơ chết.
2.  **Bộ não Chiến thuật (1-Step Lookahead + MCTS thu gọn):** 
    *   Lọc ra tất cả các ô di chuyển an toàn.
    *   Mô phỏng thử (simulate) việc bạn **Đặt bom** ở vị trí hiện tại: Nếu đặt bom xong, vòng lặp kế tiếp bạn có đường chạy thoát hay không? Quả bom đó có chặn mất lối thoát của đối phương không?
    *   Nếu phép tính dự đoán (Lookahead) trả về kết quả là Địch hết đường lui (Dead End) -> Ngay lập tức đặt bom.
3.  **Tối ưu Python:** Sử dụng các toán tử Vector hóa của Numpy thay vì vòng lặp `for` thông thường trong mảng 2D. Nếu được, dùng Numba (nếu thư viện không bị cấm) để compile mã BFS sang C++ tốc độ cực cao, giúp bạn tính được cây MCTS sâu hơn các đội khác.
