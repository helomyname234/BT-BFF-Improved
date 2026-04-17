# Kiến trúc Cải tiến (BERT-of-Theseus)

Tài liệu này lưu vết các thay đổi về kiến trúc đối với mô hình Knowledge Distillation đã thực hiện.

## 1. ProjectedMixModule — Heterogeneous Feature Alignment (Thiết kế lại thành Auxiliary Loss)
*Ban đầu:* Thêm `nn.Linear(embed_dim, embed_dim)` làm **Projection Layer** xen trực tiếp vào luồng `forward()` của Student.

**Vấn đề phát hiện:** Vì Projection Layer nằm trong luồng data của Student trong lúc Replacement Training, Student học cách tối ưu output để *sau khi đi qua Projection* thì giống Teacher. Khi `fine_tune_successor()` tháo bỏ MixModel và chạy Student thuần túy, Student bị Massive Distribution Shift → Loss bùng nổ.

**Thiết kế lại (18:16, 16/04/2026):**
- `forward()` trả thẳng **Raw Student output** — luồng data sạch hoàn toàn như Vanilla MixModule.
- Projection Layer di chuyển sang hàm `get_alignment_loss()` (Auxiliary Loss).
- Trong `module_replacement_training`, loss được tính: `L_total = L_main + 0.1 * L_alignment`.
- Ở giai đoạn Fine-tune / Deploy: `get_alignment_loss()` không bao giờ được gọi → Projection Layer **vô hình hoàn toàn**, Student không bị ảnh hưởng.

## 2. Logit Distillation (KL Divergence Loss)
Hàm loss mặc định của bài báo chỉ dùng **MSE**. 
Hiện tại đã bổ sung thêm **KL-Divergence Loss** (Hinton et al.) để làm loss hàm kết hợp trong quá trình `module_replacement_training` cũng như `fine_tune_successor`.

**Công thức tổng hợp:**
$L_{Total} = \alpha * L_{MSE} + (1 - \alpha) * L_{KL\_Div}(T)$

**Chi tiết Hyperparameters:**
- Mặc định: `T = 3.0` (Temperature để làm mềm hard label).
- `alpha = 0.5` (Trọng số hòa trộn giữa MSE và KL Div).

## 3. Cập nhật Interface của Mix Modules (Bug Fix: TypeError)
*[16/04/2026 - 09:13]*

Khắc phục lỗi `TypeError: ProjectedMixModule.forward() takes 2 positional arguments but 3 were given` do sự không đồng nhất về interface giữa `MixModel.forward()` (khi bật `use_optimization=True`) và các mix module khác nhau.
- **Giải pháp:** Bổ sung tham số `y_label: Optional[torch.Tensor] = None` vào phương thức `forward()` của các class `MixModule`, `SoftMixModule`, và `ProjectedMixModule`.
- Sự thay đổi này giúp các module trên tương thích hoàn toàn với cấu trúc gọi linh hoạt của `MixModel` khi nó truyền tham số nhãn `y_label` vào để phục vụ cho Gradient-based Optimization, mặc dù bản thân các module không tối ưu gradient kia không trực tiếp sử dụng tham số này.

## 4. Sửa Lỗi Toán Học trong OptimizedMixModule (Mathematical Bug Fix)
*[16/04/2026 - 18:16]*

**Lỗi phát hiện:**
Hàm `compute_optimal_r()` (Equations 11-12 của bài báo gốc) đã lấy *Trung bình cộng của Vector Đặc trưng không gian ẩn* (Scale ~0.01) đem trừ với *Chỉ số nguyên của Nhãn phân loại* (Class Index 0, 1, 2, 3, 4) - hai đại lượng hoàn toàn khác thứ nguyên, gây ra tỷ lệ `r` vô nghĩa và phá hỏng quá trình Distillation.

**Cách sửa:**
Thay thế `y_label` (chỉ số nhãn không tương thích) bằng **L2 Distance trong cùng không gian đặc trưng**:
- `scc_minus_target = ||student_output - teacher_output||₂` (khoảng cách thực sự giữa Student và Teacher)
- Khi `scc_minus_target` **lớn** → Student còn xa Teacher → `r = 0` (giữ Teacher dẫn dắt).
- Khi `scc_minus_target` **nhỏ** → Student đã xích lại gần → `r = r_extreme` (nhường lái cho Student).

## 5. Cấu hình cuối cùng (Final Configuration)
*[16/04/2026 - 18:16]*

Sau tất cả các quá trình thử nghiệm và fix lỗi, cấu hình chính thức được chốt là:

| Tính năng | Trạng thái | Ghi chú |
|---|---|---|
| `use_projection` | ✅ `True` | ProjectedMixModule với Auxiliary Loss (đã fix) |
| `use_optimization` | ✅ `True` | OptimizedMixModule với L2 Distance (đã fix) |
| `use_soft_replacement` | ❌ `False` | Bị vô hiệu hoá vì Projection được ưu tiên |
| `use_kd_loss` | ✅ `True` | KL-Divergence ở output cuối cùng |
| `aux_loss weight` | `0.07` | ProjectedMixModule alignment loss weight (sau khi thử 0.1, 0.03) |

## 6. Giảm Trọng số Auxiliary Alignment Loss: 0.1 → 0.03
*[16/04/2026 - 20:11 → 22:25]*

**Bằng chứng thực nghiệm (3 lần thử):**

| Class | Init | aux=0.1 | aux=0.03 | Nhận xét |
|---|---|---|---|---|
| BENIGN | 96.75% | 96.15% | **93.65%** | 0.03 crash BENIGN 3.1%! |
| DoS GoldenEye | **98.99%** | 98.29% | 98.25% | cả hai đều thấp hơn Init |
| DoS Hulk | 99.42% | 99.71% | 98.85% | 0.1 tốt hơn |
| DoS Slowhttptest | 98.76% | 98.69% | 98.76% | như nhau |
| DoS slowloris | 96.62% | 97.31% | **97.45%** | cả hai đều cải thiện |

**Phân tích cơ chế:**
- `weight cao (0.1)` → Student bị "xích cứng" vào Teacher → GoldenEye/Hulk blur (hai class gần nhau trong Teacher space)
- `weight thấp (0.03)` → Student tự do quá → BENIGN và Hulk mất ranh giới (Student tự học sai hướng)
- `Init (0.0)` → Không có áp lực alignment nào → Student học hoàn toàn từ Bernoulli Mix và MSE → BENIGN tốt nhất!

**Thử tiếp: `aux_loss_weight = 0.07`**
Lệch về phía 0.1 nhiều hơn vì dữ liệu cho thấy 0.1 chỉ tệ hơn 0.6% ở BENIGN trong khi 0.03 tệ hơn 3.1%. Điểm "ngọt" khả năng nằm gần 0.1 hơn là 0.03.

## 7. Thử Nghiệm với Auxiliary Projection Loss (Dựa trên t-SNE)
*[17/04/2026 - 00:49]*

**Kết quả thử nghiệm aux=0.07:**
- **BENIGN sập xuống 89.36%** (thấp nhất trong tất cả các lần thử nghiệm).
- BENIGN bị nhầm thành Hulk tăng vọt lên 5.85% (Init chỉ là 1.43%).

**Bằng chứng từ biểu đồ t-SNE (Successor features before classifier head):**
- Đám mây đặc trưng (feature cluster) của BENIGN (xanh dương) và DoS Hulk (nâu đỏ) bị **đan xen nhau hoàn toàn**, không có ranh giới rõ ràng.
- Các class khác như GoldenEye, slowloris, Slowhttptest phân tách rất tốt.

**Kết luận khoa học:**
Trong không gian đặc trưng trung gian (intermediate feature space) của mạng Teacher (ViT), BENIGN và Hulk có những điểm tương đồng rất lớn. Khi sử dụng Auxiliary Alignment Loss để ép Student (PoolFormer) căn chỉnh theo Teacher, ta đã vô tình ép Student thừa hưởng chính sự mơ hồ này của Teacher ở cấp độ feature map.
Ngược lại, **KD Loss** hoàn toàn ổn vì nó chỉ ép Student học theo các phân phối logit mềm (soft label) ở output cuối cùng, thời điểm mà Teacher đã phân biệt được các class bằng classifier head.

---

## 8. Siamese Network — Đổi từ Contrastive Loss sang Triplet Loss
*[Branch: Sisame_Triplet - Tháng 4/2026]*

**Thay đổi:** Đổi từ Contrastive Loss (Equation 2 của bài báo) sang Triplet Loss cho Siamese Network training.

**Lý do cải tiến (từ improve_plan.md):**
- Contrastive Loss chỉ so sánh từng cặp (Pairwise), đôi khi ép các mẫu khác loại ra quá xa mức cần thiết, làm nát latent space.
- Triplet Loss giữ được cấu trúc topology tốt hơn, đặc biệt với dữ liệu network traffic có sự chồng chéo cao giữa các loại tấn công.

**Thay đổi trong code:**
- File: `src/models/siamese_network.py`
- Thêm class `TripletLoss` (line 21-52)
- `SiameseNetwork.forward()` giờ trả về `(anchor, positive, negative)` thay vì pair
- `SiameseTripletDataset` sinh ra triplets (anchor, positive, negative) thay vì pairs
- Margin: `0.3` (config.py line 25)

**Tham chiếu:** improve_plan.md Section 3.1
