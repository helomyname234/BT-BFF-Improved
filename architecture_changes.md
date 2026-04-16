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
| `aux_loss weight` | `0.1` | ProjectedMixModule alignment loss weight |

## 6. Giảm Trọng số Auxiliary Alignment Loss: 0.1 → 0.03
*[16/04/2026 - 20:11]*

**Bằng chứng thực nghiệm (từ kết quả chạy thực tế):**

So sánh Confusion Matrix Init vs kết quả với `aux_loss_weight = 0.1`:

| Class | Init | aux=0.1 | Δ |
|---|---|---|---|
| BENIGN | 96.75% | 96.15% | ↓ -0.60% |
| DoS GoldenEye | **98.99%** | 98.29% | ↓ **-0.70%** |
| DoS Hulk | 99.42% | **99.71%** | ↑ +0.29% |
| DoS Slowhttptest | 98.76% | 98.69% | ↓ -0.07% |
| DoS slowloris | 96.62% | **97.31%** | ↑ +0.69% |

**Chẩn đoán nguyên nhân:**
Với `weight = 0.1`, Auxiliary Alignment Loss có tác động quá mạnh, cưỡng bức Student căn chỉnh đặc trưng (feature) sát với Teacher một cách thái quá. Hệ quả là ranh giới đặc trưng giữa DoS GoldenEye và DoS Hulk bị nhòe (GoldenEye → Hulk nhầm tăng từ **0.19% lên 0.86%** — gần 5 lần!). Auxiliary Loss cao cũng gây ra những spike khổng lồ trong đồ thị Replacement Loss.

**Giải pháp:**
Giảm trọng số từ `0.1` xuống `0.03` để Alignment Loss chỉ đóng vai trò **nhắc nhở nhẹ nhàng** (soft regularizer), không can thiệp sâu vào quá trình học phân loại chính.
