# Kiến trúc Cải tiến (BERT-of-Theseus)

Tài liệu này lưu vết các thay đổi về kiến trúc đối với mô hình Knowledge Distillation đã thực hiện.

## 1. ProjectedMixModule (Heterogeneous Feature Alignment)
Mô đun `MixModule` gốc giả định Teacher và Student dùng chung một không gian đặc trưng (Feature space). Tuy nhiên:
- **Predecessor (Teacher)**: Dùng Multi-Head Attention (ViT).
- **Successor (Student)**: Dùng Global Pooling (PoolFormer).

**Giải pháp:** Thêm `nn.Linear(embed_dim, embed_dim)` làm **Projection Layer** sau khối của Student. 
Layer này giúp học một phép biến đổi tịnh tiến (affine transformation) để khớp không gian đặc trưng của Student với Teacher, trước khi truyền vào khối tiếp theo. Nếu tham số random mẫu ra quyết định sử dụng module Student, module này sẽ được đi qua projection.

## 2. Logit Distillation (KL Divergence Loss)
Hàm loss mặc định của bài báo chỉ dùng **MSE**. 
Hiện tại đã bổ sung thêm **KL-Divergence Loss** (Hinton et al.) để làm loss hàm kết hợp trong quá trình `module_replacement_training` cũng như `fine_tune_successor`.

**Công thức tổng hợp:**
$L_{Total} = \alpha * L_{MSE} + (1 - \alpha) * L_{KL\_Div}(T)$

**Chi tiết Hyperparameters:**
- Mặc định: `T = 3.0` (Temperature để làm mềm hard label).
- `alpha = 0.5` (Trọng số hòa trộn giữa MSE và KL Div).

*Ghi chú:* Các thay đổi này chỉ nhằm mục đích Knowledge Distillation hiệu quả hơn. Hàm Projection của Student sẽ không cần thiết và tự bị loại bỏ khi mang mô hình Successor cuối cùng đi deploy, đảm bảo vẫn duy trì tính chất "Lightweight".

## 3. Cập nhật Interface của Mix Modules (Bug Fix: TypeError)
*[16/04/2026 - 09:13]*

Khắc phục lỗi `TypeError: ProjectedMixModule.forward() takes 2 positional arguments but 3 were given` do sự không đồng nhất về interface giữa `MixModel.forward()` (khi bật `use_optimization=True`) và các mix module khác nhau.
- **Giải pháp:** Bổ sung tham số `y_label: Optional[torch.Tensor] = None` vào phương thức `forward()` của các class `MixModule`, `SoftMixModule`, và `ProjectedMixModule`.
- Sự thay đổi này giúp các module trên tương thích hoàn toàn với cấu trúc gọi linh hoạt của `MixModel` khi nó truyền tham số nhãn `y_label` vào để phục vụ cho Gradient-based Optimization, mặc dù bản thân các module không tối ưu gradient kia không trực tiếp sử dụng tham số này.

## 4. Hủy bỏ Projection Layer & Kích hoạt toàn diện SoftMix
*[16/04/2026 - 13:14]*

**Vấn đề phát hiện (Architecture Bug):**
Phương pháp chèn thêm Linear Layer (`ProjectedMixModule`) vào Student lúc Distillation đã gây ra hiện tượng gãy vỡ phân phối đặc trưng (Massive Distribution Shift) khi gọi hàm `fine_tune_successor()`. Lý do là ở bước Deploy / Fine-tune, mô hình vứt bỏ lớp Projection, khiến mạng Student nhận vào các biểu diễn vector hoàn toàn rỗng và xa lạ (vì trước đó nó tối ưu hóa output dựa trên việc "được đi qua Linear Layer sửa lỗi"). Hậu quả là Loss tăng vọt và Recall trên các Class tấn công tụt thê thảm. 
Đồng thời, lỗi hard-code logic trong `MixModel` đã khiến `SoftMix` chưa bao giờ thực sự được gọi trúng.

**Giải pháp (Combo Mới):**
1. **Hủy bỏ hoàn toàn `use_projection`**: Đăng xuất `ProjectedMixModule` vĩnh viễn khỏi quy trình. Ép mạng Student học đối kháng bằng chính bản ngã thuần túy của kiến trúc PoolFormer.
2. **Kích hoạt cứng `SoftMixModule`**: Bật `use_soft_replacement=True` và liên thông lên vòng lặp `module_replacement_training` để scheduler hoạt động. Tham số $\alpha$ nay chính thức được trượt mượt mà (nội suy) từ 0.0 (xài 100% Teacher) lên dần 1.0 (xài 100% Student).
3. Combo này đi đôi với **Cân bằng dữ liệu nội bộ (Perfect Resampling 120k)** mang lại môi trường Distillation tốt nhất cho Model.
