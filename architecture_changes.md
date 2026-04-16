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
