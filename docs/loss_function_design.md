Dưới đây là tổng hợp chi tiết và chuẩn xác về **hàm mất mát (loss functions / training objectives)** mà 10 mô hình trong các bài báo đang sử dụng, được phân loại theo từng nhóm kiến trúc và trường phái mô hình để chúng ta sẽ tham khảo:

---

### 1. Nhóm Continuous Generative & Flow Matching (FMRec, FlowRec, GMFlowRec, PDFlow)

Các mô hình này kết hợp giữa việc tối ưu trường vector (Vector Field Regression) của Flow Matching và các hàm mất mát phân loại/xếp hạng để chống sụp đổ biểu diễn (embedding collapse) và khử trôi (debiasing).

#### 🔹 **FMRec (Flow Matching for Sequential Recommendation)**
Mô hình chuyển đổi mục tiêu học từ việc dự đoán trường vector thô sang dự đoán trực tiếp vector embedding mục tiêu \\(x_c\\). Tổng hàm loss gồm 3 thành phần:
\\[\mathcal{L} = \mathcal{L}_{FM} + \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{MSE}\\]
* **Flow Matching Loss (\\(\mathcal{L}_{FM}\\)):** Đo khoảng cách \\(L_2\\) giữa embedding do mạng nơ-ron dự đoán \\(f_\Theta(z_t, t)\\) và embedding item thực tế \\(x_c\\):
  \\[\mathcal{L}_{FM} = \mathbb{E}_{t, p_t(z|x_c), p(x_c)} \left\| f_\Theta(z_t, t) - x_c \right\|_2^2\\]
* **Cross-Entropy Regularization (\\(\mathcal{L}_{CE}\\)):** Đóng vai trò chính quy hóa để phân biệt giữa các item embeddings khác nhau, ngăn các vector \\(x_c\\) bị co cụm về một điểm duy nhất khi bảng embedding được cập nhật.
* **Reconstruction Loss (\\(\mathcal{L}_{MSE}\\)):** Đo sai số toàn phương trung bình \\(\|\hat{\mathbf{d}} - \mathbf{r}\|_2^2\\) để tái tạo lại vector tương tác nhị phân \\(\mathbf{r}\\) của người dùng từ các trạng thái ẩn trung gian của Decoder1.

---

#### 🔹 **FlowRec (Behavior-based Prior Flow Matching)**
Khác với FMRec, FlowRec bổ sung khởi tạo prior từ hành vi và một hàm loss căn chỉnh xếp hạng đơn bước. Tổng hàm loss gồm:
\\[\mathcal{L} = \mathcal{L}_{Prior} + \alpha \mathcal{L}_{CFM} + \beta \mathcal{L}_{Align}\\]
* **Prior Loss (\\(\mathcal{L}_{Prior}\\)):** Huấn luyện bộ mã hóa Transformer để tạo ra vector khởi tạo \\(x_0\\) phản ánh đúng sở thích qua hàm Cross-Entropy đối với item kế tiếp:
  \\[\mathcal{L}_{Prior} = - \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \log \frac{\exp(x_0 \cdot \mathbf{e}_{u_n})}{\sum_{i \in \mathcal{I}} \exp(x_0 \cdot \mathbf{e}_i)}\\]
* **Conditional Flow Matching Loss (\\(\mathcal{L}_{CFM}\\)):** Hồi quy trường vector \\(v_\theta(x_t, t)\\) theo vector vận tốc giải tích \\(u_t(x_t|x_1) = x_1 - x_0\\) trên quỹ đạo thẳng:
  \\[\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1} \left\| v_\theta(x_t, t) - u_t(x_t|x_1) \right\|_2^2\\]
* **Single-Step Alignment Loss (\\(\mathcal{L}_{Align}\\)):** Sử dụng phép ngoại suy Euler một bước để ước tính điểm kết thúc \\(\tilde{x}_1 = x_t + (1-t) f_\theta(x_t, t)\\), sau đó tối ưu hóa tương phản với item dương và toàn bộ item âm để tăng cường tính phân biệt thứ hạng:
  \\[\mathcal{L}_{Align} = - \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \log \frac{\exp(\tilde{x}_1 \cdot \mathbf{e}_{u_n})}{\sum_{i \in \mathcal{I}} \exp(\tilde{x}_1 \cdot \mathbf{e}_i)}\\]

---

#### 🔹 **GMFlowRec (Gaussian Mixture Flow Matching)**
Được thiết kế cho gợi ý đa miền (Multi-Domain), tối ưu hóa đồng thời trên từng miền \\(d_k\\):
\\[\mathcal{L} = \sum_{d_k \in \mathcal{D}} \left( \ell_{d_k}^{Rec} + \alpha \ell_{d_k}^{Prior} + \beta \ell_{d_k}^{GMM} \right)\\]
* **Domain-Aligned Prior Loss (\\(\ell_{d_k}^{Prior}\\)):** Ràng buộc prior thích ứng miền \\(h^{DA}\\) dự đoán đúng item kế tiếp trong miền đích \\(d_k\\) thông qua domain-specific softmax.
* **GMM Likelihood Loss (\\(\ell_{d_k}^{GMM}\\)):** Cực tiểu hóa negative log-likelihood của điểm nguồn thực tế \\(x_0\\) dưới mô hình phân phối Gaussian Mixture đa mode nhằm dẫn hướng dòng chảy ngược:
  \\[\ell_{d_k}^{GMM} = \mathbb{E}_{t, x_0, x_1} \left[ - \log q_\theta(\mu \mid x_t, h_{d_k}^{DA}, t) \right]\\]
* **Recommendation Objective (\\(\ell_{d_k}^{Rec}\\)):** Căn chỉnh vận tốc trung bình dự đoán \\(\mu\\) của hỗn hợp Gaussian với mục tiêu gợi ý thông qua hàm Cross-Entropy đối với item thực tế.

---

#### 🔹 **PDFlow (Popularity-Debiased Flow Matching)**
Tập trung khử hiện tượng trôi dạt quỹ đạo về phía các item phổ biến. Tổng loss kết hợp 3 nhánh:
\\[\mathcal{L} = \mathcal{L}_{FM} + \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{align}\\]
* **Dual Flow Matching Loss (\\(\mathcal{L}_{FM}\\)):** Tổng của hai trường vector:
  \\[\mathcal{L}_{FM} = \mathcal{L}_{FM}^{main} + \mathcal{L}_{FM}^{pop} = \mathbb{E} \left\| f_\theta(x_t, t, E_u) - e_{L+1} \right\|_2^2 + \mathbb{E} \left\| g_\phi(x_t, t, E_u) - c_{pop} \right\|_2^2\\]
  *(Trong đó \\(f_\theta\\) hồi quy về item mục tiêu \\(e_{L+1}\\), còn \\(g_\phi\\) hồi quy về trọng tâm phổ biến toàn cục \\(c_{pop}\\))*.
* **Debiased Cross-Entropy Loss (\\(\mathcal{L}_{CE}\\)):** Tính cross-entropy trên biểu diễn đã trừ bỏ thành phần phổ biến \\(\tilde{\mathbf{x}} = \hat{x} - \lambda \hat{x}_{pop}\\):
  \\[\mathcal{L}_{CE} = - \log \frac{\exp(\tilde{\mathbf{x}}^\top e_{L+1})}{\sum_{j \in \mathcal{I}} \exp(\tilde{\mathbf{x}}^\top e_j)}\\]
* **Cross-Popularity Alignment Loss (\\(\mathcal{L}_{align}\\)):** Kéo gần khoảng cách biểu diễn giữa các item phổ biến (\\(P_u^{pop}\\)) và không phổ biến (\\(P_u^{unpop}\\)) cùng xuất hiện trong một chuỗi hành vi của người dùng:
  \\[\mathcal{L}_{align} = \mathbb{E}_u \left[ \frac{1}{|P_u^{pop}| |P_u^{unpop}|} \sum_{i \in P_u^{pop}} \sum_{j \in P_u^{unpop}} \left\| e_i - e_j \right\|_2^2 \right]\\]

---

### 2. Nhóm Dynamical Systems & Selective SSM (HSR, HoloMambaRec, HyMoERec, SIGMA)

Nhóm này chủ yếu mô hình hóa chuyển đổi trạng thái ẩn (latent state evolution) ở backbone, và sử dụng hàm mất mát phân loại hoặc phân phối xác suất trên toàn bộ item catalog.

#### 🔹 **HSR (Hamiltonian Spectral Recommender)**
Mô hình hóa chuỗi bằng phương trình dao động điều hòa giảm chấn trong không gian pha. Ở đầu ra, trạng thái vị trí cuối cùng được ngoại suy một bước Euler theo động lượng \\(\hat{\mathbf{q}}_u = \mathbf{q}_L + \Delta t (\mathbf{M}^{(L)})^{-1} \mathbf{p}_L\\).
* **Hàm loss:** Sử dụng Full Softmax Cross-Entropy trên không gian pha kết hợp điều chuẩn suy giảm trọng số (Weight Decay):
  \\[\mathcal{L} = - \sum_{(u, v^+) \in \mathcal{D}} \log \frac{\exp(\hat{\mathbf{q}}_u^\top \mathbf{e}_{v^+})}{\sum_{v' \in \mathcal{V}} \exp(\hat{\mathbf{q}}_u^\top \mathbf{e}_{v'})} + \lambda \|\Theta\|_2^2\\]

---

#### 🔹 **HoloMambaRec (Holographic Reduced Representations + Mamba)**
Sử dụng circular convolution để nén thuộc tính và item vào cùng không gian.
* **Hàm loss:** Sử dụng Masked Cross-Entropy Loss (bỏ qua các vị trí padding 0) tính trên các logit dự đoán \\(z_t = y_t W_{cls}^\top\\):
  \\[\mathcal{L} = - \frac{1}{\sum_t \mathbf{1}[i_{t+1} \neq 0]} \sum_t \mathbf{1}[i_{t+1} \neq 0] \log \frac{\exp(z_{t, i_{t+1}})}{\sum_{j=1}^{|\mathcal{I}|} \exp(z_{t, j})}\\]

---

#### 🔹 **HyMoERec (Hybrid Mixture-of-Experts with Mamba)**
Để cân bằng giữa chuyên gia dùng chung (Dense) và các chuyên gia thưa (Sparse MoE) cũng như tránh sụp đổ chuyên gia (expert collapse), mô hình tối ưu hóa hàm loss kép:
\\[\mathcal{L} = \mathcal{L}_{ce} + \lambda_{lb} \sum_{e} \bar{g}_e \log \bar{g}_e\\]
* **Cross-Entropy Loss (\\(\mathcal{L}_{ce}\\)):** Tính trên output kết hợp \\(\mathbf{y} = y_{dense} + \alpha \cdot y_{MoE}\\) sau khối Adaptive Expert Fusion (AEF):
  \\[\mathcal{L}_{ce} = - \log \frac{\exp(\mathbf{y}^\top \mathbf{r}^+)}{\sum_j \exp(\mathbf{y}^\top \mathbf{r}_j)}\\]
* **Load-Balance Regularization:** Thành phần phạt entropy dựa trên xác suất định tuyến trung bình \\(\bar{g}_e\\) của expert \\(e\\) trên toàn chuỗi, thúc đẩy tất cả các chuyên gia đều được phân bổ dữ liệu đồng đều.

---

#### 🔹 **SIGMA (SelectIve Gated MAmba)**
Kết hợp Partially Flipped Mamba và Feature Extract GRU thông qua cổng DS Gate.
* **Hàm loss:** Sử dụng hàm Cross-Entropy Loss tiêu chuẩn trên mini-batch đối với phân phối xác suất dự đoán \\(P_{i, y_i}\\) thu được từ tích vô hướng giữa vector ẩn kết hợp \\(R_0\\) và ma trận embedding của toàn bộ ứng viên item:
  \\[\mathcal{L}_{CE} = - \frac{1}{B} \sum_{i=1}^B \log P_{i, y_i}, \quad \text{với } P_{ik} = \frac{\exp(logits_{ik})}{\sum_{l=1}^M \exp(logits_{il})}\\]

---

### 3. Nhóm Generative Retrieval, Semantic IDs & Foundation Model (UNIQUE, Gwhere)

Nhóm này rời rạc hóa item thành các mã ngữ nghĩa (Semantic IDs) và hợp nhất khâu truy xuất (Retrieval) với xếp hạng đa mục tiêu (Ranking).

#### 🔹 **UNIQUE (Unified Retrieval and Ranking for Mobile Baidu)**
Tối ưu hóa đa nhiệm vụ từ đầu đến cuối (End-to-End Early Fusion):
\\[\mathcal{L} = \mathcal{L}_{disc} + \alpha \mathcal{L}_{gen} + \eta \mathcal{L}_{code} + \beta \mathcal{L}_{quant}\\]
* **Generative Retrieval Loss (\\(\mathcal{L}_{gen}\\)):** Cross-entropy dự đoán mã ngữ nghĩa phẳng kế tiếp \\(c_i\\):
  \\[\mathcal{L}_{gen} = - \log p(c_i \mid \mathcal{S}_u)\\]
* **Discriminative Multi-Task Loss (\\(\mathcal{L}_{disc}\\)):** Tổng có trọng số của các mục tiêu xếp hạng công nghiệp (\\(\mathcal{L}_{disc} = \sum_{k \in \mathcal{K}} \lambda_k \mathcal{L}_k\\)):
  * Các tác vụ nhị phân (CTR, độ liên quan, tỷ lệ xem hết video): Dùng **Weighted Binary Cross-Entropy (WBCE)**.
  * Các tác vụ hồi quy liên tục (thời lượng xem - watch duration): Dùng **Mean Squared Error (MSE)**.
* **Codebook Learning Loss (\\(\mathcal{L}_{code}\\)):** Hàm mất mát huấn luyện hai tháp DSSM để embedding của mã ngữ nghĩa hấp thụ được phản hồi hành vi của người dùng.
* **Quantization Regularization (\\(\mathcal{L}_{quant}\\)):** Ràng buộc cam kết (commitment) và cân bằng tần suất sử dụng mã trong codebook phẳng.

---

#### 🔹 **Gwhere (Generative Next POI Recommendation in Amap)**
Sử dụng quy trình huấn luyện 3 giai đoạn độc lập:
1. **Contrastive Item Tokenizer Loss:** Thay thế hoàn toàn MSE của RQ-VAE bằng hàm mất mát tương phản đa phương thức **NT-Xent Loss** với nhiệt độ \\(\tau = 0.1\\) để tối đa hóa tính phân tách giữa các Semantic ID (SID) của địa điểm (POI):
   \\[\mathcal{L}_{contrastive} = - \sum_{m=1}^{|\mathcal{M}|} \log \frac{\exp(\hat{h} \cdot h_m^+ / \tau)}{\sum_{h^- \in \mathcal{B}} \exp(\hat{h} \cdot h^- / \tau)}\\]
2. **Continued Pretraining & SFT Loss:** Cross-Entropy tự hồi quy chuẩn (Autoregressive Token Generation) để LLM sinh ra chuỗi mã SID của POI mục tiêu tiếp theo dựa trên ngữ cảnh không-thời gian.
3. **EAKTO Loss (Exposure-Aware Kahneman-Tversky Optimization):** Căn chỉnh sở thích từ log hiển thị nhị phân (click vs unclick):
   \\[\mathcal{L}_{EAKTO}(\pi_\theta, \pi_{ref}) = \mathbb{E}_{x, y \sim \mathcal{D}} \left[ \lambda_y (1 - v(x, y)) + \alpha \cdot \mathbf{I}_{y \in \mathcal{Y}_D} \cdot \sigma_D(x, y) \right]\\]
   *(Trong đó tích hợp neo phần thưởng dương `Positive Reward Anchoring` để chống trôi reward và cập nhật âm có điều kiện `Conditional Negative Update` để khử nhiễu mẫu unclick)*.

---

### Bảng tóm tắt nhanh

| Mô hình | Loại bài toán | Các thành phần loss chính | Bản chất hàm loss |
| :--- | :--- | :--- | :--- |
| **FMRec** | Single-domain Flow Rec | \\(\mathcal{L}_{FM} + \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{MSE}\\) | Target-regression MSE + Item CE + Matrix Recon MSE |
| **FlowRec** | Prior-guided Flow Rec | \\(\mathcal{L}_{Prior} + \alpha \mathcal{L}_{CFM} + \beta \mathcal{L}_{Align}\\) | Prior CE + Velocity CFM + Euler 1-step Contrastive CE |
| **GMFlowRec** | Multi-domain Flow Rec | \\(\sum (\ell^{Rec} + \alpha \ell^{Prior} + \beta \ell^{GMM})\\) | Target Domain CE + Prior Softmax + GMM Log-Likelihood |
| **PDFlow** | Debiased Flow Rec | \\(\mathcal{L}_{FM} + \alpha \mathcal{L}_{CE} + \beta \mathcal{L}_{align}\\) | Dual-flow MSE + Debiased CE + Popularity Pairwise Dist |
| **HSR** | Hamiltonian Dynamics | \\(\mathcal{L}_{CE} + \lambda \|\Theta\|_2^2\\) | Softmax CE sau khi ngoại suy Euler không gian pha |
| **HoloMambaRec** | Holographic SSM | Masked Cross-Entropy | Next-item Softmax CE (loại trừ padding token) |
| **HyMoERec** | Hybrid MoE + SSM | \\(\mathcal{L}_{ce} + \lambda_{lb} \sum \bar{g}_e \log \bar{g}_e\\) | Fused Softmax CE + MoE Load Balancing Entropy |
| **SIGMA** | Gated Mamba + GRU | \\(\mathcal{L}_{CE}\\) | Batch-averaged Cross-Entropy |
| **UNIQUE** | Unified Retrieval & Rank | \\(\mathcal{L}_{disc} + \alpha \mathcal{L}_{gen} + \eta \mathcal{L}_{code} + \beta \mathcal{L}_{quant}\\) | Multi-task (BCE+MSE) + Code CE + Two-tower loss |
| **Gwhere** | LLM Generative POI | \\(\mathcal{L}_{contrastive} \to \mathcal{L}_{SFT} \to \mathcal{L}_{EAKTO}\\) | Multi-modal NT-Xent + Token CE + Exposure-aware KTO |
