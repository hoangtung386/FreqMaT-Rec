# Đánh giá tính khả thi của các ý tưởng cho FreqMaT-Rec

Tài liệu này đánh giá từng đóng góp (contribution) trong 10 paper ở `references/`, cùng các ý tưởng đã bàn trước đó, theo hai tiêu chí: **tăng hiệu năng mô hình** và **làm đóng góp của bài báo của ta**.

## Cách đọc tài liệu này (độ tin cậy)

- Mỗi paper được một agent riêng đọc toàn văn bản trích từ PDF, rồi tóm tắt. **Tôi chưa tự đối chiếu từng con số với PDF.** Trước khi trích dẫn số nào trong bài báo, hãy mở PDF kiểm tra lại.
- Những chỗ ghi "(suy luận)" là suy luận của người đọc, không phải điều paper khẳng định.
- Mọi nhận định về hiệu quả khi chuyển sang mô hình của ta (cột "Kỳ vọng") là **dự đoán chưa được đo**. Ta chưa có kết quả hội tụ nào của FreqMaT, nên chưa biết mô hình hiện tại có thắng SASRec hay không.
- Bối cảnh của ta: RecBole, full-ranking leave-one-out, loss CE chỉ ở vị trí cuối, dataset **chỉ có ID item và timestamp** (không văn bản, ảnh, thuộc tính, miền, log hiển thị-không-click), hidden 64, độ dài chuỗi 50, khoảng 1M tham số.

## 1. Kết luận ngắn

1. **Hướng 1, 2, 3 của bạn bạn đều không nên làm nguyên xi.** Flow Matching, Semantic ID và holographic binding đổi bài toán hoặc cần dữ liệu ta không có. Chi tiết ở mục 3.
2. **Ba thứ rẻ, có cơ sở và hợp mô hình:**
   - bộ lọc dao động bậc hai (từ HSR), làm thành bản **nhân quả**;
   - nhánh GRU cục bộ (từ SIGMA), vì ablation của SIGMA cho thấy nhánh này đóng góp lớn nhất;
   - expert dùng chung và deep supervision, là các công tắc nhỏ.
3. **Điểm yếu chung của cả 10 paper:** baseline thường yếu hoặc không được tune; đa số không báo seed, độ lệch chuẩn hay kiểm định ý nghĩa; nhiều "mô-đun mới" không có ablation đối chứng sạch. Hệ quả: các con số "+10%" trong các paper đó **không nên dùng làm kỳ vọng cho ta**, và ta phải tự tune SASRec/BSARec đàng hoàng trước khi tuyên bố thắng.
4. **Rủi ro về tính mới:** HSR (RecSys '26) đã công bố bộ truyền phổ dao động bậc hai. Nếu ta chỉ thay bộ lọc bằng nó thì đóng góp là **gia tăng**, không phải mới hoàn toàn. Phần ta có thể khẳng định riêng là bản nhân quả, phép so sánh với FMLP/BSARec/GFT, và sự kết hợp với nhánh Mamba/attention.

## 2. Bảng tổng hợp theo mức ưu tiên

Thang đánh giá: **Công sức** (thấp/vừa/cao), **Kỳ vọng** (dự đoán, chưa đo), **Giá trị cho bài báo** (đóng góp độc lập hay chỉ là ablation).

| # | Ý tưởng | Nguồn | Cần dữ liệu lạ? | Công sức | Kỳ vọng hiệu năng | Giá trị cho bài báo | Khuyến nghị |
|---|---|---|---|---|---|---|---|
| A | Bộ lọc dao động bậc hai `Ψ/(κ − mω² + icω)` (làm nhân quả) | HSR | Không | Thấp | Vừa phải; HSR báo gần như không đổi khi Ψ=1, nên phần vật lý mang phần lớn lợi ích | **Có thể là đóng góp chính** nếu phần nhân quả đứng vững | **Làm** |
| B | Nhánh conv + GRU cục bộ | SIGMA | Không | Thấp | Có thể đáng kể (SIGMA: bỏ GRU mất khoảng 24% HR@10 trên Beauty); nhưng RoPE-attention của ta có thể đã phủ vai trò này | Chỉ là ablation | **Làm** |
| C | Expert dùng chung + khởi động α + cân bằng entropy | HyMoERec (và DeepSeekMoE) | Không | Thấp | Nhỏ (paper báo 1–3%, chỉ 2 dataset, không ablation) | Chỉ là ablation; không được nhận là ý tưởng của ta | **Làm** (công tắc nhỏ) |
| D | Deep supervision: CE phụ trên trạng thái trung gian | FlowRec (L_Prior, L_Align), GMFlowRec | Không | Thấp | Nhỏ đến vừa (FlowRec: bỏ các CE phụ mất khoảng nửa hiệu năng, nhưng bị trộn với flow) | Ablation | **Làm** |
| E | Chỉnh logit theo độ phổ biến, và bộ metric đuôi dài | PDFlow | Không (đếm từ train) | Thấp | Gần như không đổi độ chính xác; giá trị là ở **metric đuôi dài** | Phân tích đuôi dài là điểm cộng cho bài báo | **Làm** (phần metric là chính) |
| F | Loss căn chỉnh chéo độ phổ biến `L_align` | PDFlow | Không | Thấp | Không chắc; có nguy cơ làm các embedding co lại | Ablation | Thử sau |
| G | Phép ngoại suy động lượng một bước `q_T + Δt·p_T` | HSR | Không | Thấp | Nhỏ ("small but systematic" theo HSR) | Ablation | Thử sau |
| H | Mamba lật một phần (PF-Mamba) | SIGMA | Không | Vừa | Nhỏ (SIGMA: bỏ flip mất khoảng 3% HR; chỉ tune r trên Beauty) | Ablation | Thử cuối; cần lật theo độ dài thật |
| I | Loss phụ tái dựng đa-nhãn (multi-hot) bằng BCE | FMRec | Không | Thấp | Không chắc | Ablation | Tùy chọn |
| J | Loss tương phản in-batch (NT-Xent) | Gwhere (áp cho tokenizer, ta ngoại suy) | Không | Thấp | Không chắc, dư thừa với CE | Ablation | Tùy chọn |
| K | Loss phụ dự đoán mã (codebook phẳng cân bằng) | UNIQUE | Không, nhưng vòng tròn vì chỉ có ID | Vừa | Không chắc | Ablation | Bỏ qua trừ khi rảnh |
| L | Đặc trưng không gian trên Gowalla/Foursquare | Gwhere | Chỉ check-in | Vừa | Chưa biết | Chỉ hợp lệ cho vài dataset | Hoãn |
| M | Flow Matching head (ODE nhiều bước) | FMRec, FlowRec, PDFlow, GMFlowRec | Không | **Cao** | Chưa chứng minh có ích so với CE | Là một bài báo khác | **Không làm** |
| N | Semantic ID / tokenizer phẳng / retrieval-ranking hợp nhất | UNIQUE, Gwhere | Cần nội dung và log công nghiệp | Rất cao | Không áp dụng | Ngoài phạm vi | **Không làm** |
| O | Holographic binding (tích chập vòng) | HoloMambaRec | Cần thuộc tính item | Vừa | **Bằng chứng ngược:** ablation của chính paper cho thấy binding không giúp | Không | **Không làm** |
| P | Đầu ra hỗn hợp Gauss (GMM), mặt nạ miền | GMFlowRec | Cần nhãn miền | Cao | Phần GMM chỉ thêm 0,0–0,8 điểm NDCG@10 | Không | **Không làm** |
| Q | EAKTO (căn chỉnh sở thích) | Gwhere | Cần log hiển thị-không-click và LLM | Rất cao | Không áp dụng | Không | **Không làm** |

## 3. Đánh giá từng paper

### 3.1 HSR — Hamiltonian Spectral-Temporal Dissipative Dynamics (RecSys '26)

**Đóng góp tuyên bố:** (1) mô hình hóa sở thích như hệ dao động bậc hai có tiêu tán; (2) bộ truyền phổ Hamiltonian cùng nhánh xung cục bộ; (3) ngoại suy một bước trong không gian pha; (4) thắng các mô hình bậc nhất.

**Cách làm (đủ để cài lại):** mỗi kênh có khối lượng `m`, cản `c`, độ cứng `κ`, đều qua softplus để dương. Hàm truyền `G(ω) = Ψ / (κ − mω² + icω + ε)`, với `Ψ` là tử số phức học được theo (tần số, kênh). Áp RFFT dọc trục thời gian, nhân `G`, rồi IRFFT. Cộng nhánh conv depthwise có cổng. Cuối mạng: `q̂ = q_T + Δt·M⁻¹p_T` với `p` là đạo hàm phổ.

**Kết quả báo cáo:** Beauty, Games, ML-1M; so với DIFF, HSTU, Mamba4Rec, SIGMA, SSD4Rec. Ví dụ Beauty NDCG@10 0,0566 so với 0,0527 của DIFF (+7,4%). ML-1M Hit@10 **thua** HSTU 0,39%, dù văn bản nói "tốt nhất mọi metric". Hiệu năng/hiệu suất: 4,74M tham số, 6,8 s/epoch, nhưng bộ nhớ GPU đỉnh cao hơn Mamba4Rec.

**Điểm yếu và mâu thuẫn:**
- **Không bàn về tính nhân quả.** Phép FFT trên cả chuỗi là tích chập vòng: giá trị tại vị trí `t` phụ thuộc cả vị trí tương lai và vòng ngược từ cuối về đầu. Với mô hình chỉ đọc vị trí cuối và chỉ nhận lịch sử thì không rò rỉ nhãn, nhưng đây là một thiết kế đáng nói.
- Ablation chỉ có biểu đồ cột; chỉ có văn bản nói bản `Ψ = 1` "rất gần" bản đầy đủ. Nghĩa là **bộ mẫu số vật lý mang phần lớn hiệu quả**, tử số học được thêm ít.
- Số của SIGMA mà HSR báo thấp hơn nhiều so với bài gốc SIGMA (Beauty HR@10 0,0678 so với 0,0986), nên baseline được chạy lại theo thiết lập riêng. Số tham số (4,74M) không khớp với hidden nhỏ và 12K item; hidden mặc định không được nêu.
- Không có seed, kiểm định ý nghĩa; Ψ khởi tạo, Δt, kích thước kernel conv đều không nêu.
- Mã nguồn: `github.com/asaander719/HSR` (chưa kiểm tra).

**Chuyển vào FreqMaT:** thêm `freq_type: osc` cạnh `learned`, `bsa`, `both`, `gft`. Biến thể rất nhỏ chỉ có `3·d` tham số (mẫu số) cho `d = 64`. **Phiên bản nhân quả**: thay vì FFT vòng, đệm không (zero-pad) tới ít nhất `2T`, hoặc IRFFT `G` thành nhân tích chập rồi che các độ trễ âm.

**Ghi chú về chính mô hình của ta (phân tích của tôi, chưa đo):** `FreqFilter` hiện tại của ta cũng dùng FFT vòng, nên cũng không nhân quả theo trục chuỗi, giống FMLP-Rec. Điều này **không** gây rò rỉ khi loss chỉ ở vị trí cuối, nhưng sẽ gây rò rỉ nếu sau này ta chuyển sang loss ở mọi vị trí. Nếu muốn dùng loss mọi vị trí (tăng tín hiệu huấn luyện), bộ lọc buộc phải là nhân quả.

**Phán quyết:** ứng viên số một cho đóng góp chính. Mới ở chỗ "nhân quả và có cấu trúc vật lý", không mới ở chỗ "lọc dao động".

### 3.2 SIGMA — Selective Gated Mamba (AAAI-25)

**Đóng góp tuyên bố:** (1) chỉ ra hai hạn chế của Mamba trong gợi ý: quét một chiều và ước lượng trạng thái không ổn định trên chuỗi ngắn; (2) Partially-Flipped Mamba cộng Dense Selective Gate cộng Feature-Extract GRU; (3) thắng trên 5 dataset.

**Cách làm:** lật `n` item đầu của chuỗi, giữ `r` item cuối nguyên vị trí (tốt nhất `r = 5` trên Beauty); chạy hai khối Mamba trên bản gốc và bản lật; trộn bằng cổng `SiLU(x) + sigmoid(x)`. Song song là nhánh conv rồi GRU một chiều; trộn bằng hai hệ số vô hướng học được.

**Kết quả:** Yelp, Sports, Beauty, ML-1M, Games; 5 lần chạy, kiểm định t một phía. Beauty HR@10 0,0986 (+1,96%), NDCG@10 0,0604 (+6,53%). Sports HR@10 thua FEARec 1,47%. Hiệu quả: 1 lớp đã gần tối ưu (66 ms, 3,03 GB).

**Điểm yếu:**
- **Ablation tự mâu thuẫn với câu chuyện.** Trên Beauty: bỏ flip mất khoảng 3,3% HR; bỏ DS-gate mất khoảng 3,3%; **bỏ GRU mất khoảng 24%**. Tiêu đề nhấn mạnh Mamba hai chiều, nhưng nhánh GRU quan trọng hơn khoảng 7 lần.
- Không nêu rõ: hai khối Mamba có chung trọng số không; nhánh lật có được lật ngược lại trước khi trộn không; độ dài chuỗi tối đa, batch, giao thức đánh giá.
- `r` chỉ tune trên Beauty. Chuỗi trung bình chỉ khoảng 8–9 item.
- Về tính tương thích: vì `r` item cuối giữ nguyên vị trí nên vị trí cuối vẫn là item cuối thật, đọc ở đó là hợp lệ. **Nhưng** với tensor đệm bên phải của RecBole, "n item đầu" và "r vị trí cuối" sẽ rơi vào phần đệm với người dùng ngắn. Phải lật theo độ dài thật.
- Mã nguồn: `github.com/Applied-Machine-Learning-Lab/SIMGA` (URL trong paper có lỗi chính tả; chưa kiểm tra).

**Chuyển vào FreqMaT:**
- **Nhánh conv + GRU** (khoảng 25K tham số mỗi lớp ở hidden 64) là ứng viên rẻ nhất. Thêm làm nhánh thứ ba (hoặc thay nhánh khác) trong cổng 3 chiều. Rủi ro: hiệu quả của GRU có thể chỉ là hiệu quả của "bất kỳ đường cục bộ nào", mà RoPE-attention của ta đã phủ.
- **PF-Mamba:** thêm một khối Mamba thứ hai trên chuỗi lật theo độ dài thật; chi phí gấp đôi nhánh Mamba, lợi ích nhỏ. Thử cuối.
- **Dense Selective Gate:** gần như miễn phí, nhưng cổng hợp nhất của ta có thể đã làm vai trò này. Coi như cổng thay thế, không phải đóng góp.

### 3.3 HyMoERec — Hybrid Mixture-of-Experts (AAAI-26, student abstract)

**Đóng góp tuyên bố:** (1) HyMoE: một expert dense dùng chung cộng nhiều expert thưa chuyên biệt; (2) Adaptive Expert Fusion: cổng `α` học được, khởi động tuyến tính, cộng loss cân bằng tải; (3) thắng trên ML-1M và Beauty.

**Cách làm:** `y = y_dense + α·y_MoE`, với `α = sigmoid(α_param)·min(1, t/T_warmup)`; `E = 4` expert, top-2, `T_warmup = 500` bước, `λ_lb = 0.02`; loss cân bằng là entropy âm của xác suất cổng trung bình `Σ ḡ_e log ḡ_e`.

**Kết quả:** Beauty HR@10 0,0756 so với 0,0731 của Mamba4Rec (+3,4%); NDCG@10 0,0442 so với 0,0432 (+2,3%). ML-1M cải thiện 1,2–2,7%.

**Điểm yếu:** chỉ 2 trang. Không có ablation (không tách được expert dùng chung, khởi động, `α` hay loss cân bằng); không có seed, tham số, độ trễ; baseline không có SASRec; không nêu giao thức. BERT4Rec trên Beauty chỉ 0,0243 HR@5 cho thấy thiết lập có vẻ không chuẩn. "Adaptive" thực chất là một hệ số vô hướng chung. Không có mã nguồn.

**Chuyển vào FreqMaT:** ta đã có MoE 8 expert top-2 với loss cân bằng kiểu Switch. Thêm `moe_shared: true` (một FFN dense luôn bật), `moe_warmup_steps`, và tùy chọn loss entropy. Chi phí thấp. **Không được nhận ý tưởng này là của ta** (nó đã công bố, và bắt nguồn từ DeepSeekMoE). Ghi chú: warm-up theo số bước phụ thuộc kích thước dataset (500 bước có thể là nhiều epoch trên dataset nhỏ); nên tính theo epoch.

### 3.4 FlowRec — Prior-Informed Flow Matching

**Đóng góp tuyên bố:** (1) khung Flow Matching với quỹ đạo thẳng từ prior cá nhân hóa tới mục tiêu; (2) prior dựa trên hành vi và loss căn chỉnh một bước; (3) thắng trên 4 dataset.

**Cách làm:** encoder Transformer 4 lớp cho `x0 = H[-1]`; mạng luồng là MLP 2 lớp chỉ nhận `(x_t, t)`, không thấy lịch sử; loss `L = L_Prior + α·L_CFM + β·L_Align` (CE ở `x0`, MSE trường vận tốc, CE ở `x̃1 = x_t + (1−t)·f`). Suy luận Euler 8–10 bước.

**Kết quả:** Beauty HR@5 5,91 so với 5,38 của DiffuRec (+9,87%); Toys +9,64%; Yelp +3,7%; ML-1M +1,9%. Hiệu quả: huấn luyện khoảng 1,9 lần SASRec, suy luận chậm hơn SASRec (1,27 s so với 0,88 s trên Toys).

**Điểm yếu then chốt:**
- Biến thể "Initial" (chấm điểm `x0`, không chạy luồng ở suy luận) đã gần bằng bản đầy đủ: chênh khoảng 0,23 điểm HR@5 trên Beauty (+4% tương đối).
- Bỏ `L_Prior` hay `L_Align` làm mất khoảng nửa hiệu năng. Điều này gợi ý lợi ích đến từ **CE nhiều đầu (deep supervision)**, không phải từ tính chất "luồng".
- **Không có đối chứng** "thêm đầu CE phụ nhưng không có luồng". Vì vậy lợi ích riêng của Flow Matching chưa được chứng minh.
- Baseline đáng ngờ: CL4SRec ML-1M HR@5 1,72 (rõ ràng hỏng). Beauty HR@5 là 5,9123 ở Bảng 2 nhưng 5,9125 ở Bảng 3. Không báo seed. Không có mã nguồn.

**Chuyển vào FreqMaT:** phần có thể dùng là **deep supervision** (CE phụ ở đầu ra lớp 1, trọng số nhỏ). Bộ tinh chỉnh luồng nhỏ (MLP 64 chiều) thì chi phí thấp nhưng bằng chứng yếu. Nếu thử luồng, **bắt buộc có đối chứng "đầu CE phụ không luồng"**, thứ mà không paper nào làm.

### 3.5 FMRec — Flow Matching Based Sequential Recommender (IJCAI-25)

**Đóng góp tuyên bố:** (1) lần đầu đưa Flow Matching vào gợi ý tuần tự; (2) quỹ đạo thẳng với mục tiêu "không nhiễu tương đương" (dự đoán embedding sạch, không dự đoán vận tốc); (3) decoder cùng loss tái dựng tương tác; (4) bộ lấy mẫu ODE tất định; (5) +6,53% trung bình.

**Kết quả:** Beauty HR@5 5,84 so với 5,39 của DiffuRec (+8,2%); Steam, ML-100k, Yelp. Dự đoán vận tốc (v-prediction) cho Beauty HR@5 chỉ 0,58, so với 5,84 khi dự đoán embedding sạch. Không có số liệu hiệu quả.

**Điểm yếu:** "loss đã sửa" thực chất là dự đoán x0 như DDPM; loss CE làm phần lớn việc xếp hạng; không có baseline "không luồng"; phương trình suy luận mâu thuẫn hướng thời gian với phương trình huấn luyện; ML-100k chỉ 943 người dùng nên chênh lệch nhỏ là nhiễu; bảng 2 ghi "FM4Rec". Mã: `github.com/FengLiu-1/FMRec`.

**Chuyển vào FreqMaT:** chỉ có hai thứ rẻ là loss phụ tái dựng đa-nhãn (nên dùng BCE hoặc sampled softmax thay vì MSE trên `|I|`) và nhiễu đầu vào nhỏ (`δ = 0,001`, tác dụng gần như bằng 0). Luồng đầy đủ: không làm.

### 3.6 GMFlowRec — Gaussian Mixture Flow Matching (WWW '26)

**Đóng góp tuyên bố:** khung sinh đa miền gộp luồng hỗn hợp Gauss với học biểu diễn nhận biết miền; bộ mã hóa hai mặt nạ (bất biến miền và riêng miền); gain đến 44% NDCG@5 (JD miền 2).

**Điểm yếu:**
- **Đánh giá bằng lấy mẫu:** 999 mẫu âm ngẫu nhiên trong cùng miền, không phải full-ranking. Số tuyệt đối (H@10 khoảng 15–45) không so được với số của ta; metric lấy mẫu vốn không đáng tin để xếp hạng mô hình.
- Ablation: đầu GMM đơn lẻ chỉ thêm 0,0–0,8 điểm NDCG@10 (ngoại trừ một ô JD miền 5); phần lớn lợi ích đến từ prior nhận biết miền cộng CE theo miền (khoảng 4–9 điểm trên JD).
- Bảng 1 và Bảng 2 không khớp cho cùng cấu hình (ví dụ JD miền 1 NDCG@10: 34,57 so với 34,39); learning rate 1e-4 ở văn bản chính nhưng 0,001 ở phụ lục; quy ước x0/x1 đảo ngược giữa hai mục.
- Đa phương thức (multi-modal) chưa được chứng minh: kỳ vọng hỗn hợp `Σ A_k μ_k` chỉ là một vector.

**Chuyển vào FreqMaT:** hầu như không áp dụng (cần nhãn miền). MoE 8 expert của ta đã phủ vai trò "đầu hỗn hợp". Chỉ có thể rút ra ý "CE phụ trên trạng thái trung gian" (đã ghi ở mục D).

### 3.7 PDFlow — Popularity-Debiased Flow Matching (IJCAI-26)

**Đóng góp tuyên bố:** (1) PDFlow giảm thiên lệch độ phổ biến trong gợi ý sinh; (2) Residual Popularity Vector Field và Cross-Popularity Alignment Loss; (3) thắng trên 4 dataset.

**Cách làm:** trọng tâm phổ biến `c_pop = Σ p(i)·e_i`; một Transformer thứ hai hồi quy về `c_pop`; embedding đã khử lệch `x̃ = x̂ − λ·x̂_pop` với `λ = 0,05`; loss căn chỉnh kéo gần embedding của item phổ biến và item đuôi trong cùng chuỗi.

**Kết quả:** Beauty HR@10 8,70 so với 7,74 của DiffuRec (+12,4%); Toys +14,6%; ML-100K và LastFM nhỏ. Chính paper gọi mức giảm khi bỏ từng thành phần trên Amazon là "tương đối khiêm tốn"; trên Toys, bỏ `v_pop` còn cho HR@10 cao hơn (8,64 so với 8,57).

**Điểm yếu then chốt (phân tích của người đọc):**
- Mục tiêu hồi quy `c_pop` là một **vector hằng chung cho mọi người dùng**. Phép trừ tương đương trừ `λ·(c_pop·e_j)` khỏi logit mỗi item, tức một hiệu chỉnh logit theo độ phổ biến, gần giống điều chỉnh logit theo log-tần suất. Paper không so sánh với baseline đó.
- Dạng từng bước trong ODE chỉ là tái tham số hóa của phép trừ ở điểm cuối, không phải hiệu chỉnh theo người dùng dọc theo quỹ đạo.
- "Trôi quỹ đạo" chỉ được thể hiện bằng PCA định tính; không có thước đo số; không có baseline khử lệch (IPS, logit adjustment); không có metric đuôi dài (coverage, Gini, ARP).
- Quy ước hướng thời gian và ký hiệu `λ` (hai nghĩa) không nhất quán.
- Thành phần khác biệt thật là `L_align`.
- Mã: `github.com/eqmll/PDFlow`.

**Chuyển vào FreqMaT:**
1. **Thêm bộ metric đuôi dài** (coverage@K, Gini, độ phổ biến trung bình của gợi ý, HR trên nửa item đuôi). Đây là chỗ paper này còn trống và là phân tích có giá trị cho bài của ta.
2. **Điều chỉnh logit** `score_j = x·e_j − τ·log p_j` làm đối chứng; chỉ vài dòng.
3. `L_align` ở hệ số nhỏ (0,01–0,05; PDFlow báo `β > 0,1` hơi hại). Rủi ro: kéo các embedding lại gần nhau.
4. Luồng, nhiễu đầu vào, ODE 30 bước: không làm.

### 3.8 HoloMambaRec — Scalable Sequential Recommendation under Latency and Memory Constraints (arXiv)

**Đóng góp tuyên bố:** (1) lớp embedding holographic gắn ID và thuộc tính bằng tích chập vòng, không tăng chiều; (2) bộ mã hóa SSM chọn lọc nông; (3) đường ống đánh giá thống nhất.

**Điểm yếu (nghiêm trọng):**
- **Ablation của chính paper bác bỏ đóng góp chính:** Beauty chênh 0,0007 HR@10 giữa có và không có binding; ML-1M "chỉ item" còn tốt hơn (0,1728 so với 0,1695). Thuộc tính trên Beauty là băm giả `item_id mod 50`, không mang nghĩa.
- Chỉ có SASRec và GRU4Rec làm baseline, huấn luyện đúng 10 epoch. SASRec ML-1M HR@10 chỉ 0,136, quá thấp so với thông lệ, nên mức "+24,6%" không đáng tin. Trên Beauty còn thua GRU4Rec nhiều (0,0426 so với 0,0643).
- Tiêu đề hứa "độ trễ và bộ nhớ" nhưng không đo mô hình đề xuất so với SASRec; chỉ đo trong thí nghiệm nén thất bại, nơi loss giảm xuống 0,80 mà HR sập về 0,0055. Có khả năng đó là lỗi rò rỉ mục tiêu (suy luận của người đọc).
- Báo cáo kết quả ở epoch cuối, không chọn theo validation. Nhiều tài liệu tham khảo không liên quan; thang rigor thấp.
- Mã: `github.com/adithyap/holo-mamba-rec` (chưa kiểm tra).

**Chuyển vào FreqMaT:** không dùng. Ta không có thuộc tính item; và bằng chứng của chính paper là binding không giúp. Lần duy nhất đáng ghi nhận: huấn luyện với **loss ở mọi vị trí** (teacher forcing) là một cách tăng tín hiệu đã biết (không riêng của paper này), nhưng nó đòi hỏi bộ lọc tần số phải nhân quả (xem mục 3.1). Khoảng trống để ta khai thác: **đo hiệu quả (độ trễ, bộ nhớ, tham số) so với baseline mạnh** là thứ paper hứa mà không làm.

### 3.9 UNIQUE — Unified Retrieval and Ranking (RecSys '26, Baidu)

**Đóng góp tuyên bố:** (1) khung hợp nhất sinh mã và chấm điểm có mục tiêu trong một Transformer early-fusion; (2) lượng tử hóa phẳng một lớp có phản hồi, với số hạng cân bằng sử dụng mã; (3) triển khai trên Mobile Baidu với kết quả trực tuyến.

**Dữ liệu và kết quả:** dataset công khai duy nhất là **KuaiRand-Pure** (7.583 item, 1,19M tương tác), gọi là "sanity check". HR@50 0,1510 so với 0,1453 của UniGRF-HSTU (+3,92%). Online: thời lượng xem +0,96%, lượng phân phối +1,08%, giữ chân +0,70%. Ablation: bỏ mục tiêu sinh mất 9,01% HR@50 (lớn nhất); bỏ lượng tử hóa phẳng mất 5,89%.

**Điểm yếu:** không có ablation cho số hạng cân bằng (`τ`) hay `L_code`; không nêu `τ`, `γ`, `α`, `η`, `β`; mô hình dùng đặc trưng phong phú (danh mục, nhà xuất bản, từ khóa tiêu đề, thời lượng, hiển thị); mức tăng online khiêm tốn so với "pipeline cũ" không được mô tả; hai chỗ ghi ngày triển khai khác nhau (tháng 1 và tháng 2/2026). Không có mã.

**Chuyển vào FreqMaT:** hầu như không dùng được: ta không có đa mục tiêu, tiền tố 501 token, hay âm tính từ hiển thị. Ý duy nhất thử được là loss CE phụ dự đoán mã codebook phẳng cân bằng (EMA, thước đo perplexity); nhưng với ID thuần thì codebook phải học từ chính embedding nên mang tính vòng tròn. Kỳ vọng thấp; không nên khẳng định "hợp nhất retrieval và ranking" cho mô hình một đầu tích vô hướng.

### 3.10 Gwhere — Generative Next-POI Recommendation in Amap (RecSys '26, Alibaba)

**Đóng góp tuyên bố:** (1) tokenizer ngữ nghĩa "hoàn toàn tương phản" (lượng tử hóa dư mềm, loss InfoNCE, không có loss tái dựng); (2) khung LLM end-to-end (CPT, SFT, EAKTO); (3) A/B trực tuyến +5,83% P-CTR, +6,20% U-CTR.

**Kết quả:** trên Foursquare-NYC/TKY và Gowalla-CA, chỉ có Acc@1 (không có HR@K hay NDCG). Gwhere-7B được +11,3%/+8,1%/+13,2% so với GNPR-SID, nhưng đầu vào Gwhere có hồ sơ người dùng, baseline dùng ID người dùng, nên không cùng đầu vào. Bản 0,5B (bản triển khai) chỉ hơn khiêm tốn (+2,4%/+2,1%/+6,3%). Số baseline được chép từ paper khác.

**Điểm yếu:** kết quả chính thuộc dữ liệu độc quyền; mô tả "học định danh phân cấp" không được định nghĩa trong phần phương pháp; không báo mức sử dụng codebook, tỉ lệ va chạm; ablation "w/o SID" thay bằng ID ngẫu nhiên nên không tách được tokenizer; công thức EAKTO bị lỗi định dạng trong bản trích; nửa trái và nửa phải của Bảng 5 trộn các cỡ mô hình khác nhau. Mã công bố ở `github.com/alibaba/SimCIT`, tên khác với tên paper, chưa kiểm tra.

**Chuyển vào FreqMaT:** không thể áp dụng khung LLM (10 tỷ token CPT, mô hình 0,5B–7B), EAKTO (cần log hiển thị-không-click), hay tokenizer đa phương thức. Phần có thể xét: đặc trưng không gian cho Gowalla/Foursquare (paper báo không gian là phương thức đơn lẻ có lợi nhất, nhưng chỉ ở dữ liệu độc quyền); loss NT-Xent phụ (dư thừa với CE). Không so số Acc@1 của paper này với số HR/NDCG của ta.

## 4. Các ý tưởng đã bàn trước đó (từ đầu cuộc trao đổi)

| Ý tưởng | Đánh giá sau khi đọc paper |
|---|---|
| Hướng 1: Hamiltonian Flow Matching | Phần "Hamiltonian" (HSR) dùng được ở dạng bộ lọc; phần "Flow Matching" thì không (xem 3.4–3.7). Ghép hai phần chưa có bằng chứng ở đâu. |
| Hướng 2: MoE-MambaFlow | Phần MoE (HyMoERec) và Mamba hai chiều (SIGMA) dùng được ở mức công tắc nhỏ; phần Flow, holographic, GMM thì không. |
| Hướng 3: DF-GenRec | Cần Semantic ID, nội dung và log công nghiệp; không khả thi với dữ liệu ID thuần. |
| Bộ lọc dao động bậc hai (đề xuất của tôi) | Được củng cố: HSR cung cấp công thức và bằng chứng sơ bộ; ta bổ sung bản nhân quả. |
| Expert dùng chung (đề xuất của tôi) | Có nguồn (HyMoERec/DeepSeekMoE) nhưng bằng chứng mỏng. |
| Loss căn chỉnh/tương phản (đề xuất của tôi) | Bằng chứng gián tiếp; dư thừa với CE; ưu tiên thấp. |
| Khử thiên lệch độ phổ biến (đề xuất của tôi) | PDFlow gần như tương đương điều chỉnh logit; giá trị nằm ở metric đuôi dài. |
| Mamba lật một phần | Nhỏ; cần lật theo độ dài thật. |
| Tăng số lớp lên 20 (đã thử 1 epoch) | 20 lớp chạy được (3,17 GB, 95 s/epoch, 2,63M tham số); chưa có kết luận về chất lượng. |

## 5. Kế hoạch thí nghiệm đề xuất

Nguyên tắc: **mỗi ý tưởng là một công tắc cấu hình**, có ablation "w/o X", chạy nhiều seed.

1. **Mốc so sánh (bắt buộc làm trước):** SASRec, FMLP-Rec, BSARec và FreqMaT hiện tại, hội tụ, 3 seed, trên Beauty trước. Tune SASRec đàng hoàng; hầu hết paper ở trên thắng được là nhờ baseline yếu.
2. **Đóng góp chính:** `freq_type: osc` (bản nhân quả), so với `learned`, `bsa`, `both`, `gft`; kèm bản `osc` với `Ψ = 1` (chỉ mẫu số) để kiểm chứng nhận định của HSR.
3. **Ablation phụ:** nhánh GRU cục bộ (B); expert dùng chung và khởi động (C); deep supervision (D).
4. **Phân tích bổ sung:** metric đuôi dài, hiệu quả (độ trễ, bộ nhớ, tham số) so với SASRec và Mamba4Rec, độ dài chuỗi, oversmoothing theo lớp. Có phân tích hệ số khối lượng/cản/độ cứng học được theo dataset như Fig. 7 của HSR nếu `osc` đứng vững.
5. **Thử sau, nếu còn thời gian:** ngoại suy động lượng (G), PF-Mamba (H), `L_align` (F), loss ở mọi vị trí (cần bộ lọc nhân quả).
6. **Không làm:** Flow Matching, Semantic ID, holographic binding, GMM, EAKTO.

Mỗi thí nghiệm nên báo trung bình ± độ lệch chuẩn trên ít nhất 3 seed; chênh lệch cỡ 0,001 NDCG nằm trong nhiễu.

## 6. Câu chuyện bài báo khả dĩ (nếu `osc` và các nhánh phụ có hiệu quả)

> Hành vi người dùng được mô hình hóa như một hệ dao động có tiêu tán trong miền tần số. Bộ lọc dao động **nhân quả** bổ trợ cho nhánh Mamba (phụ thuộc xa) và attention (tương tác cục bộ) qua cổng học được, với MoE thay FFN, và đạt chất lượng ngang hoặc hơn baseline với số tham số nhỏ.

Việc ghi rõ ba điều giúp bài báo thuyết phục: (i) phần nào là của người khác (HSR, SIGMA, HyMoERec); (ii) phần nào là của ta (bản nhân quả, so sánh với FMLP/BSARec/GFT, sự kết hợp ba nhánh, các phân tích); (iii) kết quả âm nếu có (ví dụ `Ψ = 1` đủ tốt như HSR gợi ý).

Nếu không có công tắc nào cải thiện đáng kể so với mốc ở bước 1, bài báo nên chuyển hướng sang **phân tích và hiệu quả** (đối chứng công bằng, đuôi dài, chi phí), thay vì tuyên bố độ chính xác tốt hơn.
