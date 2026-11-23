# Datasets

Tất cả dataset được chuyển sang định dạng RecBole (`data/processed/<tên>/<tên>.inter`, cột `user_id, item_id, rating, timestamp`).
File thô nằm ở `data/raw/<dataset>/`, kết quả ở `data/processed/`. Script: `scripts/download_raw.py` (tải) và `scripts/prepare_data.py` (chuyển đổi).

## Trạng thái

Số liệu là **trước** khi lọc 5-core của RecBole (`configs/protocol.yaml`), nên số user/item thực tế khi train sẽ ít hơn.

| Tên trong danh sách của bạn | Tên thư mục trong `data/` | Tương tác | User | Item | Trạng thái |
|---|---|---|---|---|---|
| MovieLens-1M | `ml-1m` | 1.000.209 | 6.040 | 3.706 | Đã có |
| Amazon-Beauty | `Amazon_Beauty` | 198.502 | 22.363 | 12.101 | Đã có |
| Amazon-Sports = Sports | `Amazon_Sports_and_Outdoors` | 296.337 | 35.598 | 18.357 | Đã có |
| Toys | `Amazon_Toys_and_Games` | 167.597 | 19.412 | 11.924 | Đã có |
| Amazon-Video-Games (và "Amazon-Games") | `Amazon_Video_Games` | 231.780 | 24.303 | 10.672 | Đã có |
| LastFM (1K users) | `LastFM` | 19.150.868 | 992 | 176.892 | Đã có |
| KuaiRand-1K (và "KuaiRand") | `KuaiRand_1K` | 4.429.840 | 1.000 | 1.819.489 | Đã có |
| Check-in New York | `MassiveSTEPS_NewYork` | 272.368 | 6.929 | 45.804 | Đã có |
| Check-in California | `Gowalla_CA` | 736.203 | 16.599 | 115.414 | Đã có (xấp xỉ, xem ghi chú) |
| Check-in Texas | `Gowalla_TX` | 943.102 | 17.245 | 121.145 | Đã có (xấp xỉ, xem ghi chú) |
| MovieLens-20M | `ml-20m` | 20.000.263 | 138.493 | 26.744 | Đã có |
| Amazon-Books | `Amazon_Books` | 8.898.041 | 603.668 | 367.982 | Đã có |
| Yelp | `Yelp` | 6.990.280 | — | — | Đã có (toàn bộ review, xem ghi chú) |

## Bộ rút gọn để huấn luyện

RecBole biến mọi tiền tố của lịch sử một user thành một mẫu huấn luyện, nên thời gian mỗi epoch tỉ lệ với số tương tác, và bảng 20 triệu dòng cần hàng chục GB RAM chỉ để dựng mẫu. Năm bộ lớn hoặc không hợp lý ở dạng gốc được rút gọn bằng `python scripts/make_subsets.py` (cấu hình trong `src/freqmat/data/reduce.py`, hạt giống 2024, bản gốc không bị sửa). Thứ tự áp dụng: gộp các lần lặp liên tiếp cùng item, lọc 5-core, lấy mẫu ngẫu nhiên user, giữ N tương tác gần nhất của mỗi user, lọc 5-core lần nữa.

| Bộ rút gọn | Từ | Cách rút gọn | Tương tác | User | Item |
|---|---|---|---|---|---|
| `ml-20m_sub` | `ml-20m` | 8.000 user, 100 tương tác gần nhất | 528.031 | 8.000 | 6.481 |
| `Amazon_Books_sub` | `Amazon_Books` | 100.000 user | 860.869 | 60.351 | 62.984 |
| `Yelp_sub` | `Yelp` | 50.000 user | 603.571 | 40.271 | 35.447 |
| `KuaiRand_1K_sub` | `KuaiRand_1K` | 200 lượt click gần nhất | 91.962 | 998 | 10.513 |
| `LastFM_sub` | `LastFM` | gộp lần nghe lặp liên tiếp, 200 sự kiện gần nhất | 159.890 | 987 | 7.063 |

Dùng bộ rút gọn khi huấn luyện, và ghi cách rút gọn trong bài báo. `LastFM` gốc (95% cặp user-item lặp, 19 triệu dòng) và `ml-20m`, `Amazon_Books`, `Yelp` gốc không dùng để huấn luyện trực tiếp. Số tương tác trong bảng là số sau lọc, trước khi RecBole bỏ hai tương tác cuối mỗi user cho valid/test.

## Kiểm tra bộ dữ liệu có huấn luyện được không

```bash
.venv/bin/python scripts/check_data.py -d Amazon_Beauty --steps 30
```

Script nạp dataset qua RecBole, chạy các bước huấn luyện thật với cấu hình mặc định và in số user/item, số tham số, thời gian mỗi bước, thời gian ước tính mỗi epoch, bộ nhớ GPU và RAM đỉnh.

## Yelp (tải tay, đã xong)

Yelp đã được tải và chuyển đổi. Phần dưới ghi lại cách làm lại nếu cần. Yelp yêu cầu điền form và đồng ý điều khoản nên không tải tự động được.

1. Vào https://www.yelp.com/dataset
2. Bấm **Download JSON** (hoặc trang https://www.yelp.com/dataset/download), điền họ tên, email và chấp nhận điều khoản (chỉ dùng phi thương mại/học thuật).
3. Tải về file `yelp_dataset.tar` (khoảng 4–5 GB, giải nén ra khoảng 9 GB).
4. Giải nén và lấy đúng file `yelp_academic_dataset_review.json` (khoảng 5 GB):
   ```bash
   tar -xf yelp_dataset.tar yelp_academic_dataset_review.json
   ```
5. Đặt file vào `data/raw/yelp/yelp_academic_dataset_review.json`.
6. Chuyển sang định dạng RecBole:
   ```bash
   .venv/bin/python scripts/prepare_data.py --datasets Yelp
   ```
   Lệnh này đọc file theo từng khối nên tốn ít RAM hơn bản cũ. Kết quả là `data/processed/Yelp/Yelp.inter`.

Tên file hoặc cấu trúc trang tải của Yelp có thể thay đổi theo từng phiên bản. Nếu file khác tên, hãy báo lại để chỉnh converter.

## Chạy huấn luyện trên một dataset

```bash
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Video_Games
.venv/bin/python scripts/train.py -m FreqMaT -d LastFM_sub   # độ dài 200 và batch 128 nằm trong configs/dataset/LastFM_sub.yaml
```

## Ghi chú quan trọng (đọc trước khi viết bài báo)

- **California và Texas là số liệu xấp xỉ, không trùng với bài báo nào.** Tôi không tìm thấy nguồn nào cung cấp sẵn tập check-in Texas. Hai tập này được cắt từ Gowalla toàn cầu (SNAP `loc-gowalla_totalCheckins.txt.gz`) theo khung tọa độ chữ nhật:
  - California: vĩ độ 32,5–42,0, kinh độ −124,5 đến −114,1.
  - Texas: vĩ độ 25,8–36,5, kinh độ −106,7 đến −93,5.

  Khung chữ nhật không khớp đúng biên giới bang, nên có thể lẫn vài điểm của bang lân cận (ví dụ Nevada, New Mexico, Oklahoma). Đừng so số trực tiếp với các bài dùng "Gowalla-CA". Nếu cần khớp, hãy dùng đúng bản tiền xử lý của bài đó.
- **New York** lấy từ [Massive-STEPS](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-New-York) (Foursquare, 2012–2013), không phải Foursquare-NYC của Yang et al. (TIST 2015). Hai bản có số liệu khác nhau.
- **KuaiRand-1K:** interaction là các video có `is_click = 1` trong hai log chuẩn. Log phơi bày ngẫu nhiên (`log_random_*`) bị loại vì nó dùng cho mục đích gỡ thiên lệch. Bộ này chỉ có 1.000 user nhưng rất nhiều video, nên sau lọc 5-core số item sẽ giảm mạnh.
- **LastFM-1K:** item là nghệ sĩ (không phải bài hát), timestamp là thời điểm nghe. Chuỗi rất dài nên nên tăng `MAX_ITEM_LIST_LENGTH`.
- **"Amazon-Games" trong danh sách bị mơ hồ:** tôi hiểu là Video Games (đã tải) vì Toys & Games đã có là "Toys". Nếu bạn muốn khác, hãy báo lại.
- **Yelp dùng toàn bộ review**, không lọc từ 2019-01-01 như bài BSARec, nên chưa so sánh trực tiếp được với bảng số của họ. **LastFM** là LastFM-1K, không phải HetRec-2011 mà BSARec dùng.
- **Amazon-Books và ML-20M rất lớn** (8,9 triệu và 20 triệu tương tác). Với GPU 10 GB, bảng embedding và logits có thể gây hết bộ nhớ. Tôi chưa đo trên hai bộ này, nên có thể phải giảm `train_batch_size`.
- **Giấy phép:** Amazon 2014, ML-1M/20M, Gowalla và LastFM chỉ dành cho nghiên cứu phi thương mại. KuaiRand và Yelp có điều khoản riêng, cần đọc khi công bố.
