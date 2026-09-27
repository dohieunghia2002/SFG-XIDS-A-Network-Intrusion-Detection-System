# SFG-XIDS — ứng dụng demo chạy thật

App Streamlit nạp checkpoint đã train ở giai đoạn 1 và chạy thật trên lưu lượng đưa vào.
Không có số liệu dựng sẵn trong giao diện: mọi con số tính trực tiếp từ checkpoint.

Hai chế độ, đúng theo cách một hệ thống thật được triển khai:

| Chế độ | Chạy gì | Giá |
| --- | --- | --- |
| **Giám sát theo lô** | tầng 1 cho mọi luồng: dự đoán lớp + đóng góp 4 nhóm đặc trưng (φ, bằng giá trị Shapley của nhóm) → hàng đợi cảnh báo xếp theo mức nghi vấn | ~0,6 ms/luồng (CPU), ~1.700 luồng/s |
| **Điều tra một luồng** | thêm tầng 2: Expected Gradients trong nhóm cho từng đặc trưng | ~0,2 s/luồng (GPU T4), vài giây trên CPU |

Tầng 2 đắt hơn tầng 1 khoảng ba bậc, nên nó chỉ chạy cho luồng chuyên viên chọn — đây
chính là lý do giải thích được thiết kế 2 cấp. Từ hàng đợi cảnh báo bấm
**Mở phân tích 2 cấp cho luồng này** là sang thẳng chế độ điều tra đúng luồng đó.

## Cài đặt (máy Windows, chạy một lần)

```bat
cd D:\data_fusion\app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`torch` bản CPU là đủ. `xgboost` chỉ cần cho nhánh UNSW-NB15 (`step0_unsw.py` import nó
ngay khi nạp).

## Thư mục

App không cần chép mã nguồn đi đâu cả. Nó tìm theo thứ tự: `SFG_ROOT` (biến môi trường) →
thư mục cha của app → thư mục app.

```
D:\data_fusion\
    step0_sfg.py  full_sfg.py  phase2_sfg.py  step0_unsw.py    <- mã nguồn nghiên cứu
    app\
        app_sfg_xids.py  README.md  requirements.txt
        model\nsl_sfg_tax_0.pt  unsw_sfg_tax_0.pt  ...         <- checkpoint (xem bên dưới)
        data\KDDTrain+.txt  KDDTest+.txt                       <- NSL-KDD (125.973 / 22.544 dòng)
        data\UNSW_NB15_training-set.csv  ...testing-set.csv    <- UNSW-NB15 (175.341 / 82.332 dòng)
```

Dời thư mục dự án thì đặt biến môi trường `SFG_ROOT`, hoặc sửa hai ô đường dẫn ở thanh bên.

## Checkpoint app dùng

App chỉ nạp **một loại** checkpoint: `{bộ dữ liệu}_sfg_tax_{seed}.pt` — SFG-XIDS theo taxonomy.
Các file `*_attn_*.pt`, `*_sfg_p0_*.pt`, `*_xgb_*.ubj` của giai đoạn 1 không được dùng.

Chép từ `sfg_results\full_v1\models\` sang `app\model\`:

```
nsl_sfg_tax_0.pt   nsl_sfg_tax_1.pt   nsl_sfg_tax_2.pt   nsl_sfg_tax_3.pt   nsl_sfg_tax_4.pt
unsw_sfg_tax_0.pt  unsw_sfg_tax_1.pt  unsw_sfg_tax_2.pt  unsw_sfg_tax_3.pt  unsw_sfg_tax_4.pt
```

Tổng ~7,5 MB. Chỉ demo seed 0 thì hai file seed 0 là đủ — ô **Seed mô hình** ở thanh bên chỉ
liệt kê seed thật sự có file. Thứ tự app dò thư mục checkpoint: biến môi trường `SFG_MODELS`
→ `app\model\` → `app\models\` → `sfg_results\full_v1\models\`.

`results.json` là tùy chọn, chỉ để hiện đúng σ ở thanh bên; không có thì app dùng σ = 1
(NSL-KDD) và σ = 10 (UNSW-NB15), đúng bằng giá trị đã chọn trên validation, nên kết quả
không đổi (đã kiểm: macro-F1 65,503% / 53,999% có hay không có file này).

## Chạy

```bat
streamlit run app_sfg_xids.py
```

Trình duyệt mở ở `http://localhost:8501`. Lần nạp đầu mất khoảng 20–40 giây (đọc dữ liệu,
dựng mô hình, kiểm macro-F1 trên toàn tập test), sau đó được cache.

## Đưa lưu lượng vào

**Chế độ giám sát theo lô**

1. **Lấy mẫu từ tập test** — 100 ÷ 2.000 luồng, lọc theo nhãn thật, seed lấy mẫu cố định.
2. **Tải file CSV** — mỗi dòng một luồng, cần đủ các cột đặc trưng.

Kết quả: bốn chỉ số của lô (số luồng, tỷ lệ bị gắn cảnh báo, thông lượng tầng 1, độ đúng
so với nhãn thật), lô chia theo nhãn dự đoán, nhóm đặc trưng chi phối quyết định, và hàng
đợi cảnh báo có đủ bốn cột φ — tải về CSV được.

**Chế độ điều tra một luồng**

1. **Từ tập test** — chọn lớp, lấy ngẫu nhiên hoặc gõ chỉ số dòng.
2. **Sửa tay một luồng** — lấy một dòng làm mẫu rồi sửa giá trị bất kỳ; app tiền xử lý lại
   bằng đúng tham số fit trên tập train.
3. **Tải file CSV** — chọn dòng trong file.

## Kiểm tra tính đúng ngay trên giao diện

- Thanh bên hiện **lệch tối đa của bước tiền xử lý** khi mã hóa lại tập test (phải ~5e-7):
  số này lớn thì dữ liệu hoặc từ điển đã khác lúc train.
- Mục 3 hiện **sai số đẳng thức Σφᵢ = z_c − E[z_c]** (cỡ 1e-6 ÷ 1e-7).
- Mục 4 hiện **độ lệch giữa Σ EG trong nhóm và φ của nhóm** theo % tổng độ lớn đóng góp
  (16 nút tích phân cho 0,05 ÷ 0,6%).
- φ tính theo lô và φ tính cho một luồng trùng nhau đến 5e-7, nên hàng đợi cảnh báo và
  trang điều tra không bao giờ nói hai điều khác nhau.

## Số đã đo được (CPU, nền 128 luồng, 16 nút tích phân)

| | NSL-KDD seed 0 | UNSW-NB15 seed 0 |
| --- | --- | --- |
| macro-F1 toàn tập test | 65,50% | 54,00% |
| tham số mô hình | 176.791 | 185.374 |
| tầng 1, lô 500 luồng | 0,28 s (1.769 luồng/s) | 0,29 s (1.729 luồng/s) |
| lệch tiền xử lý | 4,8e-7 | 4,8e-7 |
| sai số Σφ | 1,9e-6 | 1,9e-6 |

## Đưa lên web cho người khác xem

Streamlit Community Cloud (miễn phí): đẩy thư mục app lên một repo GitHub riêng tư, gồm
`app_sfg_xids.py`, 4 file mã nguồn, `requirements.txt`, `model\nsl_sfg_tax_0.pt` (~730 KB) và cặp file
NSL-KDD (~22 MB). Vào share.streamlit.io, chọn repo, chọn `app_sfg_xids.py`. Dùng bản
`torch` CPU trong `requirements.txt` để không vượt hạn mức bộ nhớ; nhánh UNSW-NB15 nặng
~47 MB dữ liệu nên nếu chỉ demo thì để NSL-KDD.

## Lưu ý về bộ UNSW-NB15

Vài bản trên Kaggle đặt tên hai file ngược nhau (file tên `training-set` chỉ có 82.332
dòng). App tự phát hiện theo số dòng, hoán lại đúng bộ chia chính thức và hiện cảnh báo ở
đầu trang. Bộ dữ liệu đang có trong `app\data` đặt tên đúng chiều nên không có cảnh báo.
Với bộ nào khác số dòng chuẩn, app dừng và báo lỗi thay vì chạy tiếp.

## Giới hạn

- Mô hình train trên NSL-KDD/UNSW-NB15 nên chỉ nhận đúng định dạng đặc trưng của hai bộ
  này; app không bắt gói tin trực tiếp từ card mạng. Muốn nối vào luồng thật thì cần một
  bước trích đặc trưng (Argus/Bro-Zeek như bài gốc của UNSW-NB15) đặt trước app.
- Tầng 2 trên CPU mất vài giây mỗi luồng. Giảm số luồng nền xuống 64 hoặc số nút tích phân
  xuống 8 nếu cần nhanh hơn, đổi lại độ lệch khớp 2 cấp tăng lên.
