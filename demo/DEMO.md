# Kịch bản quay clip demo SFG-XIDS

Clip khoảng **4 phút**, chạy trên NSL-KDD, seed 0, nền 128 luồng, 16 nút tích phân — đúng cấu
hình mặc định của app. Mọi con số trong file này tôi đã chạy và ghi lại trước; nếu lúc quay màn
hình hiện khác, dừng lại kiểm tra chứ đừng quay tiếp.

---

## 0 · Chuẩn bị (làm trước, không quay)

1. Mở app, **chạy trước một lượt** mỗi thao tác sẽ quay. Lần nạp đầu mất 20–40 giây vì phải đọc
   dữ liệu, dựng mô hình và tính macro-F1 trên toàn tập test; sau đó Streamlit cache lại. Quay cảnh
   chờ 40 giây là thừa.
2. Trình duyệt: phóng **110–125%** để chữ đọc được khi clip bị nén xuống 1080p. Ẩn thanh bookmark.
   Đóng các tab khác.
3. Cửa sổ **1920×1080**, giao diện tối (app đang dùng dark theme).
4. OBS Studio: 1920×1080, 30 fps, bật highlight con trỏ. Quay cả màn hình, không quay webcam.
5. Chuẩn bị sẵn một file CSV một dòng nếu muốn quay cảnh tải file: trong app vào
   **Điều tra một luồng → Từ tập test**, mở *Xem giá trị thô*, hoặc tải hàng đợi CSV ở cảnh 3 rồi
   dùng lại.
6. Không cần lồng tiếng khi quay. Ghi chú thoại bên dưới để anh đọc lại khi lồng tiếng sau, hoặc
   làm phụ đề.

---

## 1 · Mở đầu — hệ thống là gì (0:00 – 0:25)

**Hiện:** trang app vừa nạp xong, chế độ **Giám sát theo lô**.

**Chỉ vào thanh bên**, đọc to ba dòng trạng thái:

```
Thiết bị cpu · σ = 1 · nền 128 luồng
Macro-F1 test 65.50% · 176,791 tham số
Kiểm tra tiền xử lý: lệch tối đa 4.77e-07
```

**Thoại:** "Đây là SFG-XIDS chạy trên NSL-KDD. Mô hình 176 nghìn tham số, macro-F1 65,5% trên toàn
bộ 22.544 luồng của tập test. Dòng thứ ba là kiểm tra tự động: app mã hóa lại dữ liệu test rồi so
với dữ liệu lúc huấn luyện, lệch 4,8e-07 — nghĩa là tiền xử lý đang khớp, không phải tin suông."

> Dòng thứ ba đáng nhấn mạnh: nó chứng minh app đang chạy thật trên checkpoint đã train, chứ không
> phải hiển thị số dựng sẵn.

---

## 2 · Giám sát theo lô — tầng 1 (0:25 – 1:15)

**Thao tác:**

| Ô | Đặt thành |
| --- | --- |
| Số luồng | **500** |
| Chỉ lấy nhãn thật | **(tất cả)** |
| Seed lấy mẫu | **0** |

Bấm **▶ Chạy tầng 1 trên lô này**.

**Con số phải hiện ra** (đã kiểm, seed 0, 500 luồng):

| Chỉ số | Giá trị |
| --- | --- |
| Luồng đã xử lý | 500 |
| Bị gắn cảnh báo | 208 — 41,6% của lô |
| Thông lượng tầng 1 | ~900–1.800 luồng/s (~0,6–1,1 ms/luồng) |
| Đúng nhãn thật | 78,4% (macro-F1 lô 54,8%) |

> Thông lượng phụ thuộc CPU nên sẽ dao động giữa các lần chạy — đó là chỉ số duy nhất được phép
> khác. Ba chỉ số còn lại là tất định, phải ra đúng như trên.

**Thoại:** "Tầng 1 chạy cho cả 500 luồng trong chưa tới một giây — khoảng nửa mili-giây mỗi luồng.
Với mỗi luồng, ngoài nhãn dự đoán, mô hình cho luôn đóng góp của bốn nhóm đặc trưng. Đây là phần
chạy thường trực khi triển khai thật."

Kéo xuống hai biểu đồ **Lô chia theo nhãn dự đoán** và **Nhóm đặc trưng chi phối quyết định**.

**Thoại:** "Bên phải là điều mà một IDS thông thường không nói được: nhóm đặc trưng nào đang chi
phối quyết định, trên toàn bộ lô."

---

## 3 · Hàng đợi cảnh báo (1:15 – 1:50)

Kéo xuống mục **3 · Hàng đợi cảnh báo**. Để nguyên ô *Chỉ hiện luồng bị gắn cảnh báo*.

**Chỉ vào các cột**, đọc một dòng đầu bảng làm ví dụ:

**Thoại:** "Hàng đợi xếp theo mức nghi vấn, bằng 1 trừ xác suất Normal. Bốn cột φ cuối là đóng góp
của từng nhóm vào logit của lớp được dự đoán. Chuyên viên trực nhìn vào là biết ngay luồng này bị
nêu vì nhóm nào, chưa cần mở gì thêm."

Bỏ tick *Chỉ hiện luồng bị gắn cảnh báo* để thấy cả lô, rồi tick lại.

Bấm **⬇ Tải hàng đợi (CSV)** cho thấy xuất được ra file.

---

## 4 · Điều tra một luồng — tầng 2 (1:50 – 3:10)

Đây là phần quan trọng nhất của clip. Chuyển sang chế độ **Điều tra một luồng** ở thanh bên, tab
**Từ tập test**, gõ chỉ số dòng rồi bấm **▶ Phân tích luồng này**.

### 4a · Dòng 15073 — R2L, nhóm Content dẫn dắt

| Mục | Giá trị phải hiện |
| --- | --- |
| Nhãn dự đoán | **R2L** (nhãn thật R2L) |
| Xác suất | **72,3%** |
| φ bốn nhóm | Basic +0,90 · **Content +3,73** · Time +1,25 · Host +2,35 |
| Nhóm mạnh nhất | Content |
| Ba đặc trưng đầu (tầng 2) | `is_guest_login = 1.0` → **+1,845**<br>`hot = 2.0` → +0,997<br>`logged_in = 1.0` → +0,903 |
| Sai số Σφ | ~5e-07 |

**Thoại:** "Luồng này bị xếp vào R2L — remote-to-local, kẻ tấn công từ xa chiếm quyền tài khoản.
Nhóm Content đóng góp mạnh nhất. Mở nhóm Content xuống từng đặc trưng thì thấy lý do: đăng nhập
bằng tài khoản khách, và hai chỉ báo nội dung phiên bất thường. Đúng là chữ ký kinh điển của R2L —
mô hình không hề được dạy điều đó, nó học ra từ dữ liệu."

> **Cẩn thận, đừng nói quá:** đây là **một ca minh họa**, không phải bằng chứng. Trong 3.000 luồng
> tôi lấy mẫu, R2L có 71 luồng do nhóm Basic chi phối và chỉ 3 luồng do Content chi phối. Kết luận
> Lee & Stolfo trong bài báo là về **hồ sơ trung bình |φ| theo lớp** (Content 0,344 ở R2L/U2R so
> với 0,098 ở DoS/Probe, đúng ở cả 5 seed), không phải về từng luồng riêng lẻ. Nếu quay cảnh này,
> hãy nói "một ví dụ điển hình", và để kết luận thống kê cho bảng H1/H2 ở README.

### 4b · Dòng 21176 — Probe, gần như chắc chắn

| Mục | Giá trị phải hiện |
| --- | --- |
| Nhãn dự đoán | **Probe** (nhãn thật Probe), xác suất **100,0%** |
| φ bốn nhóm | **Basic +5,08** · Content +1,10 · Time +4,42 · Host +3,92 |
| Ba đặc trưng đầu | `protocol_type = icmp` → **+1,648**<br>`flag = SF` → +1,294<br>`src_bytes = 20.0` → +0,997 |

**Thoại:** "Một ca quét mạng. Giao thức ICMP, gói 20 byte — đúng dạng ping sweep. Ở đây cả Basic và
Time đều cao, hợp lý: quét mạng để lại dấu vết cả ở thông tin kết nối lẫn ở mật độ lưu lượng."

Bấm qua lại các nút **Basic / Content / Time / Host** ở mục 4 để cho thấy mở được từng nhóm.

Chỉ vào dòng chú thích dưới biểu đồ tầng 2:

**Thoại:** "Dòng này là thứ đáng tin nhất trong cả giao diện: tổng đóng góp của các đặc trưng trong
nhóm phải bằng φ của nhóm đó. Chênh lệch chỉ là sai số tính tích phân, dưới một phần trăm. Hai tầng
giải thích khớp nhau, không phải hai câu chuyện rời rạc."

### 4c · Dòng 18827 — mô hình dự đoán SAI (đừng bỏ cảnh này)

| Mục | Giá trị phải hiện |
| --- | --- |
| Nhãn thật | **Normal** |
| Nhãn dự đoán | **Probe**, xác suất **100,0%** |
| φ bốn nhóm | Basic +3,66 · Content +0,83 · **Time +4,39** · Host +3,82 |
| Ba đặc trưng đầu | `diff_srv_rate = 0.54` → **+2,330**<br>`rerror_rate = 1.0` → +0,916<br>`srv_rerror_rate = 1.0` → +0,420 |

**Thoại:** "Và đây là một ca mô hình sai — luồng này thật ra bình thường, nhưng bị gắn nhãn Probe
với xác suất gần như tuyệt đối. Lời giải thích cho biết vì sao: hơn một nửa số kết nối đi tới dịch
vụ khác nhau, và toàn bộ đều trả về lỗi. Đó đúng là hành vi của máy quét. Điểm đáng nói không phải
là mô hình sai — mọi mô hình đều sai đâu đó — mà là chuyên viên đọc được **vì sao** nó sai trong
vài giây, thay vì phải tin hoặc không tin một con số."

> Cảnh này làm clip đáng tin hơn hẳn. Người phản biện nào cũng cảnh giác với demo chỉ toàn ca thành
> công.

---

## 5 · Chốt lại (3:10 – 3:40)

Quay lại chế độ **Giám sát theo lô** để thấy hàng đợi vẫn còn nguyên.

**Thoại:** "Tóm lại: tầng 1 nửa mili-giây mỗi luồng, chạy cho mọi luồng, cho biết nhóm đặc trưng
nào chịu trách nhiệm. Tầng 2 vài giây, chỉ chạy cho luồng chuyên viên mở ra, và mở tới từng đặc
trưng. Chênh nhau ba bậc — đó là lý do lời giải thích được thiết kế hai tầng, chứ không phải để cho
đẹp. Toàn bộ số liệu trong clip này được tính trực tiếp từ checkpoint đã huấn luyện, ngay lúc bấm
nút."

---

## Bảng các dòng đã kiểm chứng (NSL-KDD, seed 0)

Dùng khi cần thay ca khác hoặc quay lại cảnh nào đó.

| Dòng | Nhãn thật | Dự đoán | Xác suất | Nhóm mạnh nhất | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 15073 | R2L | R2L | 0,723 | Content +3,73 | ca Content dẫn dắt, hiếm |
| 17142 | R2L | R2L | 0,629 | Content +3,17 | dự phòng cho 15073 |
| 21176 | Probe | Probe | 1,000 | Basic +5,08 | ping sweep ICMP |
| 6059 | Probe | Probe | 1,000 | Time +4,94 | Probe nhưng Time dẫn dắt |
| 20904 | DoS | DoS | 1,000 | Host +3,68 | DoS điển hình |
| 19895 | Normal | Normal | 1,000 | Host +2,23 | luồng sạch, φ đều và thấp |
| 14220 | U2R | U2R | 0,648 | Basic +4,98 | lớp hiếm nhất; `service = telnet` +2,603 |
| 18827 | Normal | **Probe** | 1,000 | Time +4,39 | ca dự đoán sai |

---

## 6 · Xuất clip và đưa lên GitHub

**File mp4:** xuất 1080p, H.264, khoảng 8–10 Mbps — clip 4 phút ra chừng 20–30 MB. GitHub chặn
file trên 100 MB và cảnh báo từ 50 MB, nên giữ dưới 25 MB cho nhẹ repo.

**GIF để hiện thẳng trong README:** GitHub không phát mp4 nhúng trong README, nhưng phát GIF. Cắt
lấy **10–15 giây đắt nhất** — đoạn bấm *Phân tích luồng này* ở dòng 15073 rồi biểu đồ φ và tầng 2
hiện ra — xuất GIF rộng 800 px, 10–12 fps, dưới 10 MB:

```bash
ffmpeg -i demo.mp4 -ss 00:02:05 -t 14 -vf "fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" demo/demo.gif
```

**Đặt file:**

```
demo/demo.gif      <- README hiện thẳng, dòng ![demo](demo/demo.gif) đã có sẵn
demo/demo.mp4      <- clip đầy đủ
demo/DEMO.md       <- file này
```

**Muốn mp4 phát được ngay trong README:** kéo thả file mp4 vào ô soạn một Issue bất kỳ của repo,
GitHub trả về một link `user-images.githubusercontent.com`; dán link đó vào README là clip phát
inline. Không cần commit mp4 vào repo. Cách này gọn hơn, nhất là nếu clip trên 25 MB.
