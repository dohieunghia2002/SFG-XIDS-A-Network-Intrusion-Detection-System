# Kịch bản quay clip demo SFG-XIDS

Hai phần: **phần A — NSL-KDD** (4 phút) và **phần B — UNSW-NB15** (3 phút). Quay riêng hai
clip, hoặc ghép thành một clip 7 phút. Phần B không lặp lại phần A: nó cho thấy mô hình
hành xử thế nào trên bài toán 10 lớp khó hơn nhiều, kể cả chỗ nó thất bại.

---

# PHẦN A — NSL-KDD (5 lớp)

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


---

# PHẦN B — UNSW-NB15 (10 lớp)

Chuyển **Bộ dữ liệu → UNSW-NB15 (10 lớp)** ở thanh bên. Lần đổi đầu mất 20–40 giây để nạp và
tính macro-F1 trên 82.332 luồng test — chạy trước một lượt rồi mới quay.

Phần này **không** phải bản sao của phần A. Nó cho thấy ba thứ mà NSL-KDD không cho thấy được:
mô hình xử lý 10 lớp mất cân bằng nặng ra sao, lời giải thích phơi bày artefact của bộ dữ liệu như
thế nào, và mô hình nhầm ở đâu.

## B1 · Mở đầu — bài toán khó hơn hẳn (0:00 – 0:30)

**Thanh bên phải hiện:**

```
Thiết bị cpu · σ = 10 · nền 128 luồng
Macro-F1 test 54.00% · 185,374 tham số
Kiểm tra tiền xử lý: lệch tối đa 4.77e-07
```

**Thoại:** "Cùng kiến trúc đó, giờ chạy trên UNSW-NB15: 10 lớp tấn công thay vì 5, 82 nghìn luồng
test. Macro-F1 tụt từ 65,5% xuống 54%. Đây là bộ dữ liệu khó, và tôi sẽ cho thấy khó ở chỗ nào."

> σ hiện là **10**, khác phần A (σ = 1). Đó là giá trị chọn trên validation riêng cho bộ này, không
> phải lỗi.

## B2 · Giám sát theo lô (0:30 – 1:10)

Số luồng **500** · Chỉ lấy nhãn thật **(tất cả)** · Seed lấy mẫu **0** → **▶ Chạy tầng 1**

| Chỉ số | Giá trị |
| --- | --- |
| Luồng đã xử lý | 500 |
| Bị gắn cảnh báo | **343 — 68,6% của lô** |
| Thông lượng tầng 1 | ~900–1.700 luồng/s |
| Đúng nhãn thật | **74,2%** (macro-F1 lô 48,3%) |

**Thoại:** "Gần 69% lô bị gắn cảnh báo — vì tập test UNSW-NB15 có tỷ lệ tấn công cao hơn hẳn
NSL-KDD. Thông lượng tầng 1 không đổi dù số lớp gấp đôi: vẫn dưới một mili-giây mỗi luồng."

Chỉ vào biểu đồ **Nhóm đặc trưng chi phối quyết định**.

**Thoại:** "Ở bộ này nhóm Additional — các đặc trưng đếm được tạo thêm — chi phối nhiều hơn hẳn so
với NSL-KDD. Lát nữa sẽ thấy vì sao, và đó không hẳn là tin tốt."

## B3 · Ba ca điển hình (1:10 – 2:20)

Chuyển **Điều tra một luồng → Từ tập test**.

### Dòng 445 — Reconnaissance, nhóm Time dẫn dắt

| | |
| --- | --- |
| Dự đoán | **Reconnaissance**, xác suất **100,0%** (nhãn thật đúng) |
| φ | Basic +4,53 · Content +4,38 · **Time +5,04** · Additional +2,48 |
| Ba đặc trưng đầu | `sinpkt = 90.91` → **+1,289**<br>`djit = 147.13` → +1,018<br>`sjit = 6157.30` → +1,012 |

**Thoại:** "Quét mạng. Nhóm Time dẫn dắt, và mở ra thì thấy toàn là đặc trưng về nhịp gói tin —
khoảng cách giữa các gói, độ rung. Đúng dấu vết của công cụ quét tự động: gửi đều đặn theo máy chứ
không theo nhịp người dùng."

### Dòng 65292 — Shellcode, nhóm Content dẫn dắt

| | |
| --- | --- |
| Dự đoán | **Shellcode**, xác suất **98,6%** (nhãn thật đúng) |
| φ | Basic +2,87 · **Content +4,10** · Time +0,37 · Additional +1,87 |
| Ba đặc trưng đầu | `smean = 88.0` → **+2,476**<br>`dmean = 0.0` → +0,843<br>`dwin = 0.0` → +0,332 |

**Thoại:** "Shellcode — mã khai thác nhúng trong gói tin. Ở đây nhóm Content dẫn dắt, và đặc trưng
mạnh nhất là kích thước gói trung bình chiều đi: 88 byte, trong khi chiều về bằng 0. Một payload
nhỏ gửi đi, không có phản hồi. Nhóm Time gần như không đóng góp gì — hợp lý, vì đây là chuyện nội
dung chứ không phải nhịp lưu lượng."

### Dòng 15356 — Generic, và lời giải thích phơi bày một artefact

| | |
| --- | --- |
| Dự đoán | **Generic**, xác suất **99,9%** (nhãn thật đúng) |
| φ | Basic +2,25 · Content +2,03 · Time +0,92 · **Additional +2,46** |
| Ba đặc trưng đầu | `ct_state_ttl = 2.0` → **+0,584**<br>`ct_src_dport_ltm = 43.0` → +0,428<br>`ct_srv_dst = 43.0` → +0,325 |

**Đây là cảnh đáng giá nhất của phần B.**

**Thoại:** "Lớp Generic mô hình nhận gần như tuyệt đối — F1 98,5%. Nhưng nhìn vào đặc trưng mạnh
nhất: `ct_state_ttl`, một đặc trưng dẫn xuất từ Time-To-Live. TTL là thuộc tính của môi trường tạo
dữ liệu, không phải của hành vi tấn công. Lời giải thích đang nói với chúng ta rằng mô hình dựa
vào một artefact của bộ dữ liệu — và đó chính là lý do trong bài báo chúng tôi có một thí nghiệm
riêng, bỏ hẳn ba đặc trưng TTL ra và đo lại. Một mô hình chỉ đưa ra con số độ chính xác sẽ không
bao giờ để lộ điều này."

> Nối thẳng với ablation A5 trong `sfg_results/summary.json`. Nếu anh muốn mạnh hơn nữa, mở
> **dòng 82330** (Normal, xác suất 100%): đặc trưng `is_sm_ips_ports = 1.0` một mình đóng góp
> **+4,956**, tức IP nguồn trùng IP đích và cổng trùng cổng — một artefact còn thô hơn.

## B4 · Chỗ mô hình thất bại (2:20 – 3:00)

### Dòng 62785 — Exploits bị gọi thành Shellcode

| | |
| --- | --- |
| Nhãn thật | **Exploits** |
| Dự đoán | **Shellcode**, xác suất **97,4%** |
| φ | **Basic +3,63** · Content +1,89 · Time +2,62 · Additional +1,17 |
| Ba đặc trưng đầu | `service = -` → **+1,461**<br>`dttl = 252.0` → +0,467<br>`state = FIN` → +0,393 |

**Thoại:** "Và đây là chỗ mô hình sai. Luồng này là Exploits nhưng bị gọi thành Shellcode với 97%
tự tin. Đặc trưng mạnh nhất là `service` không xác định được — đúng là thứ hai lớp này chia chung.
Exploits và Shellcode vốn chồng lấn về bản chất: shellcode thường là một phần của exploit."

**Nói thẳng con số, đừng né** — mở lại chế độ theo lô, chỉ vào biểu đồ chia theo lớp:

**Thoại:** "Trên mẫu 500 luồng này, F1 từng lớp rất chênh: Generic 98,5%, Reconnaissance 87,5%,
Normal 80,6%, Exploits 67,8% — nhưng Analysis và Worms bằng 0, Backdoor 6,2%. Macro-F1 54% là
trung bình của những con số đó. Bốn lớp hiếm nhất gần như không học được, vì tập train có quá ít
mẫu và chúng chồng lấn với Exploits. Đó là hạn chế thật của công trình này, và lời giải thích theo
nhóm giúp nhìn ra nguyên nhân chứ không chỉ báo rằng có vấn đề."

## B5 · Chốt (3:00 – 3:20)

**Thoại:** "Cùng một kiến trúc, hai bộ dữ liệu, hai câu chuyện. NSL-KDD: lời giải thích tái hiện
đúng kết luận kinh điển của Lee và Stolfo. UNSW-NB15: lời giải thích chỉ ra mô hình đang dựa vào
artefact TTL, và chỉ ra bốn lớp hiếm không học được. Cả hai điều đó đều rút ra được vì φ là giá
trị Shapley chính xác theo cấu trúc, sai số cỡ một phần triệu — không phải một ước lượng xấp xỉ
mà ta phải tin."

---

## Bảng các dòng đã kiểm chứng (UNSW-NB15, seed 0)

| Dòng | Nhãn thật | Dự đoán | Xác suất | Nhóm mạnh nhất | Ghi chú |
| --- | --- | --- | --- | --- | --- |
| 445 | Reconnaissance | Reconnaissance | 1,000 | Time +5,04 | nhịp gói tin, `sinpkt` |
| 6464 | Reconnaissance | Reconnaissance | 1,000 | Content +4,61 | dự phòng |
| 65292 | Shellcode | Shellcode | 0,986 | Content +4,10 | `smean = 88` |
| 15356 | Generic | Generic | 0,999 | Additional +2,46 | **artefact `ct_state_ttl`** |
| 82330 | Normal | Normal | 1,000 | Additional +4,99 | **artefact `is_sm_ips_ports` +4,96** |
| 278 | Exploits | Exploits | 1,000 | Additional +3,25 | `ct_state_ttl` +1,187 |
| 11918 | DoS | DoS | 0,958 | Basic +2,59 | lớp F1 chỉ 42% |
| 61746 | Fuzzers | Fuzzers | 0,994 | Basic +3,40 | |
| 62785 | Exploits | **Shellcode** | 0,974 | Basic +3,63 | sai — `service = -` |
| 11410 | Exploits | **Analysis** | 0,944 | Content +2,78 | sai — dự phòng |
| 822 | Exploits | **Backdoor** | 0,937 | Additional +4,84 | sai — Additional áp đảo |
| 80987 | Normal | **Analysis** | 0,994 | Content +2,73 | báo động giả |

**F1 từng lớp trên mẫu 3.000 luồng** (để anh biết lớp nào đáng quay): Generic 98,5 · Reconnaissance
87,5 · Normal 80,6 · Exploits 67,8 · Shellcode 42,9 · DoS 42,4 · Fuzzers 37,5 · Backdoor 6,2 ·
Analysis 0,0 · Worms 0,0. Đừng quay Analysis hay Worms — mô hình không nhận được lớp nào trong
hai lớp đó, và mẫu 3.000 luồng không có luồng Worms nào.
