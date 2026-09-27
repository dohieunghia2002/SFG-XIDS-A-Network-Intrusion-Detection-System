# ĐỀ CƯƠNG CHI TIẾT ĐỀ TÀI NGHIÊN CỨU KHOA HỌC — BẢN SỬA v3

> **Bản sửa v3 (20/09/2026) — cập nhật theo kết quả thật của giai đoạn 1 (`full_v1`) và giai đoạn 2 tối thiểu (`phase2_v1`).** Thay đổi so với v2:
> 1. Thêm **mục 10: kết quả và đánh giá tiêu chí đi tiếp/dừng** (P1, P2, P4 đạt; P3 đạt một phần). Quyết định: viết bài với cách định vị trung thực — độ chính xác ngang các mô hình mạnh, giá trị nằm ở lời giải thích chính xác và kiểm chứng được.
> 2. Sửa đóng góp (mục 2.2) cho khớp bằng chứng: tương tác cặp **không** tăng độ chính xác (A1 ≈ A2) → giữ vì chúng là số hạng thật tạo bản đồ tương tác; chia nhóm theo taxonomy **không** tăng độ chính xác so với chia ngẫu nhiên (A4) → giá trị của nó là tính diễn giải.
> 3. Tầng 2 (mục 4.4): tích phân trên **không gian embedding** của từng đặc trưng (PLR với σ = 10 dao động mạnh theo giá trị thô); phần lệch giữa tổng EG và φ chỉ là sai số cầu phương do LeakyReLU, giảm theo số nút.
> 4. Phạm vi bài báo: bỏ TabNet, EBM, giá trị Owen (để hướng phát triển); đã thêm bài toán nhị phân và FT-Transformer.
> 5. Thêm hạn chế thật: lời giải thích của mạng nơ-ron **kém ổn định qua seed hơn** XGBoost + Shapley nhóm (cả cấp nhóm lẫn cấp đặc trưng).
> 6. Quyết định khuôn bài: LaTeX, Springer LNCS (phù hợp CCIS/Scopus Q4).

> **Bản sửa v2 (18/09/2026).** Thay đổi chính so với v1:
> 1. Chia lại nhóm UNSW-NB15 theo đúng taxonomy chính thức và khớp bộ train/test 175.341 dòng (bộ này không có `stime`, `ltime`).
> 2. Thay Module 1–2 (cross-group attention + adaptive pooling γ) bằng **GAI-Head** — đầu phân loại cộng tính theo nhóm có tương tác cặp — để lời giải thích chính xác theo cấu trúc. Kiến trúc v1 giữ lại làm mô hình đối chứng **SFG-Attn**.
> 3. Bỏ các chỗ dùng trọng số attention/γ làm lời giải thích, TreeSHAP cho mạng nơ-ron, KernelSHAP trên vector ẩn.
> 4. Bỏ các con số viết trước khi chạy (nhanh hơn 8–10 lần, 2,5 s/epoch, 1,2 GB VRAM); thay bằng ước tính có ghi chú và số đo thật từ step 0 của Đề tài C.
> 5. Thêm đối chứng (chia nhóm ngẫu nhiên, bỏ TTL, XGBoost + Shapley nhóm), kiểm chứng lời giải thích và tiêu chí đi tiếp/dừng.

---

## TÊN ĐỀ TÀI
* **Tiếng Anh:** *SFG-XIDS: A Semantic Feature-Group Additive Interaction Network with Exact Hierarchical Explanations for Network Intrusion Detection*
* **Tiếng Việt:** *SFG-XIDS: Mạng tương tác cộng tính theo nhóm đặc trưng ngữ nghĩa với cơ chế giải thích phân cấp chính xác cho phát hiện xâm nhập mạng*

---

## 1. BỐI CẢNH: VÌ SAO CHUYỂN TỪ ĐỀ TÀI C (RMV-IDS) SANG ĐỀ TÀI B

### 1.1. Bài học từ step 0 của Đề tài C
Số đo thật từ `step0_v4/step0_summary.csv` (UNSW-NB15, macro-F1 10 lớp trên tập test chính thức; họ baseline chọn trên validation):

| Kịch bản | B1: XGBoost thường | B2: XGBoost + che view | B6 = BEST: XGBoost riêng từng tổ hợp | B5: MLP phẳng + view-dropout |
|---|---|---|---|---|
| Đủ view (S0) | 51,6 | 50,9 | 51,6 | 46,7 |
| Thiếu 1 view (S1, k=1) | 34,0 | 47,7 | 48,0 | 44,5 |
| Chỉ còn view lõi (S1, k=4) | 15,2 | 28,9 | 31,3 | 28,7 |

Kết luận:
1. **Vấn đề thiếu view có thật, nhưng baseline đơn giản đã xử lý gần hết.** XGBoost thường mất ~17,5 điểm khi thiếu 1 view; XGBoost có che view (B2) hoặc mô hình riêng cho từng tổ hợp (B6) chỉ mất ~3,1–3,6 điểm. Một framework mới khó chứng minh được giá trị thêm.
2. **Mạng nơ-ron kém BEST 2,2–5,6 điểm tùy kịch bản** (4,9 điểm khi đủ view). RMV-IDS là mạng nơ-ron nên phải lấp khoảng cách này trước khi nói tới vượt BEST.
3. **Chi phí $2^K$ không phải rào cản thật.** Mạng nơ-ron che view ngẫu nhiên theo batch, không cần nhân dữ liệu; riêng baseline B2 (XGBoost) phải nhân dữ liệu nhưng step 0 vẫn chạy xong.
4. **TTL không phải lối tắt bắt buộc.** Bỏ `sttl`, `dttl`, `ct_state_ttl` (`step0_v4_nottl`) chỉ làm BEST giảm 1,7 điểm (51,6 → 49,9).

### 1.2. Hệ quả cho Đề tài B
* SFG-XIDS cũng là mạng nơ-ron → **rủi ro kém XGBoost khoảng 5 điểm** như B5. Không hứa "vượt trội"; mục tiêu là ngang Flat-MLP/FT-Transformer và thu hẹp khoảng cách với XGBoost (embedding số PLR, mục 4.1). Phần chênh còn lại phải được bù bằng giá trị của lời giải thích.
* Các nhóm đặc trưng dư thừa, thay thế được cho nhau → lời giải thích cấp nhóm có thể **đổi theo seed** → cần group dropout (mục 4.3) và đo độ ổn định (mục 5.4).
* Đối thủ về giải thích không chỉ là SHAP cấp đặc trưng mà là **XGBoost + Shapley nhóm tính vét cạn** (4 nhóm chỉ có 16 liên minh): chính xác hơn mạng nơ-ron và cũng giải thích được ở cấp nhóm. SFG-XIDS phải chứng minh lợi thế của mô hình "hộp kính" ở cấp nhóm so với phương án này.
* Điểm mạnh thật của B: mô hình nhẹ (vừa Colab T4) và lời giải thích **chính xác theo cấu trúc, kiểm chứng được**, ở đúng cấp mà chuyên viên SOC cần đọc.

---

## 2. TÍNH CẤP THIẾT & ĐÓNG GÓP

### 2.1. Research gap
1. **Giải thích cấp đặc trưng khó dùng cho SOC.** Phần lớn nghiên cứu XAI cho IDS dùng SHAP/LIME trên từng đặc trưng; với 40–80 thuộc tính, chuyên viên nhận một danh sách dài các giá trị rời rạc. Giải thích theo nhóm đã có công cụ (giá trị Owen, `shap.PartitionExplainer`) nhưng là **hậu kiểm** trên mô hình hộp đen, không cho biết các nhóm tương tác với nhau thế nào bên trong mô hình.
2. **Trọng số attention/gating không phải lời giải thích.** Các mô hình fusion đa nhánh thường lấy trọng số attention hoặc gating làm "mức quan trọng của nhóm". Khi các nhóm đã được trộn qua attention và kết nối tắt, trọng số này không còn đo đóng góp của nhóm vào đầu ra (Jain & Wallace, 2019).
3. **Lời giải thích ít được kiểm chứng định lượng.** Nhiều bài chỉ trình bày biểu đồ SHAP; các tiêu chí đánh giá phương pháp giải thích trong bảo mật (Warnecke et al., 2020) ít được áp dụng.

> **Công trình gần nhất cần phân biệt:** *Explainable Feature-Group-Aware Cross-Attentive Expert Fusion for IoMT Intrusion Detection* (Sensors, 2026, doi:10.3390/s26134293) — cũng chia nhóm đặc trưng, dùng cross-attention và có phần giải thích. **[TODO: đọc toàn văn, ghi rõ khác biệt về cách chia nhóm, cơ chế fusion, cách giải thích và cách kiểm chứng.]**

### 2.2. Đóng góp (v3 — mỗi ý kèm bằng chứng ở mục 10)
1. **Chia nhóm theo taxonomy chính thức của bộ dữ liệu** (UNSW-NB15: Moustafa & Slay, 2015; NSL-KDD: Lee & Stolfo, 2000). Đối chứng chia ngẫu nhiên cho thấy chia nhóm **không** làm thay đổi độ chính xác (UNSW +0,11 ± 1,39; NSL −2,02 ± 2,24) → giá trị của taxonomy là cho lời giải thích đọc được, có "đáp án thiết kế" để kiểm chứng (NSL-KDD).
2. **GAI-Head (Group-Additive Interaction Head).** Logit phân rã **chính xác** thành đóng góp riêng của từng nhóm và tương tác của từng cặp nhóm; bản đồ $4\times4$ là các số hạng thật trong logit, không phải trọng số attention. Độ chính xác ngang các mô hình mạnh (UNSW-NB15 10 lớp: 52,35 so với LightGBM 52,68, XGBoost 51,43, FT-Transformer 48,86). Tương tác cặp không tăng độ chính xác (A1 ≈ A2) nhưng chiếm 4–17% khối lượng quy công ở từng lớp.
3. **Giải thích phân cấp chính xác, nhất quán hai tầng, được kiểm chứng.** Tầng nhóm: công thức đóng trùng Shapley nhóm vét cạn đến 1,9·10⁻⁶, chi phí 0,3 ms/mẫu. Tầng đặc trưng: Expected Gradients trong nhóm, tổng khớp φ với sai số trung vị 0,1% (giảm dần khi tăng số nút). Đối chứng: EG thông thường cộng theo nhóm làm đổi nhóm quan trọng nhất ở 15% (UNSW) / 9% (NSL) số luồng; trọng số attention của biến thể SFG-Attn chỉ khớp thứ hạng Shapley của chính nó ở mức τ = 0,29 (UNSW).

---

## 3. DỮ LIỆU & CHIA NHÓM ĐẶC TRƯNG

### 3.1. UNSW-NB15 (bộ chính)
* Dùng **bộ phân chia chính thức**: `UNSW_NB15_training-set.csv` (175.341 dòng) làm train (tách 10% phân tầng làm validation) và `UNSW_NB15_testing-set.csv` (82.332 dòng) làm test. Không trộn rồi chia ngẫu nhiên (kết quả sẽ bị thổi phồng).
* Bỏ `id`, `label`, `attack_cat` → **42 đặc trưng**. Nhãn: nhị phân (`label`) và 10 lớp (`attack_cat`: Normal + 9 loại tấn công).
* Bộ này **không có** `srcip`, `sport`, `dstip`, `dsport`, `stime`, `ltime` (chỉ có trong bộ 2,54 triệu bản ghi). Không dùng timestamp vì dễ rò rỉ nhãn.
* Chia 4 nhóm theo các nhóm chính thức trong mô tả bộ dữ liệu (Flow / Basic / Content / Time / Additional generated). Nhóm Flow trong bộ phân chia chỉ còn `proto` nên gộp vào Basic; `rate` không có trong mô tả gốc, xếp vào Basic.

| Nhóm | Số ĐT | Đặc trưng |
|---|---|---|
| G1 Basic | 15 | `proto`, `service`, `state`, `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss` |
| G2 Content | 8 | `swin`, `dwin`, `stcpb`, `dtcpb`, `smean`, `dmean`, `trans_depth`, `response_body_len` |
| G3 Time | 7 | `sjit`, `djit`, `sinpkt`, `dinpkt`, `tcprtt`, `synack`, `ackdat` |
| G4 Additional | 12 | `is_sm_ips_ports`, `ct_state_ttl`, `ct_flw_http_mthd`, `is_ftp_login`, `ct_ftp_cmd`, `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, `ct_src_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm` |

* **Lưu ý diễn giải:** `stcpb`, `dtcpb` là số thứ tự TCP khởi tạo (gần như ngẫu nhiên) — giữ đúng taxonomy nhưng không diễn giải như "payload". `sttl`, `dttl` (G1) và `ct_state_ttl` (G4) thường có độ quan trọng rất cao trên UNSW-NB15 → thí nghiệm A5 (mục 5.2) so sánh lời giải thích khi có và không có TTL.
* **Tiền xử lý (giống step 0 của Đề tài C):** `proto`, `service`, `state` dùng embedding; `log1p` cho đặc trưng không âm có đuôi dài; z-score fit trên train.

### 3.2. NSL-KDD (bộ phụ — có "đáp án thiết kế" để kiểm chứng lời giải thích)
* `KDDTrain+` (125.973 dòng), `KDDTest+` (22.544 dòng); 41 đặc trưng; nhãn 5 lớp (Normal, DoS, Probe, R2L, U2R) theo bảng ánh xạ chuẩn, gồm cả các kiểu tấn công chỉ có trong `KDDTest+`.
* 4 nhóm do Lee & Stolfo định nghĩa khi xây dựng bộ KDD'99:

| Nhóm | Cột | Đặc trưng |
|---|---|---|
| G1 Basic | 1–9 | `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent` |
| G2 Content | 10–22 | `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`, `num_outbound_cmds`*, `is_host_login`, `is_guest_login` |
| G3 Time-based traffic (cửa sổ 2 giây) | 23–31 | `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`, `srv_diff_host_rate` |
| G4 Host-based traffic (100 kết nối gần nhất tới cùng máy đích) | 32–41 | `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `dst_host_serror_rate`, `dst_host_srv_serror_rate`, `dst_host_rerror_rate`, `dst_host_srv_rerror_rate` |

\* `num_outbound_cmds` luôn bằng 0 → loại khi huấn luyện.

* **Vì sao vẫn dùng NSL-KDD dù đã cũ:** theo thiết kế gốc, nhóm Content được xây để bắt R2L/U2R (tấn công nằm trong phần dữ liệu của một kết nối), còn nhóm traffic (G3, G4) để bắt DoS/Probe (nhiều kết nối trong thời gian ngắn). Đây là "đáp án" để kiểm tra lời giải thích có hợp lý không (mục 5.4). Không dùng NSL-KDD để khẳng định hiệu năng.

---

## 4. KIẾN TRÚC MÔ HÌNH (SFG-XIDS)

```
                 Luồng mạng x
                      │
      ┌─────────┬─────┴───┬─────────┐
      ▼         ▼         ▼         ▼
     x₁        x₂        x₃        x₄
   Basic    Content     Time    Additional
      │         │         │         │
    Enc₁      Enc₂      Enc₃      Enc₄     PLR + MLP-Res riêng
      │         │         │         │
     e₁        e₂        e₃        e₄      eᵢ chỉ phụ thuộc xᵢ
      └─────────┴────┬────┴─────────┘
                     ▼
   ┌──────────────────────────────────────┐
   │ GAI-HEAD                             │
   │ 4 số hạng chính  uᵢ(eᵢ)              │
   │ 6 số hạng cặp    vᵢⱼ(eᵢ, eⱼ), i < j  │
   │ z = b + Σ uᵢ + Σ vᵢⱼ  → softmax      │
   └──────────────────┬───────────────────┘
                      ▼
   ┌──────────────────────────────────────┐
   │ GIẢI THÍCH PHÂN CẤP (chính xác)      │
   │ Tầng 1: φᵢ dạng đóng = Shapley nhóm  │
   │         + bản đồ tương tác 4×4       │
   │ Tầng 2: Expected Gradients trong Gᵢ  │
   │         Σ (đặc trưng thuộc Gᵢ) = φᵢ  │
   └──────────────────────────────────────┘
```

### 4.1. Group encoders
* Mỗi đặc trưng số qua embedding tuần hoàn **PLR** (Periodic → Linear → ReLU; Gorishniy et al., 2022) — kỹ thuật được báo cáo giúp MLP thu hẹp đáng kể khoảng cách với GBDT trên dữ liệu bảng. Đặc trưng phân loại dùng embedding thường.
* Các embedding của nhóm $i$ được nối thành $s_i$ rồi qua MLP riêng có kết nối tắt ($d = 64$):
$$h_i = \text{LeakyReLU}\big(W_i^{(1)} s_i + b_i^{(1)}\big),\qquad e_i = \text{LayerNorm}\Big(h_i + \text{LeakyReLU}\big(W_i^{(2)} h_i + b_i^{(2)}\big)\Big) \in \mathbb{R}^{d}$$
* **Điều kiện then chốt:** $e_i$ chỉ phụ thuộc $x_i$ — không trộn nhóm trước đầu phân loại. Đây là điều kiện để phân rã ở mục 4.2 và 4.4 chính xác.

### 4.2. GAI-Head — đầu phân loại cộng tính có tương tác nhóm
Với mỗi lớp $c$:
$$z_c(x) = b_c + \sum_{i=1}^{4} u_{i,c}(e_i) + \sum_{1\le i<j\le 4} v_{ij,c}(e_i, e_j)$$
* $u_i$: đóng góp riêng của nhóm $i$ (MLP nhỏ $\mathbb{R}^{d}\to\mathbb{R}^{C}$).
* $v_{ij}$: tương tác của cặp nhóm $(i,j)$ — MLP nhỏ trên $[e_i;\, e_j;\, e_i \odot e_j]$; 6 cặp. Thay cho ma trận attention của v1.
* Mô hình **không có** tương tác bậc 3–4 giữa các nhóm — giả định cần kiểm tra (P1 ở mục 9; A1–A2 ở mục 5.2). Tương tác bên trong nhóm vẫn đầy đủ nhờ encoder.
* **Định vị so với GA²M/EBM và NAM:** các mô hình này cộng tính ở cấp **đặc trưng**, chỉ cho phép tương tác bậc 2 giữa các đặc trưng. SFG-XIDS cộng tính ở cấp **nhóm ngữ nghĩa**: tương tác bậc bất kỳ trong nhóm, bậc 2 giữa các nhóm — đúng cấp mà chuyên viên SOC cần đọc.

### 4.3. Group dropout (tái dùng cơ chế che view của Đề tài C)
Khi huấn luyện, mỗi nhóm bị tắt độc lập với xác suất $p$ (chọn trong {0,1; 0,2; 0,3} theo validation): bỏ $u_i$ và mọi $v_{ij}$ chứa nhóm $i$ khỏi tổng. Mục đích: buộc từng nhóm tự mang thông tin, tránh mô hình dồn quy công vào một nhóm dư thừa một cách tùy tiện → lời giải thích ổn định hơn giữa các seed (kiểm chứng ở A3 và mục 5.4).

### 4.4. Giải thích phân cấp
**Tập nền.** Lấy $B$ mẫu train (ví dụ $B = 256$). Khi lấy kỳ vọng, các nhóm được lấy mẫu **độc lập** với nhau ($e_i'$ và $e_j'$ lấy từ các mẫu nền khác nhau).

**Tinh lọc số hạng cặp** (phân tích ANOVA hàm; Lengerich et al., 2020):
$$\bar v_{ij} = \mathbb{E}\big[v_{ij}(e_i', e_j')\big],\qquad m_{i\leftarrow j}(e_i) = \mathbb{E}_{e_j'}\big[v_{ij}(e_i, e_j')\big] - \bar v_{ij}$$
$$g_{ij}(e_i, e_j) = v_{ij}(e_i, e_j) - m_{i\leftarrow j}(e_i) - m_{j\leftarrow i}(e_j) - \bar v_{ij}$$
$m_{i\leftarrow j}$ là phần "hiệu ứng riêng" của nhóm $i$ lẫn trong số hạng cặp; $g_{ij}$ là tương tác thuần, có kỳ vọng bằng 0 theo từng biến.

**Tầng 1 — đóng góp nhóm** (cho lớp $c$, thường là lớp được dự đoán):
$$\varphi_i(x) = \big[u_i(e_i) - \mathbb{E}\,u_i(e_i')\big] + \sum_{j\ne i} m_{i\leftarrow j}(e_i) + \tfrac12\sum_{j\ne i} g_{ij}(e_i, e_j)$$
Tính chất (chứng minh ngắn đưa vào bài):
* **Đủ:** $\sum_i \varphi_i(x) = z_c(x) - \mathbb{E}[z_c]$ — logit được giải thích trọn vẹn, không có phần dư.
* **Chính xác:** $\varphi_i$ trùng với giá trị Shapley nhóm (interventional, nền độc lập theo nhóm) của chính mô hình, vì mô hình chỉ có số hạng bậc ≤ 2 giữa các nhóm và giá trị Shapley tuyến tính theo từng số hạng.
* **Bản đồ tương tác:** $M_c[i,j]$ = trung bình của $\lvert g_{ij,c}(e_i,e_j)\rvert$ trên các mẫu thuộc lớp $c$ — heatmap $4\times4$ cho từng loại tấn công.
* **Chi phí:** vài nghìn lượt tính MLP nhỏ cho mỗi mẫu, chạy theo lô trên GPU; không cần xấp xỉ kiểu KernelSHAP.

**Tầng 2 — đặc trưng trong nhóm (v3: thực hiện trên không gian embedding).** Coi $\varphi_i$ là hàm của embedding các đặc trưng nhóm $i$, $t_i = (t_k)_{k\in G_i}$ (PLR với đặc trưng số, embedding thường với đặc trưng phân loại), giữ nguyên các nhóm khác; điểm nền $t_i'$ lấy từ **đủ** 256 mẫu nền của tầng 1:
$$\text{EG}_k(x) = \mathbb{E}_{t_i'}\int_0^1 (t_k - t_k')^{\top}\,\frac{\partial \varphi_i\big(t_i' + \alpha(t_i - t_i')\big)}{\partial t_k}\,d\alpha,\qquad k \in G_i$$
Vì phép tinh lọc cho $\mathbb{E}_{t_i'}[\varphi_i(t_i')] = 0$ **đúng tuyệt đối** trên tập nền, $\sum_{k\in G_i}\text{EG}_k(x) = \varphi_i(x)$ → **nhất quán cộng tính hai tầng:** đặc trưng → nhóm → logit. Tích phân theo α tính bằng Gauss–Legendre 16 nút; sai số còn lại chỉ do cầu phương (hàm có điểm gãy của LeakyReLU): với hàm kích hoạt trơn, sai số ≈ 10⁻¹⁵. Lý do dùng không gian embedding: UNSW chọn σ = 10 cho PLR, tích phân theo giá trị thô dao động rất mạnh.

### 4.5. Mô hình đối chứng giữ từ v1: SFG-Attn
Kiến trúc v1 (4 encoder + cross-group attention + adaptive pooling γ) giữ lại để: (i) so độ chính xác; (ii) cho thấy thứ tự nhóm theo $A$ và $\gamma$ lệch khỏi Shapley nhóm tính vét cạn của chính mô hình đó — minh chứng cho gap 2.

---

## 5. THIẾT KẾ THỰC NGHIỆM

> **Trạng thái (v3).** Đã chạy: kịch bản 1 (đa lớp 7 mô hình + FT-Transformer; nhị phân 6 mô hình), kịch bản 2 (A1, A3, A4, A5, A6), kịch bản 3, kịch bản 4 (tiêu chí 1–5; tiêu chí 7 đo thời gian). **Không đưa vào bài này** (để hướng phát triển): TabNet, EBM, giá trị Owen (`shap.PartitionExplainer`). Thay cho tiêu chí 6: TreeSHAP của XGBoost làm tham chiếu ổn định cấp đặc trưng. Kết quả ở mục 10.

**Quy ước chung:** split chính thức; mọi lựa chọn (siêu tham số, $p$ của group dropout, early stopping) dựa trên macro-F1 validation; tập test niêm phong như step 0 của Đề tài C; mỗi cấu hình 5 seed, báo cáo mean ± std; chỉ kết luận "khác biệt" khi chênh lệch vượt độ lệch chuẩn giữa các seed.

### 5.1. Kịch bản 1 — Hiệu năng phát hiện
* **Mô hình:**
  * Cây: RF, XGBoost, LightGBM.
  * Hộp kính cấp đặc trưng: EBM (InterpretML).
  * Nơ-ron: Flat-MLP, TabNet, FT-Transformer.
  * Biến thể nhóm: SFG-Concat (4 encoder + ghép nối), SFG-Attn (kiến trúc v1), **SFG-XIDS** (GAI-Head).
* **Bài toán:** nhị phân và đa lớp (10 lớp UNSW-NB15; 5 lớp NSL-KDD).
* **Độ đo:** Accuracy, Macro-F1, F1 từng lớp, Precision, Recall, FAR (nhị phân).
* **Mục tiêu:** SFG-XIDS ngang Flat-MLP/FT-Transformer và cách XGBoost không quá ~3 điểm macro-F1 (UNSW-NB15, 10 lớp). Không đặt mục tiêu vượt GBDT.

### 5.2. Kịch bản 2 — Ablation & đối chứng
* **A1:** chỉ số hạng chính (bỏ $v_{ij}$) — giá trị của tương tác nhóm.
* **A2:** đầy đủ (số hạng chính + cặp).
* **A3:** bỏ group dropout.
* **A4:** chia nhóm ngẫu nhiên cùng kích thước (5 cách chia) thay cho taxonomy — giá trị của chia nhóm ngữ nghĩa, cả về độ chính xác lẫn độ ổn định/hợp lý của lời giải thích.
* **A5 (UNSW-NB15):** bỏ `sttl`, `dttl`, `ct_state_ttl` — lời giải thích thay đổi thế nào khi không còn TTL.
* **A6:** bỏ embedding PLR (MLP thường) — PLR thu hẹp khoảng cách với XGBoost được bao nhiêu.

### 5.3. Kịch bản 3 — Đóng góp & tương tác nhóm theo loại tấn công
* Với mỗi lớp: trung bình $\varphi_i$ của 4 nhóm và heatmap tương tác $M_c$.
* **NSL-KDD:** kiểm định giả thuyết đặt trước (H1, H2 ở mục 5.4).
* **UNSW-NB15:** phân tích **khám phá**, không đặt trước dự đoán; so sánh với A5.
* **Case study:** DoS, Exploits, Backdoor (UNSW-NB15); R2L, U2R (NSL-KDD). Macro-F1 10 lớp của UNSW-NB15 chỉ khoảng 50% do các lớp chồng lấn → phân tích cả mẫu đúng lẫn các cặp lớp hay nhầm (DoS / Exploits / Analysis / Backdoor).

### 5.4. Kịch bản 4 — Kiểm chứng lời giải thích
Theo các tiêu chí của Warnecke et al. (2020), điều chỉnh cho cấp nhóm:
1. **Chính xác:** so $\varphi_i$ dạng đóng với Shapley nhóm tính vét cạn trên $2^4 = 16$ liên minh (dùng tập nền nhỏ, ví dụ 16 mẫu, liệt kê đủ tổ hợp nền của các nhóm bị che — khoảng 83 nghìn lượt tính/mẫu, vẫn nhẹ) → sai lệch phải ở mức sai số số học.
2. **Nhất quán hai tầng:** $\big|\sum_{k\in G_i}\text{EG}_k - \varphi_i\big|$ nhỏ.
3. **Faithfulness của attention (trên SFG-Attn):** tương quan hạng giữa thứ tự nhóm theo $\gamma$/$A$ và theo Shapley nhóm vét cạn của cùng mô hình; thí nghiệm xóa nhóm (thay nhóm hạng 1 bằng nền) → mức giảm xác suất của lớp được dự đoán.
4. **Ổn định:** tương quan hạng của hồ sơ đóng góp nhóm theo lớp giữa 5 seed — SFG-XIDS (có/không group dropout), SFG-Attn, XGBoost + Shapley nhóm.
5. **Hợp lý (NSL-KDD), giả thuyết đặt trước theo thiết kế của Lee & Stolfo:**
   * H1: tỷ trọng $\lvert\varphi\rvert$ của G2 Content ở R2L/U2R lớn hơn ở DoS/Probe.
   * H2: ở DoS/Probe, tỷ trọng của G3 + G4 lớn hơn của G2.
6. **So với đối thủ hậu kiểm mạnh nhất:** XGBoost + Shapley nhóm vét cạn (16 liên minh) + giá trị Owen cho cấp đặc trưng (`shap.PartitionExplainer` với cây phân cấp theo 4 nhóm) — so theo các tiêu chí 2, 4, 5 và chi phí.
7. **Chi phí:** thời gian giải thích/mẫu của SFG-XIDS, XGBoost + Shapley nhóm và KernelSHAP 42 đặc trưng — báo cáo số đo thật, không đặt trước.

---

## 6. NGÂN SÁCH TÍNH TOÁN TRÊN COLAB T4 (số đo thật)
* **Tham số:** SFG-XIDS 185 nghìn (UNSW) / 177 nghìn (NSL); Flat-MLP 318/306 nghìn; SFG-Attn 124/119 nghìn; FT-Transformer 939/924 nghìn.
* **Thời gian huấn luyện (TB/lần, kể cả giải thích tầng 1 nếu có):** SFG-XIDS 46 s (UNSW) / 22 s (NSL); XGBoost 25/13 s; FT-Transformer 371/191 s (đa lớp), tới gần 10 phút ở UNSW nhị phân.
* **Chi phí giải thích/mẫu:** φ dạng đóng 0,3 ms; EG tầng 2 (4 nhóm, 16 nút, 256 điểm nền) 0,22 s; EG phẳng 4,6 ms; TreeSHAP 4–7 ms.
* **Tổng:** giai đoạn 1 khoảng 2–3 giờ; giai đoạn 2 khoảng 2 giờ 30 (FT-Transformer chiếm khoảng một nửa). Cả hai script chạy tiếp được khi Colab ngắt phiên.

---

## 7. CẤU TRÚC DỰ KIẾN CỦA BÀI BÁO

> **v3:** bản thảo đã viết theo cấu trúc này, tiếng Anh, khuôn Springer LNCS (`paper/sfg_xids_lncs.tex`). Kết quả đưa vào: đa lớp + nhị phân (8 bảng), ablation A1–A6, hồ sơ nhóm theo lớp, kiểm chứng tầng 1 và tầng 2, hạn chế về độ ổn định.
1. **Abstract:** SOC cần lời giải thích ở cấp hành vi; trọng số attention không phải lời giải thích; đề xuất SFG-XIDS phân rã logit theo nhóm ngữ nghĩa với tương tác cặp và giải thích phân cấp chính xác; kết quả: độ chính xác ngang các mô hình nơ-ron mạnh, lời giải thích chính xác, ổn định, hợp lý với thiết kế đặc trưng.
2. **Introduction.**
3. **Related Work:** ML/DL cho NIDS; fusion theo nhóm/đa nhánh cho IDS (gồm Sensors 2026); XAI cho IDS và đánh giá phương pháp giải thích trong bảo mật; mô hình cộng tính (GA²M/EBM, NAM) và giá trị Owen; tranh luận attention-as-explanation.
4. **Phương pháp:** chia nhóm theo taxonomy; group encoders + PLR; GAI-Head; group dropout; giải thích phân cấp (kèm chứng minh: $\varphi$ = Shapley nhóm, nhất quán hai tầng).
5. **Thực nghiệm:** setup; hiệu năng; ablation; phân tích nhóm theo loại tấn công; kiểm chứng lời giải thích; case study.
6. **Discussion & Threats to Validity:** khoảng cách với XGBoost; artifact TTL của UNSW-NB15; NSL-KDD đã cũ; giả định nền độc lập giữa các nhóm (có thể tạo mẫu ngoài phân phối); bỏ tương tác bậc 3–4; chỉ 2 bộ dữ liệu.
7. **Conclusion & Future Work:** thêm bộ dữ liệu hiện đại (CIC-IDS2017, các bộ NetFlow v2); kết hợp với kịch bản thiếu nhóm của Đề tài C.

---

## 8. MỤC TIÊU CÔNG BỐ

> **v3 — quyết định:** viết theo khuôn Springer LNCS, hợp các hội thảo xuất bản trong CCIS/LNCS (Scopus Q4). Nếu nộp IEEE KSE/RIVF/ATC (6 trang), cần rút gọn: giữ bảng 2, 5, 7 và gộp phần kiểm chứng.
* **Hội thảo phù hợp:** IEEE KSE, IEEE RIVF, IEEE ATC. **[TODO: kiểm tra hạn nộp.]**
* **Tạp chí:** IJICS (Inderscience) là mục tiêu vừa sức. JISA (Elsevier) và MDPI Sensors/Electronics **không** thuộc nhóm Q4/Q3 như v1 ghi, khó hơn đáng kể — nên bổ sung ít nhất một bộ dữ liệu hiện đại trước khi nhắm tới. Sensors cũng là nơi đã đăng công trình gần nhất ở mục 2.1.

---

## 9. LỘ TRÌNH & TIÊU CHÍ ĐI TIẾP / DỪNG

> **Trạng thái v3:** Bước 0 (pilot), giai đoạn 1 (kịch bản 1–4, 5 seed) và giai đoạn 2 tối thiểu (EG tầng 2, nhị phân, FT-Transformer) đã xong; đánh giá tiêu chí ở mục 10.5. Bước 1 (đọc toàn văn bài Sensors 2026) **vẫn TODO**: khi viết v3, trang MDPI đang bảo trì và PMC chặn truy cập tự động, nên phần so sánh trong bài chỉ dựa trên tên bài và cần kiểm lại.

**Bước 0 — Pilot (1 phiên Colab, trước khi viết bài).** Tái dùng pipeline dữ liệu và cơ chế niêm phong test của `step0_unsw.py`. Chạy XGBoost, Flat-MLP, SFG-Attn, SFG-XIDS trên cả 2 bộ dữ liệu, 3 seed, thêm 3 cách chia nhóm ngẫu nhiên.

Đi tiếp khi đạt **cả bốn** tiêu chí:
* **P1:** macro-F1 (UNSW-NB15, 10 lớp) của SFG-XIDS không thấp hơn Flat-MLP quá 1 điểm và cách XGBoost không quá ~3 điểm.
* **P2:** chia theo taxonomy không thua chia ngẫu nhiên quá một độ lệch chuẩn giữa các seed.
* **P3:** hồ sơ đóng góp nhóm khác nhau rõ giữa các lớp tấn công; nhóm hạng 1 của phần lớn các lớp giữ nguyên qua các seed.
* **P4 (NSL-KDD):** tỷ trọng G2 Content ở R2L/U2R cao hơn ở DoS/Probe.

Nếu không đạt:
* **P1:** tăng dung lượng encoder/embedding; nếu vẫn cách XGBoost quá 3 điểm thì SFG-XIDS không còn đủ lợi thế so với XGBoost + Shapley nhóm → cân nhắc lại hướng đề tài.
* **P3/P4:** báo cáo như một phát hiện về sự dư thừa/artifact của dữ liệu và cân nhắc lại trọng tâm bài báo.

**Bước 1.** Đọc toàn văn bài Sensors 2026; hoàn thiện Related Work và điểm khác biệt.
**Bước 2.** Chạy đầy đủ kịch bản 1–4 (5 seed).
**Bước 3.** Viết bài và nộp.

---

## 10. KẾT QUẢ GIAI ĐOẠN 1–2 VÀ ĐÁNH GIÁ TIÊU CHÍ (v3)
Nguồn: `sfg_results/full_v1/summary.json`, `sfg_results/phase2_v1/summary_p2.json`. TB ± SD qua 5 seed trên tập test chính thức.

### 10.1. Đa lớp (macro-F1)
| Mô hình | UNSW-NB15 (10 lớp) | NSL-KDD (5 lớp) |
|---|---|---|
| XGBoost | 51,43 ± 0,29 | 57,79 ± 0,97 |
| LightGBM | 52,68 ± 0,29 | 54,23 ± 0,71 |
| Random Forest | 48,50 ± 0,33 | 47,86 ± 0,41 |
| Flat-MLP (PLR) | 52,28 ± 0,55 | 58,76 ± 2,21 |
| FT-Transformer | 48,86 ± 0,52 | 64,26 ± 3,15 |
| SFG-Concat | 52,42 ± 0,80 | 55,98 ± 2,70 |
| SFG-Attn (v1) | 52,16 ± 0,70 | 59,63 ± 3,63 |
| **SFG-XIDS** | **52,35 ± 1,51** | **62,88 ± 1,89** |

### 10.2. Nhị phân (macro-F1 / DR / FAR, %)
| Mô hình | UNSW-NB15 | NSL-KDD |
|---|---|---|
| XGBoost | 87,59 / 97,46 / 23,48 | 81,16 / 69,16 / 2,96 |
| LightGBM | 87,75 / 97,75 / 23,47 | 78,78 / 64,93 / 2,80 |
| Random Forest | 86,47 / 98,71 / 27,12 | 77,37 / 62,55 / 2,83 |
| Flat-MLP | 87,25 / 96,85 / 23,50 | 77,29 / 63,08 / 3,76 |
| FT-Transformer | 88,58 / 96,96 / 20,87 | 80,57 / 69,66 / 4,97 |
| SFG-XIDS | 87,69 / 97,64 / 23,47 | 78,98 / 67,56 / 5,90 |

### 10.3. Ablation (hiệu ghép cặp theo seed, macro-F1 đa lớp; UNSW / NSL)
* A1 chỉ số hạng chính: −0,90 ± 1,38 / +0,78 ± 3,62 → tương tác cặp không đổi độ chính xác.
* A3 bỏ group dropout: +0,08 ± 1,33 / −5,30 ± 2,64 → có ích rõ ở NSL.
* A4 taxonomy − chia ngẫu nhiên: +0,11 ± 1,39 / −2,02 ± 2,24.
* A5 bỏ TTL (UNSW): SFG-XIDS −0,72; XGBoost −1,44; Flat-MLP −0,72.
* A6 bỏ phần tuần hoàn của PLR: −5,07 ± 2,66 / +1,65 ± 3,26.

### 10.4. Giải thích
* **Tầng 1:** φ dạng đóng so với Shapley vét cạn lệch tối đa 1,9·10⁻⁶; completeness 3,8·10⁻⁶; Monte Carlo 128 tổ hợp lệch hồ sơ (L1) 0,019 / 0,029.
* **Attention (SFG-Attn):** Kendall τ giữa trọng số attention A và Shapley của chính mô hình 0,29 / 0,49; với γ 0,55 / 0,55. Xóa nhóm hạng 1 theo A làm giảm xác suất 0,33, theo φ 0,54 (UNSW).
* **Hợp lý (NSL):** H1 (Content ở R2L/U2R 0,344 > DoS/Probe 0,098) và H2 (traffic ở DoS/Probe 0,626 > 0,098) đạt 5/5 seed với mọi phương pháp.
* **Tầng 2 (16 nút):** |ΣEG − φ| / Σ|φ| trung vị 1,1·10⁻³ / 1,2·10⁻³, p99 1,8·10⁻² / 9,7·10⁻³; trung vị giảm từ 7,3·10⁻³ (4 nút) xuống 7,7·10⁻⁵ (128 nút).
* **EG phẳng cộng theo nhóm so với φ:** đổi nhóm hạng 1 ở 14,8 ± 3,3% / 8,6 ± 5,1% số mẫu; EG phân cấp chỉ 1,7% / 1,0% (do sai số cầu phương khi hai nhóm gần bằng nhau).
* **Xóa đặc trưng (giảm p lớp thật, k = 1/3/5, UNSW):** EG phân cấp 0,139 / 0,373 / 0,437, bằng EG phẳng; ngẫu nhiên 0,014 / 0,037 / 0,066.
* **Ổn định qua seed:** cấp nhóm, tỷ lệ lớp có cùng nhóm hạng 1 ở mọi seed: SFG-XIDS 40%, A3 80%, XGBoost 100%; cấp đặc trưng Kendall τ: SFG-XIDS 0,70 / 0,82, TreeSHAP của XGBoost 0,86 / 0,91 → **hạn chế cần nêu trong bài**.
* **Hợp lý cấp đặc trưng (NSL):** R2L → `logged_in`, `num_failed_logins`; U2R → `root_shell`; Probe → `diff_srv_rate`, `dst_host_diff_srv_rate`; DoS → `flag`, `count`, `dst_host_rerror_rate`.

### 10.5. Tiêu chí đi tiếp/dừng
* **P1 ĐẠT:** SFG-XIDS 52,35 so với Flat-MLP 52,28 và XGBoost 51,43.
* **P2 ĐẠT:** taxonomy − ngẫu nhiên +0,11 (UNSW); −2,02 ± 2,24 (NSL, trong 1 SD).
* **P3 ĐẠT MỘT PHẦN:** hồ sơ khác nhau rõ giữa các lớp; nhóm hạng 1 theo đa số seed giữ nguyên ở 100% lớp nhưng giống nhau ở cả 5 seed chỉ 40%.
* **P4 ĐẠT:** 0,344 > 0,098 ở 5/5 seed.
* **Quyết định:** đi tiếp và viết bài. Định vị: không vượt trội về độ chính xác; giá trị nằm ở lời giải thích chính xác, nhất quán hai tầng, kiểm chứng được; nêu rõ hạn chế về độ ổn định.

---

## TÀI LIỆU THAM KHẢO CHÍNH
1. Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS*.
2. Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE CISDA*.
3. Lee, W., & Stolfo, S. J. (2000). A framework for constructing features and models for intrusion detection systems. *ACM TISSEC*, 3(4).
4. Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
5. Warnecke, A., Arp, D., Wressnegger, C., & Rieck, K. (2020). Evaluating Explanation Methods for Deep Learning in Security. *IEEE EuroS&P*.
6. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate intelligible models with pairwise interactions. *KDD*.
7. Agarwal, R., et al. (2021). Neural Additive Models: Interpretable Machine Learning with Neural Nets. *NeurIPS*.
8. Lengerich, B., Tan, S., Chang, C.-H., Hooker, G., & Caruana, R. (2020). Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models. *AISTATS*.
9. Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. M., & Lee, S.-I. (2021). Improving performance of deep learning models with axiomatic attribution priors and expected gradients. *Nature Machine Intelligence*.
10. Owen, G. (1977). Values of games with a priori unions. In R. Henn & O. Moeschlin (Eds.), *Mathematical Economics and Game Theory* (pp. 76–88). Springer.
11. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *NeurIPS*.
12. Gorishniy, Y., Rubachev, I., & Babenko, A. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*.
13. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets & Benchmarks*.
14. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*.
15. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
16. Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
17. Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv:1909.09223*.
18. *Explainable Feature-Group-Aware Cross-Attentive Expert Fusion for IoMT Intrusion Detection.* Sensors, 26(13), 4293 (2026). doi:10.3390/s26134293. **[TODO: bổ sung tác giả sau khi đọc toàn văn.]**
19. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
20. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
21. Jullum, M., Redelmeier, A., & Aas, K. (2021). groupShapley: Efficient prediction explanation with Shapley values for feature groups. *arXiv:2106.12228*.
22. Yang, Z., Zhang, A., & Sudjianto, A. (2021). GAMI-Net: An explainable neural network based on generalized additive models with structured interactions. *Pattern Recognition*, 120, 108192.
23. Chang, C.-H., Caruana, R., & Goldenberg, A. (2022). NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning. *ICLR*.
24. Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP-IJCNLP*.
25. Wang, M., Zheng, K., Yang, Y., & Wang, X. (2020). An Explainable Machine Learning Framework for Intrusion Detection Systems. *IEEE Access*, 8, 73127–73141.
26. Neupane, S., et al. (2022). Explainable Intrusion Detection Systems (X-IDS): A Survey of Current Methods, Challenges, and Opportunities. *IEEE Access*, 10, 112392–112415.
27. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
28. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
