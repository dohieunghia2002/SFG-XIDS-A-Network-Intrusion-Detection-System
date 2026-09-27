#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMV-IDS - Bước 0 (cổng quyết định) trên UNSW-NB15  [đề cương v2, mục 5.0]
Phiên bản script: 6

KỶ LUẬT TÁCH DỮ LIỆU (bắt buộc, được cưỡng chế bằng code):
    TRAIN      -> fit mô hình và mọi tham số tiền xử lý.
    VALIDATION -> chọn p_drop (B2), learning rate (B5), và chọn MỘT họ baseline duy nhất
                  (B2 hoặc B6) cho TOÀN BỘ thí nghiệm, KHÓA trước khi chạm vào TEST.
    TEST       -> CHỈ chấm điểm các lựa chọn đã chốt.
    BOOTSTRAP  -> lấy lại mẫu trên nhãn dự đoán test ĐÃ CỐ ĐỊNH, không chọn lại mô hình.
Nhãn dự đoán trên test được "niêm phong" trong PredStore: mọi lời gọi đọc nhãn test
trước khi định tuyến được chốt sẽ ném lỗi. Không có điểm số nào trên test được tính
trước thời điểm mở niêm phong.

Các khối chạy:
  B1 : XGBoost train trên dữ liệu đầy đủ; view thiếu để NaN. Chỉ để tham khảo,
       KHÔNG tham gia định tuyến.
  B2 : XGBoost + che view. Nhân dữ liệu theo 16 tổ hợp, trọng số theo p_drop; tổ hợp
       thiếu V3 có thêm bản đổi http->ssl; p_drop chọn trên VALIDATION.
  B6 : mỗi tổ hợp view một XGBoost riêng (+1 mô hình cho chế độ S2a). Lúc suy luận đã
       biết view nào thiếu nên B6 là đối thủ dùng được thật.
  B5 : MLP phẳng (PyTorch): điền 0 + vector mặt nạ + view-dropout (dùng p_drop của B2;
       learning rate chọn trong {1e-3, 3e-4} trên VALIDATION).
  C2 : lượng thông tin của từng view, và cây 1 tầng trên từng đặc trưng (chẩn đoán).

CHỌN MỘT HỌ BASELINE DUY NHẤT (thay cho max(B2, B6) trên test ở bản cũ - đó là oracle test):
  - Quy tắc gộp trên VALIDATION, chốt trước: trung bình cộng KHÔNG trọng số của macro-F1
    trên ĐÚNG 16 tổ hợp view (mỗi tổ hợp một lần, có cả "1111"), KHÔNG có S2a_tls.
    Lý do: 9 kịch bản chồng lấn nhau nên gộp theo kịch bản sẽ đếm vài tổ hợp hai lần;
    16 tổ hợp là phân hoạch đầy đủ, không chồng lấn của mọi điều kiện vận hành. "1111"
    được giữ vì nó là điều kiện vận hành thật và là mốc tham chiếu của luật A.
  - Chọn B2 nếu mean_val(B2) >= mean_val(B6), ngược lại chọn B6; hòa -> B2 (mô hình đơn).
  - Họ đã chọn được KHÓA và dùng cho MỌI kịch bản S0/S1/S2. Nếu họ B6 được chọn thì bên
    trong B6 vẫn dùng mô hình riêng theo tổ hợp view (định tuyến nội bộ của B6), gồm cả
    mô hình cho chế độ S2a. Không bao giờ đổi họ giữa các kịch bản.
  - S2a_tls chỉ để đánh giá: không tham gia chọn p_drop, chọn lr, hay chọn họ.
  - Việc khóa họ diễn ra TRƯỚC khi mở niêm phong test. Cột BEST = họ đã khóa.

GIẢ ĐỊNH VỀ MÔ HÌNH S2a CỦA HỌ B6 (chỉ ghi rõ ở đây, KHÔNG đổi hành vi):
  - S2a_tls không tham gia chọn họ, chọn p_drop hay chọn learning rate.
  - Nếu họ B6 được KHÓA thì B6 được phép có một mô hình riêng cho chế độ S2a, huấn luyện
    trên dữ liệu TRAIN đã biến đổi tương ứng (bỏ V3 và đổi service http->ssl). Giả định đi
    kèm: chế độ mã hóa là điều kiện triển khai đã biết, giống như việc biết view nào thiếu.
    Đây là mô hình nội bộ của họ B6, không phải một lựa chọn giữa B2 và B6.
  - Nếu họ B2 được KHÓA thì mô hình S2a của B6 không được dùng ở bất kỳ đâu.

Luật quyết định (điểm số trên test của BEST đã định tuyến; CI 95% bằng bootstrap test):
  A. Bài toán có thật: BEST giảm >= 1 điểm macro-F1 so với S0 ở ít nhất một kịch bản
     S2a/S2b/S2c, và cận dưới khoảng tin cậy của mức giảm > 0.
  B. Mạng nơ-ron có cửa thắng: B5 kém BEST trung bình <= 2 điểm trên 9 kịch bản S0-S2
     (điểm ước lượng, không dùng CI; CI chỉ in ra để tham khảo).
  Chẩn đoán thêm, KHÔNG tham gia verdict: missing_only_gap = B5 kém BEST trung bình trên
  8 kịch bản thiếu view (bỏ S0).
  Kết luận:
    - A sai                -> DỪNG.
    - A đúng, B sai        -> không làm RMV-IDS, chuyển sang bài benchmark.
    - Cả A và B đều đúng   -> TIẾP TỤC.

Chạy trên Colab (Runtime -> Change runtime type -> T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install -U xgboost kagglehub
    %cd /content/drive/MyDrive/rmv_ids      # thư mục trên Drive chứa file step0_unsw.py
    !python step0_unsw.py --selftest        # kiểm tra nội bộ, không cần dữ liệu
    !python step0_unsw.py --out_dir step0_v4
    !python step0_unsw.py --out_dir step0_v4_nottl --no_ttl
Chỉ kết luận khi hai lần chạy cho cùng quyết định. Chạy mỗi lệnh MỘT lần với cấu hình
mặc định; đừng đổi tham số rồi chạy lại chỉ vì đã thấy kết quả trên tập test.

Dữ liệu:
  - Mặc định script tự tải bằng kagglehub.
  - Nếu không tải được, tải thủ công từ
    https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 rồi truyền --data_dir.
  - Train/test xác định theo TÊN FILE (training-set / testing-set), không theo kích thước.
    Số dòng phải đúng bản chuẩn 175.341 / 82.332, nếu không script dừng (trừ khi có
    --skip_row_check). Script KHÔNG tự đảo vai hai file.

Mẹo:
  - Chạy thử toàn bộ pipeline trên mẫu nhỏ: thêm --quick (dùng --out_dir riêng).
  - Bị ngắt giữa chừng: chạy lại đúng lệnh cũ, script tự bỏ qua phần đã xong.
  - Thời gian ước lượng trên T4: khoảng 25-45 phút mỗi lần chạy.
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")
xgb.set_config(verbosity=0)

SCRIPT_VERSION = 6

# ----------------------------------------------------------------------------
# Cấu hình view theo đề cương v2 (mục 3.1): 42 đặc trưng, mỗi đặc trưng 1 view
# ----------------------------------------------------------------------------
VIEWS = {
    "V0": ["dur", "proto", "service", "is_sm_ips_ports"],                  # core, luôn có
    "V1": ["sbytes", "spkts", "sttl", "sload", "sloss", "sinpkt", "sjit",
           "swin", "stcpb", "smean"],                                        # chiều nguồn
    "V2": ["dbytes", "dpkts", "dttl", "dload", "dloss", "dinpkt", "djit",
           "dwin", "dtcpb", "dmean", "synack", "ackdat", "tcprtt", "state",
           "rate", "ct_state_ttl"],                                          # chiều đích & 2 chiều
    "V3": ["trans_depth", "response_body_len", "ct_flw_http_mthd",
           "is_ftp_login", "ct_ftp_cmd"],                                    # tầng ứng dụng
    "V4": ["ct_srv_src", "ct_srv_dst", "ct_src_ltm", "ct_dst_ltm",
           "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],        # ngữ cảnh kết nối
}
DROPPABLE = ["V1", "V2", "V3", "V4"]   # V3 ở vị trí 2 (dùng trong B5)
CAT_COLS = ["proto", "service", "state"]
TTL_COLS = ["sttl", "dttl", "ct_state_ttl"]
# Tên cột ở một số bản phân phối khác -> tên trong bản training/testing-set
ALIASES = {"sintpkt": "sinpkt", "dintpkt": "dinpkt", "smeansz": "smean",
           "dmeansz": "dmean", "res_bdy_len": "response_body_len"}
# Kịch bản theo nguyên nhân (mục 5.2): (các view thiếu, có đổi service http->ssl không)
S2 = {
    "S2a_tls": (["V3"], True),
    "S2b_onesided": (["V2", "V3"], False),
    "S2c_coldstart": (["V4"], False),
    "S2d_worst": (["V2", "V3", "V4"], False),
}
REALISTIC = ["S2a_tls", "S2b_onesided", "S2c_coldstart"]   # kịch bản một nguyên nhân (luật A)
# 16 tổ hợp; ký tự thứ i = '1' nghĩa là view DROPPABLE[i] có mặt
PATTERNS = ["".join(b) for b in itertools.product("10", repeat=4)]
EVAL_KEYS = PATTERNS + ["S2a_tls"]
FULL = "1111"
LOSS_MIN = 1.0     # luật A: mức giảm tối thiểu (điểm macro-F1)
NN_GAP_MAX = 2.0   # luật B: B5 được phép kém BEST tối đa (điểm, trung bình 9 kịch bản)
MODELS = ["B1", "B2", "B6", "B5"]
FAMILIES = ["B2", "B6"]                # hai họ baseline tham gia lựa chọn
# Tập gộp để chọn họ trên validation: đúng 16 tổ hợp view, mỗi tổ hợp một lần, có "1111",
# KHÔNG có S2a_tls. Cùng tập này cũng được dùng để chọn p_drop (B2) và lr (B5), nên cả ba
# lựa chọn nằm trên cùng một mặt đánh giá.
FAMILY_SELECT_KEYS = PATTERNS
SCENARIOS = ["S0"] + [f"S1_k{k}" for k in (1, 2, 3, 4)] + list(S2)
MISSING_SCENARIOS = [s for s in SCENARIOS if s != "S0"]   # 8 kịch bản, chỉ dùng cho chẩn đoán
B5_LRS = (1e-3, 3e-4)                  # lr thử cho B5, chọn theo validation
EXPECTED_ROWS = (175341, 82332)        # số dòng bản chuẩn UNSW-NB15 (train, test)
LABEL_MISMATCH_MAX = 0.001             # tỷ lệ lệch label/attack_cat tối đa cho phép
CIC_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/index.html"

AUDIT = []                             # nhật ký bằng chứng, in ở cuối và lưu vào JSON


def audit(msg):
    AUDIT.append(msg)


def missing_views(pat):
    return [v for v, b in zip(DROPPABLE, pat) if b == "0"]


def pattern_of(missing):
    return "".join("0" if v in missing else "1" for v in DROPPABLE)


def key_setting(key):
    """Khóa đánh giá -> (tổ hợp view, có đổi http->ssl không)."""
    return (pattern_of(S2["S2a_tls"][0]), True) if key == "S2a_tls" else (key, False)


# ----------------------------------------------------------------------------
# Tiện ích chung
# ----------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="RMV-IDS bước 0 trên UNSW-NB15")
    ap.add_argument("--data_dir", default=None,
                    help="Thư mục chứa 2 file training/testing-set (bỏ trống: tự tải bằng kagglehub)")
    ap.add_argument("--out_dir", default="step0_results")
    ap.add_argument("--no_ttl", action="store_true",
                    help="Bỏ sttl, dttl, ct_state_ttl (đối chứng C1)")
    ap.add_argument("--p_grid", default="0.2,0.3,0.4",
                    help="Các giá trị p_drop để chọn cho B2, cách nhau bởi dấu phẩy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000, help="Số mẫu bootstrap")
    ap.add_argument("--device", default=None, help="cuda hoặc cpu (mặc định: tự nhận)")
    ap.add_argument("--quick", action="store_true", help="Chạy thử nhanh trên mẫu nhỏ")
    ap.add_argument("--skip_row_check", action="store_true",
                    help="Bỏ kiểm tra số dòng chuẩn (chỉ dùng khi chạy thử trên dữ liệu giả)")
    ap.add_argument("--check_cic", action="store_true",
                    help="Chẩn đoán phụ: thử mở trang tải CIC-IDS2017 bản sửa (mặc định: tắt)")
    ap.add_argument("--selftest", action="store_true",
                    help="Chạy các kiểm tra nội bộ (không cần dữ liệu) rồi thoát")
    return ap.parse_args()


def detect_device():
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return "cuda"
    except Exception:
        return "cpu"


def check_cic():
    """Chẩn đoán phụ, KHÔNG nằm trong đường chạy của bước 0 (chỉ chạy khi có --check_cic)."""
    try:
        req = urllib.request.Request(CIC_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"[CIC-IDS2017 bản sửa] Trang phản hồi HTTP {r.status}: {CIC_URL}")
    except Exception as e:
        print(f"[CIC-IDS2017 bản sửa] KHÔNG mở được trang ({e}).")


def mf1(y, p, n_cls, w=None):
    """Macro-F1 (điểm %) tính từ ma trận nhầm lẫn; w = trọng số mẫu (bootstrap / lọc dòng).
    Lớp không có mẫu thật lẫn mẫu dự đoán được tính F1 = 0 (như sklearn, zero_division=0)."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p).astype(np.int64)
    cm = np.bincount(y * n_cls + p, weights=w, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    tp = np.diag(cm)
    d = cm.sum(0) + cm.sum(1)                      # = 2TP + FP + FN
    return float(100 * np.mean(np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0.0)))


def summarize(res):
    """Từ điểm theo từng khóa đánh giá -> điểm theo kịch bản S0, S1_k, S2."""
    out = {"S0": res[FULL]}
    for k in range(1, 5):
        out[f"S1_k{k}"] = float(np.mean([res[p] for p in PATTERNS if p.count("0") == k]))
    for name, (miss, _) in S2.items():
        out[name] = res["S2a_tls"] if name == "S2a_tls" else res[pattern_of(miss)]
    return out


class PredStore:
    """Lưu nhãn dự đoán vào 1 file .npz, khóa '<split>__<model>__<key>'.

    Nhãn trên TEST bị NIÊM PHONG: get_test() ném lỗi cho tới khi unseal_test() được gọi,
    và unseal_test() chỉ được gọi sau khi họ baseline (chọn trên validation) đã được KHÓA.
    Nhờ vậy không có lựa chọn nào trong script có thể nhìn thấy test."""

    def __init__(self, path):
        self.path = path
        self.d = {}
        self._test_open = False
        if path.exists():
            with np.load(path) as z:
                self.d = {k: z[k] for k in z.files}

    @staticmethod
    def _k(split, model, key):
        return f"{split}__{model}__{key}"

    def put(self, split, model, key, pred):
        assert split in ("val", "test")
        self.d[self._k(split, model, key)] = np.asarray(pred).astype(np.int8)

    def has(self, split, model, key):
        return self._k(split, model, key) in self.d

    def get_val(self, model, key):
        return self.d[self._k("val", model, key)].astype(np.int64)

    def unseal_test(self, reason):
        self._test_open = True
        audit(f"Mở niêm phong nhãn dự đoán TEST lúc {time.strftime('%H:%M:%S')} - {reason}")

    def get_test(self, model, key):
        if not self._test_open:
            raise RuntimeError("RÒ RỈ TEST: đọc nhãn dự đoán trên test trước khi chốt lựa chọn.")
        return self.d[self._k("test", model, key)].astype(np.int64)

    def save(self):
        tmp = self.path.with_name(self.path.stem + "_tmp.npz")
        np.savez(tmp, **self.d)
        tmp.replace(self.path)


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def get_data_dir(arg):
    if arg:
        return arg
    try:
        import kagglehub
        path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
        print(f"Đã tải dữ liệu vào: {path}")
        return path
    except Exception as e:
        sys.exit(f"Không tự tải được bằng kagglehub ({e}).\n"
                 "Hãy tải thủ công từ https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 "
                 "rồi chạy lại với --data_dir <thư mục>.")


def find_csvs(data_dir):
    """Tìm cặp training/testing-set NẰM CÙNG MỘT THƯ MỤC, xác định vai trò theo TÊN FILE."""
    sets = {}
    for p in sorted(Path(data_dir).rglob("*.csv")):
        name = p.name.lower()
        kind = ("tr" if ("training-set" in name or "training_set" in name) else
                "te" if ("testing-set" in name or "testing_set" in name) else None)
        if kind:
            sets.setdefault(p.parent, {}).setdefault(kind, []).append(p)
    pairs = [(v["tr"][0], v["te"][0]) for v in sets.values()
             if len(v.get("tr", [])) == 1 and len(v.get("te", [])) == 1]
    if not pairs:
        sys.exit(f"Không tìm thấy đúng một cặp training-set/testing-set trong cùng thư mục "
                 f"dưới {data_dir}. Hãy chỉ rõ bằng --data_dir.")
    sizes = {(a.stat().st_size, b.stat().st_size) for a, b in pairs}
    if len(sizes) > 1:
        sys.exit("Tìm thấy nhiều bộ training/testing-set KHÁC NHAU:\n  "
                 + "\n  ".join(str(a.parent) for a, _ in pairs)
                 + "\nHãy chỉ rõ thư mục bằng --data_dir.")
    if len(pairs) > 1:
        print(f"LƯU Ý: có {len(pairs)} bản giống nhau, dùng bản trong {pairs[0][0].parent}")
    return pairs[0]


def read_split(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
    if "attack_cat" not in df.columns:
        sys.exit(f"{path.name} không có cột attack_cat - không phải bản training/testing-set.")
    cat = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["attack_cat"] = cat.where(cat != "", "Normal").replace({"Backdoors": "Backdoor"})
    return df


def check_rows(n_tr, n_te, skip):
    exp_tr, exp_te = EXPECTED_ROWS
    if (n_tr, n_te) == (exp_tr, exp_te):
        audit(f"Số dòng đúng bản chuẩn: train {n_tr:,}, test {n_te:,}.")
        return
    if skip:
        audit(f"BỎ QUA kiểm tra số dòng (--skip_row_check): train {n_tr:,}, test {n_te:,}.")
        print(f"CẢNH BÁO: bỏ qua kiểm tra số dòng chuẩn (train {n_tr:,}, test {n_te:,}).")
        return
    if (n_tr, n_te) == (exp_te, exp_tr):
        sys.exit(f"Số dòng cho thấy hai file BỊ ĐẢO TÊN (train {n_tr:,}, test {n_te:,}; "
                 f"bản chuẩn là {exp_tr:,}/{exp_te:,}). Script không tự đảo vai để giữ nguyên "
                 "cách chia chuẩn - hãy đặt lại tên file cho đúng rồi chạy lại.")
    sys.exit(f"Số dòng không khớp bản chuẩn: train {n_tr:,} (cần {exp_tr:,}), "
             f"test {n_te:,} (cần {exp_te:,}). Nếu cố ý dùng dữ liệu khác, thêm --skip_row_check.")


def row_hash(df, cols):
    """Hash từng dòng sau khi chuẩn hóa kiểu (để so trùng giữa 2 file không lệch dtype)."""
    t = pd.DataFrame({c: (df[c].astype(str).str.strip() if c in CAT_COLS + ["attack_cat"]
                          else pd.to_numeric(df[c], errors="coerce").astype("float64"))
                      for c in cols})
    return pd.util.hash_pandas_object(t, index=False).to_numpy()


def encode(fit_df, dfs, feats):
    """Biến phân loại -> category (từ điển CHỈ lấy từ fit_df = train, thêm '__unk__');
    biến số -> float32, ô không hợp lệ -> 0 (để NaN CHỈ mang nghĩa 'view bị thiếu')."""
    dtypes = {}
    for c in feats:
        if c in CAT_COLS:
            vals = fit_df[c].fillna("__nan__").astype(str).str.strip()
            dtypes[c] = pd.CategoricalDtype(sorted(vals.unique().tolist()) + ["__unk__"])
    out, n_bad = [], 0
    for df in dfs:
        cols = {}
        for c in feats:
            if c in dtypes:
                s = df[c].fillna("__nan__").astype(str).str.strip()
                s = s.where(s.isin(dtypes[c].categories), "__unk__")
                cols[c] = pd.Categorical(s.to_numpy(), dtype=dtypes[c])
            else:
                v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32", copy=True)
                bad = ~np.isfinite(v)
                n_bad += int(bad.sum())
                v[bad] = 0.0
                cols[c] = v
        out.append(pd.DataFrame(cols))
    if n_bad:
        print(f"CẢNH BÁO: {n_bad:,} ô số rỗng/không hợp lệ trong dữ liệu gốc đã thay bằng 0.")
    return out, dtypes


def apply_pattern(X, pat, views, remap_http=False, drop=False):
    """Che (NaN) hoặc bỏ hẳn (drop=True) các view thiếu; tùy chọn đổi service http->ssl."""
    X = X.copy()
    cols = [c for v in missing_views(pat) for c in views[v]]
    if drop:
        X = X.drop(columns=cols)
    else:
        for c in cols:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = pd.Categorical.from_codes(np.full(len(X), -1), dtype=X[c].dtype)
            else:
                X[c] = np.full(len(X), np.nan, dtype="float32")
    if remap_http and "service" in X.columns:
        s = X["service"]
        target = "ssl" if "ssl" in s.cat.categories else "__unk__"
        X["service"] = s.where(s != "http", target)
    return X


def class_weights(y, n_cls):
    freq = np.bincount(y, minlength=n_cls).astype(float)
    sw = (1.0 / np.sqrt(np.maximum(freq, 1)))[y]
    return (sw / sw.mean()).astype("float32")


def load_data(args):
    tr_path, te_path = find_csvs(get_data_dir(args.data_dir))
    train, test = read_split(tr_path), read_split(te_path)
    print(f"Train: {tr_path.name} {train.shape} | Test: {te_path.name} {test.shape}")
    check_rows(len(train), len(test), args.skip_row_check)

    all_feats = [c for v in VIEWS.values() for c in v]
    for name, df in (("train", train), ("test", test)):
        have = set(df.columns) - {"id", "attack_cat", "label"}
        miss, extra = set(all_feats) - have, have - set(all_feats)
        if miss or extra:
            sys.exit(f"[{name}] Cột không khớp đề cương. Thiếu: {sorted(miss)} | "
                     f"Thừa: {sorted(extra)}\nHãy dùng đúng bản training/testing-set.")
        if "label" in df.columns:      # nhãn dùng cho thí nghiệm là attack_cat
            lab = pd.to_numeric(df["label"], errors="coerce")
            bad = int(((df["attack_cat"] == "Normal") != (lab == 0)).sum())
            rate = bad / len(df)
            print(f"[{name}] label lệch attack_cat: {bad:,} dòng ({rate:.4%})")
            audit(f"Nhất quán label/attack_cat [{name}]: {bad:,} dòng lệch ({rate:.4%}), "
                  f"ngưỡng cho phép {LABEL_MISMATCH_MAX:.2%}.")
            if rate > LABEL_MISMATCH_MAX:
                sys.exit(f"[{name}] tỷ lệ lệch {rate:.4%} vượt ngưỡng {LABEL_MISMATCH_MAX:.2%}. "
                         "Ground truth của thí nghiệm là attack_cat; hãy kiểm tra lại file dữ liệu.")

    views = {v: [c for c in cols if not (args.no_ttl and c in TTL_COLS)]
             for v, cols in VIEWS.items()}
    feats = [c for v in views.values() for c in v]

    # C3 (chẩn đoán): dòng test trùng hệt một dòng train, so trên 42 đặc trưng + attack_cat
    h_tr = row_hash(train, all_feats + ["attack_cat"])
    h_te = row_hash(test, all_feats + ["attack_cat"])
    dup_full = np.isin(h_te, h_tr)
    fingerprint = {"train_rows": int(len(train)), "test_rows": int(len(test)),
                   "sha1": hashlib.sha1(h_tr.tobytes() + h_te.tobytes()).hexdigest()[:16]}

    classes = sorted(train["attack_cat"].unique().tolist())
    if "Normal" not in classes:
        sys.exit(f"Không thấy lớp Normal. Các lớp: {classes}")
    unseen = sorted(set(test["attack_cat"]) - set(classes))
    if unseen:
        sys.exit(f"Test có lớp không có trong train: {unseen}")
    cid = {c: i for i, c in enumerate(classes)}
    y_all = train["attack_cat"].map(cid).to_numpy(dtype=np.int64)
    y_te = test["attack_cat"].map(cid).to_numpy(dtype=np.int64)

    if args.quick:  # lấy mẫu nhỏ, giữ tỷ lệ lớp
        if len(train) > 20000:
            keep, _ = train_test_split(np.arange(len(train)), train_size=20000,
                                       stratify=y_all, random_state=args.seed)
            train, y_all = train.iloc[keep].reset_index(drop=True), y_all[keep]
        if len(test) > 10000:
            keep, _ = train_test_split(np.arange(len(test)), train_size=10000,
                                       stratify=y_te, random_state=args.seed)
            test, y_te, dup_full = test.iloc[keep].reset_index(drop=True), y_te[keep], dup_full[keep]

    i_tr, i_va = train_test_split(np.arange(len(train)), test_size=0.1,
                                  stratify=y_all, random_state=args.seed)
    tr_df = train.iloc[i_tr].reset_index(drop=True)
    va_df = train.iloc[i_va].reset_index(drop=True)
    (Xtr, Xva, Xte), dtypes = encode(tr_df, [tr_df, va_df, test], feats)
    audit(f"Tiền xử lý fit trên TRAIN ({len(tr_df):,} dòng): từ điển biến phân loại, "
          "log1p/z-score của B5. Validation và test chỉ được biến đổi theo tham số đó.")
    if "http" not in dtypes["service"].categories:
        print("CẢNH BÁO: train không có service 'http'; kịch bản S2a sẽ không đổi gì.")
    if "ssl" not in dtypes["service"].categories:
        print("CẢNH BÁO: train không có service 'ssl'; S2a sẽ đổi http -> '__unk__'.")

    n_cls = len(classes)
    print("Số mẫu mỗi lớp (train | test):")
    for c in classes:
        print(f"    {c:<15}{int((y_all == cid[c]).sum()):>9,} |{int((y_te == cid[c]).sum()):>8,}")
    n_dup = int(dup_full.sum())
    print(f"Đặc trưng dùng: {len(feats)} | train {len(i_tr):,} / val {len(i_va):,} / "
          f"test {len(y_te):,} | dòng test trùng hệt train: {n_dup:,} ({n_dup / len(y_te):.3%})")
    audit(f"Chia dữ liệu: train {len(i_tr):,}, validation {len(i_va):,} (10% phân tầng của "
          f"file train), test {len(y_te):,} (file test chuẩn, không đụng tới).")
    audit(f"Trùng lặp chéo train-test: {n_dup:,} dòng ({n_dup / len(y_te):.3%}) - theo đề cương "
          "đây là chẩn đoán, không loại khỏi dữ liệu; báo cáo có thêm bảng trên tập test đã bỏ trùng.")
    return dict(Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=y_all[i_tr], yva=y_all[i_va], yte=y_te,
                wtr=class_weights(y_all[i_tr], n_cls), wva=class_weights(y_all[i_va], n_cls),
                views=views, feats=feats, classes=classes, n_cls=n_cls,
                normal=cid["Normal"], n_dup=n_dup, dup_mask=dup_full, fingerprint=fingerprint)


# ----------------------------------------------------------------------------
# XGBoost
# ----------------------------------------------------------------------------
def new_xgb(cfg):
    return xgb.XGBClassifier(
        n_estimators=60 if cfg["quick"] else 2000,
        learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device=cfg["device"], enable_categorical=True,
        max_cat_to_onehot=1, objective="multi:softprob", eval_metric="mlogloss",
        early_stopping_rounds=10 if cfg["quick"] else 50,
        random_state=cfg["seed"], n_jobs=-1)


def fit(cfg, Xtr, ytr, wtr, Xva, yva, wva, tag=""):
    """Fit trên TRAIN; VALIDATION chỉ dùng cho early stopping."""
    t = time.time()
    m = new_xgb(cfg)
    m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)],
          sample_weight_eval_set=[wva], verbose=False)
    print(f"    {tag}: {len(Xtr):,} dòng x {Xtr.shape[1]} cột, "
          f"best_iter={m.best_iteration}, {time.time() - t:.0f}s", flush=True)
    return m


def xgb_preds(model, X, views, keys, drop_key=None):
    """Nhãn dự đoán cho từng khóa đánh giá. drop_key != None: mô hình chỉ có cột của
    tổ hợp đó (B6), nên bỏ hẳn cột thay vì để NaN."""
    out = {}
    for key in keys:
        pat, remap = key_setting(key)
        Xk = apply_pattern(X, pat, views, remap_http=remap, drop=drop_key is not None)
        out[key] = np.asarray(model.predict(Xk))
    return out


def replicate(X, y, w_cls, views, p):
    """Nhân dữ liệu theo 16 tổ hợp; trọng số = xác suất tổ hợp dưới p_drop x trọng số lớp.
    Tổ hợp thiếu V3 có thêm một bản đổi http->ssl, chia đôi trọng số
    (khớp với view-dropout của B5 và RMV-IDS)."""
    parts, ws = [], []
    for pat in PATTERNS:
        k = pat.count("0")
        wp = (p ** k) * ((1 - p) ** (4 - k))
        variants = [False, True] if "V3" in missing_views(pat) else [False]
        for remap in variants:
            parts.append(apply_pattern(X, pat, views, remap_http=remap))
            ws.append(w_cls * wp / len(variants))
    w = np.concatenate(ws)
    return (pd.concat(parts, ignore_index=True), np.tile(y, len(parts)),
            (w / w.mean()).astype("float32"))


def run_b1(D, cfg, store):
    print("\n[B1] XGBoost trên dữ liệu đầy đủ (tham khảo, không tham gia định tuyến)")
    m = fit(cfg, D["Xtr"], D["ytr"], D["wtr"], D["Xva"], D["yva"], D["wva"], "B1")
    for k, p in xgb_preds(m, D["Xte"], D["views"], EVAL_KEYS).items():
        store.put("test", "B1", k, p)
    store.save()
    return {"done": True}


def run_b2(D, cfg, p_grid, store):
    print("\n[B2] XGBoost + che view (16 tổ hợp + 8 bản đổi http->ssl)")
    best, val_scores = None, {}
    for p in p_grid:
        Xr, yr, wr = replicate(D["Xtr"], D["ytr"], D["wtr"], D["views"], p)
        Xv, yv, wv = replicate(D["Xva"], D["yva"], D["wva"], D["views"], p)
        m = fit(cfg, Xr, yr, wr, Xv, yv, wv, f"B2 p_drop={p}")
        del Xr, Xv
        # Chọn p CHỈ trên validation, và CHỈ trên 16 tổ hợp view (không dùng S2a_tls)
        vp = xgb_preds(m, D["Xva"], D["views"], PATTERNS)
        score = float(np.mean([mf1(D["yva"], vp[pat], D["n_cls"]) for pat in PATTERNS]))
        val_scores[str(p)] = score
        print(f"    macro-F1 validation (trung bình 16 tổ hợp) = {score:.2f}")
        if best is None or score > best[0]:
            best = (score, p, m, vp)
    _, p_best, m, vp = best
    print(f"    -> chọn p_drop = {p_best} (trên validation)")
    for k, pr in vp.items():
        store.put("val", "B2", k, pr)
    for k, pr in xgb_preds(m, D["Xte"], D["views"], EVAL_KEYS).items():
        store.put("test", "B2", k, pr)
    store.save()
    audit(f"B2: p_drop chọn trên VALIDATION (16 tổ hợp, không dùng S2a_tls). "
          f"p_grid={p_grid}, val={val_scores}, p_best={p_best}.")
    return {"done": True, "p_drop": p_best, "val_scores": val_scores}


def run_b6(D, cfg, results, store, save):
    print("\n[B6] Mỗi tổ hợp view một mô hình riêng")
    # Khóa "S2a_tls" là mô hình theo CHẾ ĐỘ mã hóa của họ B6: train trên dữ liệu TRAIN đã bỏ
    # V3 và đổi http->ssl. Chỉ được dùng nếu họ B6 được KHÓA; S2a không tham gia lựa chọn nào.
    b6 = results.setdefault("B6", {"keys_done": []})
    for key in EVAL_KEYS:
        if key in b6["keys_done"]:
            continue
        pat, remap = key_setting(key)

        def view_of(X):
            return apply_pattern(X, pat, D["views"], remap_http=remap, drop=True)

        m = fit(cfg, view_of(D["Xtr"]), D["ytr"], D["wtr"],
                view_of(D["Xva"]), D["yva"], D["wva"], f"B6 {key}")
        if key != "S2a_tls":     # S2a_tls không tham gia lựa chọn nào -> không cần nhãn val
            store.put("val", "B6", key, np.asarray(m.predict(view_of(D["Xva"]))))
        store.put("test", "B6", key, np.asarray(m.predict(view_of(D["Xte"]))))
        store.save()
        b6["keys_done"].append(key)
        save()
    b6["done"] = True
    save()


# ----------------------------------------------------------------------------
# B5: MLP phẳng + view-dropout (PyTorch)
# ----------------------------------------------------------------------------
def mask_inputs(xn, xc, m, remap, spec):
    """Áp mặt nạ view cho B5.
    m: (B,4), 1 = V1..V4 có mặt. remap: (B,) bool, đổi service http->ssl.
    spec: num_v (view của từng cột số), cat_v (view của từng cột phân loại),
          cat_n (chỉ số 'bị che' của từng cột phân loại), s_idx, http_code, tgt_code."""
    import torch
    full = torch.cat([torch.ones_like(m[:, :1]), m], 1)          # V0 luôn có
    xn = xn * full[:, spec["num_v"]]                               # cột số của view thiếu -> 0
    xc = xc.clone()
    for i, v in enumerate(spec["cat_v"]):
        if v > 0:
            xc[:, i] = torch.where(full[:, v] > 0, xc[:, i],
                                   torch.full_like(xc[:, i], spec["cat_n"][i]))
    if spec["http_code"] >= 0:
        hit = remap & (xc[:, spec["s_idx"]] == spec["http_code"])
        xc[hit, spec["s_idx"]] = spec["tgt_code"]
    return xn, xc, m


def run_b5(D, cfg, p_drop, store):
    import torch
    import torch.nn as nn

    print(f"\n[B5] MLP phẳng + view-dropout (p_drop = {p_drop}, lấy từ B2)")
    dev = torch.device("cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    feats, n_cls = D["feats"], D["n_cls"]
    num_cols = [c for c in feats if c not in CAT_COLS]
    cat_cols = [c for c in feats if c in CAT_COLS]
    vid = {c: int(v[1:]) for v, cols in D["views"].items() for c in cols}   # "V2" -> 2

    # Số: log1p cho cột không âm (xét trên TRAIN), rồi z-score với tham số fit trên TRAIN
    nonneg = D["Xtr"][num_cols].to_numpy(np.float64).min(0) >= 0

    def num_prep(X):
        A = X[num_cols].to_numpy(np.float64).copy()
        A[:, nonneg] = np.log1p(np.clip(A[:, nonneg], 0, None))
        return A

    L = num_prep(D["Xtr"])
    mu, sd = L.mean(0), L.std(0)
    sd[sd < 1e-6] = 1.0

    def to_t(X):
        xn = np.clip((num_prep(X) - mu) / sd, -10, 10).astype(np.float32)
        xc = np.stack([X[c].cat.codes.to_numpy().astype(np.int64) for c in cat_cols], 1)
        return torch.tensor(xn, device=dev), torch.tensor(xc, device=dev)

    Tr, Va, Te = to_t(D["Xtr"]), to_t(D["Xva"]), to_t(D["Xte"])
    ytr = torch.tensor(D["ytr"], device=dev)
    num_v = torch.tensor([vid[c] for c in num_cols], device=dev)
    cat_v = [vid[c] for c in cat_cols]
    cat_n = [len(D["Xtr"][c].cat.categories) for c in cat_cols]   # chỉ số cat_n[i] = "bị che"
    s_idx = cat_cols.index("service")
    svc = list(D["Xtr"]["service"].cat.categories)
    http_code = svc.index("http") if "http" in svc else -1
    tgt_code = svc.index("ssl") if "ssl" in svc else svc.index("__unk__")
    v3_pos = DROPPABLE.index("V3")
    spec = dict(num_v=num_v, cat_v=cat_v, cat_n=cat_n, s_idx=s_idx,
                http_code=http_code, tgt_code=tgt_code)

    def masked(xn, xc, m, remap):
        return mask_inputs(xn, xc, m, remap, spec)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            # chỉ số n-1 = '__unk__' (giá trị chưa gặp khi train) -> vector 0 cố định;
            # chỉ số n = 'bị che' (được học nhờ view-dropout)
            self.embs = nn.ModuleList([nn.Embedding(n + 1, 8 if n > 20 else 4, padding_idx=n - 1)
                                       for n in cat_n])
            d = len(num_cols) + sum(e.embedding_dim for e in self.embs) + len(DROPPABLE)
            layers = []
            for h in (256, 256, 128):
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
                d = h
            self.net = nn.Sequential(*layers, nn.Linear(d, n_cls))

        def forward(self, xn, xc, m):
            e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
            return self.net(torch.cat([xn, *e, m], 1))

    @torch.no_grad()
    def predict(model, T, key):
        model.eval()
        pat, remap = key_setting(key)
        n = len(T[0])
        m = torch.tensor([[float(b) for b in pat]], device=dev).expand(n, -1)
        r = torch.full((n,), remap, dtype=torch.bool, device=dev)
        out = []
        for s in range(0, n, 8192):
            xn, xc, mm = masked(T[0][s:s + 8192], T[1][s:s + 8192], m[s:s + 8192], r[s:s + 8192])
            out.append(model(xn, xc, mm).argmax(1))
        return torch.cat(out).cpu().numpy()

    freq = np.bincount(D["ytr"], minlength=n_cls).astype(float)
    cw = torch.tensor(1 / np.sqrt(np.maximum(freq, 1)), dtype=torch.float32, device=dev)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    n, bs = len(ytr), 1024
    max_ep, patience = (3, 2) if cfg["quick"] else (50, 8)

    def train_one(lr):
        torch.manual_seed(cfg["seed"])          # một seed cho mỗi lr (bước 0 là cổng rẻ)
        model = MLP().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best, best_state, bad, t0 = -1.0, None, 0, time.time()
        for ep in range(1, max_ep + 1):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                if len(idx) < 2:          # BatchNorm cần >= 2 mẫu
                    continue
                m = (torch.rand(len(idx), len(DROPPABLE), device=dev) >= p_drop).float()
                remap = (m[:, v3_pos] == 0) & (torch.rand(len(idx), device=dev) < 0.5)
                xn, xc, mm = masked(Tr[0][idx], Tr[1][idx], m, remap)
                loss = loss_fn(model(xn, xc, mm), ytr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            # early stopping + chọn lr: CHỈ trên validation, CHỈ 16 tổ hợp (không có S2a_tls)
            score = float(np.mean([mf1(D["yva"], predict(model, Va, pat), n_cls)
                                   for pat in PATTERNS]))
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if ep % 5 == 0 or ep == 1 or bad >= patience or ep == max_ep:
                print(f"    lr={lr:g} epoch {ep}: macro-F1 val (TB 16 tổ hợp) = {score:.2f} "
                      f"(tốt nhất {best:.2f})", flush=True)
            if bad >= patience:
                break
        model.load_state_dict(best_state)
        print(f"    lr={lr:g}: {ep} epoch, {time.time() - t0:.0f}s, val tốt nhất = {best:.2f}")
        return best, model

    runs = {lr: train_one(lr) for lr in B5_LRS}
    lr_best = max(runs, key=lambda k: runs[k][0])
    best, model = runs[lr_best]
    print(f"    -> chọn lr = {lr_best:g} (trên validation)")
    for key in EVAL_KEYS:
        store.put("test", "B5", key, predict(model, Te, key))
    store.save()
    val_scores = {f"{k:g}": v[0] for k, v in runs.items()}
    audit(f"B5: learning rate chọn trên VALIDATION (16 tổ hợp, không dùng S2a_tls). "
          f"val={val_scores}, lr_best={lr_best:g}. p_drop lấy từ B2 = {p_drop} (theo đề cương). "
          "Mỗi lr chỉ 1 seed: CI bootstrap KHÔNG phản ánh dao động theo seed.")
    return {"done": True, "p_drop": p_drop, "lr": lr_best, "val_score": best,
            "val_scores": val_scores}


# ----------------------------------------------------------------------------
# C2: thông tin từng view (chẩn đoán, không tham gia quyết định)
# ----------------------------------------------------------------------------
def run_c2(D, cfg):
    print("\n[C2] Thông tin từng view (chẩn đoán)")
    nm = D["normal"]
    yb_va = D["yva"] != nm
    out = {"per_view": {}, "stumps": {}}
    for v, cols in D["views"].items():
        m = fit(cfg, D["Xtr"][cols], D["ytr"], D["wtr"],
                D["Xva"][cols], D["yva"], D["wva"], f"C2 {v}")
        pred = np.asarray(m.predict(D["Xva"][cols]))       # chấm trên VALIDATION
        out["per_view"][v] = {
            "macro_f1_val": mf1(D["yva"], pred, D["n_cls"]),
            "bin_bal_acc_val": float(100 * balanced_accuracy_score(yb_va, pred != nm)),
        }
    # Cây 1 tầng trên từng đặc trưng số (nhị phân), fit trên train, chấm trên validation
    yb_tr = D["ytr"] != nm
    ranked = []
    for c in [c for c in D["feats"] if c not in CAT_COLS]:
        st = DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                    random_state=cfg["seed"]).fit(D["Xtr"][[c]], yb_tr)
        p = st.predict(D["Xva"][[c]])
        ranked.append((balanced_accuracy_score(yb_va, p), c, p))
    ranked.sort(key=lambda t: -t[0])
    for ba, c, p in ranked[:5] + [r for r in ranked[5:] if r[1] == "sttl"]:
        out["stumps"][c] = {"val_bal_acc": float(100 * ba),
                            "val_acc": float(100 * accuracy_score(yb_va, p))}
    majority = bool(yb_tr.mean() >= 0.5)
    out["majority_val_acc"] = float(100 * np.mean(yb_va == majority))
    audit("C2 (chẩn đoán) chấm điểm trên VALIDATION, không đụng test.")
    return out


# ----------------------------------------------------------------------------
# Chọn MỘT họ baseline (B2 hoặc B6) - CHỈ dựa trên validation, KHÓA trước khi mở niêm phong
# ----------------------------------------------------------------------------
def select_family_on_validation(D, store):
    """Chọn MỘT họ baseline (B2 hoặc B6) cho toàn bộ thí nghiệm, CHỈ dựa trên validation.

    Quy tắc gộp (chốt trước, không đổi): trung bình cộng không trọng số của macro-F1 trên
    đúng 16 tổ hợp view trong FAMILY_SELECT_KEYS - mỗi tổ hợp một lần, có cả "1111",
    không có S2a_tls. Hòa -> B2. Họ được chọn dùng cho mọi kịch bản; nếu là B6 thì bên
    trong B6 vẫn dùng mô hình riêng theo tổ hợp view."""
    tab = pd.DataFrame([{"pattern": pat,
                         **{f"val_{f}": mf1(D["yva"], store.get_val(f, pat), D["n_cls"])
                            for f in FAMILIES}}
                        for pat in FAMILY_SELECT_KEYS])
    means = {f: float(tab[f"val_{f}"].mean()) for f in FAMILIES}
    family = "B2" if means["B2"] >= means["B6"] else "B6"
    tab["thang_tung_to_hop"] = np.where(tab["val_B2"] >= tab["val_B6"], "B2", "B6")  # tham khảo
    audit("Chọn họ baseline: trung bình KHÔNG trọng số macro-F1 trên 16 tổ hợp view của "
          f"VALIDATION (không dùng S2a_tls). val trung bình B2={means['B2']:.2f}, "
          f"B6={means['B6']:.2f} -> KHÓA họ {family} cho mọi kịch bản S0/S1/S2. "
          "Không đổi họ giữa các kịch bản.")
    return family, means, tab


# ----------------------------------------------------------------------------
# Chấm điểm trên test (sau khi đã chốt) + bootstrap + luật quyết định
# ----------------------------------------------------------------------------
def decide(D, store, family, n_boot, seed):
    y, n_cls = D["yte"], D["n_cls"]
    preds = {m: {k: store.get_test(m, k) for k in EVAL_KEYS} for m in MODELS}
    preds["BEST"] = dict(preds[family])      # họ đã KHÓA; B6 tự định tuyến nội bộ theo tổ hợp
    for m, pk in preds.items():
        for k, p in pk.items():
            assert len(p) == len(y), f"Số dòng nhãn dự đoán {m}/{k} không khớp tập test."

    def scores(w=None):
        return {m: summarize({k: mf1(y, p, n_cls, w) for k, p in pk.items()})
                for m, pk in preds.items()}

    def gate(sc):
        b, f = sc["BEST"], sc["B5"]
        loss = {s: b["S0"] - b[s] for s in REALISTIC}
        gap = float(np.mean([b[s] - f[s] for s in SCENARIOS]))                  # luật B
        gap_missing = float(np.mean([b[s] - f[s] for s in MISSING_SCENARIOS]))  # chẩn đoán
        return loss, gap, gap_missing

    point = scores()
    loss, gap, gap_missing = gate(point)
    rng = np.random.default_rng(seed)
    L, G, GM = {s: [] for s in REALISTIC}, [], []
    t = time.time()
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(np.float64)
        lb, gb, gmb = gate(scores(w))     # KHÔNG chọn lại họ/mô hình bên trong bootstrap
        for s in REALISTIC:
            L[s].append(lb[s])
        G.append(gb)
        GM.append(gmb)
    print(f"    bootstrap {n_boot} lần trên nhãn dự đoán đã cố định: {time.time() - t:.0f}s")
    audit(f"Bootstrap {n_boot} lần: chỉ lấy lại mẫu các dòng test với nhãn dự đoán đã cố định "
          "của họ đã khóa; không có lựa chọn họ hay mô hình nào bên trong vòng lặp.")
    ci = {s: np.percentile(L[s], [2.5, 97.5]).tolist() for s in REALISTIC}
    real = {s: bool(loss[s] >= LOSS_MIN and ci[s][0] > 0) for s in REALISTIC}
    dup_scores = scores((~D["dup_mask"]).astype(np.float64)) if D["n_dup"] else None
    return {"family": family, "scores": point, "dup_free_scores": dup_scores,
            "loss": loss, "loss_ci": ci, "loss_real": real,
            "gap_best_minus_B5": gap, "gap_ci": np.percentile(G, [2.5, 97.5]).tolist(),
            "missing_only_gap": gap_missing,
            "missing_only_gap_ci": np.percentile(GM, [2.5, 97.5]).tolist(),
            "A_problem_real": any(real.values()), "B_nn_viable": bool(gap <= NN_GAP_MAX)}


def report(results, D, out_dir, args, dec, fam_table):
    S = dec["scores"]
    cols = MODELS + ["BEST"]
    rows = [{"scenario": s, **{m: S[m][s] for m in cols},
             "BEST_minus_B5": S["BEST"][s] - S["B5"][s]} for s in S["BEST"]]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "step0_summary.csv", index=False)
    fam_table.to_csv(out_dir / "step0_family_selection.csv", index=False)

    pd.set_option("display.width", 130)
    fm = results["family_val_mean"]
    print("\n============ CHỌN HỌ BASELINE (chỉ dùng VALIDATION, đã KHÓA) ============")
    print(fam_table.round(2).to_string(index=False))
    print(f"Trung bình 16 tổ hợp: B2 = {fm['B2']:.2f} | B6 = {fm['B6']:.2f} -> KHÓA họ "
          f"{results['family']} cho mọi kịch bản (S2a_tls không tham gia chọn). "
          "Cột 'thang_tung_to_hop' chỉ để tham khảo, KHÔNG dùng để đổi họ.")

    print("\n================ KẾT QUẢ TRÊN TEST (macro-F1, điểm %) ================")
    print(f"Thiết lập: {'BỎ TTL (C1)' if args.no_ttl else 'đầy đủ đặc trưng'} | "
          f"p_drop = {results['B2']['p_drop']} | lr B5 = {results['B5']['lr']:g} | "
          f"BEST = họ {results['family']} đã khóa trên validation (KHÔNG phải max trên test)")
    print(df.round(2).to_string(index=False))
    if dec["dup_free_scores"]:
        d2 = dec["dup_free_scores"]
        print(f"\n[C3] Bảng phụ trên tập test đã bỏ {D['n_dup']:,} dòng trùng train "
              "(chẩn đoán, không dùng cho quyết định):")
        print(pd.DataFrame([{"scenario": s, **{m: d2[m][s] for m in cols}}
                            for s in d2["BEST"]]).round(2).to_string(index=False))

    if max(S["BEST"][s] for s in MISSING_SCENARIOS) > S["BEST"]["S0"]:
        print("LƯU Ý: BEST ở một kịch bản thiếu view cao hơn BEST ở S0. Vì chỉ dùng MỘT họ cho "
              "mọi kịch bản nên đây không phải hiệu ứng đổi họ, mà là tính chất của dữ liệu "
              "(bỏ một view có thể bỏ luôn nhiễu). Luật A khi đó ra số âm ở kịch bản đó.")

    c2 = results["C2"]
    print("\n[C2] XGBoost trên từng view riêng lẻ (chấm trên validation):")
    for v, r in c2["per_view"].items():
        print(f"    {v}: macro-F1 = {r['macro_f1_val']:.2f} | "
              f"balanced acc nhị phân = {r['bin_bal_acc_val']:.2f}")
    print(f"[C2] Cây 1 tầng (nhị phân, validation); đoán toàn lớp đa số được acc = "
          f"{c2['majority_val_acc']:.2f}:")
    for c, r in c2["stumps"].items():
        print(f"    {c:<18} val acc = {r['val_acc']:.2f} | "
              f"val balanced acc = {r['val_bal_acc']:.2f}")

    print("\n================ QUYẾT ĐỊNH (mục 5.0) ================")
    print(f"Luật A - BEST giảm bao nhiêu so với S0 (cần >= {LOSS_MIN} và CI 95% > 0):")
    for s in REALISTIC:
        lo, hi = dec["loss_ci"][s]
        print(f"    {s:<14} {dec['loss'][s]:6.2f}  CI [{lo:.2f}, {hi:.2f}]  "
              f"{'ĐẠT' if dec['loss_real'][s] else '-'}")
    lo, hi = dec["gap_ci"]
    print(f"Luật B - B5 kém BEST trung bình {dec['gap_best_minus_B5']:.2f} điểm trên 9 kịch bản "
          f"(cần <= {NN_GAP_MAX}): {'ĐẠT' if dec['B_nn_viable'] else 'KHÔNG ĐẠT'}. "
          f"CI 95% [{lo:.2f}, {hi:.2f}] chỉ để tham khảo, luật B dùng điểm ước lượng.")
    mlo, mhi = dec["missing_only_gap_ci"]
    print(f"[Chẩn đoán, KHÔNG tham gia verdict] missing_only_gap (8 kịch bản thiếu view, bỏ S0) "
          f"= {dec['missing_only_gap']:.2f} điểm, CI 95% [{mlo:.2f}, {mhi:.2f}].")

    if not dec["A_problem_real"]:
        verdict, msg = "DỪNG", ("Thiếu view không làm mô hình tốt nhất giảm đáng kể ở kịch bản "
                                "thực tế nào: không có bài toán để giải.")
    elif not dec["B_nn_viable"]:
        verdict, msg = "KHÔNG LÀM RMV-IDS", (
            "Bài toán có thật, nhưng MLP kém mô hình cây quá xa: kiến trúc nơ-ron khó thắng. "
            "Chuyển sang phương án dự phòng: bài benchmark (đóng góp 1 và 3).")
    else:
        verdict, msg = "TIẾP TỤC", ("Bài toán có thật và mạng nơ-ron còn cửa. "
                                    "RMV-IDS phải vượt cột BEST ở từng kịch bản.")
    print(f"\n>>> {verdict}: {msg}")
    if args.quick:
        print("!!! Chế độ --quick (mẫu nhỏ, B5 ít epoch): chỉ để kiểm tra pipeline, "
              "KHÔNG dùng kết luận này.")
    if not args.no_ttl:
        print("Lưu ý: chạy thêm với --no_ttl (out_dir khác). "
              "Chỉ kết luận khi hai lần chạy cho cùng quyết định.")

    print("\n================ AUDIT: bằng chứng không rò rỉ test ================")
    for i, line in enumerate(AUDIT, 1):
        print(f"[{i}] {line}")
    results["decision"] = {"verdict": verdict, "message": msg, **dec}
    results["audit"] = list(AUDIT)


# ----------------------------------------------------------------------------
# Kiểm tra nội bộ (--selftest): không cần dữ liệu
# ----------------------------------------------------------------------------
def selftest():
    from sklearn.metrics import f1_score
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

    print("Kiểm tra nội bộ:")
    rng = np.random.default_rng(0)
    same = all(abs(mf1(y := rng.integers(0, 9, 200), p := rng.integers(0, 9, 200), 10)
                   - 100 * f1_score(y, p, average="macro", labels=list(range(10)),
                                    zero_division=0)) < 1e-9 for _ in range(50))
    check("macro-F1 khớp sklearn (kể cả khi có lớp vắng mặt)", same)
    check("16 tổ hợp view, không trùng", len(PATTERNS) == 16 and len(set(PATTERNS)) == 16)
    check("S2a_tls ánh xạ đúng tổ hợp thiếu V3",
          key_setting("S2a_tls") == ("1101", True) and key_setting("1010") == ("1010", False))

    tmp = Path("._selftest_preds.npz")
    st = PredStore(tmp)
    st.put("val", "B2", "1111", np.zeros(5)); st.put("val", "B6", "1111", np.ones(5))
    st.put("test", "B2", "1111", np.zeros(5))
    try:
        st.get_test("B2", "1111")
        check("niêm phong test: đọc trước khi chốt phải lỗi", False)
    except RuntimeError:
        check("niêm phong test: đọc trước khi chốt phải lỗi", True)
    st.unseal_test("selftest")
    check("sau khi mở niêm phong thì đọc được", len(st.get_test("B2", "1111")) == 5)
    check("nhãn validation đọc được mọi lúc", len(st.get_val("B6", "1111")) == 5)

    class FakeD(dict):
        pass
    D = FakeD(yva=np.array([0, 0, 1, 1]), n_cls=2)
    s2 = PredStore(tmp)                        # store mới -> nhãn test còn NIÊM PHONG
    for pat in PATTERNS:                       # B2 đúng 4/4, B6 đúng 2/4 -> phải chọn B2
        s2.put("val", "B2", pat, np.array([0, 0, 1, 1]))
        s2.put("val", "B6", pat, np.array([0, 0, 0, 0]))
        s2.put("test", "B2", pat, np.array([1, 1, 0, 0]))   # nhãn test ngược hẳn lại
        s2.put("test", "B6", pat, np.array([0, 0, 1, 1]))
    fam, means, _ = select_family_on_validation(D, s2)
    check("chọn họ B2 khi B2 thắng trên validation, dù trên test thì B6 tốt hơn", fam == "B2")
    check("việc chọn họ không đọc nhãn test (store vẫn niêm phong sau khi chọn)",
          not s2._test_open)
    for pat in PATTERNS:                       # đổi vai -> phải chọn B6
        s2.put("val", "B2", pat, np.array([0, 0, 0, 0]))
        s2.put("val", "B6", pat, np.array([0, 0, 1, 1]))
    fam2, _, _ = select_family_on_validation(D, s2)
    check("chọn họ B6 khi B6 thắng trên validation", fam2 == "B6")
    for pat in PATTERNS:                       # hòa -> tie-break về B2
        s2.put("val", "B6", pat, np.array([0, 0, 0, 0]))
    fam3, _, tab3 = select_family_on_validation(D, s2)
    check("hòa trên validation -> chọn B2", fam3 == "B2")
    check("tập gộp đúng 16 tổ hợp view, không có S2a_tls",
          len(tab3) == 16 and "S2a_tls" not in set(tab3["pattern"]))
    check("9 kịch bản cho luật B, 8 kịch bản cho chẩn đoán missing_only_gap",
          len(SCENARIOS) == 9 and len(MISSING_SCENARIOS) == 8 and "S0" not in MISSING_SCENARIOS)
    tmp.unlink(missing_ok=True)

    # --- chống oracle test ngay trong decide(): BEST luôn theo họ đã KHÓA -------
    import contextlib
    import io

    y = np.array([0, 0, 1, 1] * 25)
    good, bad = y.copy(), np.zeros_like(y)
    noise = np.random.default_rng(1).integers(0, 2, len(y))
    Dt = FakeD(yte=y, n_cls=2, dup_mask=np.zeros(len(y), bool), n_dup=0)

    def fake_store(b2, b6, b5):
        s = PredStore(Path("._selftest_decide.npz"))
        for k in EVAL_KEYS:
            s.put("test", "B1", k, bad)
            s.put("test", "B2", k, b2)
            s.put("test", "B6", k, b6)
            s.put("test", "B5", k, b5)
        s.unseal_test("selftest")
        return s

    def run_decide(b2, b6, b5, family, n_boot=20, seed=0):
        with contextlib.redirect_stdout(io.StringIO()):
            return decide(Dt, fake_store(b2, b6, b5), family, n_boot, seed)

    def eq_best(dec, fam):
        return all(abs(dec["scores"]["BEST"][s] - dec["scores"][fam][s]) < 1e-12
                   for s in SCENARIOS)

    d1 = run_decide(good, bad, bad, "B2")
    check("BEST bằng đúng họ đã KHÓA (B2 khóa, B2 đang tốt hơn trên test)", eq_best(d1, "B2"))
    d2 = run_decide(bad, good, bad, "B2")          # B6 tốt hơn HẲN trên test
    check("KHÓA B2 nhưng B6 tốt hơn nhiều trên test -> BEST vẫn là B2",
          eq_best(d2, "B2") and d2["scores"]["BEST"]["S0"] < d2["scores"]["B6"]["S0"] - 10)
    d3 = run_decide(good, bad, bad, "B6")          # B2 tốt hơn HẲN trên test
    check("KHÓA B6 nhưng B2 tốt hơn nhiều trên test -> BEST vẫn là B6",
          eq_best(d3, "B6") and d3["scores"]["BEST"]["S0"] < d3["scores"]["B2"]["S0"] - 10)
    check("decide() ghi lại đúng họ đã khóa", d2["family"] == "B2" and d3["family"] == "B6")

    mid = y.copy()
    mid[::4] = 1 - mid[::4]                       # họ được khóa chỉ ở mức trung bình
    dA = run_decide(mid, bad, noise, "B2", n_boot=50, seed=7)
    dB = run_decide(mid, good, noise, "B2", n_boot=50, seed=7)   # B6 (không khóa) tốt hơn hẳn
    check("bootstrap chỉ dùng BEST cố định: đổi hẳn nhãn test của họ KHÔNG khóa thì luật A, "
          "luật B và mọi CI không đổi",
          dA["loss"] == dB["loss"] and dA["loss_ci"] == dB["loss_ci"]
          and dA["gap_best_minus_B5"] == dB["gap_best_minus_B5"]
          and dA["gap_ci"] == dB["gap_ci"]
          and dA["missing_only_gap"] == dB["missing_only_gap"]
          and dA["missing_only_gap_ci"] == dB["missing_only_gap_ci"])
    Path("._selftest_decide.npz").unlink(missing_ok=True)

    try:
        import torch
        spec = dict(num_v=torch.tensor([0, 1, 2, 3, 4]), cat_v=[0, 0, 2], cat_n=[5, 4, 3],
                    s_idx=1, http_code=2, tgt_code=3)
        a, b, _ = mask_inputs(torch.ones(2, 5), torch.tensor([[0, 2, 1]] * 2),
                              torch.tensor([[0., 1, 0, 1], [1, 1, 1, 1]]),
                              torch.tensor([True, False]), spec)
        check("che view cho B5: cột số về 0, cột phân loại về chỉ số 'bị che', remap http->ssl",
              a.tolist() == [[1, 0, 1, 0, 1], [1, 1, 1, 1, 1]] and
              b.tolist() == [[0, 3, 1], [0, 2, 1]])
    except ImportError:
        check("PyTorch có sẵn (cần cho B5)", False)

    print("KẾT QUẢ:", "TẤT CẢ PASS" if ok else "CÓ KIỂM TRA THẤT BẠI")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"device": args.device or detect_device(), "seed": args.seed, "quick": args.quick}
    p_grid = [float(x) for x in args.p_grid.split(",") if x.strip()]
    print(f"Script v{SCRIPT_VERSION} | XGBoost {xgb.__version__} | pandas {pd.__version__} "
          f"| device = {cfg['device']}")
    if cfg["device"] == "cpu" and not args.quick:
        print("CẢNH BÁO: không thấy GPU, chạy CPU sẽ rất lâu. "
              "Trên Colab hãy chọn Runtime -> T4 GPU.")
    try:
        import torch  # noqa: F401  (cần cho B5; Colab có sẵn)
    except ImportError:
        sys.exit("Thiếu PyTorch (cần cho B5). Cài bằng: pip install torch")
    if args.check_cic:
        check_cic()
    audit(f"Hằng số quyết định cố định trong file: LOSS_MIN={LOSS_MIN}, "
          f"NN_GAP_MAX={NN_GAP_MAX}, p_grid={p_grid}, B5_LRS={list(B5_LRS)}, seed={args.seed}.")
    audit("S2a_tls là kịch bản đánh giá giữ riêng: không dùng để chọn p_drop, chọn lr, "
          "hay chọn họ baseline.")

    res_path = out_dir / "step0_results.json"
    meta = {"version": SCRIPT_VERSION, "no_ttl": args.no_ttl, "p_grid": p_grid,
            "seed": args.seed, "quick": args.quick, "skip_row_check": args.skip_row_check}
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    if results and results.get("meta") != meta:
        sys.exit(f"{res_path} chứa kết quả của cấu hình/phiên bản khác "
                 f"{results.get('meta')}. Hãy dùng --out_dir mới.")
    results["meta"] = meta
    results["env"] = {"xgboost": xgb.__version__, "pandas": pd.__version__,
                      "device": cfg["device"]}
    store = PredStore(out_dir / "step0_preds.npz")
    # Kết quả trong JSON chỉ dùng lại khi nhãn dự đoán tương ứng có đủ trong file npz
    need = {"B1": [("test", EVAL_KEYS)],
            "B2": [("val", PATTERNS), ("test", EVAL_KEYS)],
            "B5": [("test", EVAL_KEYS)]}
    for m, specs in need.items():
        if m in results and not all(store.has(sp, m, k) for sp, ks in specs for k in ks):
            print(f"LƯU Ý: thiếu nhãn dự đoán của {m} trong {store.path.name} -> chạy lại {m}.")
            results.pop(m)
    if "B2" not in results:
        results.pop("B5", None)          # B5 dùng p_drop của B2
        results.pop("family", None)
    done6 = results.get("B6", {}).get("keys_done", [])
    for k in list(done6):
        if not store.has("test", "B6", k) or (k != "S2a_tls" and not store.has("val", "B6", k)):
            done6.remove(k)
            results["B6"]["done"] = False
            results.pop("family", None)

    def save():
        tmp = res_path.with_name(res_path.stem + "_tmp.json")
        tmp.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(res_path)

    if "B2" in results:
        audit(f"B2 dùng lại kết quả lần chạy trước: p_grid={p_grid}, "
              f"val={results['B2']['val_scores']}, p_best={results['B2']['p_drop']} "
              "(đã chọn trên VALIDATION).")
    if "B5" in results:
        audit(f"B5 dùng lại kết quả lần chạy trước: lr={results['B5']['lr']:g}, "
              f"val={results['B5']['val_scores']} (đã chọn trên VALIDATION); 1 seed mỗi lr.")

    D = load_data(args)
    if results.get("data", D["fingerprint"]) != D["fingerprint"]:
        sys.exit(f"Dữ liệu khác với lần chạy trước trong {out_dir} "
                 f"({results['data']} != {D['fingerprint']}). Hãy dùng --out_dir mới.")
    results["data"] = D["fingerprint"]
    t0 = time.time()

    # --- GIAI ĐOẠN 1: fit trên train, chọn trên validation -------------------
    if "B1" not in results:
        results["B1"] = run_b1(D, cfg, store)
        save()
    if "B2" not in results:
        results["B2"] = run_b2(D, cfg, p_grid, store)
        save()
    if not results.get("B6", {}).get("done"):
        run_b6(D, cfg, results, store, save)
    if "B5" not in results:
        results["B5"] = run_b5(D, cfg, results["B2"]["p_drop"], store)
        save()
    if "C2" not in results:
        results["C2"] = run_c2(D, cfg)
        save()

    # --- GIAI ĐOẠN 2: chốt (KHÓA) họ baseline, chỉ nhìn validation -----------
    print("\n[Chọn họ baseline] So B2 và B6 trên VALIDATION (trung bình 16 tổ hợp view)")
    family, fam_means, fam_table = select_family_on_validation(D, store)
    if results.get("family", family) != family:
        sys.exit(f"Họ baseline đã khóa trước đó ({results['family']}) khác lần chạy này "
                 f"({family}). Đừng chạy tiếp trên thư mục này.")
    results["family"] = family
    results["family_val_mean"] = fam_means
    results["family_table"] = fam_table.to_dict("records")
    save()
    print(f"    -> KHÓA họ {family} (val trung bình B2={fam_means['B2']:.2f}, "
          f"B6={fam_means['B6']:.2f}); đã ghi vào {res_path.name}")

    # --- GIAI ĐOẠN 3: chỉ bây giờ mới được chạm vào test --------------------
    store.unseal_test(f"đã khóa họ baseline {family} trên validation và ghi ra file")
    print("\n[Quyết định] Chấm điểm test và bootstrap")
    dec = decide(D, store, family, 200 if args.quick else args.n_boot, args.seed)
    report(results, D, out_dir, args, dec, fam_table)
    save()
    print(f"\nXong sau {(time.time() - t0) / 60:.1f} phút. Kết quả: {res_path}, "
          f"{out_dir / 'step0_summary.csv'}, {out_dir / 'step0_family_selection.csv'}")


if __name__ == "__main__":
    main()
