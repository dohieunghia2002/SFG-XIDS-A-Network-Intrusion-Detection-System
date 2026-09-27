#!/usr/bin/env python3
"""
SFG-XIDS - BƯỚC 0 (pilot đi tiếp / dừng), theo đề cương v2 mục 9.

Bốn tiêu chí (quyết định trước khi chạy, không đổi sau khi thấy kết quả test). Mỗi tiêu chí nhận đúng một
trạng thái PASS / FAIL / NOT_EVALUABLE. Thiếu kết quả, giá trị NaN, không đủ mẫu hoặc bước đánh giá
thất bại -> NOT_EVALUABLE, không bao giờ là PASS. Chỉ khi cả 4 tiêu chí đều PASS mới ĐI TIẾP.
  P1  UNSW-NB15, macro-F1 10 lớp: SFG-XIDS >= Flat-MLP - 1 điểm VÀ XGBoost - SFG-XIDS <= 3 điểm
      (trung bình các seed; cần đủ kết quả ở mọi seed).
  P2  UNSW-NB15, so sánh ghép cặp theo seed:
          D_s = F1_taxonomy,s - TB_r(F1_ngẫu_nhiên,r,s)    (TB trên mọi cách chia ngẫu nhiên r ở seed s)
          PASS nếu mean(D) >= -max(SD(D), 1), với SD mẫu (ddof=1): chấp nhận thua trong phạm vi
          nhiễu giữa các seed, hoặc thua không quá 1 điểm macro-F1 (cùng dung sai 1 điểm của P1).
      Cần đủ kết quả taxonomy và mọi cách chia ngẫu nhiên ở mọi seed, và >= 2 seed; nếu không ->
      NOT_EVALUABLE. Cùng seed thì taxonomy và ngẫu nhiên có cùng khởi tạo, thứ tự batch và các lần
      bốc ngẫu nhiên của group dropout (kích thước nhóm như nhau), nên seed là đơn vị ghép cặp.
  P3  UNSW-NB15: hồ sơ đóng góp nhóm khác nhau giữa các lớp (>= 2 nhóm hạng 1 khác nhau) và ổn định
      (>= 70% số lớp giữ nguyên nhóm hạng 1 qua mọi seed). Chỉ xét lớp có >= 30 mẫu test được giải
      thích, tức min(số mẫu test của lớp, n_per_class); với n_per_class = 400 điều này tương đương
      "lớp có >= 30 mẫu test". Không lớp nào đủ 30 mẫu -> NOT_EVALUABLE (khi chạy thật: LỖI
      PROTOCOL/DỮ LIỆU).
  P4  NSL-KDD: tỷ trọng nhóm Content ở R2L/U2R cao hơn ở DoS/Probe (thiết kế của Lee & Stolfo).
      Quần thể của P4: mẫu test có kiểu tấn công xuất hiện trong CẢ KDDTrain+ và KDDTest+. Kiểu chỉ
      có ở KDDTest+ bị loại khỏi P4 nhưng vẫn nằm trong tập test khi tính macro-F1. Báo cáo liệt kê
      kiểu tấn công của train, của test, được P4 đánh giá và bị loại.
  P3/P4: bước giải thích bị lỗi, hoặc kiểm tra φ không đạt (sai số so với vét cạn và sai số
  completeness phải hữu hạn và < 1e-3) -> NOT_EVALUABLE.
  Không đạt -> xem gợi ý in ở cuối.

Kiểm tra dữ liệu và chẩn đoán (KHÔNG phải tiêu chí, không đổi quyết định đi tiếp/dừng):
  - UNSW-NB15: tập nhãn của train và của test phải đúng bằng 10 lớp chuẩn (so khớp tên); sai -> dừng.
  - UNSW-NB15: XGBoost nên gần 51.57 (B1 của step 0, step0_v4/step0_summary.csv); lệch > 2 điểm ->
    in cảnh báo để kiểm tra tiền xử lý/nhãn. Bỏ qua ở --quick.
  - SFG-XIDS với p = 0 (không group dropout), 3 seed, cùng quy trình: chỉ để xem group dropout có
    làm giảm F1 hay không. Không tham gia P1-P4 và không dùng để chọn lại p cho Step 0.

Mô hình (3 seed; macro-F1 trên test chính thức; epoch và σ chọn trên validation):
  XGBoost (cấu hình B1 của step0_unsw.py), Flat-MLP (cấu hình B5, không che view),
  SFG-Attn (kiến trúc v1; trong Step 0 chỉ dùng để so độ chính xác), SFG-XIDS (GAI-Head) theo
  taxonomy + 3 cách chia nhóm ngẫu nhiên.
  Mọi mạng nơ-ron dùng embedding số PLR; σ chọn trên validation bằng Flat-MLP seed 0.
  Ánh xạ kiểu tấn công NSL-KDD -> 5 lớp: NSL_MAP. Các kiểu mà tài liệu xếp khác nhau được xếp:
  snmpgetattack, snmpguess, httptunnel, worm -> R2L.

Ghi chú cài đặt:
  - Group dropout: với xác suất p, thay toàn bộ đặc trưng của một nhóm bằng giá trị của một mẫu
    KHÁC trong batch (rút đều trong n - 1 mẫu còn lại, không bao giờ là chính mẫu đó; nhóm "không
    mang thông tin"), khớp với cách tính Shapley nhóm.
    p = 0.2 là giá trị CỐ ĐỊNH của pilot (hằng số P_SWAP), quyết định trước khi chạy: không có tùy chọn
    dòng lệnh để đổi, cấu hình có p khác 0.2 bị từ chối trước khi huấn luyện, không chọn trên
    validation và không đổi theo kết quả test. Full experiment mới chọn p trong {0.1; 0.2; 0.3} trên
    validation (đề cương mục 4.3). Nhánh chẩn đoán p = 0 là nhánh riêng, không đi qua P_SWAP.
  - Lời giải thích: đóng góp nhóm φ dạng đóng (đề cương mục 4.4) cho logit đã trừ trung bình các
    lớp, tính cho LỚP THẬT. Hồ sơ lớp = tỷ trọng mean|φ| của từng nhóm, dùng cùng tập mẫu test
    (tối đa 400 mẫu/lớp) và cùng tập nền 256 mẫu train cho mọi seed.
  - Script tự kiểm tra: φ dạng đóng phải khớp Shapley vét cạn 16 liên minh.

Phạm vi Step 0: độ chính xác của 4 mô hình, φ cấp nhóm (kèm kiểm tra vét cạn và completeness) và
P1-P4. CHƯA gồm, thuộc full experiment: Expected Gradients cấp đặc trưng (tầng 2, mục 4.4), kiểm
chứng faithfulness của SFG-Attn (thứ hạng γ/A so với Shapley nhóm, thí nghiệm xóa nhóm), ablation
A1-A6, bài toán nhị phân, 5 seed. Step 0 không kiểm chứng toàn bộ phương pháp.

Chạy trên Colab (Runtime -> Change runtime type -> T4 GPU), cùng thư mục với step0_unsw.py:
    from google.colab import drive; drive.mount('/content/drive')
    !pip -q install -U xgboost kagglehub
    %cd /content/drive/MyDrive/rmv_ids
    !python step0_sfg.py --selftest                     # vài giây, không cần dữ liệu
    !python step0_sfg.py --quick --out_dir sfg_quick    # chạy thử cả pipeline trên mẫu nhỏ
    !python step0_sfg.py --out_dir sfg_step0            # chạy thật (ước tính 45-75 phút trên T4)
"SELFTEST: PASS" chỉ có nghĩa các kiểm tra cài đặt đều đạt, KHÔNG có nghĩa P1-P4 đạt.
Bị ngắt giữa chừng: chạy lại đúng lệnh cũ, script bỏ qua các lần chạy đã xong.
Dữ liệu: UNSW-NB15 tải như step0_unsw.py (kagglehub, hoặc --unsw_dir). NSL-KDD tải bằng kagglehub
(hassan06/nslkdd), lỗi thì tải từ GitHub; hoặc tự tải KDDTrain+.txt, KDDTest+.txt rồi dùng --nsl_dir.
Kết quả: <out_dir>/results.json (từng lần chạy), summary.json, profiles_<bộ dữ liệu>.csv.
"""
import argparse
import itertools
import json
import math
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.exit("Thiếu PyTorch. Cài bằng: pip install torch")
try:
    import xgboost as xgb
except ImportError:
    xgb = None

SCRIPT_VERSION = 5

# ----------------------------------------------------------------------------
# Nhóm đặc trưng theo đề cương v2 (mục 3)
# ----------------------------------------------------------------------------
UNSW_GROUPS = {
    "Basic": ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
              "sttl", "dttl", "sload", "dload", "sloss", "dloss"],
    "Content": ["swin", "dwin", "stcpb", "dtcpb", "smean", "dmean", "trans_depth",
                "response_body_len"],
    "Time": ["sjit", "djit", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat"],
    "Additional": ["is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
                   "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                   "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],
}
UNSW_CAT = ["proto", "service", "state"]

NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]
NSL_GROUPS = {
    "Basic": NSL_COLS[0:9],
    "Content": [c for c in NSL_COLS[9:22] if c != "num_outbound_cmds"],   # luôn bằng 0 -> loại
    "Time": NSL_COLS[22:31],
    "Host": NSL_COLS[31:41],
}
NSL_CAT = ["protocol_type", "service", "flag"]
NSL_ROWS = (125973, 22544)
NSL_KAGGLE = "hassan06/nslkdd"
NSL_URLS = {"KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
            "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"}
NSL_MAP = {"normal": "Normal"}
for _cat, _names in {
        "DoS": "back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm",
        "Probe": "ipsweep nmap portsweep satan mscan saint",
        "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster named sendmail "
               "snmpgetattack snmpguess xlock xsnoop httptunnel worm",
        "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm"}.items():
    NSL_MAP.update({n: _cat for n in _names.split()})

SEEDS, RAND_SEEDS, SIGMAS = [0, 1, 2], [100, 101, 102], [0.01, 0.1, 1.0]
CFG = dict(lr=1e-3, bs=1024, max_ep=40, patience=6, d_emb=16, n_freq=32, d=64, n_bg=256,
           n_per_class=400, n_exact=16, n_bg_exact=6, n_estimators=2000, es_rounds=50)
QUICK = dict(max_ep=3, patience=2, n_bg=32, n_per_class=50, n_exact=4, n_bg_exact=3,
             n_estimators=60, es_rounds=10, seeds=[0, 1], rand_seeds=[100], sigmas=[0.1])
UNSW_CLASSES = {"Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Analysis",
                "Backdoor", "Shellcode", "Worms"}        # 10 lớp chuẩn của bộ training/testing-set
PHI_TOL = 1e-3          # kiểm tra φ (vét cạn, completeness); không đạt -> P3/P4 NOT_EVALUABLE
P2_MARGIN = 1.0        # dung sai thực tế của P2 (điểm macro-F1), cùng mức 1 điểm của P1
P_SWAP = 0.2           # group dropout của Step 0: CỐ ĐỊNH, quyết định trước pilot, không có tùy chọn dòng lệnh
XGB_REF, XGB_REF_TOL = 51.57, 2.0    # CHỈ chẩn đoán: B1 (S0) trong step0_v4/step0_summary.csv
NSL_WATCH = ["snmpguess", "smurf", "httptunnel", "worm", "buffer_overflow", "loadmodule", "perl",
             "rootkit", "ps", "sqlattack", "xterm"]     # in riêng cách xử lý trong báo cáo quần thể P4


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def check_unsw_classes(y_tr, y_te):
    """Tập nhãn attack_cat của train và của test phải đúng bằng 10 lớp chuẩn (so khớp tên chính xác)."""
    for name, y in (("train", y_tr), ("test", y_te)):
        got = set(pd.Series(list(y)).astype(str))
        miss, extra = sorted(UNSW_CLASSES - got), sorted(got - UNSW_CLASSES)
        if miss or extra:
            sys.exit(f"UNSW-NB15 [{name}]: tập lớp không đúng 10 lớp chuẩn. Thiếu: {miss} | "
                     f"Thừa hoặc sai tên: {extra}")


def load_unsw(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import step0_unsw as s0
    except ImportError:
        sys.exit("Cần file step0_unsw.py cùng thư mục (dùng lại phần tải/đọc UNSW-NB15).")
    tr_path, te_path = s0.find_csvs(s0.get_data_dir(args.unsw_dir))
    tr, te = s0.read_split(tr_path), s0.read_split(te_path)
    s0.check_rows(len(tr), len(te), args.skip_row_check)
    check_unsw_classes(tr["attack_cat"], te["attack_cat"])      # sau khi chuẩn hóa nhãn
    print(f"UNSW-NB15: train {len(tr):,} | test {len(te):,} ({tr_path.parent})")
    return tr, te, "attack_cat"


def find_nsl(d):
    tr = te = None
    for p in Path(d).rglob("*"):
        n = p.name.lower()
        if n in ("kddtrain+.txt", "kddtrain+.csv"):
            tr = p
        elif n in ("kddtest+.txt", "kddtest+.csv"):
            te = p
    return tr, te


def load_nsl(args, out_dir):
    tr = te = None
    if args.nsl_dir:
        tr, te = find_nsl(args.nsl_dir)
        if not (tr and te):
            sys.exit(f"Không thấy KDDTrain+.txt / KDDTest+.txt trong {args.nsl_dir}")
    else:
        try:
            import kagglehub
            tr, te = find_nsl(kagglehub.dataset_download(NSL_KAGGLE))
        except Exception as e:
            print(f"kagglehub không tải được NSL-KDD ({e}); thử GitHub.")
        if not (tr and te):
            cache = out_dir / "nsl_kdd"
            cache.mkdir(parents=True, exist_ok=True)
            try:
                for name, url in NSL_URLS.items():
                    if not (cache / name).exists():
                        urllib.request.urlretrieve(url, cache / name)
            except Exception as e:
                sys.exit(f"Không tải được NSL-KDD ({e}). Hãy tải KDDTrain+.txt, KDDTest+.txt "
                         "rồi chạy lại với --nsl_dir <thư mục>.")
            tr, te = cache / "KDDTrain+.txt", cache / "KDDTest+.txt"
    dfs = []
    for p in (tr, te):
        df = pd.read_csv(p, header=None, low_memory=False)
        if str(df.iat[0, 0]).strip().lower() == "duration":        # file có dòng tiêu đề
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] not in (42, 43):
            sys.exit(f"{p.name}: cần 42-43 cột, file có {df.shape[1]} cột.")
        df.columns = NSL_COLS + ["label", "difficulty"][: df.shape[1] - 41]
        lab = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
        unknown = sorted(set(lab) - set(NSL_MAP))
        if unknown:
            sys.exit(f"{p.name}: nhãn chưa có trong bảng ánh xạ 5 lớp: {unknown}")
        df["y_name"] = lab.map(NSL_MAP)
        df["attack_type"] = lab             # tên kiểu tấn công gốc, dùng cho quần thể P4
        dfs.append(df)
    n = (len(dfs[0]), len(dfs[1]))
    if n != NSL_ROWS and not args.skip_row_check:
        sys.exit(f"Số dòng NSL-KDD {n} khác bản chuẩn {NSL_ROWS}. Thêm --skip_row_check nếu cố ý.")
    print(f"NSL-KDD: train {n[0]:,} | test {n[1]:,} ({tr.parent})")
    return dfs[0], dfs[1], "y_name"


def nsl_population(tr_types, te_types):
    """Quần thể P4: kiểu tấn công có trong CẢ KDDTrain+ và KDDTest+ ('normal' không phải kiểu tấn công)."""
    trc, tec = pd.Series(list(tr_types)).value_counts(), pd.Series(list(te_types)).value_counts()
    tr_set, te_set = set(trc.index) - {"normal"}, set(tec.index) - {"normal"}
    return {"train": sorted(tr_set), "test": sorted(te_set), "evaluated": sorted(te_set & tr_set),
            "excluded_unseen": sorted(te_set - tr_set), "train_only": sorted(tr_set - te_set),
            "map": {t: NSL_MAP[t] for t in sorted(tr_set | te_set)},
            "train_counts": {t: int(trc[t]) for t in sorted(tr_set)},
            "test_counts": {t: int(tec[t]) for t in sorted(te_set)}}


def prepare(tr, te, label_col, groups, cat_cols, quick=False, split_seed=42, keep_col=None):
    """Chia validation 10% (phân tầng, cố định cho mọi seed); mọi tham số tiền xử lý fit trên train."""
    feats = [c for g in groups.values() for c in g]
    miss = [c for c in feats if c not in tr.columns or c not in te.columns]
    if miss:
        sys.exit(f"Thiếu cột: {miss}")
    classes = sorted(tr[label_col].unique().tolist())
    unseen = sorted(set(te[label_col]) - set(classes))
    if unseen:
        sys.exit(f"Test có lớp không có trong train: {unseen}")
    cid = {c: i for i, c in enumerate(classes)}
    y_all = tr[label_col].map(cid).to_numpy(np.int64, copy=True)
    y_te = te[label_col].map(cid).to_numpy(np.int64, copy=True)
    if quick:
        if len(tr) > 20000:
            k, _ = train_test_split(np.arange(len(tr)), train_size=20000, stratify=y_all,
                                    random_state=split_seed)
            tr, y_all = tr.iloc[k].reset_index(drop=True), y_all[k]
        if len(te) > 10000:
            k, _ = train_test_split(np.arange(len(te)), train_size=10000, stratify=y_te,
                                    random_state=split_seed)
            te, y_te = te.iloc[k].reset_index(drop=True), y_te[k]
    te_extra = te[keep_col].astype(str).to_numpy() if keep_col else None   # khớp dòng test cuối
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.1, stratify=y_all,
                                  random_state=split_seed)
    va, tr = tr.iloc[i_va].reset_index(drop=True), tr.iloc[i_tr].reset_index(drop=True)
    y = {"tr": y_all[i_tr], "va": y_all[i_va], "te": y_te}
    frames = {"tr": tr, "va": va, "te": te}

    nums = [c for c in feats if c not in cat_cols]
    cats = [c for c in feats if c in cat_cols]
    vocab = {c: sorted(tr[c].fillna("__nan__").astype(str).str.strip().unique().tolist())
             + ["__unk__"] for c in cats}

    def enc_cat(df):
        cols = []
        for c in cats:
            idx = {v: i for i, v in enumerate(vocab[c])}
            s = df[c].fillna("__nan__").astype(str).str.strip().map(idx)
            cols.append(s.fillna(len(vocab[c]) - 1).to_numpy(np.int64))
        return np.stack(cols, 1) if cols else np.zeros((len(df), 0), np.int64)

    def raw_num(df):
        A = df[nums].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64, copy=True)
        A[~np.isfinite(A)] = 0.0
        return A

    R = {k: raw_num(d) for k, d in frames.items()}
    C = {k: enc_cat(d) for k, d in frames.items()}
    nonneg = R["tr"].min(0) >= 0

    def lg(A):
        A = A.copy()
        A[:, nonneg] = np.log1p(np.clip(A[:, nonneg], 0, None))
        return A

    L = lg(R["tr"])
    mu, sd = L.mean(0), L.std(0)
    sd[sd < 1e-6] = 1.0
    Z = {k: np.clip((lg(R[k]) - mu) / sd, -10, 10).astype(np.float32) for k in R}

    def xgb_frame(k):   # XGBoost: số giữ nguyên, phân loại dạng category (từ điển của train)
        d = {c: R[k][:, j].astype(np.float32) for j, c in enumerate(nums)}
        for j, c in enumerate(cats):
            d[c] = pd.Categorical.from_codes(C[k][:, j], categories=vocab[c])
        return pd.DataFrame(d)[feats]

    freq = np.bincount(y["tr"], minlength=len(classes)).astype(float)
    w = 1.0 / np.sqrt(np.maximum(freq, 1))
    w = w / w[y["tr"]].mean()          # cùng cách đặt trọng số lớp với step0_unsw.py
    return dict(classes=classes, n_cls=len(classes), groups=groups, feats=feats, nums=nums,
                cats=cats, cards=[len(vocab[c]) for c in cats], Z=Z, C=C,
                F={k: xgb_frame(k) for k in R}, y=y, cw=w.astype(np.float32), te_extra=te_extra)


def random_grouping(feats, sizes, seed):
    perm = np.random.RandomState(seed).permutation(feats).tolist()
    out, s = {}, 0
    for g, k in enumerate(sizes):
        out[f"R{g + 1}"] = perm[s:s + k]
        s += k
    return out


def group_index(D, grouping):
    return [(torch.tensor([D["nums"].index(c) for c in cols if c in D["nums"]], dtype=torch.long),
             torch.tensor([D["cats"].index(c) for c in cols if c in D["cats"]], dtype=torch.long))
            for cols in grouping.values()]


# ----------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------
class FeatEmb(nn.Module):
    """PLR (Periodic -> Linear -> ReLU) cho cột số; Embedding cho cột phân loại."""

    def __init__(self, n_num, cards, d_emb, n_freq, sigma):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.w = nn.Parameter(torch.randn(n_num, 2 * n_freq, d_emb) / math.sqrt(2 * n_freq))
        self.b = nn.Parameter(torch.zeros(n_num, d_emb))
        self.cat = nn.ModuleList([nn.Embedding(k, d_emb) for k in cards])
        self.d_emb = d_emb

    def forward(self, xn, xc):
        v = 2 * math.pi * self.freq[None] * xn[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)
        ne = torch.relu(torch.einsum("nfk,fkd->nfd", v, self.w) + self.b)
        if len(self.cat):
            ce = torch.stack([e(xc[:, j]) for j, e in enumerate(self.cat)], 1)
        else:
            ce = xn.new_zeros(xn.shape[0], 0, self.d_emb)
        return ne, ce


class FlatMLP(nn.Module):
    """Cấu hình như B5 (256-256-128, BatchNorm, ReLU, Dropout 0.1), đầu vào PLR."""

    def __init__(self, n_num, cards, n_cls, cfg, sigma):
        super().__init__()
        self.emb = FeatEmb(n_num, cards, cfg["d_emb"], cfg["n_freq"], sigma)
        layers, prev = [], (n_num + len(cards)) * cfg["d_emb"]
        for h in (256, 256, 128):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, n_cls))

    def forward(self, xn, xc):
        ne, ce = self.emb(xn, xc)
        return self.net(torch.cat([ne.flatten(1), ce.flatten(1)], 1))


class GroupEnc(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.l1, self.l2, self.ln = nn.Linear(d_in, d), nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, s):
        h = F.leaky_relu(self.l1(s))
        return self.ln(h + F.leaky_relu(self.l2(h)))


class GroupedBase(nn.Module):
    """Embedding từng đặc trưng + encoder riêng từng nhóm: e_i chỉ phụ thuộc x_i."""

    def __init__(self, n_num, cards, gidx, cfg, sigma):
        super().__init__()
        self.emb = FeatEmb(n_num, cards, cfg["d_emb"], cfg["n_freq"], sigma)
        self.G = len(gidx)
        for g, (ni, ci) in enumerate(gidx):
            self.register_buffer(f"ni{g}", ni.clone())
            self.register_buffer(f"ci{g}", ci.clone())
        self.enc = nn.ModuleList([GroupEnc((len(ni) + len(ci)) * cfg["d_emb"], cfg["d"])
                                  for ni, ci in gidx])

    def groups(self):
        return [(getattr(self, f"ni{g}"), getattr(self, f"ci{g}")) for g in range(self.G)]

    def encode(self, xn, xc):
        ne, ce = self.emb(xn, xc)
        n, out = xn.shape[0], []
        for g, (ni, ci) in enumerate(self.groups()):
            parts = []
            if ni.numel():
                parts.append(ne.index_select(1, ni).reshape(n, -1))
            if ci.numel():
                parts.append(ce.index_select(1, ci).reshape(n, -1))
            out.append(self.enc[g](torch.cat(parts, 1)))
        return out


class SFGAttn(GroupedBase):
    """Kiến trúc v1: cross-group attention + adaptive group pooling (γ)."""

    def __init__(self, n_num, cards, gidx, n_cls, cfg, sigma):
        super().__init__(n_num, cards, gidx, cfg, sigma)
        d = cfg["d"]
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1))
        self.cls = nn.Sequential(nn.Linear(d, 64), nn.LeakyReLU(), nn.Linear(64, n_cls))

    def forward(self, xn, xc):
        E = torch.stack(self.encode(xn, xc), 1)                               # n x G x d
        A = torch.softmax(self.q(E) @ self.k(E).transpose(1, 2) / math.sqrt(E.shape[-1]), -1)
        Eo = self.ln(E + A @ self.v(E))
        gamma = torch.softmax(self.pool(Eo).squeeze(-1), 1)
        return self.cls((gamma[..., None] * Eo).sum(1))


class SFGXIDS(GroupedBase):
    """GAI-Head: z = b + Σ u_i(e_i) + Σ_{i<j} v_ij(e_i, e_j)."""

    def __init__(self, n_num, cards, gidx, n_cls, cfg, sigma):
        super().__init__(n_num, cards, gidx, cfg, sigma)
        d = cfg["d"]
        self.u = nn.ModuleList([nn.Linear(d, n_cls) for _ in range(self.G)])
        self.pairs = list(itertools.combinations(range(self.G), 2))
        self.v = nn.ModuleList([nn.Sequential(nn.Linear(3 * d, 64), nn.LeakyReLU(),
                                              nn.Linear(64, n_cls)) for _ in self.pairs])
        self.bias = nn.Parameter(torch.zeros(n_cls))

    def pair(self, k, ei, ej):
        return self.v[k](torch.cat([ei, ej, ei * ej], -1))

    def head(self, es):
        z = self.bias + sum(self.u[i](es[i]) for i in range(self.G))
        for k, (i, j) in enumerate(self.pairs):
            z = z + self.pair(k, es[i], es[j])
        return z

    def forward(self, xn, xc):
        return self.head(self.encode(xn, xc))


def group_swap(xn, xc, groups, p, gen):
    """Group dropout: mỗi nhóm, với xác suất p, lấy giá trị của một mẫu KHÁC trong batch.
    Nguồn rút đều trong n - 1 mẫu còn lại: src = (dòng + k) mod n với k ~ U{1, ..., n-1}, nên src != dòng.
    Mặt nạ Bernoulli(p) và thứ tự rút số ngẫu nhiên giữ nguyên như bản trước."""
    n = xn.shape[0]
    if n < 2:
        raise ValueError("group_swap cần batch có ít nhất 2 mẫu (không có mẫu khác để lấy)")
    xn2, xc2 = xn.clone(), xc.clone()
    for ni, ci in groups:
        rows = (torch.rand(n, device=xn.device, generator=gen) < p).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        src = (rows + torch.randint(1, n, (rows.numel(),), device=xn.device, generator=gen)) % n
        if ni.numel():
            xn2[rows[:, None], ni[None, :]] = xn[src[:, None], ni[None, :]]
        if ci.numel():
            xc2[rows[:, None], ci[None, :]] = xc[src[:, None], ci[None, :]]
    return xn2, xc2


# ----------------------------------------------------------------------------
# Giải thích cấp nhóm (đề cương v2, mục 4.4)
# ----------------------------------------------------------------------------
def nan_max(a, b):
    """max giữ NaN: max() của Python bỏ qua NaN tùy thứ tự đối số, ví dụ max(1e-7, nan) = 1e-7."""
    return float("nan") if math.isnan(a) or math.isnan(b) else max(a, b)


@torch.no_grad()
def group_phi(model, xn, xc, bn, bc, cls, chunk=128):
    """φ dạng đóng cho logit (đã trừ trung bình các lớp) của lớp cls; nền độc lập theo nhóm.
    Trả về (φ: n x G, sai số lớn nhất của đẳng thức Σ_i φ_i = z - E[z])."""
    model.eval()
    Eb, B, G = model.encode(bn, bc), bn.shape[0], model.G
    ubar = [model.u[i](Eb[i]).mean(0) for i in range(G)]
    vbar = [model.pair(k, Eb[i].repeat_interleave(B, 0), Eb[j].repeat(B, 1)).mean(0)
            for k, (i, j) in enumerate(model.pairs)]
    base = model.bias + sum(ubar) + sum(vbar)
    out, err = [], 0.0
    for s in range(0, xn.shape[0], chunk):
        E = model.encode(xn[s:s + chunk], xc[s:s + chunk])
        n = E[0].shape[0]
        phi = [model.u[i](E[i]) - ubar[i] for i in range(G)]
        for k, (i, j) in enumerate(model.pairs):
            V = model.pair(k, E[i], E[j])
            Ai = model.pair(k, E[i].repeat_interleave(B, 0), Eb[j].repeat(n, 1)).view(n, B, -1).mean(1)
            Aj = model.pair(k, Eb[i].repeat(n, 1), E[j].repeat_interleave(B, 0)).view(n, B, -1).mean(1)
            g = V - Ai - Aj + vbar[k]                      # tương tác thuần
            phi[i] = phi[i] + (Ai - vbar[k]) + 0.5 * g
            phi[j] = phi[j] + (Aj - vbar[k]) + 0.5 * g
        P = torch.stack(phi, 1)                            # n x G x C
        err = nan_max(err, (P.sum(1) - (model.head(E) - base)).abs().max().item())
        P = P - P.mean(2, keepdim=True)
        out.append(P[torch.arange(n, device=P.device), :, cls[s:s + chunk]])
    return torch.cat(out), err


@torch.no_grad()
def group_phi_brute(model, xn, xc, bn, bc, cls):
    """Shapley nhóm vét cạn 2^G liên minh; liệt kê đủ tổ hợp nền của các nhóm bị che."""
    model.eval()
    E, Eb = model.encode(xn, xc), model.encode(bn, bc)
    G, n, B = model.G, xn.shape[0], bn.shape[0]
    val = {}
    for S in range(1 << G):
        masked = [g for g in range(G) if not (S >> g) & 1]
        combos = torch.tensor(list(itertools.product(range(B), repeat=len(masked))) if masked
                              else [[]], dtype=torch.long, device=xn.device)
        m = combos.shape[0]
        es = [Eb[g][combos[:, masked.index(g)]].repeat(n, 1) if g in masked
              else E[g].repeat_interleave(m, 0) for g in range(G)]
        val[S] = model.head(es).view(n, m, -1).mean(1)
    phi = torch.zeros(n, G, val[0].shape[1], dtype=val[0].dtype, device=xn.device)
    for g in range(G):
        for S in range(1 << G):
            if (S >> g) & 1:
                continue
            k = bin(S).count("1")
            w = math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G)
            phi[:, g] += w * (val[S | (1 << g)] - val[S])
    phi = phi - phi.mean(2, keepdim=True)
    return phi[torch.arange(n, device=phi.device), :, cls]


def class_profiles(model, D, T, cfg, dev):
    rng = np.random.RandomState(1234)           # cùng mẫu nền và mẫu test cho mọi seed
    bi = torch.as_tensor(rng.choice(len(D["y"]["tr"]), min(cfg["n_bg"], len(D["y"]["tr"])),
                                    replace=False), device=dev)
    bn, bc = T["xn_tr"][bi], T["xc_tr"][bi]
    mask = D.get("explain_mask")        # NSL-KDD: chỉ kiểu tấn công có trong cả train và test (P4)
    prof, counts, err = [], [], 0.0
    for c in range(D["n_cls"]):
        idx = np.where(D["y"]["te"] == c if mask is None else (D["y"]["te"] == c) & mask)[0]
        if len(idx) > cfg["n_per_class"]:
            idx = rng.choice(idx, cfg["n_per_class"], replace=False)
        if len(idx) == 0:
            prof.append([float("nan")] * model.G)
            counts.append(0)
            continue
        ti = torch.as_tensor(idx, device=dev)
        cls = torch.full((len(idx),), c, dtype=torch.long, device=dev)
        phi, e = group_phi(model, T["xn_te"][ti], T["xc_te"][ti], bn, bc, cls)
        a = phi.abs().mean(0).cpu().numpy()
        tot = float(a.sum())
        # Tổng bằng 0 hoặc không hữu hạn -> hồ sơ không xác định (NaN); không che bằng epsilon.
        prof.append((a / tot).tolist() if math.isfinite(tot) and tot > 0 else [float("nan")] * model.G)
        counts.append(int(len(idx)))
        err = nan_max(err, e)
    return prof, counts, err


def exactness(model, D, T, cfg, dev):
    rng = np.random.RandomState(99)
    bi = torch.as_tensor(rng.choice(len(D["y"]["tr"]), cfg["n_bg_exact"], replace=False), device=dev)
    ti = torch.as_tensor(rng.choice(len(D["y"]["te"]), cfg["n_exact"], replace=False), device=dev)
    xn, xc, bn, bc = T["xn_te"][ti], T["xc_te"][ti], T["xn_tr"][bi], T["xc_tr"][bi]
    with torch.no_grad():
        cls = model(xn, xc).argmax(1)
    a, _ = group_phi(model, xn, xc, bn, bc, cls)
    return (a - group_phi_brute(model, xn, xc, bn, bc, cls)).abs().max().item()


# ----------------------------------------------------------------------------
# Huấn luyện & đánh giá
# ----------------------------------------------------------------------------
def macro_f1(y, p, n_cls):
    return float(100 * f1_score(y, p, average="macro", labels=list(range(n_cls)), zero_division=0))


@torch.no_grad()
def predict(model, xn, xc, bs=8192):
    model.eval()
    return torch.cat([model(xn[s:s + bs], xc[s:s + bs]).argmax(1)
                      for s in range(0, xn.shape[0], bs)]).cpu().numpy()


def train_nn(model, T, D, dev, seed, cfg, p_swap=0.0):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(weight=T["cw"])
    groups = model.groups() if p_swap > 0 else None
    ntr, best, bad, ep_best, state = T["y_tr"].shape[0], -1.0, 0, 0, None
    for ep in range(1, cfg["max_ep"] + 1):
        model.train()
        perm = torch.randperm(ntr, device=dev, generator=gen)
        for s in range(0, ntr, cfg["bs"]):
            idx = perm[s:s + cfg["bs"]]
            if idx.numel() < 2:
                continue
            xn, xc = T["xn_tr"][idx], T["xc_tr"][idx]
            if groups:
                xn, xc = group_swap(xn, xc, groups, p_swap, gen)
            loss = lossf(model(xn, xc), T["y_tr"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        f1 = macro_f1(D["y"]["va"], predict(model, T["xn_va"], T["xc_va"]), D["n_cls"])
        if f1 > best + 1e-6:
            best, bad, ep_best = f1, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(state)
    return best, ep_best


def run_dataset(ds, D, R, cfg, dev, save):
    if cfg.get("p_swap") != P_SWAP:       # Step 0: p_swap cố định, không chạy với giá trị khác
        sys.exit(f"Step 0 chỉ chạy với p_swap = {P_SWAP} (quyết định trước pilot); nhận {cfg.get('p_swap')}.")
    if xgb is None:
        sys.exit("Thiếu xgboost. Cài bằng: pip install -U xgboost")
    T = {f"{a}_{k}": torch.as_tensor(D[b][k], device=dev)
         for a, b in (("xn", "Z"), ("xc", "C"), ("y", "y")) for k in ("tr", "va", "te")}
    T["cw"] = torch.as_tensor(D["cw"], device=dev)
    n_num, cards, C, seeds = len(D["nums"]), D["cards"], D["n_cls"], cfg["seeds"]
    sizes = [len(v) for v in D["groups"].values()]
    groupings = {"tax": D["groups"]}
    groupings.update({f"rand{r}": random_grouping(D["feats"], sizes, r) for r in cfg["rand_seeds"]})
    R[f"{ds}|groupings"] = groupings

    def log(key, r):
        extra = f"{r['epochs']} epoch" if "epochs" in r else f"{r['trees']} cây"
        print(f"  [{ds}] {key}: val {r['val_f1']:.2f} | test {r['test_f1']:.2f} | {extra} | "
              f"{r['seconds']}s", flush=True)

    def nn_run(key, make, seed, sigma, swap=False, explain=False):
        fk = f"{ds}|{key}"
        if fk in R:
            return
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make().to(dev)
        p_eff = cfg["p_swap"] if swap else 0.0          # nhánh chính 0.2; nhánh chẩn đoán sfg0 và các mô hình khác 0
        best, ep = train_nn(model, T, D, dev, seed, cfg, p_eff)
        pt = predict(model, T["xn_te"], T["xc_te"])
        r = dict(val_f1=best, test_f1=macro_f1(D["y"]["te"], pt, C),
                 test_acc=float(100 * accuracy_score(D["y"]["te"], pt)), epochs=ep, sigma=sigma,
                 p_swap=p_eff)
        if explain:
            try:
                r["profile"], r["counts"], r["sum_err"] = class_profiles(model, D, T, cfg, dev)
                if seed == seeds[0]:
                    r["exact_err"] = exactness(model, D, T, cfg, dev)
                r["profile_population"] = D.get("profile_population", "all_test")
            except Exception as e:   # lỗi giải thích -> P3/P4 = NOT_EVALUABLE, không bịa kết quả
                for k in ("profile", "counts", "sum_err", "exact_err"):
                    r.pop(k, None)
                r["explain_error"] = f"{type(e).__name__}: {e}"
                print(f"  [{ds}] LỖI khi tính lời giải thích ({key}): {r['explain_error']}", flush=True)
        r["seconds"] = round(time.time() - t0, 1)
        R[fk] = r
        save()
        log(key, r)

    # 1) Chọn σ của PLR trên validation (Flat-MLP, seed đầu tiên)
    for sg in cfg["sigmas"]:
        nn_run(f"flat|sigma{sg}|{seeds[0]}", lambda: FlatMLP(n_num, cards, C, cfg, sg), seeds[0], sg)
    sigma = max(cfg["sigmas"], key=lambda s: R[f"{ds}|flat|sigma{s}|{seeds[0]}"]["val_f1"])
    R[f"{ds}|sigma"] = sigma
    print(f"  [{ds}] σ của PLR chọn trên validation: {sigma}")
    R.setdefault(f"{ds}|flat|tax|{seeds[0]}", dict(R[f"{ds}|flat|sigma{sigma}|{seeds[0]}"]))

    # 2) Các mô hình chính, theo taxonomy
    g_tax = group_index(D, groupings["tax"])
    for seed in seeds:
        key = f"{ds}|xgb|tax|{seed}"
        if key not in R:
            t0 = time.time()
            m = xgb.XGBClassifier(
                n_estimators=cfg["n_estimators"], learning_rate=0.1, max_depth=8, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", device=dev.type, enable_categorical=True,
                max_cat_to_onehot=1, objective="multi:softprob", eval_metric="mlogloss",
                early_stopping_rounds=cfg["es_rounds"], random_state=seed)
            m.fit(D["F"]["tr"], D["y"]["tr"], sample_weight=D["cw"][D["y"]["tr"]],
                  eval_set=[(D["F"]["va"], D["y"]["va"])],
                  sample_weight_eval_set=[D["cw"][D["y"]["va"]]], verbose=False)
            pv, pt = m.predict(D["F"]["va"]), m.predict(D["F"]["te"])
            R[key] = dict(val_f1=macro_f1(D["y"]["va"], pv, C), test_f1=macro_f1(D["y"]["te"], pt, C),
                          test_acc=float(100 * accuracy_score(D["y"]["te"], pt)),
                          trees=int(m.best_iteration) + 1, seconds=round(time.time() - t0, 1))
            save()
            log(f"xgb|tax|{seed}", R[key])
        nn_run(f"flat|tax|{seed}", lambda: FlatMLP(n_num, cards, C, cfg, sigma), seed, sigma)
        nn_run(f"attn|tax|{seed}", lambda: SFGAttn(n_num, cards, g_tax, C, cfg, sigma), seed, sigma)
        nn_run(f"sfg|tax|{seed}", lambda: SFGXIDS(n_num, cards, g_tax, C, cfg, sigma), seed, sigma,
               swap=True, explain=True)

    # 3) SFG-XIDS với các cách chia ngẫu nhiên (cùng kích thước nhóm)
    for r in cfg["rand_seeds"]:
        g_r = group_index(D, groupings[f"rand{r}"])
        for seed in seeds:
            nn_run(f"sfg|rand{r}|{seed}", lambda: SFGXIDS(n_num, cards, g_r, C, cfg, sigma), seed,
                   sigma, swap=True)

    # 4) Chẩn đoán, KHÔNG tham gia P1-P4: SFG-XIDS theo taxonomy, không group dropout (p = 0)
    for seed in seeds:
        nn_run(f"sfg0|tax|{seed}", lambda: SFGXIDS(n_num, cards, g_tax, C, cfg, sigma), seed, sigma)


# ----------------------------------------------------------------------------
# Tổng hợp & kết luận
# ----------------------------------------------------------------------------
STATUSES = ("PASS", "FAIL", "NOT_EVALUABLE")


def _vals(R, ds, model, grouping, seeds):
    return [R[f"{ds}|{model}|{grouping}|{s}"]["test_f1"] for s in seeds
            if f"{ds}|{model}|{grouping}|{s}" in R]


def _ms(x):
    """(TB, SD mẫu ddof=1). SD không xác định (NaN) khi có < 2 giá trị - không thay bằng 0."""
    return (float(np.mean(x)) if x else float("nan"),
            float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"))


def _fmt(v, spec):
    return format(v, spec) if math.isfinite(v) else "n/a".rjust(len(format(0.0, spec)))


def xgb_reference(vals, context):
    """Chẩn đoán, KHÔNG phải tiêu chí: XGBoost UNSW-NB15 nên gần B1 của step 0."""
    out = {"ref": XGB_REF, "tol": XGB_REF_TOL, "value": None, "flag": None}
    if context != "real":
        return dict(out, note=f"bỏ qua ở chế độ {context} (dữ liệu nhỏ)")
    if not vals or not all(math.isfinite(v) for v in vals):
        return dict(out, flag=True, note="thiếu kết quả XGBoost -> kiểm tra lại")
    m = float(np.mean(vals))
    flag = abs(m - XGB_REF) > XGB_REF_TOL
    return dict(out, value=m, flag=flag,
                note=f"XGBoost {m:.2f} vs tham chiếu step 0 {XGB_REF} (lệch {m - XGB_REF:+.2f}, ngưỡng "
                     f"±{XGB_REF_TOL})" + (" -> LỆCH LỚN: kiểm tra tiền xử lý/nhãn" if flag else " -> OK"))


def nsl_population_lines(pop):
    def by(types, counts):
        return "; ".join(f"{c}: " + (", ".join(f"{t}({counts.get(t, 0)})" for t in types
                                               if pop["map"][t] == c) or "-")
                         for c in ("DoS", "Probe", "R2L", "U2R"))

    def status(t):
        if t in pop["evaluated"]:
            return "đánh giá trong P4"
        if t in pop["excluded_unseen"]:
            return "loại khỏi P4: không có trong KDDTrain+"
        if t in pop["train_only"]:
            return "chỉ có ở KDDTrain+, không có trong test"
        return "không có trong dữ liệu"

    trc, tec = pop["train_counts"], pop["test_counts"]
    return [f"KDDTrain+ ({len(pop['train'])} kiểu, số dòng train) - {by(pop['train'], trc)}",
            f"KDDTest+ ({len(pop['test'])} kiểu, số dòng test) - {by(pop['test'], tec)}",
            f"P4 đánh giá ({len(pop['evaluated'])} kiểu có ở cả hai) - {by(pop['evaluated'], tec)}",
            f"Loại khỏi P4 ({len(pop['excluded_unseen'])} kiểu chỉ có ở test) - "
            f"{by(pop['excluded_unseen'], tec)}",
            "Kiểm tra riêng: " + "; ".join(f"{t} -> {NSL_MAP[t]} ({status(t)})" for t in NSL_WATCH)]


def evaluate_criteria(R, cfg, datasets, context):
    """P1-P4 -> {Pk: (trạng thái, chi tiết)}, trạng thái thuộc STATUSES.
    Thiếu kết quả, NaN, không đủ mẫu hoặc bước đánh giá thất bại -> NOT_EVALUABLE, không bao giờ là
    PASS. context: "real" | "quick" | "selftest"; khi "real", NOT_EVALUABLE do dữ liệu/kết quả được
    báo là LỖI PROTOCOL/DỮ LIỆU. Nhánh chẩn đoán p = 0 (khóa "sfg0") không được đọc ở đây."""
    seeds, nan = cfg["seeds"], float("nan")

    def ne(reason):
        return "NOT_EVALUABLE", ("LỖI PROTOCOL/DỮ LIỆU - " if context == "real" else "") + reason

    def finite(x):
        return all(math.isfinite(v) for v in x)

    def explain_problem(ds):
        """Lý do bước giải thích của ds không dùng được cho P3/P4; None nếu dùng được."""
        recs = [R.get(f"{ds}|sfg|tax|{s}") for s in seeds]
        if any(r is None for r in recs):
            return "thiếu kết quả SFG-XIDS (taxonomy) ở một số seed"
        errs = [r["explain_error"] for r in recs if "explain_error" in r]
        if errs:
            return f"bước giải thích bị lỗi: {errs[0]}"
        if not all("profile" in r and "counts" in r for r in recs):
            return "thiếu hồ sơ đóng góp nhóm"
        errv = [recs[0].get("exact_err", nan)] + [r.get("sum_err", nan) for r in recs]
        if not all(math.isfinite(v) and v < PHI_TOL for v in errv):
            return (f"kiểm tra φ không đạt: sai số lớn nhất {float(np.max(errv)):.1e} "
                    f"(cần hữu hạn và < {PHI_TOL:g})")
        return None

    def profiles(ds):
        recs = [R[f"{ds}|sfg|tax|{s}"] for s in seeds]
        return [np.array(r["profile"], dtype=float) for r in recs], np.array(recs[0]["counts"])

    crit = {}
    if "unsw" in datasets:
        sfg, flat, xg = (_vals(R, "unsw", m, "tax", seeds) for m in ("sfg", "flat", "xgb"))
        if not len(sfg) == len(flat) == len(xg) == len(seeds) or not finite(sfg + flat + xg):
            crit["P1"] = ne("thiếu kết quả ở một số seed hoặc có giá trị không hữu hạn")
        else:
            m_s, m_f, m_x = np.mean(sfg), np.mean(flat), np.mean(xg)
            crit["P1"] = ("PASS" if m_s >= m_f - 1 and m_x - m_s <= 3 else "FAIL",
                          f"SFG-XIDS {m_s:.2f} vs Flat-MLP {m_f:.2f} (cần >= Flat - 1); "
                          f"XGBoost - SFG-XIDS = {m_x - m_s:.2f} (cần <= 3)")
        rgs = [f"rand{r}" for r in cfg["rand_seeds"]]
        have = all(f"unsw|sfg|{g}|{s}" in R for g in ["tax"] + rgs for s in seeds)
        if len(seeds) < 2 or not rgs or not have:
            crit["P2"] = ne(f"cần kết quả taxonomy và mọi cách chia ngẫu nhiên ({len(rgs)}) ở mọi seed, "
                            f"và >= 2 seed (có {len(seeds)} seed)")
        else:
            d = [R[f"unsw|sfg|tax|{s}"]["test_f1"]
                 - float(np.mean([R[f"unsw|sfg|{g}|{s}"]["test_f1"] for g in rgs])) for s in seeds]
            if not finite(d):
                crit["P2"] = ne("D_s không hữu hạn (có NaN)")
            else:
                m_d, s_d = float(np.mean(d)), float(np.std(d, ddof=1))
                thr = max(s_d, P2_MARGIN)
                crit["P2"] = ("PASS" if m_d >= -thr else "FAIL",
                              f"D_s = F1 taxonomy - TB {len(rgs)} cách chia ngẫu nhiên cùng seed = "
                              f"[{', '.join(f'{v:+.2f}' for v in d)}]; mean(D) = {m_d:+.2f}, "
                              f"SD(D) = {s_d:.2f}; cần mean(D) >= -max(SD(D), {P2_MARGIN:g}) = {-thr:+.2f}")
        prob = explain_problem("unsw")
        if prob:
            crit["P3"] = ne(prob)
        else:
            profs, counts = profiles("unsw")
            valid = counts >= 30
            if not valid.any():
                crit["P3"] = ne(f"không có lớp nào có >= 30 mẫu test được giải thích (nhiều nhất "
                                f"{int(counts.max())}; n_per_class = {cfg['n_per_class']})")
            elif not all(np.isfinite(p[valid]).all() for p in profs):
                crit["P3"] = ne("hồ sơ đóng góp nhóm không xác định (NaN) ở lớp có >= 30 mẫu")
            else:
                top = np.stack([p[valid].argmax(1) for p in profs])            # seed x lớp hợp lệ
                same = float((top == top[0]).all(0).mean())
                distinct = len(set(np.mean([p[valid] for p in profs], 0).argmax(1).tolist()))
                crit["P3"] = ("PASS" if same >= 0.7 and distinct >= 2 else "FAIL",
                              f"{100 * same:.0f}% lớp giữ nguyên nhóm hạng 1 qua các seed (cần >= 70%); "
                              f"{distinct} nhóm hạng 1 khác nhau giữa các lớp (cần >= 2); "
                              f"xét {int(valid.sum())}/{len(counts)} lớp có >= 30 mẫu")
    else:
        crit.update({k: ("NOT_EVALUABLE", "chưa chạy UNSW-NB15") for k in ("P1", "P2", "P3")})
    if "nsl" in datasets:
        prob = explain_problem("nsl")
        classes, gn = R.get("nsl|classes", []), R.get("nsl|group_names", [])
        need = ["R2L", "U2R", "DoS", "Probe"]
        if prob:
            crit["P4"] = ne(prob)
        elif not R.get("nsl|attack_types") or not all(
                R[f"nsl|sfg|tax|{s}"].get("profile_population") == "seen_train_types" for s in seeds):
            crit["P4"] = ne("hồ sơ NSL-KDD không được tính trên quần thể P4 "
                            "(kiểu tấn công có trong cả KDDTrain+ và KDDTest+)")
        elif not set(need) <= set(classes) or not {"Content", "Time", "Host"} <= set(gn):
            crit["P4"] = ne("thiếu lớp hoặc nhóm cần cho P4")
        else:
            profs, counts = profiles("nsl")
            ci = {c: i for i, c in enumerate(classes)}
            gi, gt, gh = gn.index("Content"), gn.index("Time"), gn.index("Host")
            n_txt = ", ".join(f"{c} {int(counts[ci[c]])}" for c in need)
            if not all(np.isfinite(p[[ci[c] for c in need]]).all() for p in profs):
                crit["P4"] = ne(f"hồ sơ đóng góp nhóm không xác định (NaN) ở R2L/U2R/DoS/Probe "
                                f"(số mẫu được giải thích: {n_txt})")
            else:
                a = [(p[ci["R2L"], gi] + p[ci["U2R"], gi]) / 2 for p in profs]
                b = [(p[ci["DoS"], gi] + p[ci["Probe"], gi]) / 2 for p in profs]
                t = [(p[ci["DoS"], gt] + p[ci["DoS"], gh] + p[ci["Probe"], gt] + p[ci["Probe"], gh]) / 2
                     for p in profs]
                crit["P4"] = ("PASS" if np.mean(a) > np.mean(b) else "FAIL",
                              f"Content ở R2L/U2R = {100 * np.mean(a):.1f}% vs DoS/Probe = "
                              f"{100 * np.mean(b):.1f}% (theo seed: "
                              f"{', '.join(f'{100 * x:.0f}/{100 * y:.0f}' for x, y in zip(a, b))}); "
                              f"số mẫu được giải thích: {n_txt}. "
                              f"Tham khảo H2: Time+Host ở DoS/Probe = {100 * np.mean(t):.1f}%")
    else:
        crit["P4"] = ("NOT_EVALUABLE", "chưa chạy NSL-KDD")
    return crit


def verdict_of(crit):
    """ĐI TIẾP chỉ khi cả P1-P4 đều là PASS; FAIL và NOT_EVALUABLE được liệt kê riêng."""
    if all(crit.get(k, ("NOT_EVALUABLE", ""))[0] == "PASS" for k in ("P1", "P2", "P3", "P4")):
        return "ĐI TIẾP: chạy đầy đủ kịch bản 1-4 của đề cương (5 seed)"
    parts = []
    if any(s == "NOT_EVALUABLE" and d.startswith("LỖI PROTOCOL") for s, d in crit.values()):
        parts.append("LỖI PROTOCOL/DỮ LIỆU, cần kiểm tra trước khi kết luận")
    for status, label in (("NOT_EVALUABLE", "KHÔNG ĐÁNH GIÁ ĐƯỢC"), ("FAIL", "KHÔNG ĐẠT")):
        ks = [k for k, (s, _) in crit.items() if s == status]
        if ks:
            parts.append(f"{label}: {', '.join(ks)}")
    return "CHƯA ĐI TIẾP - " + "; ".join(parts)


def summarize(R, cfg, datasets, out_dir, context):
    seeds = cfg["seeds"]
    summary = {"version": SCRIPT_VERSION, "config": R.get("config"), "datasets": {}}
    print("\n================ SFG-XIDS BƯỚC 0 - KẾT QUẢ (macro-F1 % trên test) ================")
    for ds in datasets:
        rows = {"XGBoost": _vals(R, ds, "xgb", "tax", seeds), "Flat-MLP": _vals(R, ds, "flat", "tax", seeds),
                "SFG-Attn (v1)": _vals(R, ds, "attn", "tax", seeds),
                "SFG-XIDS (taxonomy)": _vals(R, ds, "sfg", "tax", seeds),
                "SFG-XIDS (ngẫu nhiên)": [v for r in cfg["rand_seeds"]
                                          for v in _vals(R, ds, "sfg", f"rand{r}", seeds)],
                "SFG-XIDS p=0 (chẩn đoán)": _vals(R, ds, "sfg0", "tax", seeds)}
        print(f"\n{ds.upper()} (σ PLR = {R.get(f'{ds}|sigma')}):")
        for name, x in rows.items():
            m, s = _ms(x)
            note = "   <- chẩn đoán, không tham gia P1-P4" if "chẩn đoán" in name else ""
            print(f"  {name:<24} {_fmt(m, '6.2f')} ± {_fmt(s, '4.2f')}   (n={len(x)}){note}")
        summary["datasets"][ds] = {k: dict(zip(("mean", "std"), _ms(x))) for k, x in rows.items()}
        if ds == "unsw":
            ref = xgb_reference(rows["XGBoost"], context)
            print(f"  Đối chiếu (chẩn đoán, không phải tiêu chí): {ref['note']}")
            summary["datasets"][ds]["xgb_reference_check"] = ref
        if ds == "nsl" and R.get("nsl|attack_types"):
            print("  Quần thể P4 (NSL-KDD), dạng kiểu(số dòng):")
            for line in nsl_population_lines(R["nsl|attack_types"]):
                print(f"    {line}")
            summary["datasets"][ds]["p4_population"] = R["nsl|attack_types"]
        classes, gnames = R[f"{ds}|classes"], R[f"{ds}|group_names"]
        profs = [np.array(R[f"{ds}|sfg|tax|{s}"]["profile"]) for s in seeds
                 if "profile" in R.get(f"{ds}|sfg|tax|{s}", {})]
        exact = R.get(f"{ds}|sfg|tax|{seeds[0]}", {}).get("exact_err", float("nan"))
        errs = [R[f"{ds}|sfg|tax|{s}"]["sum_err"] for s in seeds if "sum_err" in R.get(f"{ds}|sfg|tax|{s}", {})]
        sum_err = float(np.max(errs)) if errs else float("nan")              # np.max giữ NaN
        phi_ok = all(math.isfinite(v) and v < PHI_TOL for v in (exact, sum_err))
        print(f"  Kiểm tra φ: |dạng đóng - vét cạn| = {exact:.1e}; |Σφ - (z - E z)| = {sum_err:.1e}"
              + ("  -> OK" if phi_ok else "  -> KHÔNG ĐẠT hoặc thiếu kết quả (P3/P4 sẽ NOT_EVALUABLE)"))
        csv_rows = []
        if profs:
            mean_prof = np.mean(profs, 0)
            pop_note = "; chỉ mẫu có kiểu tấn công đã có trong KDDTrain+" if ds == "nsl" else ""
            print(f"  Tỷ trọng đóng góp nhóm theo lớp (%, TB các seed{pop_note}) | nhóm hạng 1 theo từng seed:")
            print("  " + f"{'lớp':<16}" + "".join(f"{g[:10]:>11}" for g in gnames))
            for c, name in enumerate(classes):
                tops = "/".join(gnames[int(np.argmax(p[c]))][:4] if np.isfinite(p[c]).all() else "-"
                                for p in profs)
                print("  " + f"{name:<16}" + "".join(f"{100 * v:11.1f}" for v in mean_prof[c]) + f"   {tops}")
                for s, p in zip(seeds, profs):
                    csv_rows.append({"class": name, "seed": s, **{g: p[c][i] for i, g in enumerate(gnames)}})
            pd.DataFrame(csv_rows).to_csv(out_dir / f"profiles_{ds}.csv", index=False)
        summary["datasets"][ds].update(sigma=R.get(f"{ds}|sigma"), exact_err=exact, sum_err=sum_err)

    crit = evaluate_criteria(R, cfg, datasets, context)
    verdict = verdict_of(crit)
    print("\n================ TIÊU CHÍ ĐI TIẾP / DỪNG ================")
    for k, (status, msg) in crit.items():
        print(f"  {k}: {status:<13} {msg}")
    print(f"\nKẾT LUẬN: {verdict}")
    hints = {"P1": "tăng dung lượng encoder/embedding (chọn lại trên validation, không nhìn test); nếu "
                   "vẫn cách XGBoost > 3 điểm thì SFG-XIDS không đủ lợi thế so với XGBoost + Shapley "
                   "nhóm -> cân nhắc lại hướng đề tài.",
             "P2": "chia nhóm ngữ nghĩa không giúp gì về độ chính xác; chỉ còn lý do diễn giải -> "
                   "cần bằng chứng ở P3/P4.",
             "P3": "lời giải thích không ổn định/không phân biệt được lớp. Không đổi p_swap rồi chạy lại "
                   "theo kết quả test (p = 0.2 đã được quyết định cho pilot); báo cáo như một phát hiện về dư "
                   "thừa/artifact của dữ liệu và cân nhắc lại trọng tâm bài báo.",
             "P4": "lời giải thích không khớp thiết kế Lee & Stolfo: xem lại profiles_nsl.csv trước "
                   "khi quyết định."}
    for k, (status, _) in crit.items():
        if status == "FAIL":
            print(f"  - {k}: {hints[k]}")
    print("\nPhạm vi Step 0: chưa kiểm chứng Expected Gradients cấp đặc trưng, faithfulness của SFG-Attn "
          "(γ/A so với Shapley nhóm, xóa nhóm) và ablation A1-A6 - các hạng mục này thuộc full experiment.")
    summary.update(criteria={k: {"status": s, "pass": {"PASS": True, "FAIL": False}.get(s), "detail": d}
                             for k, (s, d) in crit.items()}, verdict=verdict)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return crit, verdict


# ----------------------------------------------------------------------------
# Tự kiểm tra (không cần dữ liệu)
# ----------------------------------------------------------------------------
def selftest():
    """Kiểm tra cài đặt. "SELFTEST: PASS" chỉ nghĩa các kiểm tra này đạt, KHÔNG nghĩa P1-P4 đạt."""
    import warnings
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    def exit_msg(fn):
        try:
            fn()
        except SystemExit as e:
            return str(e)
        return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u = [c for g in UNSW_GROUPS.values() for c in g]
        check("UNSW-NB15: 42 đặc trưng, nhóm 15/8/7/12, không trùng",
              len(u) == 42 == len(set(u)) and [len(v) for v in UNSW_GROUPS.values()] == [15, 8, 7, 12])
        n = [c for g in NSL_GROUPS.values() for c in g]
        check("NSL-KDD: 40 đặc trưng (bỏ num_outbound_cmds), nhóm 9/12/9/10, không trùng",
              len(n) == 40 == len(set(n)) and [len(v) for v in NSL_GROUPS.values()] == [9, 12, 9, 10])
        ok10 = sorted(UNSW_CLASSES)
        check("UNSW-NB15: đúng 10 lớp chuẩn ở train và test -> qua kiểm tra",
              len(ok10) == 10 and exit_msg(lambda: check_unsw_classes(ok10, ok10)) is None)
        msg = exit_msg(lambda: check_unsw_classes([c if c != "Worms" else "Worm" for c in ok10], ok10)) or ""
        check("UNSW-NB15: sai tên một lớp (Worm) -> dừng, báo rõ lớp thiếu và lớp thừa",
              "train" in msg and "['Worms']" in msg and "['Worm']" in msg)
        msg = exit_msg(lambda: check_unsw_classes(ok10, [c for c in ok10 if c != "Analysis"])) or ""
        check("UNSW-NB15: test thiếu một lớp (Analysis) -> dừng", "test" in msg and "['Analysis']" in msg)

        torch.manual_seed(0)
        cfg = dict(CFG, d=8, d_emb=4, n_freq=4)
        E0 = torch.tensor([], dtype=torch.long)
        gidx = [(torch.tensor([0, 1]), torch.tensor([0])), (torch.tensor([2]), E0),
                (torch.tensor([3, 4]), torch.tensor([1])), (torch.tensor([5]), E0)]
        m = SFGXIDS(6, [3, 4], gidx, 3, cfg, 1.0).double()
        xn, bn = torch.randn(7, 6, dtype=torch.float64), torch.randn(4, 6, dtype=torch.float64)
        xc = torch.stack([torch.randint(0, 3, (7,)), torch.randint(0, 4, (7,))], 1)
        bc = torch.stack([torch.randint(0, 3, (4,)), torch.randint(0, 4, (4,))], 1)
        cls = torch.randint(0, 3, (7,))
        a, err = group_phi(m, xn, xc, bn, bc, cls, chunk=3)
        d = (a - group_phi_brute(m, xn, xc, bn, bc, cls)).abs().max().item()
        check(f"φ dạng đóng khớp Shapley vét cạn 16 liên minh (max|Δ| = {d:.1e})", d < 1e-9)
        check(f"Σφ = z - E[z] (max|Δ| = {err:.1e})", err < 1e-9)
        nan = float("nan")
        check("nan_max giữ NaN ở cả hai vị trí đối số",
              math.isnan(nan_max(1e-7, nan)) and math.isnan(nan_max(nan, 1e-7)) and nan_max(1.0, 2.0) == 2.0)

        gen = torch.Generator().manual_seed(0)
        xs, cs = group_swap(xn.float(), xc, [gidx[0]], 1.0, gen)
        check("group dropout chỉ thay đúng các cột của nhóm được chọn",
              torch.equal(xs[:, 2:], xn.float()[:, 2:]) and torch.equal(cs[:, 1], xc[:, 1])
              and not torch.equal(xs[:, :2], xn.float()[:, :2]))
        ids = torch.arange(7, dtype=torch.float32)[:, None].repeat(1, 6)     # mỗi dòng mang số hiệu riêng
        cid = torch.arange(7)[:, None].repeat(1, 2)
        never_self = True
        for t in range(300):
            n_t = 2 + t % 6                                                   # batch 2..7 mẫu
            xs_t, cs_t = group_swap(ids[:n_t], cid[:n_t], gidx, 1.0, torch.Generator().manual_seed(t))
            own = torch.arange(n_t)[:, None]
            for ni, ci in gidx:
                never_self &= bool((xs_t[:, ni] != own.float()).all()) if ni.numel() else True
                never_self &= bool((cs_t[:, ci] != own).all()) if ci.numel() else True
        check("group dropout không bao giờ lấy chính dòng đích làm nguồn (300 batch, n = 2..7, mọi nhóm, "
              "p = 1)", never_self)
        xa, ca = group_swap(ids, cid, gidx, 0.5, torch.Generator().manual_seed(5))
        xb, cb = group_swap(ids, cid, gidx, 0.5, torch.Generator().manual_seed(5))
        mask0 = torch.rand(7, generator=torch.Generator().manual_seed(5)) < 0.5   # mặt nạ của nhóm đầu
        changed0 = (xa[:, gidx[0][0]] != ids[:, gidx[0][0]]).any(1)
        check("group dropout: cùng seed cho cùng kết quả; các dòng bị thay đúng bằng mặt nạ Bernoulli(p)",
              torch.equal(xa, xb) and torch.equal(ca, cb) and torch.equal(changed0, mask0))
        try:
            group_swap(ids[:1], cid[:1], gidx, 1.0, torch.Generator().manual_seed(0))
            raised = False
        except ValueError:
            raised = True
        check("group dropout: batch 1 mẫu (không có mẫu khác) -> báo lỗi, không tự chép", raised)

        m0 = SFGXIDS(6, [3, 4], gidx, 3, cfg, 1.0)
        for prm in m0.parameters():
            torch.nn.init.zeros_(prm)
        D0 = {"y": {"tr": np.zeros(4, np.int64), "te": np.array([0, 0, 1, 1, 2, 2])}, "n_cls": 3}
        T0 = {"xn_tr": bn.float(), "xc_tr": bc, "xn_te": xn.float()[:6], "xc_te": xc[:6]}
        p0, c0, _ = class_profiles(m0, D0, T0, dict(n_bg=4, n_per_class=400), torch.device("cpu"))
        check("hồ sơ lớp = NaN (không xác định) khi mọi φ bằng 0, không che bằng epsilon",
              all(math.isnan(v) for row in p0 for v in row) and c0 == [2, 2, 2])
        D0["explain_mask"] = np.array([True, False, True, True, False, False])
        _, c0m, _ = class_profiles(m0, D0, T0, dict(n_bg=4, n_per_class=400), torch.device("cpu"))
        check("P4: chỉ giải thích các mẫu test được phép (mask kiểu tấn công đã thấy)", c0m == [1, 2, 0])

        pop = nsl_population(["normal", "neptune", "smurf", "guess_passwd", "buffer_overflow", "spy"],
                             ["normal", "neptune", "smurf", "snmpguess", "httptunnel", "worm", "ps",
                              "buffer_overflow"])
        check("P4: tập đánh giá = kiểu có ở cả train và test; loại kiểu chỉ có ở test; không tính 'normal'",
              pop["evaluated"] == ["buffer_overflow", "neptune", "smurf"]
              and pop["excluded_unseen"] == ["httptunnel", "ps", "snmpguess", "worm"]
              and pop["train_only"] == ["guess_passwd", "spy"] and "normal" not in pop["map"])
        check("ánh xạ: snmpguess, httptunnel, worm -> R2L; smurf -> DoS; ps, sqlattack, xterm -> U2R",
              [NSL_MAP[t] for t in ("snmpguess", "httptunnel", "worm", "smurf", "ps", "sqlattack", "xterm")]
              == ["R2L", "R2L", "R2L", "DoS", "U2R", "U2R", "U2R"])
        check("chẩn đoán XGBoost: gần 51.57 -> OK; lệch > 2 -> cảnh báo; chế độ quick -> bỏ qua",
              xgb_reference([51.0, 52.0], "real")["flag"] is False
              and xgb_reference([48.0], "real")["flag"] is True and xgb_reference([51.0], "quick")["flag"] is None)

        # Logic P1-P4 trên số liệu dựng sẵn, biết trước đáp án (không phải kết quả nghiên cứu)
        pA = [[.4, .2, .2, .2], [.2, .4, .2, .2], [.2, .2, .4, .2]]
        pA2 = [[.4, .2, .2, .2], [.2, .4, .2, .2], [.2, .2, .2, .4]]      # chỉ lớp thứ 3 đổi hạng 1
        pB = [[.2, .4, .2, .2], [.2, .4, .2, .2], [.2, .2, .4, .2]]       # lớp thứ 1 đổi hạng 1
        nslP = [[.1, .1, .4, .4], [.25] * 4, [.1, .1, .4, .4], [.1, .7, .1, .1], [.1, .7, .1, .1]]
        nslF = [[.1, .7, .1, .1], [.25] * 4, [.1, .7, .1, .1], [.1, .1, .4, .4], [.1, .1, .4, .4]]
        nslN = nslP[:4] + [[nan] * 4]

        def case(sfg, flat, xg, rands, profs, counts, nsl=None, exact=1e-7, nsl_exact=1e-7,
                 nsl_pop="seen_train_types", extra=None):
            seeds, R = list(range(len(sfg))), {}
            R["nsl|classes"] = ["DoS", "Normal", "Probe", "R2L", "U2R"]
            R["nsl|group_names"] = ["Basic", "Content", "Time", "Host"]
            R["nsl|attack_types"] = pop
            for s in seeds:
                R[f"unsw|sfg|tax|{s}"] = {"test_f1": sfg[s], "profile": profs[s], "counts": counts,
                                          "sum_err": 1e-7, "exact_err": exact}
                R[f"unsw|flat|tax|{s}"], R[f"unsw|xgb|tax|{s}"] = {"test_f1": flat[s]}, {"test_f1": xg[s]}
                for i, rv in enumerate(rands):
                    R[f"unsw|sfg|rand{100 + i}|{s}"] = {"test_f1": rv[s]}
                if nsl is not None:
                    R[f"nsl|sfg|tax|{s}"] = {"profile": nsl, "counts": [100] * 5, "sum_err": 1e-7,
                                             "exact_err": nsl_exact, "profile_population": nsl_pop}
            R.update(extra or {})
            c_ = dict(seeds=seeds, rand_seeds=[100 + i for i in range(len(rands))], n_per_class=400)
            crit = evaluate_criteria(R, c_, ["unsw"] + (["nsl"] if nsl is not None else []), "real")
            return {k: v[0] for k, v in crit.items()}, crit

        base = ([50, 51, 52], [51] * 3, [53, 54, 55], [[52] * 3], [pA, pA2, pA], [30, 30, 29])
        s1, c1 = case(*base, nslP)
        check("tiêu chí: P1 PASS tại ngưỡng; P2 ghép cặp PASS tại ngưỡng mean(D) = -max(SD(D), 1); P3 tính lớp 30 "
              "mẫu, bỏ lớp 29 mẫu; P4 PASS", s1 == dict(P1="PASS", P2="PASS", P3="PASS", P4="PASS"))
        check("tiêu chí: cả 4 PASS -> ĐI TIẾP", verdict_of(c1).startswith("ĐI TIẾP"))
        _, c1d = case(*base, nslP, extra={f"unsw|sfg0|tax|{s}": {"test_f1": 0.0} for s in range(3)})
        check("nhánh chẩn đoán p = 0 không ảnh hưởng P1-P4 (F1 = 0 vẫn cho kết quả y hệt)", c1d == c1)
        s2, _ = case([50, 51, 52], [51] * 3, [53, 54, 55.3], [[52.5] * 3], [pA, pB, pA], [100] * 3, nslF)
        check("tiêu chí: vượt ngưỡng -> P1, P2 FAIL; nhóm hạng 1 đổi theo seed -> P3 FAIL; P4 FAIL",
              s2 == dict(P1="FAIL", P2="FAIL", P3="FAIL", P4="FAIL"))
        tax, rr = [50, 53, 56], [[51, 54, 57], [52, 55, 58]]     # ngẫu nhiên hơn đúng 1.5 điểm ở mọi seed
        old_pass = bool(np.mean(tax) >= np.mean(rr[0] + rr[1]) - np.std(tax, ddof=1))
        s6, c6 = case(tax, [53] * 3, [54] * 3, rr, [pA] * 3, [100] * 3)
        check("P2 ghép cặp khác công thức cũ: cũ PASS, ghép cặp FAIL (D = -1.50 ở mọi seed, TB 2 cách chia "
              "ngẫu nhiên ở cùng seed, vượt dung sai 1 điểm)",
              old_pass and s6["P2"] == "FAIL" and "[-1.50, -1.50, -1.50]" in c6["P2"][1])
        s9, _ = case([50, 52, 54], [52] * 3, [53] * 3, [[50.5, 52.5, 54.5]], [pA] * 3, [100] * 3)
        check("P2: thua đều 0.5 điểm (SD(D) = 0) -> PASS nhờ dung sai 1 điểm", s9["P2"] == "PASS")
        s10, c10 = case([50, 51, 52], [51] * 3, [53] * 3, [[54, 52.5, 51]], [pA] * 3, [100] * 3)
        check("P2: nhiễu giữa các seed lớn (mean(D) = -1.5, SD(D) = 2.5) -> PASS nhờ vế SD",
              s10["P2"] == "PASS" and "SD(D) = 2.50" in c10["P2"][1])
        s7, c7 = case(*base, nslP, exact=5e-3, nsl_exact=nan)
        check("kiểm tra φ không đạt (UNSW 5e-3, NSL NaN) -> P3, P4 NOT_EVALUABLE -> không ĐI TIẾP",
              s7["P3"] == "NOT_EVALUABLE" and s7["P4"] == "NOT_EVALUABLE" and "kiểm tra φ" in c7["P3"][1]
              and not verdict_of(c7).startswith("ĐI TIẾP"))
        s8, _ = case(*base, nslP, nsl_pop="all_test")
        check("P4 = NOT_EVALUABLE nếu hồ sơ NSL không tính trên quần thể kiểu tấn công đã thấy",
              s8["P4"] == "NOT_EVALUABLE")
        s3, c3 = case([50, 51, 52], [51] * 3, [53, 54, 55], [[52] * 3], [pA] * 3, [29] * 3, nslN)
        check("tiêu chí: không lớp nào >= 30 mẫu -> P3 NOT_EVALUABLE, báo LỖI PROTOCOL/DỮ LIỆU khi chạy "
              "thật; hồ sơ NaN -> P4 NOT_EVALUABLE; kết luận không phải ĐI TIẾP",
              s3["P3"] == "NOT_EVALUABLE" and c3["P3"][1].startswith("LỖI PROTOCOL")
              and s3["P4"] == "NOT_EVALUABLE" and not verdict_of(c3).startswith("ĐI TIẾP"))
        s4, _ = case([50, nan, 52], [51] * 3, [53, 54, 55], [[52] * 3], [pA, [[nan] * 4] * 3, pA], [100] * 3)
        check("tiêu chí: NaN không bao giờ thành PASS (P1, P2, P3 -> NOT_EVALUABLE)",
              s4 == dict(P1="NOT_EVALUABLE", P2="NOT_EVALUABLE", P3="NOT_EVALUABLE", P4="NOT_EVALUABLE"))
        s5, _ = case([51], [51], [53], [[52]], [pA], [100] * 3)
        check("tiêu chí: 1 seed -> không tính được SD(D) -> P2 NOT_EVALUABLE", s5["P2"] == "NOT_EVALUABLE")

        if xgb is None:
            check("xgboost đã cài (cần cho chạy thử đường ống)", False)
        else:   # chạy thử toàn bộ đường ống trên dữ liệu giả dạng UNSW-NB15 (CPU, rất nhỏ)
            rng = np.random.RandomState(0)

            def fake(nr):
                df = pd.DataFrame({c: rng.exponential(5, nr) for c in u if c not in UNSW_CAT})
                df["proto"] = rng.choice(["tcp", "udp", "arp"], nr)
                df["service"] = rng.choice(["-", "http", "dns"], nr)
                df["state"] = rng.choice(["FIN", "INT", "CON"], nr)
                df["attack_cat"] = rng.choice(["Normal", "DoS", "Exploits", "Generic"], nr)
                return df

            D = prepare(fake(900), fake(400), "attack_cat", UNSW_GROUPS, UNSW_CAT)
            tcfg = dict(CFG, **QUICK)
            tcfg.update(max_ep=1, n_bg=8, n_per_class=10, n_exact=2, n_bg_exact=2, n_estimators=5,
                        es_rounds=2, p_swap=0.2, bs=256)
            R = {"unsw|classes": D["classes"], "unsw|group_names": list(D["groups"])}
            run_dataset("unsw", D, R, tcfg, torch.device("cpu"), lambda: None)
            need = [f"unsw|{m_}|tax|{s}" for m_ in ("xgb", "flat", "attn", "sfg", "sfg0") for s in tcfg["seeds"]]
            check("chạy thử đường ống: đủ kết quả các mô hình, gồm nhánh chẩn đoán p = 0",
                  all(k in R for k in need))
            ex = R["unsw|sfg|tax|0"]["exact_err"]
            check(f"chạy thử: φ khớp vét cạn trên mô hình đã huấn luyện (max|Δ| = {ex:.1e})", ex < 1e-4)
            check("hồ sơ UNSW-NB15 ghi rõ quần thể = toàn bộ test",
                  R["unsw|sfg|tax|0"].get("profile_population") == "all_test")
            import contextlib
            import io
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                cli_msg = exit_msg(lambda: parse_args(["--p_swap", "0.3"]))
            check("p_swap cố định: dòng lệnh không còn --p_swap (truyền --p_swap 0.3 bị từ chối)",
                  cli_msg is not None and "unrecognized arguments: --p_swap 0.3" in err_buf.getvalue())
            R_rej = {}
            msgs = [exit_msg(lambda v=v: run_dataset("unsw", D, R_rej, dict(tcfg, p_swap=v),
                                                     torch.device("cpu"), lambda: None)) or ""
                    for v in (0.3, 0.0, 0.25)]
            check("p_swap cố định: run_dataset từ chối p_swap = 0.3 / 0.0 / 0.25 trước khi huấn luyện",
                  all("p_swap = 0.2" in m for m in msgs) and R_rej == {})
            check("p_swap = 0.2 vẫn chạy: các run SFG-XIDS chính ghi p_swap = 0.2; nhánh chẩn đoán sfg0 "
                  "ghi p_swap = 0.0",
                  all(R[f"unsw|sfg|{g}|{s}"]["p_swap"] == 0.2 for g in ("tax", "rand100") for s in tcfg["seeds"])
                  and all(R[f"unsw|sfg0|tax|{s}"]["p_swap"] == 0.0 for s in tcfg["seeds"]))
            print("\n--- Tổng hợp trên dữ liệu giả: chỉ kiểm tra logic, P1-P4 dưới đây KHÔNG có ý nghĩa "
                  "nghiên cứu ---")
            crit, verdict = summarize(R, tcfg, ["unsw"], Path(tempfile.mkdtemp()), "selftest")
            print("--- hết phần dữ liệu giả ---\n")
            st = {k: v[0] for k, v in crit.items()}
            check("dữ liệu giả: P1-P4 đều có trạng thái PASS/FAIL/NOT_EVALUABLE",
                  set(st) == {"P1", "P2", "P3", "P4"} and all(v in STATUSES for v in st.values()))
            check("dữ liệu giả: P1, P2 đánh giá được với 2 seed (PASS hoặc FAIL, không NaN)",
                  st["P1"] in ("PASS", "FAIL") and st["P2"] in ("PASS", "FAIL"))
            check(f"dữ liệu giả: P3 = NOT_EVALUABLE vì không lớp nào có >= 30 mẫu được giải thích "
                  f"(n_per_class = {tcfg['n_per_class']})", st["P3"] == "NOT_EVALUABLE")
            check("dữ liệu giả: P4 = NOT_EVALUABLE vì không chạy NSL-KDD", st["P4"] == "NOT_EVALUABLE")
            check("dữ liệu giả: kết luận không phải ĐI TIẾP khi còn tiêu chí NOT_EVALUABLE",
                  not verdict.startswith("ĐI TIẾP"))

            orig = globals()["class_profiles"]

            def boom(*args, **kwargs):
                raise RuntimeError("lỗi giả lập trong bước giải thích")

            print("--- chạy lại đường ống với bước giải thích bị làm lỗi có chủ đích ---")
            globals()["class_profiles"] = boom
            try:
                R2 = {"unsw|classes": D["classes"], "unsw|group_names": list(D["groups"])}
                run_dataset("unsw", D, R2, tcfg, torch.device("cpu"), lambda: None)
            finally:
                globals()["class_profiles"] = orig
            c_err = evaluate_criteria(R2, tcfg, ["unsw"], "real")
            check("lỗi trong bước giải thích -> ghi explain_error, P3 = NOT_EVALUABLE, không ĐI TIẾP",
                  "explain_error" in R2["unsw|sfg|tax|0"] and "profile" not in R2["unsw|sfg|tax|0"]
                  and c_err["P3"][0] == "NOT_EVALUABLE" and "giải thích bị lỗi" in c_err["P3"][1]
                  and not verdict_of(c_err).startswith("ĐI TIẾP"))

    rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    for w in rw:
        print(f"     RuntimeWarning: {w.message} ({Path(w.filename).name}:{w.lineno})")
    check(f"không có RuntimeWarning nào (mean of empty slice, chia cho 0, ...): có {len(rw)}", not rw)
    for cat, msg in sorted({(w.category.__name__, str(w.message)) for w in caught
                            if not issubclass(w.category, RuntimeWarning)}):
        print(f"     (cảnh báo khác, không tính là lỗi) {cat}: {msg}")
    n_fail = results.count(False)
    print(f"\nSELFTEST: {'PASS' if n_fail == 0 else 'FAIL'} ({len(results) - n_fail} PASS, {n_fail} FAIL / "
          f"{len(results)} kiểm tra). Chỉ có nghĩa các kiểm tra cài đặt đạt, KHÔNG có nghĩa P1-P4 đạt.")
    sys.exit(0 if n_fail == 0 else 1)


# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFG-XIDS bước 0 (pilot đi tiếp/dừng)")
    ap.add_argument("--unsw_dir", default=None, help="Thư mục chứa training/testing-set UNSW-NB15")
    ap.add_argument("--nsl_dir", default=None, help="Thư mục chứa KDDTrain+.txt, KDDTest+.txt")
    ap.add_argument("--datasets", default="unsw,nsl", help="unsw, nsl hoặc unsw,nsl")
    ap.add_argument("--out_dir", default="sfg_step0")
    ap.add_argument("--device", default=None, help="cuda hoặc cpu (mặc định: tự nhận)")
    ap.add_argument("--quick", action="store_true", help="Chạy thử nhanh trên mẫu nhỏ")
    ap.add_argument("--skip_row_check", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="Tự kiểm tra (không cần dữ liệu)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.selftest:
        selftest()
    cfg = dict(CFG, seeds=SEEDS, rand_seeds=RAND_SEEDS, sigmas=SIGMAS, p_swap=P_SWAP)
    if args.quick:
        cfg.update(QUICK)
    if cfg["n_per_class"] < 30:          # P3 chỉ xét lớp có >= 30 mẫu test được giải thích
        sys.exit("Cấu hình sai: n_per_class < 30 thì P3 không bao giờ đánh giá được.")
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print("CẢNH BÁO: không thấy GPU - sẽ rất chậm. Trên Colab chọn Runtime -> T4 GPU.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "results.json"
    R = json.loads(rpath.read_text()) if rpath.exists() else {}
    conf = dict(version=SCRIPT_VERSION, quick=args.quick, p_swap=P_SWAP)
    if R.get("config", conf) != conf:
        sys.exit(f"{out} chứa kết quả của cấu hình khác {R['config']}; hãy dùng --out_dir khác.")
    R["config"] = conf

    def save():
        tmp = rpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(R))
        tmp.replace(rpath)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in datasets:
        t0 = time.time()
        if ds == "unsw":
            D = prepare(*load_unsw(args), UNSW_GROUPS, UNSW_CAT, args.quick)
        elif ds == "nsl":
            tr, te, lab = load_nsl(args, out)
            pop = nsl_population(tr["attack_type"], te["attack_type"])     # tính trên file đầy đủ
            D = prepare(tr, te, lab, NSL_GROUPS, NSL_CAT, args.quick, keep_col="attack_type")
            D["explain_mask"] = np.isin(D["te_extra"], pop["evaluated"] + ["normal"])
            D["profile_population"] = "seen_train_types"
            R["nsl|attack_types"] = pop
        else:
            sys.exit(f"Bộ dữ liệu không hỗ trợ: {ds}")
        R[f"{ds}|classes"], R[f"{ds}|group_names"] = D["classes"], list(D["groups"])
        print(f"\n=== {ds}: {len(D['y']['tr']):,} train | {len(D['y']['va']):,} val | "
              f"{len(D['y']['te']):,} test | {D['n_cls']} lớp ===")
        run_dataset(ds, D, R, cfg, dev, save)
        print(f"  [{ds}] xong sau {(time.time() - t0) / 60:.1f} phút")
    summarize(R, cfg, datasets, out, "quick" if args.quick else "real")


if __name__ == "__main__":
    main()
