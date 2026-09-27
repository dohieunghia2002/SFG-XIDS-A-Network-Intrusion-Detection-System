#!/usr/bin/env python3
"""
SFG-XIDS - FULL EXPERIMENT, GIAI ĐOẠN 1 (đề cương v2: kịch bản 1-4, bài toán đa lớp, 5 seed).
Dùng lại nguyên các thành phần đã kiểm tra của step0_sfg.py v5 (dữ liệu, tiền xử lý, mô hình, huấn
luyện, φ dạng đóng, Shapley vét cạn); không sửa file đó.

Quyết định giao thức (đưa ra TRƯỚC khi chạy, không đổi sau khi thấy kết quả test):
  1. 5 seed: 0-4. Tập test chính thức. Mọi lựa chọn chỉ dựa trên macro-F1 validation; hòa -> giá trị
     đứng trước trong lưới.
  2. σ của PLR chọn trong {0.3; 1; 3; 10} bằng Flat-MLP seed 0 (lưới mở rộng vì pilot chọn σ = 1 ở biên);
     dùng chung cho mọi mạng nơ-ron của cùng bộ dữ liệu, kể cả ablation.
  3. p của group dropout chọn trong {0.1; 0.2; 0.3} bằng SFG-XIDS (taxonomy) seed 0 (đề cương mục 4.3);
     dùng cho mọi biến thể SFG-XIDS có group dropout. A3 dùng p = 0. SFG-Concat, SFG-Attn và Flat-MLP
     không dùng group dropout (như pilot).
  4. max_ep = 60 (pilot 40); patience 6, lr 1e-3, batch 1024, trọng số lớp như pilot.
  5. A5 (bỏ sttl, dttl, ct_state_ttl; chỉ UNSW-NB15) chạy ĐẦU TIÊN, với σ và p đã chọn trên dữ liệu đầy
     đủ. Mô hình: XGBoost, Flat-MLP, SFG-XIDS. Nhóm còn lại: Basic 13, Content 8, Time 7, Additional 11.
  6. A4: 5 cách chia nhóm ngẫu nhiên (seed 100-104), cùng kích thước nhóm với taxonomy.
  7. Mô hình cây dùng cấu hình cố định, không tinh chỉnh: XGBoost (như pilot/B1), LightGBM, Random Forest.
  8. Giải thích dùng cùng tập nền (256 mẫu train) và cùng mẫu test (tối đa 400/lớp; NSL-KDD chỉ kiểu tấn
     công có trong cả KDDTrain+ và KDDTest+) như pilot, cho logit đã trừ TB các lớp của LỚP THẬT.
     SFG-XIDS: φ dạng đóng. XGBoost, SFG-Attn: Shapley nhóm Monte Carlo (nền độc lập theo nhóm; 128 tổ
     hợp nền rút một lần, dùng chung cho mọi liên minh, mọi mẫu, mọi mô hình). SFG-XIDS cũng được tính
     bằng Monte Carlo để so trên cùng một bộ ước lượng. XGBoost dự đoán qua DMatrix dựng thẳng từ mảng,
     cột xếp theo đúng tên cột lúc huấn luyện, trên thiết bị lúc huấn luyện (GPU trên Colab).

Kịch bản:
  1  Hiệu năng: XGBoost, Random Forest, LightGBM, Flat-MLP, SFG-Concat, SFG-Attn, SFG-XIDS.
  2  Ablation: A1 chỉ số hạng chính (bỏ v_ij), A3 không group dropout, A4 chia ngẫu nhiên, A5 bỏ TTL,
     A6 bỏ phần tuần hoàn của PLR (Linear -> ReLU).
  3  Hồ sơ đóng góp nhóm và bản đồ tương tác |g_ij| theo lớp (SFG-XIDS; UNSW đầy đủ, UNSW bỏ TTL, NSL).
  4  Kiểm chứng giải thích: φ dạng đóng = Shapley vét cạn; completeness; Monte Carlo so với dạng đóng;
     độ ổn định qua 5 seed (SFG-XIDS, A3, SFG-Attn, XGBoost: tỷ lệ lớp giữ nhóm hạng 1 ở mọi seed và ở
     đa số seed, Kendall τ, SD tỷ trọng giữa các seed); thiết kế Lee & Stolfo trên NSL (H1, H2);
     faithfulness của attention (γ, A so với Shapley nhóm của chính SFG-Attn; thí nghiệm xóa nhóm).
  Báo cáo: TB ± SD 5 seed; so sánh hai mô hình bằng hiệu ghép cặp theo seed (TB ± SD). Không có tiêu chí
  đạt/không đạt ở giai đoạn này; H1/H2 báo SUPPORTED / NOT_SUPPORTED / NOT_EVALUABLE.
CHƯA gồm (giai đoạn 2): Expected Gradients cấp đặc trưng, FT-Transformer / TabNet / EBM, bài toán nhị
phân, giá trị Owen (shap.PartitionExplainer).

Chạy trên Colab (T4 GPU), cùng thư mục với step0_sfg.py (v5) và step0_unsw.py:
    !pip -q install -U xgboost kagglehub lightgbm
    (chạy lại cell tạo /content/unsw_std như khi chạy pilot)
    !python -u full_sfg.py --selftest
    !python -u full_sfg.py --quick --out_dir full_quick --unsw_dir /content/unsw_std
    !python -u full_sfg.py --out_dir full_v1 --unsw_dir /content/unsw_std
Bị ngắt: chạy lại đúng lệnh cũ, script bỏ qua các lần chạy đã xong. Ước tính 2-3 giờ trên T4.
Kết quả: <out_dir>/results.json, summary.json, runs.csv, per_class_f1.csv, profiles.csv, heatmaps.csv,
models/ (trọng số các mô hình có giải thích, dùng cho giai đoạn 2).
"""
import argparse
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import step0_sfg as S
except ImportError:
    sys.exit("Cần step0_sfg.py (v5) và step0_unsw.py cùng thư mục với full_sfg.py.")
torch, nn = S.torch, S.nn
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

SCRIPT_VERSION = 2
SEEDS, SIGMAS, P_GRID = [0, 1, 2, 3, 4], [0.3, 1.0, 3.0, 10.0], [0.1, 0.2, 0.3]
RAND_SEEDS = [100, 101, 102, 103, 104]
TTL = ["sttl", "dttl", "ct_state_ttl"]
NOTTL_GROUPS = {g: [c for c in cols if c not in TTL] for g, cols in S.UNSW_GROUPS.items()}
MC_SEED, DEL_SEED = 2024, 2025
CFG = dict(S.CFG, max_ep=60, n_mc=128, n_del=16, rf_trees=300,
           seeds=SEEDS, sigmas=SIGMAS, p_grid=P_GRID, rand_seeds=RAND_SEEDS)
QUICK = dict(max_ep=3, patience=2, n_bg=32, n_per_class=50, n_exact=4, n_bg_exact=3, n_estimators=60,
             es_rounds=10, n_mc=16, n_del=4, rf_trees=20, seeds=[0, 1], sigmas=[1.0], p_grid=[0.2],
             rand_seeds=[100])
PROTOCOL_KEYS = ["seeds", "sigmas", "p_grid", "rand_seeds", "max_ep", "patience", "lr", "bs", "n_bg",
                 "n_per_class", "n_mc", "n_del", "rf_trees", "n_estimators", "es_rounds"]
TREES = ("xgb", "rf", "lgbm")
MAIN = ["xgb", "rf", "lgbm", "flat", "concat", "attn", "sfg"]
ABL = ["sfg_main", "sfg_p0", "sfg_lr"]
NAMES = {"xgb": "XGBoost", "rf": "Random Forest", "lgbm": "LightGBM", "flat": "Flat-MLP",
         "concat": "SFG-Concat", "attn": "SFG-Attn (v1)", "sfg": "SFG-XIDS",
         "sfg_main": "A1 chỉ số hạng chính", "sfg_p0": "A3 không group dropout",
         "sfg_lr": "A6 bỏ tuần hoàn (LR)"}
EXPLAINERS = [("SFG-XIDS, φ dạng đóng", "sfg", "profile"), ("SFG-XIDS, Monte Carlo", "sfg", "mc_profile"),
              ("A3 (p = 0), φ dạng đóng", "sfg_p0", "profile"), ("SFG-Attn, Monte Carlo", "attn", "mc_profile"),
              ("XGBoost, Monte Carlo", "xgb", "mc_profile")]
TRAINED = [0]          # số lần huấn luyện thật (selftest dùng để kiểm tra chạy tiếp)
NAN = float("nan")


# ----------------------------------------------------------------------------
# Mô hình bổ sung
# ----------------------------------------------------------------------------
class SFGConcat(S.GroupedBase):
    """Đối chứng SFG-Concat: 4 encoder nhóm, ghép nối rồi MLP."""

    def __init__(self, n_num, cards, gidx, n_cls, cfg, sigma):
        super().__init__(n_num, cards, gidx, cfg, sigma)
        self.head = nn.Sequential(nn.Linear(self.G * cfg["d"], 128), nn.LeakyReLU(), nn.Linear(128, n_cls))

    def forward(self, xn, xc):
        return self.head(torch.cat(self.encode(xn, xc), 1))


class SFGMain(S.SFGXIDS):
    """A1: GAI-Head chỉ còn các số hạng chính u_i (bỏ mọi v_ij)."""

    def __init__(self, *args):
        super().__init__(*args)
        self.pairs, self.v = [], nn.ModuleList()


class LREmb(nn.Module):
    """A6: bỏ phần tuần hoàn của PLR, còn Linear -> ReLU cho từng đặc trưng số (cùng d_emb)."""

    def __init__(self, n_num, cards, d_emb):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d_emb))
        self.b = nn.Parameter(torch.zeros(n_num, d_emb))
        self.cat = nn.ModuleList([nn.Embedding(k, d_emb) for k in cards])
        self.d_emb = d_emb

    def forward(self, xn, xc):
        ne = torch.relu(xn[..., None] * self.w + self.b)
        if len(self.cat):
            return ne, torch.stack([e(xc[:, j]) for j, e in enumerate(self.cat)], 1)
        return ne, xn.new_zeros(xn.shape[0], 0, self.d_emb)


def build(kind, D, gidx, cfg, sigma):
    n_num, cards, C = len(D["nums"]), D["cards"], D["n_cls"]
    if kind == "flat":
        return S.FlatMLP(n_num, cards, C, cfg, sigma)
    if kind == "concat":
        return SFGConcat(n_num, cards, gidx, C, cfg, sigma)
    if kind == "attn":
        return S.SFGAttn(n_num, cards, gidx, C, cfg, sigma)
    if kind == "sfg_main":
        return SFGMain(n_num, cards, gidx, C, cfg, sigma)
    m = S.SFGXIDS(n_num, cards, gidx, C, cfg, sigma)            # sfg, sfg_p0, sfg_lr
    if kind == "sfg_lr":
        m.emb = LREmb(n_num, cards, cfg["d_emb"])
    return m


def run_tree(kind, D, seed, cfg, dev):
    w, ytr, yva = D["cw"], D["y"]["tr"], D["y"]["va"]
    if kind == "xgb":
        m = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"], learning_rate=0.1, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", device=dev.type, enable_categorical=True,
            max_cat_to_onehot=1, objective="multi:softprob", eval_metric="mlogloss",
            early_stopping_rounds=cfg["es_rounds"], random_state=seed)
        m.fit(D["F"]["tr"], ytr, sample_weight=w[ytr], eval_set=[(D["F"]["va"], yva)],
              sample_weight_eval_set=[w[yva]], verbose=False)
        return m, {"trees": int(m.best_iteration) + 1}, m.predict(D["F"]["va"]), m.predict(D["F"]["te"])
    if kind == "lgbm":
        m = lgb.LGBMClassifier(n_estimators=cfg["n_estimators"], learning_rate=0.05, num_leaves=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                               n_jobs=-1, verbose=-1)
        m.fit(D["F"]["tr"], ytr, sample_weight=w[ytr], eval_set=[(D["F"]["va"], yva)],
              eval_sample_weight=[w[yva]], callbacks=[lgb.early_stopping(cfg["es_rounds"], verbose=False)])
        return (m, {"trees": int(m.best_iteration_ or cfg["n_estimators"])}, m.predict(D["F"]["va"]),
                m.predict(D["F"]["te"]))
    X = {k: np.hstack([D["Z"][k], D["C"][k]]) for k in ("tr", "va", "te")}    # rf: số + mã phân loại
    m = RandomForestClassifier(n_estimators=cfg["rf_trees"], n_jobs=-1, random_state=seed)
    m.fit(X["tr"], ytr, sample_weight=w[ytr])
    return m, {"trees": cfg["rf_trees"]}, m.predict(X["va"]), m.predict(X["te"])


def metrics(y, p, C):
    lab = list(range(C))
    return dict(test_f1=S.macro_f1(y, p, C), test_acc=float(100 * accuracy_score(y, p)),
                test_prec=float(100 * precision_score(y, p, average="macro", labels=lab, zero_division=0)),
                test_rec=float(100 * recall_score(y, p, average="macro", labels=lab, zero_division=0)),
                f1_per_class=[float(v) for v in 100 * f1_score(y, p, average=None, labels=lab, zero_division=0)])


# ----------------------------------------------------------------------------
# Giải thích
# ----------------------------------------------------------------------------
def shares(a):
    a = np.asarray(a, dtype=float)
    tot = float(a.sum())
    return (a / tot).tolist() if math.isfinite(tot) and tot > 0 else [NAN] * len(a)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def explain_set(D, cfg):
    """Mẫu nền và mẫu test theo lớp, cùng chuỗi RandomState(1234) với S.class_profiles."""
    rng = np.random.RandomState(1234)
    ntr, mask = len(D["y"]["tr"]), D.get("explain_mask")
    bi = rng.choice(ntr, min(cfg["n_bg"], ntr), replace=False)
    per = []
    for c in range(D["n_cls"]):
        idx = np.where(D["y"]["te"] == c if mask is None else (D["y"]["te"] == c) & mask)[0]
        if len(idx) > cfg["n_per_class"]:
            idx = rng.choice(idx, cfg["n_per_class"], replace=False)
        per.append(idx)
    return bi, per


@torch.no_grad()
def phi_g(model, xn, xc, bn, bc, chunk=128):
    """Cùng công thức với S.group_phi, trả về đầy đủ: φ (n x G x C) và tương tác thuần g (n x P x C), cả hai
    đã trừ TB các lớp, cùng sai số lớn nhất của đẳng thức Σφ = z - E[z]."""
    model.eval()
    Eb, G = model.encode(bn, bc), model.G

    def grid(k, left, right):
        """v_k(left[a], right[b]) cho mọi a, b -> (len(left), len(right), C). Trục 0 luôn ứng với left, trục 1
        luôn ứng với right, với mọi kích thước lô."""
        na, nb = left.shape[0], right.shape[0]
        lft = left[:, None, :].expand(na, nb, left.shape[1]).reshape(na * nb, -1)
        rgt = right[None, :, :].expand(na, nb, right.shape[1]).reshape(na * nb, -1)
        return model.pair(k, lft, rgt).view(na, nb, -1)

    ubar = [model.u[i](Eb[i]).mean(0) for i in range(G)]
    vbar = [grid(k, Eb[i], Eb[j]).mean((0, 1)) for k, (i, j) in enumerate(model.pairs)]   # TB mọi cặp nền
    base = model.bias + sum(ubar) + sum(vbar)
    Ps, Gs, err = [], [], 0.0
    for s in range(0, xn.shape[0], chunk):
        E = model.encode(xn[s:s + chunk], xc[s:s + chunk])
        phi, gl = [model.u[i](E[i]) - ubar[i] for i in range(G)], []
        for k, (i, j) in enumerate(model.pairs):
            V = model.pair(k, E[i], E[j])
            Ai = grid(k, E[i], Eb[j]).mean(1)      # (n, B, C) -> TB_b v(e_i[a], e_j'[b]), rút gọn trục nền
            Aj = grid(k, Eb[i], E[j]).mean(0)      # (B, n, C) -> TB_b v(e_i'[b], e_j[a]), rút gọn trục nền
            g = V - Ai - Aj + vbar[k]
            phi[i] = phi[i] + (Ai - vbar[k]) + 0.5 * g
            phi[j] = phi[j] + (Aj - vbar[k]) + 0.5 * g
            gl.append(g)
        P = torch.stack(phi, 1)
        err = S.nan_max(err, (P.sum(1) - (model.head(E) - base)).abs().max().item())
        Ps.append(P - P.mean(2, keepdim=True))
        if gl:
            Gt = torch.stack(gl, 1)
            Gs.append(Gt - Gt.mean(2, keepdim=True))
    return torch.cat(Ps), (torch.cat(Gs) if Gs else None), err


def closed_explain(model, D, T, ES, dev):
    bi, per = ES
    bt = torch.as_tensor(bi, device=dev)
    bn, bc = T["xn_tr"][bt], T["xc_tr"][bt]
    out = {"profile": [], "counts": [], "heat": [], "phi_abs_sum": [], "sum_err": 0.0}
    for c, idx in enumerate(per):
        out["counts"].append(int(len(idx)))
        if len(idx) == 0:
            out["profile"].append([NAN] * model.G)
            out["heat"].append([NAN] * len(model.pairs))
            out["phi_abs_sum"].append(NAN)
            continue
        ti = torch.as_tensor(idx, device=dev)
        Pf, Gf, e = phi_g(model, T["xn_te"][ti], T["xc_te"][ti], bn, bc)
        a = Pf[:, :, c].abs().mean(0).double().cpu().numpy()
        out["profile"].append(shares(a))
        out["phi_abs_sum"].append(float(a.sum()))
        out["heat"].append(Gf[:, :, c].abs().mean(0).double().cpu().tolist() if Gf is not None else [])
        out["sum_err"] = S.nan_max(out["sum_err"], e)
    return out


def mc_shapley(predict, xn, xc, bn, bc, groups, M, seed, chunk=64):
    """Shapley nhóm của một mô hình bất kỳ, nền độc lập theo nhóm, cho logit đã trừ TB các lớp.
    M = None: liệt kê đủ B^G tổ hợp nền (chính xác; chỉ dùng khi B nhỏ). M số nguyên: M tổ hợp nền rút một
    lần (seed cố định), dùng chung cho mọi liên minh và mọi mẫu. predict(xn, xc) -> logit (n x C).
    Trả về φ: n x G x C."""
    G, B = len(groups), len(bn)
    combos = (np.array(list(itertools.product(range(B), repeat=G)), dtype=np.int64) if M is None
              else np.random.RandomState(seed).randint(0, B, size=(M, G)))
    K, out = len(combos), []
    wts = [math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G) for k in range(G)]
    for s in range(0, len(xn), chunk):
        x_n, x_c = xn[s:s + chunk], xc[s:s + chunk]
        n, val = len(x_n), {}
        for S_ in range(1 << G):
            masked = [g for g in range(G) if not (S_ >> g) & 1]
            if not masked:
                val[S_] = np.asarray(predict(x_n, x_c), dtype=np.float64)
                continue
            Xn, Xc = np.repeat(x_n, K, 0), np.repeat(x_c, K, 0)
            for g in masked:
                ni, ci = groups[g]
                src = np.tile(combos[:, g], n)
                if len(ni):
                    Xn[:, ni] = bn[src][:, ni]
                if len(ci):
                    Xc[:, ci] = bc[src][:, ci]
            val[S_] = np.asarray(predict(Xn, Xc), dtype=np.float64).reshape(n, K, -1).mean(1)
        phi = np.zeros((n, G, val[0].shape[1]))
        for g in range(G):
            for S_ in range(1 << G):
                if not (S_ >> g) & 1:
                    phi[:, g] += wts[bin(S_).count("1")] * (val[S_ | (1 << g)] - val[S_])
        out.append(phi - phi.mean(2, keepdims=True))
    return np.concatenate(out)


def nn_predict(model, dev, bs=65536):
    @torch.no_grad()
    def f(xn, xc):
        model.eval()
        return np.concatenate([model(torch.as_tensor(xn[s:s + bs], device=dev),
                                     torch.as_tensor(xc[s:s + bs], device=dev)).double().cpu().numpy()
                               for s in range(0, len(xn), bs)])
    return f


def xgb_predict(m, D):
    """Logit thô (output_margin) của XGBoost từ mảng số gốc (cột theo D["nums"]) và mã phân loại (cột theo
    D["cats"]). Ghép thẳng thành ma trận theo đúng tên và thứ tự cột lúc huấn luyện rồi đưa vào DMatrix (XGBoost
    kiểm tra lại tên cột khi dự đoán); không dựng DataFrame trong vòng lặp. Dự đoán trên thiết bị lúc huấn luyện
    và chỉ dùng các cây đến vòng tốt nhất, như m.predict."""
    booster = m.get_booster()
    names, types = booster.feature_names, booster.feature_types
    if names is None or list(names) != list(D["feats"]):
        raise ValueError(f"tên cột của mô hình XGBoost không khớp D['feats']: {names}")
    pos_n = np.array([names.index(c) for c in D["nums"]], dtype=np.int64)
    pos_c = np.array([names.index(c) for c in D["cats"]], dtype=np.int64)
    try:
        it = (0, int(m.best_iteration) + 1)
    except AttributeError:                 # không dùng early stopping -> dùng mọi cây
        it = (0, 0)

    def f(xn, xc):
        X = np.empty((len(xn), len(names)), dtype=np.float32)
        X[:, pos_n] = xn
        X[:, pos_c] = xc                   # mã phân loại = mã category của pandas lúc huấn luyện
        dm = xgb.DMatrix(X, feature_names=names, feature_types=types, enable_categorical=True)
        return booster.predict(dm, output_margin=True, iteration_range=it)
    return f


def mc_explain(predict, x_te, c_te, x_bg, c_bg, groups, per, cfg):
    prof, Ps, idxs = [], [], []
    for c, idx in enumerate(per):
        if len(idx) == 0:
            prof.append([NAN] * len(groups))
            continue
        P = mc_shapley(predict, x_te[idx], c_te[idx], x_bg, c_bg, groups, cfg["n_mc"], MC_SEED)
        prof.append(shares(np.abs(P[:, :, c]).mean(0)))
        Ps.append(P)
        idxs.append(idx)
    return prof, Ps, idxs


@torch.no_grad()
def attn_forward(model, xn, xc):
    """Như S.SFGAttn.forward, trả thêm γ (n x G) và ma trận attention A (n x G x G)."""
    model.eval()
    E = torch.stack(model.encode(xn, xc), 1)
    A = torch.softmax(model.q(E) @ model.k(E).transpose(1, 2) / math.sqrt(E.shape[-1]), -1)
    Eo = model.ln(E + A @ model.v(E))
    gamma = torch.softmax(model.pool(Eo).squeeze(-1), 1)
    return model.cls((gamma[..., None] * Eo).sum(1)), gamma, A


def deletion_drops(predict, xn, xc, bn, bc, groups, pred, K, seed):
    """drop[i, g] = p(pred_i | x_i) - TB_k p(pred_i | x_i với nhóm g lấy từ mẫu nền b_k)."""
    n, ar = len(xn), np.arange(len(xn))
    bidx = np.random.RandomState(seed).randint(0, len(bn), size=K)
    base = softmax(np.asarray(predict(xn, xc), dtype=np.float64))[ar, pred]
    drops = np.zeros((n, len(groups)))
    for g, (ni, ci) in enumerate(groups):
        Xn, Xc = np.repeat(xn, K, 0), np.repeat(xc, K, 0)
        src = np.tile(bidx, n)
        if len(ni):
            Xn[:, ni] = bn[src][:, ni]
        if len(ci):
            Xc[:, ci] = bc[src][:, ci]
        p = softmax(np.asarray(predict(Xn, Xc), dtype=np.float64)).reshape(n, K, -1)[ar, :, pred].mean(1)
        drops[:, g] = base - p
    return drops


def kendall_tau_b(a, b):
    conc = disc = ta = tb = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return (conc - disc) / den if den > 0 else NAN


def attn_faith(model, dev, xn, xc, bn, bc, groups, P, K, seed):
    """Faithfulness của attention: xếp hạng nhóm theo γ và theo A (TB attention nhận được) so với |φ| Shapley
    của chính SFG-Attn cho lớp dự đoán; thí nghiệm xóa nhóm hạng 1."""
    logits, gamma, A = attn_forward(model, torch.as_tensor(xn, device=dev), torch.as_tensor(xc, device=dev))
    logits = logits.double().cpu().numpy()
    sc = {"gamma": gamma.double().cpu().numpy(), "attn": A.mean(1).double().cpu().numpy()}
    pred, n, ar = logits.argmax(1), len(xn), np.arange(len(xn))
    sc["phi"] = np.abs(P[ar, :, pred])
    drops = deletion_drops(nn_predict(model, dev), xn, xc, bn, bc, groups, pred, K, seed)
    rnd = np.random.RandomState(seed + 1).randint(0, len(groups), size=n)
    res = {"n": int(n), "drop_random": float(drops[ar, rnd].mean())}
    for k in ("gamma", "attn"):
        res[f"top1_{k}"] = float((sc[k].argmax(1) == sc["phi"].argmax(1)).mean())
        taus = [t for t in (kendall_tau_b(sc[k][i], sc["phi"][i]) for i in range(n)) if math.isfinite(t)]
        res[f"tau_{k}"], res[f"tau_{k}_n"] = (float(np.mean(taus)) if taus else NAN), len(taus)
    for k in ("phi", "gamma", "attn"):
        res[f"drop_{k}"] = float(drops[ar, sc[k].argmax(1)].mean())
    return res


def explain_run(kind, model, ctx, cfg, dev):
    D, T, (_, per), gr = ctx["D"], ctx["T"], ctx["ES"], ctx["g_np"]
    out = {"profile_population": D.get("profile_population", "all_test"), "counts": [int(len(i)) for i in per]}
    if kind in ("sfg", "sfg_p0"):
        out.update(closed_explain(model, D, T, ctx["ES"], dev))
        out["exact_err"] = S.exactness(model, D, T, cfg, dev)
        if kind == "sfg":
            out["mc_profile"] = mc_explain(nn_predict(model, dev), ctx["Z_te"], ctx["C_te"], ctx["Z_bg"],
                                           ctx["C_bg"], gr, per, cfg)[0]
    elif kind == "attn":
        prof, Ps, idxs = mc_explain(nn_predict(model, dev), ctx["Z_te"], ctx["C_te"], ctx["Z_bg"], ctx["C_bg"],
                                    gr, per, cfg)
        ii = np.concatenate(idxs)
        out["mc_profile"] = prof
        out["faith"] = attn_faith(model, dev, ctx["Z_te"][ii], ctx["C_te"][ii], ctx["Z_bg"], ctx["C_bg"], gr,
                                  np.concatenate(Ps), cfg["n_del"], DEL_SEED)
    elif kind == "xgb":
        out["mc_profile"] = mc_explain(xgb_predict(model, D), ctx["R_te"], ctx["C_te"], ctx["R_bg"],
                                       ctx["C_bg"], gr, per, cfg)[0]
    return out


# ----------------------------------------------------------------------------
# Chạy
# ----------------------------------------------------------------------------
def make_ctx(ds, D, cfg, dev):
    T = {f"{a}_{k}": torch.as_tensor(D[b][k], device=dev)
         for a, b in (("xn", "Z"), ("xc", "C"), ("y", "y")) for k in ("tr", "va", "te")}
    T["cw"] = torch.as_tensor(D["cw"], device=dev)
    ES = explain_set(D, cfg)
    bi, sizes = ES[0], [len(v) for v in D["groups"].values()]
    groupings = {"tax": D["groups"]}
    groupings.update({f"rand{r}": S.random_grouping(D["feats"], sizes, r) for r in cfg["rand_seeds"]})
    raw = {k: D["F"][k][D["nums"]].to_numpy(np.float32) for k in ("tr", "te")}
    return dict(ds=ds, D=D, T=T, ES=ES, groupings=groupings,
                gidx={g: S.group_index(D, v) for g, v in groupings.items()},
                g_np=[(ni.numpy(), ci.numpy()) for ni, ci in S.group_index(D, D["groups"])],
                Z_te=D["Z"]["te"], C_te=D["C"]["te"], Z_bg=D["Z"]["tr"][bi], C_bg=D["C"]["tr"][bi],
                R_te=raw["te"], R_bg=raw["tr"][bi])


def do_run(R, save, ctx, kind, grouping, seed, cfg, dev, sigma=None, p=0.0, key=None, explain=False,
           models_dir=None):
    ds, D, T, C = ctx["ds"], ctx["D"], ctx["T"], ctx["D"]["n_cls"]
    key = key or f"{ds}|{kind}|{grouping}|{seed}"
    if key in R:
        return R[key]
    TRAINED[0] += 1
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kind in TREES:
        model, info, pv, pt = run_tree(kind, D, seed, cfg, dev)
        r = dict(val_f1=S.macro_f1(D["y"]["va"], pv, C), **metrics(D["y"]["te"], pt, C), **info)
    else:
        model = build(kind, D, ctx["gidx"][grouping], cfg, sigma).to(dev)
        best, ep = S.train_nn(model, T, D, dev, seed, cfg, p)
        pt = S.predict(model, T["xn_te"], T["xc_te"])
        r = dict(val_f1=best, **metrics(D["y"]["te"], pt, C), epochs=ep, sigma=sigma, p_swap=p)
    if explain:
        try:
            r.update(explain_run(kind, model, ctx, cfg, dev))
        except Exception as e:     # lỗi giải thích -> ghi lại, không bịa kết quả
            r["explain_error"] = f"{type(e).__name__}: {e}"
            print(f"  [{ds}] LỖI khi tính lời giải thích ({key}): {r['explain_error']}", flush=True)
        if models_dir is not None:
            models_dir.mkdir(parents=True, exist_ok=True)
            f = models_dir / key.replace("|", "_")
            if kind == "xgb":
                model.save_model(str(f) + ".ubj")
            else:
                torch.save(model.state_dict(), str(f) + ".pt")
    r["seconds"] = round(time.time() - t0, 1)
    R[key] = r
    save()
    extra = f"{r['epochs']} epoch" if "epochs" in r else f"{r['trees']} cây"
    print(f"  [{ds}] {key.split('|', 1)[1]}: val {r['val_f1']:.2f} | test {r['test_f1']:.2f} | {extra} | "
          f"{r['seconds']}s", flush=True)
    return r


def run_all(Ds, R, cfg, dev, save, models_dir=None):
    """Thứ tự theo quyết định giao thức: chọn σ, p -> A5 (UNSW) -> kịch bản 1 -> A1, A3, A6 -> A4."""
    ctxs = {ds: make_ctx(ds, D, cfg, dev) for ds, D in Ds.items()}
    for ds, c in ctxs.items():
        R[f"{ds}|groupings"] = c["groupings"]
    s0, sp = cfg["seeds"][0], {}

    def block(ds, kinds, sigma, p, grouping="tax", explain_kinds=("sfg", "sfg_p0", "attn", "xgb")):
        for kind in kinds:
            for seed in cfg["seeds"]:
                do_run(R, save, ctxs[ds], kind, grouping, seed, cfg, dev, sigma=sigma,
                       p=p if kind in ("sfg", "sfg_main", "sfg_lr") else 0.0,
                       explain=grouping == "tax" and kind in explain_kinds, models_dir=models_dir)

    for ds in ("unsw", "nsl"):
        if ds not in ctxs:
            continue
        for sg in cfg["sigmas"]:
            do_run(R, save, ctxs[ds], "flat", "tax", s0, cfg, dev, sigma=sg, key=f"{ds}|sel|sigma{sg}")
        sigma = max(cfg["sigmas"], key=lambda v: R[f"{ds}|sel|sigma{v}"]["val_f1"])
        for pv in cfg["p_grid"]:
            do_run(R, save, ctxs[ds], "sfg", "tax", s0, cfg, dev, sigma=sigma, p=pv, key=f"{ds}|sel|p{pv}")
        p = max(cfg["p_grid"], key=lambda v: R[f"{ds}|sel|p{v}"]["val_f1"])
        R[f"{ds}|sigma"], R[f"{ds}|p"] = sigma, p
        sp[ds] = (sigma, p)
        print(f"  [{ds}] chọn trên validation: σ = {sigma}, p = {p}", flush=True)
        if ds == "unsw" and "unsw_nottl" in ctxs:
            R["unsw_nottl|sigma"], R["unsw_nottl|p"] = sigma, p
            block("unsw_nottl", ["xgb", "flat", "sfg"], sigma, p, explain_kinds=("sfg",))
        block(ds, MAIN, sigma, p)
        block(ds, ABL, sigma, p)
    for ds, (sigma, p) in sp.items():
        for r in cfg["rand_seeds"]:
            block(ds, ["sfg"], sigma, p, grouping=f"rand{r}")


# ----------------------------------------------------------------------------
# Tổng hợp
# ----------------------------------------------------------------------------
def stability(profs, counts):
    """Độ ổn định hồ sơ đóng góp nhóm qua các seed, chỉ xét lớp có >= 30 mẫu được giải thích. Chỉ để mô tả,
    không phải tiêu chí đạt/không đạt. top1_same: tỷ lệ lớp có cùng nhóm hạng 1 ở MỌI seed; top1_majority: tỷ lệ
    lớp có một nhóm đứng hạng 1 ở hơn một nửa số seed (5 seed: >= 3); kendall: Kendall τ-b TB giữa mọi cặp seed;
    share_sd: SD (ddof=1) giữa các seed của tỷ trọng từng nhóm, TB trên các lớp và các nhóm."""
    if profs is None or len(profs) < 2:
        return {"status": "NOT_EVALUABLE", "reason": "cần hồ sơ của >= 2 seed"}
    valid = np.asarray(counts) >= 30
    if not valid.any():
        return {"status": "NOT_EVALUABLE", "reason": "không lớp nào có >= 30 mẫu được giải thích"}
    Pv = [np.asarray(p, dtype=float)[valid] for p in profs]
    if not all(np.isfinite(p).all() for p in Pv):
        return {"status": "NOT_EVALUABLE", "reason": "hồ sơ có NaN"}
    top = np.stack([p.argmax(1) for p in Pv])
    taus = [kendall_tau_b(Pv[a][c], Pv[b][c]) for c in range(len(Pv[0]))
            for a, b in itertools.combinations(range(len(Pv)), 2)]
    if not all(math.isfinite(t) for t in taus):
        return {"status": "NOT_EVALUABLE", "reason": "τ không xác định (hồ sơ hằng)"}
    ns, G = len(Pv), Pv[0].shape[1]
    maj = [np.bincount(top[:, c], minlength=G).max() > ns / 2 for c in range(top.shape[1])]
    return {"status": "OK", "top1_same": float((top == top[0]).all(0).mean()), "top1_majority": float(np.mean(maj)),
            "kendall": float(np.mean(taus)), "share_sd": float(np.std(np.stack(Pv), axis=0, ddof=1).mean()),
            "n_classes": int(valid.sum())}


def lee_stolfo(profs, classes, gnames):
    need = ["R2L", "U2R", "DoS", "Probe"]
    if not profs or not set(need) <= set(classes) or not {"Content", "Time", "Host"} <= set(gnames):
        return {"status": "NOT_EVALUABLE", "reason": "thiếu hồ sơ, lớp hoặc nhóm"}
    ci = {c: i for i, c in enumerate(classes)}
    gi, gt, gh = (gnames.index(x) for x in ("Content", "Time", "Host"))
    a, b, t = [], [], []
    for p in (np.asarray(p, dtype=float) for p in profs):
        if not np.isfinite(p[[ci[c] for c in need]]).all():
            return {"status": "NOT_EVALUABLE", "reason": "hồ sơ có NaN ở R2L/U2R/DoS/Probe"}
        a.append((p[ci["R2L"], gi] + p[ci["U2R"], gi]) / 2)
        b.append((p[ci["DoS"], gi] + p[ci["Probe"], gi]) / 2)
        t.append((p[ci["DoS"], gt] + p[ci["DoS"], gh] + p[ci["Probe"], gt] + p[ci["Probe"], gh]) / 2)
    ma, mb, mt = float(np.mean(a)), float(np.mean(b)), float(np.mean(t))
    return {"status": "OK", "content_r2l_u2r": ma, "content_dos_probe": mb, "traffic_dos_probe": mt,
            "H1": "SUPPORTED" if ma > mb else "NOT_SUPPORTED", "H2": "SUPPORTED" if mt > mb else "NOT_SUPPORTED",
            "H1_seeds": int(sum(x > y for x, y in zip(a, b))), "H2_seeds": int(sum(x > y for x, y in zip(t, b))),
            "n_seeds": len(a)}


def summarize(R, cfg, dss, out_dir):
    seeds = cfg["seeds"]
    rep = {"version": SCRIPT_VERSION, "config": R.get("config"), "selection": {}, "k1": {}, "k2": {}, "k3": {},
           "k4": {}}

    def vals(ds, m, g="tax", f="test_f1"):
        ks = [f"{ds}|{m}|{g}|{s}" for s in seeds]
        if not all(k in R for k in ks):
            return None
        v = [R[k][f] for k in ks]
        return v if all(math.isfinite(x) for x in v) else None

    def ms(v):
        if not v:
            return (NAN, NAN)
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else NAN)

    def fmt(t, w=6):
        a, b = (format(x, f"{w}.2f") if math.isfinite(x) else "n/a".rjust(w) for x in t)
        return f"{a} ± {b}"

    def gather(ds, m, field):
        recs = [R.get(f"{ds}|{m}|tax|{s}") for s in seeds]
        if any(r is None or "explain_error" in r or field not in r for r in recs):
            return None, None
        return [r[field] for r in recs], recs[0]["counts"]

    dsm = [d for d in ("unsw", "nsl") if d in dss]
    print("\n================ KỊCH BẢN 1 - HIỆU NĂNG (macro-F1 % trên test, TB ± SD "
          f"{len(seeds)} seed; Δ = mô hình - SFG-XIDS, ghép cặp theo seed) ================")
    for ds in dsm:
        rep["selection"][ds] = {"sigma": R.get(f"{ds}|sigma"), "p": R.get(f"{ds}|p")}
        ref, rows = vals(ds, "sfg"), {}
        print(f"\n{ds.upper()} (σ = {R.get(f'{ds}|sigma')}, p = {R.get(f'{ds}|p')}):")
        for m in MAIN:
            v, acc = vals(ds, m), vals(ds, m, f="test_acc")
            d = ms([x - y for x, y in zip(v, ref)]) if v and ref and m != "sfg" else (NAN, NAN)
            rows[m] = {"f1": ms(v), "acc": ms(acc), "delta_vs_sfg": d}
            print(f"  {NAMES[m]:<16} F1 {fmt(ms(v))}   acc {fmt(ms(acc))}"
                  + (f"   Δ {fmt(d)}" if m != "sfg" else ""))
        rep["k1"][ds] = rows

    print("\n================ KỊCH BẢN 2 - ABLATION (Δ = biến thể - SFG-XIDS đầy đủ, ghép cặp theo seed) "
          "================")
    for ds in dsm:
        ref, rows = vals(ds, "sfg"), {}
        print(f"\n{ds.upper()}: SFG-XIDS đầy đủ F1 {fmt(ms(ref))}")
        for m in ABL:
            v = vals(ds, m)
            d = ms([x - y for x, y in zip(v, ref)]) if v and ref else (NAN, NAN)
            rows[m] = {"f1": ms(v), "delta": d}
            print(f"  {NAMES[m]:<24} F1 {fmt(ms(v))}   Δ {fmt(d)}")
        rv = [vals(ds, "sfg", f"rand{r}") for r in cfg["rand_seeds"]]
        if ref and all(rv):
            D_s = [ref[i] - float(np.mean([v[i] for v in rv])) for i in range(len(seeds))]
            rows["A4"] = {"f1_pooled": ms([x for v in rv for x in v]), "D": ms(D_s)}
            print(f"  A4 chia ngẫu nhiên ({len(rv)} cách x {len(seeds)} seed) F1 {fmt(ms([x for v in rv for x in v]))}"
                  f"   D = taxonomy - TB các cách chia cùng seed: {fmt(ms(D_s))}")
        rep["k2"][ds] = rows
    if "unsw" in dss and "unsw_nottl" in dss:
        print("\nA5 bỏ TTL (UNSW-NB15), Δ = bỏ TTL - đầy đủ, cùng mô hình, ghép cặp theo seed:")
        rep["k2"]["A5"] = {}
        for m in ("xgb", "flat", "sfg"):
            v5, vf = vals("unsw_nottl", m), vals("unsw", m)
            d = ms([x - y for x, y in zip(v5, vf)]) if v5 and vf else (NAN, NAN)
            rep["k2"]["A5"][m] = {"f1_nottl": ms(v5), "delta": d}
            print(f"  {NAMES[m]:<16} F1 bỏ TTL {fmt(ms(v5))}   Δ {fmt(d)}")

    print("\n================ KỊCH BẢN 3 - HỒ SƠ ĐÓNG GÓP NHÓM (SFG-XIDS, φ dạng đóng, % TB ± SD giữa các "
          "seed) ================")
    prof_rows, heat_rows = [], []
    for ds in [d for d in ("unsw", "unsw_nottl", "nsl") if d in dss]:
        profs, counts = gather(ds, "sfg", "profile")
        heats, _ = gather(ds, "sfg", "heat")
        sums, _ = gather(ds, "sfg", "phi_abs_sum")
        if profs is None:
            print(f"\n{ds.upper()}: thiếu hồ sơ (xem explain_error trong results.json)")
            continue
        cls, gn = R[f"{ds}|classes"], R[f"{ds}|group_names"]
        pn = [f"{gn[i][:5]}×{gn[j][:5]}" for i, j in itertools.combinations(range(len(gn)), 2)]
        Pst = np.stack([np.asarray(p, float) for p in profs])
        mp = Pst.mean(0)
        sp = Pst.std(0, ddof=1) if len(profs) > 1 else np.full_like(mp, NAN)
        print(f"\n{ds.upper()}:  lớp (n) | " + " ".join(f"{g[:10]:>10}" for g in gn)
              + " | hạng 1 theo seed | cặp tương tác mạnh nhất (|g| / Σ|φ|)")
        rep["k3"][ds] = {}
        for c, name in enumerate(cls):
            tops = "/".join(gn[int(np.argmax(p[c]))][:4] if np.isfinite(p[c]).all() else "-" for p in profs)
            rel = [np.asarray(h[c], float) / s[c] for h, s in zip(heats, sums)
                   if len(h[c]) and math.isfinite(s[c]) and s[c] > 0]
            strong = ""
            if rel and len(rel) == len(seeds):
                mr = np.mean(rel, 0)
                strong = f"{pn[int(np.argmax(mr))]} {100 * float(np.max(mr)):.1f}%"
                for s, h, sm in zip(seeds, heats, sums):
                    for k, nm in enumerate(pn):
                        heat_rows.append({"ds": ds, "seed": s, "class": name, "pair": nm, "mean_abs_g": h[c][k],
                                          "rel_to_sum_abs_phi": h[c][k] / sm[c]})
            print(f"  {name[:12]:<12} ({counts[c]:>3}) | "
                  + " ".join(f"{100 * v:5.1f}±{100 * d:4.1f}" for v, d in zip(mp[c], sp[c])) + f" | {tops} | {strong}")
            rep["k3"][ds][name] = {"profile_mean": mp[c].tolist(), "profile_sd": sp[c].tolist(),
                                   "top1_by_seed": tops, "strongest_pair": strong}

    print("\n================ KỊCH BẢN 4 - KIỂM CHỨNG GIẢI THÍCH ================")
    ex = [v["exact_err"] for k, v in R.items() if isinstance(v, dict) and "exact_err" in v]
    se = [v["sum_err"] for k, v in R.items() if isinstance(v, dict) and "sum_err" in v]
    mex, mse = (float(np.max(ex)) if ex else NAN), (float(np.max(se)) if se else NAN)
    ok = all(math.isfinite(x) and x < S.PHI_TOL for x in (mex, mse))
    n_err = sum(1 for v in R.values() if isinstance(v, dict) and "explain_error" in v)
    print(f"  φ dạng đóng vs vét cạn: max {mex:.1e}; completeness: max {mse:.1e} ({len(ex)} lần chạy) -> "
          + ("OK" if ok else "KHÔNG ĐẠT hoặc thiếu") + f"; lần chạy có lỗi giải thích: {n_err}")
    rep["k4"]["exactness"] = {"max_exact_err": mex, "max_sum_err": mse, "ok": ok, "explain_errors": n_err}
    for ds in dsm:
        cls, gn = R[f"{ds}|classes"], R[f"{ds}|group_names"]
        a, _ = gather(ds, "sfg", "profile")
        b, counts = gather(ds, "sfg", "mc_profile")
        rep["k4"][ds] = {}
        if a and b:
            valid = np.asarray(counts) >= 30
            l1 = [float(np.abs(np.asarray(x, float)[c] - np.asarray(y, float)[c]).sum())
                  for x, y in zip(a, b) for c in range(len(cls)) if valid[c]]
            l1 = [v for v in l1 if math.isfinite(v)]
            mv = float(np.mean(l1)) if l1 else NAN
            rep["k4"][ds]["mc_vs_closed_L1"] = mv
            print(f"\n{ds.upper()}: Monte Carlo vs dạng đóng (SFG-XIDS), khoảng cách L1 TB giữa hai hồ sơ: "
                  + (f"{mv:.3f}" if math.isfinite(mv) else "n/a"))
        else:
            print(f"\n{ds.upper()}:")
        print("  Độ ổn định qua các seed (lớp >= 30 mẫu): tỷ lệ lớp giữ nhóm hạng 1 ở mọi seed | ở đa số seed | "
              "Kendall τ TB | SD tỷ trọng TB")
        rep["k4"][ds]["stability"], rep["k4"][ds]["lee_stolfo"] = {}, {}
        for label, m, field in EXPLAINERS:
            profs, counts = gather(ds, m, field)
            st = stability(profs, counts)
            rep["k4"][ds]["stability"][label] = st
            print(f"    {label:<26} " + (
                f"{100 * st['top1_same']:4.0f}% | {100 * st['top1_majority']:4.0f}% | τ = {st['kendall']:.2f} | "
                f"SD {100 * st['share_sd']:.1f} điểm %" if st["status"] == "OK" else f"NOT_EVALUABLE ({st['reason']})"))
        if ds == "nsl":
            print("  Thiết kế Lee & Stolfo: H1 Content ở R2L/U2R > DoS/Probe; H2 Time+Host ở DoS/Probe > Content")
            for label, m, field in EXPLAINERS:
                profs, _ = gather(ds, m, field)
                h = lee_stolfo(profs, cls, gn)
                rep["k4"][ds]["lee_stolfo"][label] = h
                print(f"    {label:<26} " + (
                    f"H1 {h['H1']} ({100 * h['content_r2l_u2r']:.1f}% vs {100 * h['content_dos_probe']:.1f}%, "
                    f"{h['H1_seeds']}/{h['n_seeds']} seed); H2 {h['H2']} ({100 * h['traffic_dos_probe']:.1f}%, "
                    f"{h['H2_seeds']}/{h['n_seeds']} seed)" if h["status"] == "OK" else f"NOT_EVALUABLE ({h['reason']})"))
        fa, _ = gather(ds, "attn", "faith")
        if fa:
            keys = ["top1_gamma", "top1_attn", "tau_gamma", "tau_attn", "drop_phi", "drop_gamma", "drop_attn",
                    "drop_random"]
            agg = {k: ms([f[k] for f in fa if math.isfinite(f[k])]) for k in keys}
            rep["k4"][ds]["attention_faithfulness"] = agg
            print("  Faithfulness của attention (SFG-Attn, lớp dự đoán; TB ± SD các seed):")
            print(f"    trùng nhóm hạng 1 với |φ|: γ {fmt(agg['top1_gamma'], 5)} | A {fmt(agg['top1_attn'], 5)}"
                  f"   Kendall τ với |φ|: γ {fmt(agg['tau_gamma'], 5)} | A {fmt(agg['tau_attn'], 5)}")
            print(f"    xóa nhóm hạng 1, xác suất lớp dự đoán giảm: theo |φ| {fmt(agg['drop_phi'], 5)} | theo γ "
                  f"{fmt(agg['drop_gamma'], 5)} | theo A {fmt(agg['drop_attn'], 5)} | nhóm ngẫu nhiên "
                  f"{fmt(agg['drop_random'], 5)}")

    run_rows, pc_rows = [], []
    for k, v in R.items():
        parts = k.split("|")
        if len(parts) == 4 and isinstance(v, dict) and "test_f1" in v:
            run_rows.append({"ds": parts[0], "model": parts[1], "grouping": parts[2], "seed": parts[3],
                             **{f: v.get(f) for f in ("val_f1", "test_f1", "test_acc", "test_prec", "test_rec",
                                                      "epochs", "trees", "sigma", "p_swap", "seconds")}})
            for c, f1v in zip(R.get(f"{parts[0]}|classes", []), v.get("f1_per_class", [])):
                pc_rows.append({"ds": parts[0], "model": parts[1], "grouping": parts[2], "seed": parts[3],
                                "class": c, "f1": f1v})
            for field in ("profile", "mc_profile"):
                if field in v:
                    for c, (name, row) in enumerate(zip(R[f"{parts[0]}|classes"], v[field])):
                        prof_rows.append({"ds": parts[0], "model": parts[1], "field": field, "seed": parts[3],
                                          "class": name, "count": v["counts"][c],
                                          **dict(zip(R[f"{parts[0]}|group_names"], row))})
    for name, rows in (("runs", run_rows), ("per_class_f1", pc_rows), ("profiles", prof_rows),
                       ("heatmaps", heat_rows)):
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\nĐã ghi summary.json, runs.csv, per_class_f1.csv, profiles.csv, heatmaps.csv vào {out_dir}")
    return rep


def check_config(R, conf):
    """Không trộn kết quả của hai cấu hình khác nhau trong cùng một thư mục."""
    old = R.get("config", conf)
    return None if old == conf else f"thư mục chứa kết quả của cấu hình khác: {old}"


# ----------------------------------------------------------------------------
# Tự kiểm tra (chỉ phần mới; phần dùng lại từ step0_sfg.py đã có selftest riêng)
# ----------------------------------------------------------------------------
def selftest():
    """"SELFTEST: PASS" chỉ nghĩa các kiểm tra cài đặt đạt, không nói gì về kết quả thực nghiệm."""
    import warnings
    from scipy.stats import kendalltau
    results = []

    def check(name, cond):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name, flush=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check("A5: bỏ đúng 3 cột TTL, nhóm 13/8/7/11",
              [len(v) for v in NOTTL_GROUPS.values()] == [13, 8, 7, 11]
              and not set(TTL) & {c for v in NOTTL_GROUPS.values() for c in v})
        feats = [c for v in S.UNSW_GROUPS.values() for c in v]
        rg = [S.random_grouping(feats, [15, 8, 7, 12], r) for r in RAND_SEEDS]
        check("A4: 5 cách chia ngẫu nhiên khác nhau, đúng kích thước, mỗi cách phủ đủ 42 đặc trưng một lần",
              all([len(v) for v in g.values()] == [15, 8, 7, 12] and sorted(c for v in g.values() for c in v)
                  == sorted(feats) for g in rg) and len({json.dumps(g, sort_keys=True) for g in rg}) == 5)

        torch.manual_seed(0)
        cfg = dict(CFG, d=8, d_emb=4, n_freq=4)
        E0 = torch.tensor([], dtype=torch.long)
        gidx = [(torch.tensor([0, 1]), torch.tensor([0])), (torch.tensor([2]), E0),
                (torch.tensor([3, 4]), torch.tensor([1])), (torch.tensor([5]), E0)]
        gnp = [(ni.numpy(), ci.numpy()) for ni, ci in gidx]
        m = S.SFGXIDS(6, [3, 4], gidx, 3, cfg, 1.0).double()
        xn, bn = torch.randn(7, 6, dtype=torch.float64), torch.randn(4, 6, dtype=torch.float64)
        xc = torch.stack([torch.randint(0, 3, (7,)), torch.randint(0, 4, (7,))], 1)
        bc = torch.stack([torch.randint(0, 3, (4,)), torch.randint(0, 4, (4,))], 1)
        cls = torch.randint(0, 3, (7,))
        Pf, Gf, err = phi_g(m, xn, xc, bn, bc, chunk=3)
        ref, _ = S.group_phi(m, xn, xc, bn, bc, cls, chunk=3)
        d1 = (Pf[torch.arange(7), :, cls] - ref).abs().max().item()
        check(f"phi_g trùng S.group_phi (max|Δ| = {d1:.1e}) và completeness (max|Δ| = {err:.1e})",
              d1 < 1e-12 and err < 1e-9 and Gf.shape == (7, 6, 3))
        mm = SFGMain(6, [3, 4], gidx, 3, cfg, 1.0).double()
        a, _ = S.group_phi(mm, xn, xc, bn, bc, cls)
        d2 = (a - S.group_phi_brute(mm, xn, xc, bn, bc, cls)).abs().max().item()
        check(f"A1 (không v_ij): φ dạng đóng = Shapley vét cạn (max|Δ| = {d2:.1e})",
              d2 < 1e-9 and len(mm.pairs) == 0 and phi_g(mm, xn, xc, bn, bc)[1] is None)
        f = nn_predict(m, torch.device("cpu"))
        Pe = mc_shapley(f, xn.numpy(), xc.numpy(), bn.numpy(), bc.numpy(), gnp, None, 0)
        d3 = float(np.abs(Pe - Pf.numpy()).max())
        check(f"Shapley hộp đen, liệt kê đủ tổ hợp nền = φ dạng đóng (max|Δ| = {d3:.1e})", d3 < 1e-9)
        worst, worst_s0 = 0.0, 0.0
        for n_, ch, B_ in ((5, 128, 9), (11, 4, 2), (1, 1, 3), (9, 2, 9)):
            xa, ba = torch.randn(n_, 6, dtype=torch.float64), torch.randn(B_, 6, dtype=torch.float64)
            ca = torch.stack([torch.randint(0, 3, (n_,)), torch.randint(0, 4, (n_,))], 1)
            cb = torch.stack([torch.randint(0, 3, (B_,)), torch.randint(0, 4, (B_,))], 1)
            Pa = phi_g(m, xa, ca, ba, cb, chunk=ch)[0]
            Pb = mc_shapley(f, xa.numpy(), ca.numpy(), ba.numpy(), cb.numpy(), gnp, None, 0)
            ka = torch.randint(0, 3, (n_,))
            worst = max(worst, float(np.abs(Pa.numpy() - Pb).max()))
            worst_s0 = max(worst_s0, (Pa[torch.arange(n_), :, ka]
                                      - S.group_phi(m, xa, ca, ba, cb, ka, chunk=ch)[0]).abs().max().item())
        check(f"phi_g đúng với mọi kích thước lô và nền (n < B, n > B, lô cuối lẻ, n = 1): max|Δ| so với liệt kê "
              f"đủ = {worst:.1e}, so với S.group_phi = {worst_s0:.1e}", worst < 1e-9 and worst_s0 < 1e-12)
        Pm = mc_shapley(f, xn.numpy(), xc.numpy(), bn.numpy(), bc.numpy(), gnp, 4000, 1)
        d4, sc = float(np.abs(Pm - Pe).max()), float(np.abs(Pe).max())
        check(f"Shapley Monte Carlo (M = 4000) gần giá trị chính xác (max|Δ| = {d4:.3f}, max|φ| = {sc:.3f})",
              d4 < 0.1 * sc)
        ma = S.SFGAttn(6, [3, 4], gidx, 3, cfg, 1.0)
        lg, gm, A = attn_forward(ma, xn.float(), xc)
        check("attn_forward cho đúng logit của SFGAttn; γ và mỗi hàng của A có tổng 1",
              torch.allclose(lg, ma(xn.float(), xc), atol=1e-6) and torch.allclose(gm.sum(1), torch.ones(7))
              and torch.allclose(A.sum(2), torch.ones(7, 4)))
        xnn, xcc = xn.float().numpy(), xc.numpy()
        dr = deletion_drops(nn_predict(ma, torch.device("cpu")), xnn[:1], xcc[:1], np.repeat(xnn[:1], 5, 0),
                            np.repeat(xcc[:1], 5, 0), gnp, np.array([0]), 3, 0)
        check("xóa nhóm: nền trùng hệt mẫu -> mức giảm đúng bằng 0", np.abs(dr).max() < 1e-7)
        rng = np.random.RandomState(0)
        pairs = [(rng.randint(0, 3, 4), rng.randint(0, 3, 4)) for _ in range(300)]
        pairs = [(u, v) for u, v in pairs if len(set(u)) > 1 and len(set(v)) > 1]
        d5 = max(abs(kendall_tau_b(u, v) - kendalltau(u, v)[0]) for u, v in pairs)
        check(f"Kendall τ-b khớp scipy trên {len(pairs)} cặp có hạng bằng nhau (max|Δ| = {d5:.1e})", d5 < 1e-12)
        p1 = [[.4, .3, .2, .1], [.1, .2, .3, .4], [.25, .25, .3, .2]]
        p2 = [[.4, .3, .2, .1], [.2, .1, .3, .4], [.3, .25, .25, .2]]
        st = stability([p1, p2], [100, 100, 10])
        check("độ ổn định: lớp < 30 mẫu bị bỏ; 2 lớp giữ hạng 1 (mọi seed và đa số); τ = TB(1, 2/3); "
              "SD tỷ trọng = 0.2 / (8·√2)",
              st["status"] == "OK" and st["top1_same"] == 1.0 and st["top1_majority"] == 1.0
              and abs(st["kendall"] - 5 / 6) < 1e-12 and abs(st["share_sd"] - 0.2 / (8 * math.sqrt(2))) < 1e-12
              and st["n_classes"] == 2)

        def row(g):
            return [.4 if i == g else .2 for i in range(4)]

        st5 = stability([[row(a), row(b)] for a, b in ((0, 0), (0, 0), (0, 1), (0, 1), (1, 2))], [100, 100])
        check("độ ổn định 5 seed: lớp có hạng 1 ở 4/5 seed tính là đa số, lớp chia 2/2/1 thì không; "
              "không lớp nào giữ hạng 1 ở mọi seed",
              st5["status"] == "OK" and st5["top1_same"] == 0.0 and st5["top1_majority"] == 0.5)
        check("độ ổn định: NaN, 1 seed, không lớp nào >= 30 mẫu -> NOT_EVALUABLE",
              stability([p1, [[NAN] * 4] * 3], [100] * 3)["status"] == "NOT_EVALUABLE"
              and stability([p1], [100] * 3)["status"] == "NOT_EVALUABLE"
              and stability([p1, p2], [10] * 3)["status"] == "NOT_EVALUABLE")
        cl, gn = ["DoS", "Normal", "Probe", "R2L", "U2R"], ["Basic", "Content", "Time", "Host"]
        good = [[.1, .1, .4, .4], [.25] * 4, [.1, .1, .4, .4], [.1, .7, .1, .1], [.1, .7, .1, .1]]
        bad = [[.1, .7, .1, .1], [.25] * 4, [.1, .7, .1, .1], [.1, .1, .4, .4], [.1, .1, .4, .4]]
        h1, h2, h3 = lee_stolfo([good] * 3, cl, gn), lee_stolfo([bad] * 3, cl, gn), lee_stolfo([good[:4] + [[NAN] * 4]], cl, gn)
        check("Lee & Stolfo: H1, H2 SUPPORTED / NOT_SUPPORTED đúng; NaN -> NOT_EVALUABLE",
              (h1["H1"], h1["H2"], h2["H1"], h2["H2"], h3["status"])
              == ("SUPPORTED", "SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED", "NOT_EVALUABLE"))
        c0 = dict(version=1, quick=False)
        check("cấu hình khác trong cùng thư mục -> bị từ chối",
              check_config({"config": c0}, c0) is None and check_config({"config": c0}, dict(c0, quick=True)))

        if xgb is None or lgb is None:
            check("xgboost và lightgbm đã cài (cần cho chạy thử đường ống)", False)
        else:
            rs = np.random.RandomState(0)
            uf = [c for v in S.UNSW_GROUPS.values() for c in v]

            def fake_unsw(nr):
                df = pd.DataFrame({c: rs.exponential(5, nr) for c in uf if c not in S.UNSW_CAT})
                df["proto"], df["service"] = rs.choice(["tcp", "udp", "arp"], nr), rs.choice(["-", "http"], nr)
                df["state"] = rs.choice(["FIN", "INT", "CON"], nr)
                df["attack_cat"] = rs.choice(["Normal", "DoS", "Exploits", "Generic"], nr)
                return df

            def fake_nsl(nr, types):
                df = pd.DataFrame({c: rs.exponential(2, nr) for c in S.NSL_COLS})
                df["protocol_type"], df["service"] = rs.choice(["tcp", "udp"], nr), rs.choice(["http", "ftp"], nr)
                df["flag"] = rs.choice(["SF", "S0"], nr)
                df["attack_type"] = rs.choice(types, nr)
                df["y_name"] = df["attack_type"].map(S.NSL_MAP)
                return df

            seen = ["normal", "neptune", "ipsweep", "guess_passwd", "buffer_overflow"]
            utr, ute = fake_unsw(900), fake_unsw(400)
            ntr, nte = fake_nsl(900, seen), fake_nsl(700, seen + ["snmpguess", "ps"])
            Ds = {"unsw": S.prepare(utr, ute, "attack_cat", S.UNSW_GROUPS, S.UNSW_CAT),
                  "unsw_nottl": S.prepare(utr, ute, "attack_cat", NOTTL_GROUPS, S.UNSW_CAT)}
            pop = S.nsl_population(ntr["attack_type"], nte["attack_type"])
            Dn = S.prepare(ntr, nte, "y_name", S.NSL_GROUPS, S.NSL_CAT, keep_col="attack_type")
            Dn["explain_mask"] = np.isin(Dn["te_extra"], pop["evaluated"] + ["normal"])
            Dn["profile_population"] = "seen_train_types"
            Ds["nsl"] = Dn
            tcfg = dict(CFG, **QUICK)
            tcfg.update(max_ep=1, n_bg=8, n_per_class=40, n_exact=2, n_bg_exact=2, n_estimators=5, es_rounds=2,
                        n_mc=8, n_del=2, rf_trees=5, bs=256)
            cpu = torch.device("cpu")
            ctx = make_ctx("unsw", Ds["unsw"], tcfg, cpu)
            mt = build("sfg", ctx["D"], ctx["gidx"]["tax"], tcfg, 1.0)
            S.train_nn(mt, ctx["T"], ctx["D"], cpu, 0, tcfg, 0.2)
            mine = closed_explain(mt, ctx["D"], ctx["T"], ctx["ES"], cpu)["profile"]
            theirs = S.class_profiles(mt, ctx["D"], ctx["T"], tcfg, cpu)[0]
            check("closed_explain dùng đúng mẫu nền/mẫu test và cho cùng hồ sơ với S.class_profiles",
                  np.allclose(np.asarray(mine), np.asarray(theirs), atol=1e-6, equal_nan=True))
            Du = ctx["D"]
            mx = xgb.XGBClassifier(n_estimators=60, learning_rate=0.3, max_depth=4, tree_method="hist",
                                   device="cpu", enable_categorical=True, max_cat_to_onehot=1,
                                   objective="multi:softprob", eval_metric="mlogloss", early_stopping_rounds=3,
                                   random_state=0)
            mx.fit(Du["F"]["tr"], Du["y"]["tr"], eval_set=[(Du["F"]["va"], Du["y"]["va"])], verbose=False)
            fx = xgb_predict(mx, Du)
            dxg = float(np.abs(fx(ctx["R_te"], ctx["C_te"]) - mx.predict(Du["F"]["te"], output_margin=True)).max())
            cut = mx.best_iteration + 1 < mx.get_booster().num_boosted_rounds()
            check(f"xgb_predict (DMatrix, không DataFrame) = m.predict trên dòng gốc, đúng số cây tốt nhất "
                  f"(max|Δ| = {dxg:.1e}; early stopping đã cắt bớt cây: {cut})", dxg < 1e-5 and cut)
            rtr = Du["F"]["tr"][Du["nums"]].to_numpy(np.float32)
            ti, bj = np.arange(6), np.arange(10, 16)
            ni, ci = ctx["g_np"][0]                          # nhóm Basic: có cả cột số và cột phân loại
            Xn, Xc = ctx["R_te"][ti].copy(), ctx["C_te"][ti].copy()
            Xn[:, ni] = rtr[bj][:, ni]
            Xc[:, ci] = Du["C"]["tr"][bj][:, ci]
            gcols = set(Du["groups"]["Basic"])
            ref_df = pd.DataFrame({c: (Du["F"]["tr"][c].iloc[bj] if c in gcols else Du["F"]["te"][c].iloc[ti])
                                   .reset_index(drop=True) for c in Du["feats"]})
            dmix = float(np.abs(fx(Xn, Xc) - mx.predict(ref_df, output_margin=True)).max())
            check(f"xgb_predict trên dòng trộn nhóm (trộn theo chỉ số cột) = m.predict trên DataFrame trộn theo TÊN "
                  f"cột (max|Δ| = {dmix:.1e})", dmix < 1e-5 and len(ni) > 0 and len(ci) > 0)

            tmp = Path(tempfile.mkdtemp())
            R = {f"{ds}|{k}": v for ds, D in Ds.items()
                 for k, v in (("classes", D["classes"]), ("group_names", list(D["groups"])))}
            R["nsl|attack_types"] = pop
            print("--- chạy thử toàn bộ đường ống trên dữ liệu giả (CPU) ---", flush=True)
            n0 = TRAINED[0]
            run_all(Ds, R, tcfg, cpu, lambda: None, tmp / "models")
            n1 = TRAINED[0]
            s = tcfg["seeds"]
            need = ([f"{d}|{m}|tax|{x}" for d in ("unsw", "nsl") for m in MAIN + ABL for x in s]
                    + [f"unsw_nottl|{m}|tax|{x}" for m in ("xgb", "flat", "sfg") for x in s]
                    + [f"{d}|sfg|rand100|{x}" for d in ("unsw", "nsl") for x in s])
            check(f"chạy thử: đủ {len(need)} lần chạy chính (kể cả A5 và A4), chọn σ và p",
                  all(k in R for k in need) and all(f"{d}|{v}" in R for d in ("unsw", "nsl") for v in ("sigma", "p")))
            exp = [f"{d}|{m}|tax|{x}" for d in ("unsw", "nsl") for m in ("sfg", "sfg_p0", "attn", "xgb") for x in s]
            exp += [f"unsw_nottl|sfg|tax|{x}" for x in s]
            check("chạy thử: mọi lần chạy cần giải thích đều có kết quả, không có explain_error",
                  all("explain_error" not in R[k] and ("profile" in R[k] or "mc_profile" in R[k]) for k in exp))
            check("chạy thử: có φ dạng đóng + Monte Carlo (SFG-XIDS), Monte Carlo + faithfulness (SFG-Attn)",
                  all("mc_profile" in R[f"unsw|sfg|tax|{x}"] and "heat" in R[f"unsw|sfg|tax|{x}"]
                      and "faith" in R[f"nsl|attn|tax|{x}"] for x in s))
            ex = max(R[k]["exact_err"] for k in exp if "exact_err" in R[k])
            check(f"chạy thử: φ dạng đóng = vét cạn trên mọi mô hình đã huấn luyện (max|Δ| = {ex:.1e})", ex < 1e-4)
            check("chạy thử: p = p đã chọn cho SFG-XIDS, A1, A6; p = 0 cho A3 và các mô hình khác",
                  all(R[f"unsw|{m}|tax|0"]["p_swap"] == R["unsw|p"] for m in ("sfg", "sfg_main", "sfg_lr"))
                  and all(R[f"unsw|{m}|tax|0"]["p_swap"] == 0.0 for m in ("sfg_p0", "flat", "concat", "attn")))
            check("chạy thử: đã lưu trọng số các mô hình có giải thích",
                  len(list((tmp / "models").glob("*.pt"))) == 3 * 2 * len(s) + len(s)
                  and len(list((tmp / "models").glob("*.ubj"))) == 2 * len(s))
            run_all(Ds, R, tcfg, cpu, lambda: None, tmp / "models")
            check(f"chạy lại: không huấn luyện lại lần nào ({n1 - n0} lần lúc đầu, {TRAINED[0] - n1} lần sau)",
                  TRAINED[0] == n1 and n1 > n0)
            rep = summarize(R, tcfg, list(Ds), tmp)
            files = ["summary.json", "runs.csv", "per_class_f1.csv", "profiles.csv", "heatmaps.csv"]
            check("tổng hợp: đủ 4 kịch bản, đủ file, mọi trạng thái hợp lệ",
                  all(rep[k] for k in ("k1", "k2", "k3", "k4")) and all((tmp / f).exists() for f in files)
                  and all(v["status"] in ("OK", "NOT_EVALUABLE")
                          for d in ("unsw", "nsl") for v in rep["k4"][d]["stability"].values()))

    rw = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    for w in rw:
        print(f"     RuntimeWarning: {w.message} ({Path(w.filename).name}:{w.lineno})")
    check(f"không có RuntimeWarning nào: có {len(rw)}", not rw)
    n_fail = results.count(False)
    print(f"\nSELFTEST: {'PASS' if n_fail == 0 else 'FAIL'} ({len(results) - n_fail} PASS, {n_fail} FAIL / "
          f"{len(results)} kiểm tra). Chỉ nói về cài đặt, không nói gì về kết quả thực nghiệm.")
    sys.exit(0 if n_fail == 0 else 1)


# ----------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFG-XIDS full experiment, giai đoạn 1")
    ap.add_argument("--unsw_dir", default=None, help="Thư mục chứa training/testing-set UNSW-NB15 (tên chuẩn)")
    ap.add_argument("--nsl_dir", default=None, help="Thư mục chứa KDDTrain+.txt, KDDTest+.txt")
    ap.add_argument("--datasets", default="unsw,nsl", help="unsw, nsl hoặc unsw,nsl")
    ap.add_argument("--out_dir", default="full_v1")
    ap.add_argument("--device", default=None, help="cuda hoặc cpu (mặc định: tự nhận)")
    ap.add_argument("--quick", action="store_true", help="Chạy thử nhanh trên mẫu nhỏ")
    ap.add_argument("--skip_row_check", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="Tự kiểm tra (không cần dữ liệu)")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.selftest:
        selftest()
    if xgb is None or lgb is None:
        sys.exit("Thiếu xgboost hoặc lightgbm. Cài bằng: pip install -U xgboost lightgbm")
    cfg = dict(CFG)
    if args.quick:
        cfg.update(QUICK)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print("CẢNH BÁO: không thấy GPU - sẽ rất chậm. Trên Colab chọn Runtime -> T4 GPU.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "results.json"
    R = json.loads(rpath.read_text()) if rpath.exists() else {}
    conf = dict(version=SCRIPT_VERSION, step0_version=S.SCRIPT_VERSION, quick=args.quick,
                **{k: cfg[k] for k in PROTOCOL_KEYS})
    msg = check_config(R, json.loads(json.dumps(conf)))
    if msg:
        sys.exit(f"{out}: {msg}; hãy dùng --out_dir khác.")
    R["config"] = conf

    def save():
        tmp = rpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(R))
        tmp.replace(rpath)

    Ds, datasets = {}, [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "unsw" in datasets:
        tr, te, lab = S.load_unsw(args)
        Ds["unsw"] = S.prepare(tr, te, lab, S.UNSW_GROUPS, S.UNSW_CAT, args.quick)
        Ds["unsw_nottl"] = S.prepare(tr, te, lab, NOTTL_GROUPS, S.UNSW_CAT, args.quick)
    if "nsl" in datasets:
        tr, te, lab = S.load_nsl(args, out)
        pop = S.nsl_population(tr["attack_type"], te["attack_type"])
        D = S.prepare(tr, te, lab, S.NSL_GROUPS, S.NSL_CAT, args.quick, keep_col="attack_type")
        D["explain_mask"] = np.isin(D["te_extra"], pop["evaluated"] + ["normal"])
        D["profile_population"] = "seen_train_types"
        R["nsl|attack_types"] = pop
        Ds["nsl"] = D
    if not Ds:
        sys.exit("Không có bộ dữ liệu nào để chạy (--datasets).")
    for ds, D in Ds.items():
        R[f"{ds}|classes"], R[f"{ds}|group_names"] = D["classes"], list(D["groups"])
    t0 = time.time()
    run_all(Ds, R, cfg, dev, save, out / "models")
    print(f"\nHuấn luyện xong sau {(time.time() - t0) / 60:.1f} phút.")
    summarize(R, cfg, list(Ds), out)


if __name__ == "__main__":
    main()
