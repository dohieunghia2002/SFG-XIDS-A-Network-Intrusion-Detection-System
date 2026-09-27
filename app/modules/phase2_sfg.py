#!/usr/bin/env python3
"""
SFG-XIDS - GIAI ĐOẠN 2 (TỐI THIỂU), đề cương v2:
  (1) Expected Gradients cấp đặc trưng = tầng 2 của giải thích phân cấp (mục 4.4; mục 5.4 tiêu chí 2).
  (2) Bài toán nhị phân Normal/Attack (kịch bản 1).
  (3) Baseline FT-Transformer (Gorishniy et al., 2021), đa lớp và nhị phân.
Dùng lại nguyên full_sfg.py (giai đoạn 1) và step0_sfg.py (v5); không sửa hai file đó. Tầng 2 chạy trên đúng các
mô hình SFG-XIDS và XGBoost đã lưu ở giai đoạn 1 (<p1_dir>/models), với đúng tập nền của tầng 1.

Quyết định giao thức (đưa ra TRƯỚC khi chạy, không đổi sau khi thấy kết quả test):
  1. Tầng 2. Hàm được giải thích là φ_g của tầng 1 (logit đã trừ TB các lớp của LỚP THẬT), coi là hàm của các đặc
     trưng nhóm g, các nhóm khác giữ nguyên. Điểm nền: đủ 256 mẫu nền của tầng 1 (cũng là tập dùng để tinh lọc số
     hạng cặp), nên Σ_{k∈g} EG_k = φ_g, chỉ còn sai số cầu phương. Đường tích phân thẳng trong không gian embedding
     của từng đặc trưng (PLR với đặc trưng số, embedding với đặc trưng phân loại); attribution của một đặc trưng =
     tổng theo các chiều embedding của nó. Lý do: PLR có thành phần tuần hoàn tần số cao (UNSW chọn σ = 10) nên tích
     phân theo giá trị thô dao động mạnh, cần rất nhiều nút. α: Gauss-Legendre 16 nút. Hàm có bước nhảy đạo hàm
     (LeakyReLU) nên sai số cầu phương giảm cỡ 1/m chứ không triệt tiêu; báo cáo sai số này và độ hội tụ với
     4/8/16/32/64/128 nút trên khoảng 16 mẫu (mô hình seed 0).
  2. Mẫu giải thích: tối đa 100 mẫu/lớp rút ngẫu nhiên (RandomState(4321)) từ tập mẫu test của tầng 1 (NSL-KDD: chỉ
     kiểu tấn công có trong cả train và test). Mô hình: 5 SFG-XIDS (seed 0-4) của giai đoạn 1.
  3. Đối chứng "EG phẳng": EG thông thường trên toàn bộ logit của cùng mô hình, cùng số nút, cùng không gian
     embedding, nền độc lập theo nhóm (điểm nền b ghép nhóm g từ mẫu nền π_g(b), π_g là hoán vị cố định) - cùng phân
     phối can thiệp với tầng 1. Cộng EG phẳng theo nhóm rồi so với φ: tỷ lệ trái dấu (chỉ xét nhóm có
     |φ_g| >= 5% Σ_g|φ_g|), tỷ lệ khác nhóm hạng 1 (theo |.|), khoảng cách L1 giữa tỷ trọng |.| của các nhóm.
  4. Faithfulness cấp đặc trưng (xóa đặc trưng): thay k = 1, 3, 5 đặc trưng có attribution CÓ DẤU lớn nhất (bằng
     chứng ủng hộ lớp thật) bằng giá trị của 16 mẫu nền cố định; đo mức giảm TB xác suất lớp thật. So thứ tự theo
     EG phân cấp, EG phẳng và ngẫu nhiên.
  5. Ổn định qua 5 seed: Kendall τ-b và Jaccard top-5 giữa mọi cặp seed của vectơ TB |attribution| theo lớp. Tham
     chiếu: TreeSHAP (pred_contribs) của 5 mô hình XGBoost đã lưu, cùng mẫu, logit đã trừ TB các lớp.
  6. Nhị phân: Attack = attack_cat (UNSW) / nhãn 5 lớp (NSL) khác Normal. Mô hình: XGBoost, LightGBM, Random Forest,
     Flat-MLP, SFG-XIDS, FT-Transformer; 5 seed; giao thức như giai đoạn 1 (validation 10% phân tầng theo nhãn nhị
     phân, chọn theo macro-F1 validation, trọng số lớp, test niêm phong). σ (PLR) và p (group dropout) lấy đúng giá
     trị đã chọn ở bài toán đa lớp của cùng bộ dữ liệu (không chọn lại). XGBoost giữ nguyên siêu tham số, chỉ đổi
     objective sang binary:logistic (multi:softprob không dùng được cho 2 lớp qua wrapper sklearn).
     Độ đo: Accuracy, Precision/Recall(DR)/F1 của lớp Attack, FAR = FP/(FP+TN), macro-F1, ROC-AUC.
  7. FT-Transformer: cấu hình mặc định của bài gốc, không tinh chỉnh (d_token 192, 3 khối, 8 đầu, attention dropout
     0,2, FFN ReGLU 256, FFN dropout 0,1, residual dropout 0; AdamW lr 1e-4, weight decay 1e-5 trừ tokenizer,
     LayerNorm, bias; batch 512); tối đa 60 epoch, patience 10 (lr nhỏ hơn 10 lần các mạng khác); cùng trọng số lớp
     và cách chọn theo macro-F1 validation; AMP fp16 trên GPU.
Báo cáo: TB ± SD (ddof = 1) qua 5 seed; so hai mô hình bằng hiệu ghép cặp theo seed. Không có tiêu chí đạt/không đạt.

Chạy trên Colab (T4 GPU). Cùng thư mục: phase2_sfg.py, full_sfg.py, step0_sfg.py (v5), step0_unsw.py và thư mục
full_v1 của giai đoạn 1 (cần results.json và models/).
    !pip -q install -U xgboost lightgbm kagglehub
    (chạy lại cell tạo /content/unsw_std như giai đoạn 1)
    !python -u phase2_sfg.py --selftest
    !python -u phase2_sfg.py --quick --out_dir p2_quick --unsw_dir /content/unsw_std
    !python -u phase2_sfg.py --p1_dir full_v1 --out_dir phase2_v1 --unsw_dir /content/unsw_std
Bị ngắt: chạy lại đúng lệnh cũ, script bỏ qua phần đã xong. --parts eg,bin,ftt để chạy riêng từng phần.
Hết bộ nhớ GPU ở tầng 2: script tự giảm lô; nếu vẫn lỗi thêm --eg_rows 1048576.
Kết quả (<out_dir>): results.json, summary_p2.json, runs_p2.csv, eg_features.csv, eg/<ds>_<seed>.npz (attribution
từng mẫu, dùng vẽ case study).
"""
import argparse
import contextlib
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import full_sfg as P1
except ImportError as _e:
    sys.exit(f"Cần full_sfg.py, step0_sfg.py (v5), step0_unsw.py cùng thư mục với phase2_sfg.py ({_e}).")
S = P1.S
torch, nn = S.torch, S.nn
TF = torch.nn.functional
xgb, lgb = P1.xgb, P1.lgb

SCRIPT_VERSION = 2
NAN = float("nan")
XGB_TOL = 1e-3        # dung sai (tương đối theo max|margin|, tối thiểu 1e-3) cho Σ contrib = margin
BIN_MODELS = ["xgb", "lgbm", "rf", "flat", "sfg", "ftt"]
NAMES = dict(P1.NAMES, ftt="FT-Transformer")
EG_SEED, PERM_SEED, RAND_SEED, DEL_SEED = 4321, 2026, 2027, P1.DEL_SEED
CFG = dict(P1.CFG, n_eg=100, eg_nodes=16, conv_nodes=[4, 8, 16, 32, 64, 128], n_conv=16, del_k=[1, 3, 5],
           n_del_bg=16, sign_min=0.05, top_k=5, eg_rows=2 ** 22, eg_pts=2 ** 15,
           ftt_d=192, ftt_blocks=3, ftt_heads=8, ftt_adrop=0.2, ftt_fdrop=0.1, ftt_lr=1e-4, ftt_wd=1e-5,
           ftt_bs=512, ftt_max_ep=60, ftt_patience=10)
QUICK = dict(P1.QUICK, n_eg=6, eg_nodes=8, conv_nodes=[4, 8], n_conv=4, n_del_bg=4, eg_rows=2 ** 18,
             eg_pts=2 ** 13, ftt_d=32, ftt_blocks=1, ftt_heads=4, ftt_bs=256, ftt_max_ep=2, ftt_patience=1)
PROTOCOL_KEYS = ["seeds", "max_ep", "patience", "lr", "bs", "n_bg", "n_per_class", "n_estimators", "es_rounds",
                 "rf_trees", "n_eg", "eg_nodes", "conv_nodes", "n_conv", "del_k", "n_del_bg", "sign_min", "top_k",
                 "ftt_d", "ftt_blocks", "ftt_heads", "ftt_adrop", "ftt_fdrop", "ftt_lr", "ftt_wd", "ftt_bs",
                 "ftt_max_ep", "ftt_patience"]


def ms(v):
    return S._ms([float(x) for x in v if x is not None and math.isfinite(x)])


def clock(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


# ----------------------------------------------------------------------------
# FT-Transformer
# ----------------------------------------------------------------------------
class FTBlock(nn.Module):
    def __init__(self, d, heads, adrop, fdrop, first):
        super().__init__()
        self.ln1 = nn.Identity() if first else nn.LayerNorm(d)      # khối đầu: không LayerNorm trước attention
        self.qkv, self.out, self.ln2 = nn.Linear(d, 3 * d), nn.Linear(d, d), nn.LayerNorm(d)
        dh = int(d * 4 / 3)
        self.ff1, self.ff2 = nn.Linear(d, 2 * dh), nn.Linear(dh, d)
        self.heads, self.adrop, self.fdrop = heads, adrop, nn.Dropout(fdrop)

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = TF.scaled_dot_product_attention(q, k, v, dropout_p=self.adrop if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(n, t, d))
        a, b = self.ff1(self.ln2(x)).chunk(2, -1)                    # ReGLU
        return x + self.ff2(self.fdrop(a * TF.relu(b)))


class FTTransformer(nn.Module):
    """Tokenizer tuyến tính cho đặc trưng số, embedding + bias cho đặc trưng phân loại, token [CLS], khối
    Transformer pre-norm, đầu LayerNorm -> ReLU -> Linear trên [CLS]."""

    def __init__(self, n_num, cards, n_cls, cfg):
        super().__init__()
        d = cfg["ftt_d"]
        s = 1 / math.sqrt(d)
        self.num_w = nn.Parameter(torch.empty(n_num, d).uniform_(-s, s))
        self.num_b = nn.Parameter(torch.empty(n_num, d).uniform_(-s, s))
        self.cat = nn.ModuleList([nn.Embedding(k, d) for k in cards])
        for e in self.cat:
            nn.init.uniform_(e.weight, -s, s)
        self.cat_b = nn.Parameter(torch.empty(len(cards), d).uniform_(-s, s))
        self.cls_tok = nn.Parameter(torch.empty(1, 1, d).uniform_(-s, s))
        self.blocks = nn.ModuleList([FTBlock(d, cfg["ftt_heads"], cfg["ftt_adrop"], cfg["ftt_fdrop"], i == 0)
                                     for i in range(cfg["ftt_blocks"])])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, n_cls))

    def forward(self, xn, xc):
        tok = [self.cls_tok.expand(xn.shape[0], 1, -1), xn[..., None] * self.num_w + self.num_b]
        if len(self.cat):
            tok.append(torch.stack([e(xc[:, j]) for j, e in enumerate(self.cat)], 1) + self.cat_b)
        x = torch.cat(tok, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x[:, 0])


def amp_ctx(dev, on):
    return torch.autocast("cuda", dtype=torch.float16) if on and dev.type == "cuda" else contextlib.nullcontext()


def make_scaler(on):
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=on)


@torch.no_grad()
def nn_logits(model, xn, xc, bs=8192):
    model.eval()
    amp = isinstance(model, FTTransformer)
    out = []
    for s in range(0, xn.shape[0], bs):
        with amp_ctx(xn.device, amp):
            out.append(model(xn[s:s + bs], xc[s:s + bs]).float())
    return torch.cat(out)


def train_ftt(model, T, D, dev, seed, cfg):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    dec, nodec = [], []
    for name, prm in model.named_parameters():
        (nodec if prm.ndim < 2 or name.startswith(("num_", "cat", "cls_tok")) else dec).append(prm)
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": cfg["ftt_wd"]},
                             {"params": nodec, "weight_decay": 0.0}], lr=cfg["ftt_lr"])
    amp = dev.type == "cuda"
    scaler, lossf = make_scaler(amp), nn.CrossEntropyLoss(weight=T["cw"])
    ntr, best, bad, ep_best, state = T["y_tr"].shape[0], -1.0, 0, 0, None
    for ep in range(1, cfg["ftt_max_ep"] + 1):
        model.train()
        perm = torch.randperm(ntr, device=dev, generator=gen)
        for s in range(0, ntr, cfg["ftt_bs"]):
            idx = perm[s:s + cfg["ftt_bs"]]
            if idx.numel() < 2:
                continue
            with amp_ctx(dev, amp):
                loss = lossf(model(T["xn_tr"][idx], T["xc_tr"][idx]).float(), T["y_tr"][idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        f1 = S.macro_f1(D["y"]["va"], nn_logits(model, T["xn_va"], T["xc_va"]).argmax(1).cpu().numpy(), D["n_cls"])
        if f1 > best + 1e-6:
            best, bad, ep_best = f1, 0, ep
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["ftt_patience"]:
                break
    model.load_state_dict(state)
    return best, ep_best


# ----------------------------------------------------------------------------
# Tầng 2: Expected Gradients trong nhóm (và đối chứng EG phẳng)
# ----------------------------------------------------------------------------
def cz(z, c):
    """Logit lớp c đã trừ TB các lớp (cùng quy ước với φ của tầng 1)."""
    return z[:, c] - z.mean(1)


def centered_w(lin, c):
    return lin.weight[c] - lin.weight.mean(0), lin.bias[c] - lin.bias.mean()


def pair_parts(model, k, c):
    """Tách lớp đầu của v_k: W1 = [Wa | Wb | Wc] ứng với [e_i; e_j; e_i ⊙ e_j]; lớp cuối gộp thành vectơ (w2, b2) của
    logit lớp c đã trừ TB các lớp."""
    l1, act, l2 = model.v[k][0], model.v[k][1], model.v[k][2]
    d, W = l1.in_features // 3, l1.weight
    w2, b2 = centered_w(l2, c)
    return dict(Wa=W[:, :d], Wb=W[:, d:2 * d], Wc=W[:, 2 * d:], b1=l1.bias, w2=w2, b2=b2, slope=act.negative_slope)


def pair_rows(pp, eg, eo, g_first):
    """v_k,c(e_g, e_o) theo từng hàng (eg, eo: P x d) -> P. g_first: nhóm g đứng trước trong cặp k."""
    Wg, Wo = (pp["Wa"], pp["Wb"]) if g_first else (pp["Wb"], pp["Wa"])
    pre = eg @ Wg.T + eo @ Wo.T + (eg * eo) @ pp["Wc"].T + pp["b1"]
    return TF.leaky_relu(pre, pp["slope"]) @ pp["w2"] + pp["b2"]


def pair_grid_mean(pp, eg, Eo, g_first):
    """TB_b v_k,c(e_g, Eo[b]) với mọi e_g (eg: P x d, Eo: B x d hằng) -> P. Gộp cả lớp đầu vào một phép nhân ma
    trận: pre[p, h, b] = Σ_d M[p, h, d] Eo[b, d] + c[p, h], M = Wc ⊙ e_g + W_o, c = W_g e_g + b1 -> chỉ tạo một
    tensor lớn P x 64 x B."""
    Wg, Wo = (pp["Wa"], pp["Wb"]) if g_first else (pp["Wb"], pp["Wa"])
    M = torch.cat([pp["Wc"][None] * eg[:, None, :] + Wo[None], (eg @ Wg.T + pp["b1"])[:, :, None]], 2)
    Eo1 = torch.cat([Eo, Eo.new_ones(Eo.shape[0], 1)], 1)
    h = TF.leaky_relu(M @ Eo1.T, pp["slope"], inplace=True)
    return (pp["w2"] @ h).mean(-1) + pp["b2"]


def gl_nodes(m, dev):
    x, w = np.polynomial.legendre.leggauss(m)
    return (torch.tensor((x + 1) / 2, dtype=torch.float32, device=dev),
            torch.tensor(w / 2, dtype=torch.float32, device=dev))


def group_tokens(ne, ce, ni, ci):
    """Embedding các đặc trưng của một nhóm theo đúng thứ tự encoder (số theo ni, rồi phân loại theo ci)."""
    parts = ([ne.index_select(1, ni)] if ni.numel() else []) + ([ce.index_select(1, ci)] if ci.numel() else [])
    return torch.cat(parts, 1)


def forward_tokens(model, tok, nnum):
    """Logit của SFG-XIDS từ embedding của mọi đặc trưng (tok: n x (số + phân loại) x d_emb)."""
    n, ne, ce = tok.shape[0], tok[:, :nnum], tok[:, nnum:]
    return model.head([model.enc[g](group_tokens(ne, ce, ni, ci).reshape(n, -1))
                       for g, (ni, ci) in enumerate(model.groups())])


def geometry(D, model, dev):
    feats = D["feats"]
    gcol = np.zeros(len(feats), dtype=np.int64)
    for g, cols in enumerate(D["groups"].values()):
        for f in cols:
            gcol[feats.index(f)] = g
    cols = [[feats.index(f) for f in [D["nums"][j] for j in ni.tolist()] + [D["cats"][j] for j in ci.tolist()]]
            for ni, ci in model.groups()]
    return dict(gcol=gcol, onehot=np.eye(model.G)[gcol], cols=cols,
                cols_all=[feats.index(f) for f in D["nums"] + D["cats"]],
                col_num=torch.as_tensor([feats.index(f) for f in D["nums"]], dtype=torch.long, device=dev),
                col_cat=torch.as_tensor([feats.index(f) for f in D["cats"]], dtype=torch.long, device=dev))


def hier_eg(model, xn, xc, bn, bc, c, m, rows, cols):
    """Tầng 2. A[s, f] = EG của φ_g(x_s) theo đặc trưng f (g = nhóm của f), lớp c. φ_g phụ thuộc nhóm g qua
    h_g(e_g) = u_g(e_g) + ½ Σ_{cặp k∋g} [v_k(e_g, e_o) + TB_b v_k(e_g, e_o^(b))] (e_o của chính mẫu, e_o^(b) của
    mẫu nền); phần còn lại là hằng số khi chỉ nhóm g thay đổi. Nền: đủ B mẫu nền; α: Gauss-Legendre m nút."""
    dev, n, B = xn.device, xn.shape[0], bn.shape[0]
    al, wl = gl_nodes(m, dev)
    with torch.no_grad():
        E, Eb = model.encode(xn, xc), model.encode(bn, bc)
        (ne_x, ce_x), (ne_b, ce_b) = model.emb(xn, xc), model.emb(bn, bc)
    pp = [pair_parts(model, k, c) for k in range(len(model.pairs))]
    A = torch.zeros(n, sum(len(v) for v in cols), device=dev)
    mc = min(m, max(1, rows // (B * B)))                  # số nút mỗi lô
    spc = max(1, rows // (m * B * B)) if mc == m else 1   # số mẫu mỗi lô
    for g, (ni, ci) in enumerate(model.groups()):
        wu, bu = centered_w(model.u[g], c)
        terms = [(k, i == g, j if i == g else i) for k, (i, j) in enumerate(model.pairs) if g in (i, j)]
        Xg, Xb = group_tokens(ne_x, ce_x, ni, ci), group_tokens(ne_b, ce_b, ni, ci)
        kg, de = Xg.shape[1], Xg.shape[2]
        s = 0
        while s < n:
            ns = min(spc, n - s)
            try:
                dX = Xg[s:s + ns, None] - Xb[None]                                       # ns x B x kg x de
                acc = torch.zeros(ns, kg, device=dev)
                for j in range(0, m, mc):
                    a, w = al[j:j + mc], wl[j:j + mc]
                    q = a.shape[0]
                    path = (Xb[None, None] + a.view(-1, 1, 1, 1, 1) * dX[None]).reshape(-1, kg * de)
                    path.requires_grad_(True)
                    eg = model.enc[g](path)                                               # P x d, P = q·ns·B
                    h = eg @ wu + bu
                    for k, first, o in terms:
                        eo = E[o][s:s + ns][None, :, None].expand(q, ns, B, -1).reshape(-1, eg.shape[1])
                        h = h + 0.5 * (pair_rows(pp[k], eg, eo, first) + pair_grid_mean(pp[k], eg, Eb[o], first))
                    grad, = torch.autograd.grad((h.view(q, ns, B) * w.view(-1, 1, 1)).sum(), path)
                    acc += (dX[None] * grad.view(q, ns, B, kg, de)).sum((0, 2, 4))
                A[s:s + ns, cols[g]] = acc / B
                s += ns
            except torch.cuda.OutOfMemoryError:
                dX = acc = path = eg = h = grad = None
                torch.cuda.empty_cache()
                if spc > 1:
                    spc //= 2
                elif mc > 1:
                    mc = max(1, mc // 2)
                else:
                    raise
                print(f"    hết bộ nhớ GPU ở tầng 2 -> lô còn {spc} mẫu x {mc} nút", flush=True)
    return A


def flat_eg(model, xn, xc, bn, bc, c, m, pts, geo, perms):
    """Đối chứng: EG thông thường trên toàn bộ logit lớp c (đã trừ TB các lớp), nền độc lập theo nhóm (điểm nền b
    ghép nhóm g từ mẫu nền perms[g][b]). Trả về (A: n x F theo D['feats'], vế phải completeness z_c(x) - TB_b z_c)."""
    dev, n, B = xn.device, xn.shape[0], bn.shape[0]
    al, wl = gl_nodes(m, dev)
    with torch.no_grad():
        (ne_x, ce_x), (ne_b, ce_b) = model.emb(xn, xc), model.emb(bn, bc)
        ne_p, ce_p = ne_b.clone(), ce_b.clone()
        for g, (ni, ci) in enumerate(model.groups()):
            pg = torch.as_tensor(perms[g], device=dev)
            if ni.numel():
                ne_p[:, ni] = ne_b[pg][:, ni]
            if ci.numel():
                ce_p[:, ci] = ce_b[pg][:, ci]
        nnum = ne_x.shape[1]
        X, Xb = torch.cat([ne_x, ce_x], 1), torch.cat([ne_p, ce_p], 1)
        target = cz(forward_tokens(model, X, nnum), c) - cz(forward_tokens(model, Xb, nnum), c).mean()
    nt, de = X.shape[1], X.shape[2]
    A = torch.zeros(n, nt, device=dev)
    spc = max(1, pts // (m * B))
    for s in range(0, n, spc):
        ns = min(spc, n - s)
        dX = X[s:s + ns, None] - Xb[None]
        path = (Xb[None, None] + al.view(-1, 1, 1, 1, 1) * dX[None]).reshape(-1, nt, de)
        path.requires_grad_(True)
        z = cz(forward_tokens(model, path, nnum), c)
        grad, = torch.autograd.grad((z.view(m, ns, B) * wl.view(-1, 1, 1)).sum(), path)
        A[s:s + ns] = (dX[None] * grad.view(m, ns, B, nt, de)).sum((0, 2, 4)) / B
    out = torch.zeros_like(A)
    out[:, geo["cols_all"]] = A
    return out, target


@torch.no_grad()
def deletion(model, xn, xc, bn, bc, cls, orders, ks, K, geo, seed):
    """Mức giảm TB p(lớp thật) khi thay top-k đặc trưng (theo từng thứ tự) bằng giá trị của K mẫu nền cố định."""
    dev, n = xn.device, xn.shape[0]
    src = torch.as_tensor(np.random.RandomState(seed).randint(0, bn.shape[0], size=K), device=dev).repeat(n)
    ar = torch.arange(n, device=dev)
    p0 = torch.softmax(model(xn, xc), 1)[ar, cls]
    xn_r, xc_r, bn_s, bc_s = xn.repeat_interleave(K, 0), xc.repeat_interleave(K, 0), bn[src], bc[src]
    out = {}
    for name, order in orders.items():
        res = []
        for k in ks:
            mask = torch.zeros(n, order.shape[1], dtype=torch.bool, device=dev)
            mask.scatter_(1, order[:, :k], True)
            Xn = torch.where(mask[:, geo["col_num"]].repeat_interleave(K, 0), bn_s, xn_r)
            Xc = torch.where(mask[:, geo["col_cat"]].repeat_interleave(K, 0), bc_s, xc_r)
            p = torch.softmax(model(Xn, Xc), 1).view(n, K, -1)[ar, :, cls].mean(1)
            res.append(float((p0 - p).mean()))
        out[name] = res
    return out


def xgb_contribs(path, D, frame, xr, xc, cls):
    """TreeSHAP (pred_contribs) của mô hình XGBoost đã lưu, logit lớp cls đã trừ TB các lớp -> (n x F, kiểm tra).
    DMatrix dựng từ ĐÚNG loại DataFrame lúc huấn luyện (D['F']: cột phân loại là pd.Categorical cùng danh mục của
    train), để XGBoost tự khớp mã danh mục như lúc huấn luyện. Kiểm tra:
      sum_err     = max |Σ contrib (kể cả bias) - margin|, cùng DMatrix;
      np_vs_frame = max |margin từ mảng số + mã phân loại (đường xgb_predict của giai đoạn 1) - margin từ DataFrame|."""
    bst = xgb.Booster()
    bst.load_model(str(path))
    names, types = bst.feature_names, bst.feature_types
    if names is None or list(names) != list(D["feats"]) or list(frame.columns) != list(names):
        raise ValueError("tên/thứ tự cột của mô hình XGBoost không khớp D['feats']")
    bad = [f for f, t in zip(names, types) if (t == "c") != (f in D["cats"])]
    if bad or any(not isinstance(frame[f].dtype, pd.CategoricalDtype) for f in D["cats"]):
        raise ValueError(f"kiểu cột phân loại không khớp mô hình đã lưu: {bad}")
    try:
        it = (0, int(bst.best_iteration) + 1)
    except (AttributeError, TypeError, ValueError):
        it = (0, 0)
    dm = xgb.DMatrix(frame, enable_categorical=True)
    ph = np.asarray(bst.predict(dm, pred_contribs=True, iteration_range=it), dtype=np.float64)
    mg = np.asarray(bst.predict(dm, output_margin=True, iteration_range=it), dtype=np.float64)
    if ph.ndim != 3:
        raise ValueError(f"pred_contribs có dạng {ph.shape}, cần n x C x (F+1)")
    X = np.empty((len(xr), len(names)), dtype=np.float32)
    X[:, [names.index(f) for f in D["nums"]]] = xr
    X[:, [names.index(f) for f in D["cats"]]] = xc
    mg_np = np.asarray(bst.predict(xgb.DMatrix(X, feature_names=names, feature_types=types, enable_categorical=True),
                                   output_margin=True, iteration_range=it), dtype=np.float64)
    chk = dict(sum_err=float(np.abs(ph.sum(2) - mg).max()), np_vs_frame=float(np.abs(mg_np - mg).max()),
               margin_absmax=float(np.abs(mg).max()), iteration_range=list(it))
    ph = ph - ph.mean(1, keepdims=True)
    return ph[np.arange(len(cls)), cls, :-1], chk


def eg_subset(ES, n_eg):
    rng = np.random.RandomState(EG_SEED)
    return [np.sort(rng.choice(idx, min(n_eg, len(idx)), replace=False)) if len(idx) else np.asarray(idx, np.int64)
            for idx in ES[1]]


def grp_cmp(A, phi, sign_min):
    """So quy công nhóm A (n x G) với φ (n x G): trái dấu, khác nhóm hạng 1, L1 giữa tỷ trọng |.|."""
    tot = np.abs(phi).sum(1, keepdims=True) + 1e-12
    big = np.abs(phi) >= sign_min * tot
    sh = lambda M: np.abs(M) / (np.abs(M).sum(1, keepdims=True) + 1e-12)
    return dict(sign_disagree=float((np.sign(A)[big] != np.sign(phi)[big]).mean()) if big.any() else NAN,
                top1_disagree=float((np.abs(A).argmax(1) != np.abs(phi).argmax(1)).mean()),
                share_L1=float(np.abs(sh(A) - sh(phi)).sum(1).mean()))


def per_class(M, cls, C, fn):
    return [fn(M[cls == c]).tolist() if (cls == c).any() else [NAN] * M.shape[1] for c in range(C)]


def eg_one(ctx, model, xgb_path, sub, cfg, dev):
    D, T = ctx["D"], ctx["T"]
    bt = torch.as_tensor(ctx["ES"][0], device=dev)
    bn, bc = T["xn_tr"][bt], T["xc_tr"][bt]
    geo, B, C = geometry(D, model, dev), len(bt), D["n_cls"]
    perms = [np.random.RandomState(PERM_SEED + g).permutation(B) for g in range(model.G)]
    tm = dict(phi=0.0, hier=0.0, flat=0.0, xgb=NAN)
    keep = {k: [] for k in ("phi", "hier", "flat", "flat_target", "idx", "cls")}
    dsum = {o: np.zeros(len(cfg["del_k"])) for o in ("hier", "flat", "random")}
    for c, idx in enumerate(sub):
        if len(idx) == 0:
            continue
        ti = torch.as_tensor(idx, device=dev)
        xn, xc = T["xn_te"][ti], T["xc_te"][ti]
        t = clock(dev)
        P = P1.phi_g(model, xn, xc, bn, bc)[0]
        tm["phi"] += clock(dev) - t
        t = clock(dev)
        H = hier_eg(model, xn, xc, bn, bc, c, cfg["eg_nodes"], cfg["eg_rows"], geo["cols"])
        tm["hier"] += clock(dev) - t
        t = clock(dev)
        Fl, ft = flat_eg(model, xn, xc, bn, bc, c, cfg["eg_nodes"], cfg["eg_pts"], geo, perms)
        tm["flat"] += clock(dev) - t
        rnd = torch.as_tensor(np.argsort(np.random.RandomState(RAND_SEED + c).rand(len(idx), H.shape[1]), 1),
                              device=dev)
        cls = torch.full((len(idx),), c, dtype=torch.long, device=dev)
        dd = deletion(model, xn, xc, bn, bc, cls,
                      {"hier": torch.argsort(-H, 1), "flat": torch.argsort(-Fl, 1), "random": rnd},
                      cfg["del_k"], cfg["n_del_bg"], geo, DEL_SEED)
        for o in dsum:
            dsum[o] += np.asarray(dd[o]) * len(idx)
        for k, v in (("phi", P[:, :, c]), ("hier", H), ("flat", Fl), ("flat_target", ft)):
            keep[k].append(v.double().cpu().numpy())
        keep["idx"].append(np.asarray(idx, dtype=np.int64))
        keep["cls"].append(np.full(len(idx), c, dtype=np.int64))
    arr = {k: np.concatenate(v) for k, v in keep.items()}
    phi, H, Fl, cls, n = arr["phi"], arr["hier"], arr["flat"], arr["cls"], len(arr["cls"])
    X, xstat, xchk = None, "không có xgboost hoặc mô hình XGBoost", None
    if xgb is not None and xgb_path is not None:
        try:
            t = time.time()
            X, xchk = xgb_contribs(xgb_path, D, D["F"]["te"].iloc[arr["idx"]], ctx["R_te"][arr["idx"]],
                                   ctx["C_te"][arr["idx"]], cls)
            tm["xgb"] = time.time() - t
            tol = XGB_TOL * max(1.0, xchk["margin_absmax"])
            xstat = "OK" if xchk["sum_err"] < tol else f"LỆCH: max|Σ contrib - margin| = {xchk['sum_err']:.1e}"
            if xchk["np_vs_frame"] >= tol:        # đường mảng của giai đoạn 1 lệch DataFrame -> báo to, không che
                print(f"  CẢNH BÁO: margin XGBoost từ mảng (đường xgb_predict của giai đoạn 1) lệch margin từ "
                      f"DataFrame {xchk['np_vs_frame']:.1e} - Shapley nhóm Monte Carlo của XGBoost ở giai đoạn 1 cần "
                      f"xem lại.", flush=True)
            if xstat != "OK":
                X = None
        except Exception as e:                    # ghi lại, không bịa kết quả
            xstat = f"{type(e).__name__}: {e}"
    arr["xgb"] = X if X is not None else np.zeros((0, H.shape[1]))
    tot = np.abs(phi).sum(1) + 1e-12
    Hg, Fg = H @ geo["onehot"], Fl @ geo["onehot"]
    rel, relf = np.abs(Hg - phi) / tot[:, None], np.abs(Fl.sum(1) - arr["flat_target"]) / tot
    r = dict(n=[int((cls == c).sum()) for c in range(C)],
             hier_err=dict(median=float(np.median(rel)), p99=float(np.quantile(rel, 0.99)), max=float(rel.max()),
                           max_abs=float(np.abs(Hg - phi).max())),
             flat_err=dict(median=float(np.median(relf)), max=float(relf.max())),
             flat_total_gap=float(np.median(np.abs(arr["flat_target"] - phi.sum(1)) / tot)),
             hier_vs_phi=grp_cmp(Hg, phi, cfg["sign_min"]), flat_vs_phi=grp_cmp(Fg, phi, cfg["sign_min"]),
             deletion={o: (v / n).tolist() for o, v in dsum.items()},
             imp_hier=per_class(np.abs(H), cls, C, lambda a: a.mean(0)),
             imp_hier_signed=per_class(H, cls, C, lambda a: a.mean(0)),
             imp_flat=per_class(np.abs(Fl), cls, C, lambda a: a.mean(0)),
             imp_xgb=per_class(np.abs(X), cls, C, lambda a: a.mean(0)) if X is not None else None,
             xgb_status=xstat, xgb_check=xchk, time_per_sample={k: v / n for k, v in tm.items()})
    return r, arr


def eg_conv(ctx, model, sub, cfg, dev):
    """Sai số completeness của tầng 2 theo số nút Gauss-Legendre (vài mẫu mỗi lớp)."""
    D, T = ctx["D"], ctx["T"]
    bt = torch.as_tensor(ctx["ES"][0], device=dev)
    bn, bc = T["xn_tr"][bt], T["xc_tr"][bt]
    geo = geometry(D, model, dev)
    per = math.ceil(cfg["n_conv"] / max(1, sum(len(i) > 0 for i in sub)))
    out = {}
    for m in cfg["conv_nodes"]:
        rels, t, n = [], 0.0, 0
        for c, idx in enumerate(sub):
            idx = idx[:per]
            if len(idx) == 0:
                continue
            ti = torch.as_tensor(idx, device=dev)
            xn, xc = T["xn_te"][ti], T["xc_te"][ti]
            phi = P1.phi_g(model, xn, xc, bn, bc)[0][:, :, c].double().cpu().numpy()
            t0 = clock(dev)
            H = hier_eg(model, xn, xc, bn, bc, c, m, cfg["eg_rows"], geo["cols"]).double().cpu().numpy()
            t += clock(dev) - t0
            rels.append(np.abs(H @ geo["onehot"] - phi) / (np.abs(phi).sum(1, keepdims=True) + 1e-12))
            n += len(idx)
        rel = np.concatenate(rels)
        out[str(m)] = dict(median=float(np.median(rel)), max=float(rel.max()), sec_per_sample=t / n, n=n)
    return out


def sfg_model(ds, ctx, seed, cfg, dev, p1, R1, sigma, p, quick):
    D, T = ctx["D"], ctx["T"]
    model = P1.build("sfg", D, ctx["gidx"]["tax"], cfg, sigma).to(dev)
    f = p1 / "models" / f"{ds}_sfg_tax_{seed}.pt"
    if quick:                                     # dữ liệu --quick khác giai đoạn 1 -> huấn luyện nhanh tại chỗ
        torch.manual_seed(seed)
        np.random.seed(seed)
        S.train_nn(model, T, D, dev, seed, cfg, p)
    elif f.exists():
        model.load_state_dict(torch.load(f, map_location=dev))
    else:
        sys.exit(f"Không thấy {f} (mô hình SFG-XIDS của giai đoạn 1). Kiểm tra --p1_dir.")
    f1_now = S.macro_f1(D["y"]["te"], S.predict(model, T["xn_te"], T["xc_te"]), D["n_cls"])
    f1_p1 = NAN if quick else float(R1.get(f"{ds}|sfg|tax|{seed}", {}).get("test_f1", NAN))
    if not quick and not abs(f1_now - f1_p1) < 0.05:
        print(f"  CẢNH BÁO [{ds}] seed {seed}: macro-F1 test của mô hình nạp lại {f1_now:.2f} khác giai đoạn 1 "
              f"({f1_p1:.2f}) - dữ liệu hoặc tiền xử lý có thể đã khác.", flush=True)
    return model, f1_now, f1_p1


def xgb_model(ctx, seed, cfg, dev, p1, out, quick):
    if xgb is None:
        return None
    if not quick:
        f = p1 / "models" / f"{ctx['ds']}_xgb_tax_{seed}.ubj"
        return f if f.exists() else None
    f = out / "quick_models" / f"{ctx['ds']}_xgb_{seed}.ubj"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        P1.run_tree("xgb", ctx["D"], seed, cfg, dev)[0].save_model(str(f))
    return f


def run_eg(ctx, R, R1, cfg, dev, p1, out, save, quick):
    ds, D = ctx["ds"], ctx["D"]
    sub = eg_subset(ctx["ES"], cfg["n_eg"])
    sigma, p = R1.get(f"{ds}|sigma", 1.0), R1.get(f"{ds}|p", 0.2)
    R[f"{ds}|feats"], R[f"{ds}|classes"] = list(D["feats"]), [str(c) for c in D["classes"]]
    R[f"{ds}|groups"] = {g: list(v) for g, v in D["groups"].items()}
    (out / "eg").mkdir(parents=True, exist_ok=True)
    for i, seed in enumerate(cfg["seeds"]):
        key, ckey = f"eg|{ds}|{seed}", f"eg_conv|{ds}"
        need_conv = i == 0 and ckey not in R
        if key in R and not need_conv:
            continue
        model, f1_now, f1_p1 = sfg_model(ds, ctx, seed, cfg, dev, p1, R1, sigma, p, quick)
        model.eval()
        model.requires_grad_(False)
        if key not in R:
            t0 = time.time()
            r, arr = eg_one(ctx, model, xgb_model(ctx, seed, cfg, dev, p1, out, quick), sub, cfg, dev)
            r.update(f1_now=f1_now, f1_p1=f1_p1, seconds=round(time.time() - t0, 1))
            np.savez_compressed(out / "eg" / f"{ds}_{seed}.npz", **arr)
            R[key] = r
            save()
            d = r["deletion"]
            print(f"  [{ds}] EG seed {seed}: |ΣEG - φ|/Σ|φ| trung vị {r['hier_err']['median']:.1e}, max "
                  f"{r['hier_err']['max']:.1e} | EG phẳng: trái dấu {r['flat_vs_phi']['sign_disagree']:.3f}, khác "
                  f"hạng 1 {r['flat_vs_phi']['top1_disagree']:.3f} | xóa k={cfg['del_k'][0]}: phân cấp "
                  f"{d['hier'][0]:.3f}, phẳng {d['flat'][0]:.3f}, ngẫu nhiên {d['random'][0]:.3f} | TreeSHAP: "
                  f"{r['xgb_status']} | {r['seconds']}s", flush=True)
        if need_conv:
            R[ckey] = eg_conv(ctx, model, sub, cfg, dev)
            save()
            print(f"  [{ds}] hội tụ theo số nút: " + ", ".join(
                f"m={m}: {v['median']:.1e}/{v['max']:.1e}" for m, v in R[ckey].items()), flush=True)


# ----------------------------------------------------------------------------
# Kịch bản 1 bổ sung: nhị phân và FT-Transformer
# ----------------------------------------------------------------------------
def metrics_bin(y, prob, pred, pos):
    yt, yp = np.asarray(y) == pos, np.asarray(pred) == pos
    tp, fp, fn, tn = (int((yt & yp).sum()), int((~yt & yp).sum()), int((yt & ~yp).sum()), int((~yt & ~yp).sum()))
    prec = 100 * tp / (tp + fp) if tp + fp else 0.0
    rec = 100 * tp / (tp + fn) if tp + fn else 0.0
    return dict(test_acc=100 * (tp + tn) / len(yt), test_prec=prec, test_rec=rec,
                test_f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                test_far=100 * fp / (fp + tn) if fp + tn else NAN,
                test_macro_f1=S.macro_f1(y, pred, 2),
                test_auc=100 * float(roc_auc_score(yt.astype(int), prob[:, pos])) if 0 < yt.sum() < len(yt) else NAN,
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn))


def run_tree_bin(kind, D, seed, cfg, dev):
    """Cây cho bài toán nhị phân. XGBoost: đúng siêu tham số của giai đoạn 1 nhưng objective binary:logistic -
    multi:softprob của giai đoạn 1 cần num_class, mà wrapper sklearn chỉ đặt num_class khi có > 2 lớp (lỗi
    'num_class should be greater equal to 1'). LightGBM, Random Forest: dùng nguyên run_tree của giai đoạn 1."""
    if kind != "xgb":
        return P1.run_tree(kind, D, seed, cfg, dev)
    w, ytr, yva = D["cw"], D["y"]["tr"], D["y"]["va"]
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"], learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device=dev.type, enable_categorical=True, max_cat_to_onehot=1,
        objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=cfg["es_rounds"],
        random_state=seed)
    m.fit(D["F"]["tr"], ytr, sample_weight=w[ytr], eval_set=[(D["F"]["va"], yva)],
          sample_weight_eval_set=[w[yva]], verbose=False)
    return m, {"trees": int(m.best_iteration) + 1}, m.predict(D["F"]["va"]), m.predict(D["F"]["te"])


def check_labels(D, T, binary):
    """Trọng số lớp phải đúng số lớp của CHÍNH bài toán này và đúng công thức 1/sqrt(tần suất lớp trên train) của
    prepare(); bài toán nhị phân phải có lớp đúng là ['Attack', 'Normal'] (Attack = chỉ số 0 = lớp dương)."""
    C = D["n_cls"]
    freq = np.bincount(D["y"]["tr"], minlength=C).astype(float)
    ref = 1.0 / np.sqrt(np.maximum(freq, 1))
    ref = ref / ref[D["y"]["tr"]].mean()
    cw = T["cw"].detach().cpu().numpy()
    if len(D["cw"]) != C or cw.shape != (C,) or not np.allclose(cw, ref, rtol=1e-5):
        sys.exit(f"Trọng số lớp không khớp phân phối nhãn train ({C} lớp): {cw} vs {ref}")
    if binary and list(D["classes"]) != ["Attack", "Normal"]:
        sys.exit(f"Bài toán nhị phân cần lớp ['Attack', 'Normal'], nhận {D['classes']}")


def fit_eval(R, save, ctx, kind, seed, cfg, dev, sigma=None, p=0.0):
    ds, D, T = ctx["ds"], ctx["D"], ctx["T"]
    key = f"{ds}|{kind}|tax|{seed}"
    if key in R:
        return R[key]
    binary, C = ds.endswith("_bin"), D["n_cls"]
    check_labels(D, T, binary)
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if kind in P1.TREES:
        m, info, pv, pt = (run_tree_bin if binary else P1.run_tree)(kind, D, seed, cfg, dev)
        r = dict(val_f1=S.macro_f1(D["y"]["va"], pv, C), **info)
        if binary:                                # cột xác suất xếp theo chỉ số lớp (m.classes_), không giả định
            P = np.asarray(m.predict_proba(np.hstack([D["Z"]["te"], D["C"]["te"]]) if kind == "rf"
                                           else D["F"]["te"]), dtype=np.float64)
            prob = np.zeros((len(P), C))
            prob[:, np.asarray(getattr(m, "classes_", np.arange(C))).astype(int)] = P
    else:
        if kind == "ftt":
            model = FTTransformer(len(D["nums"]), D["cards"], C, cfg).to(dev)
            best, ep = train_ftt(model, T, D, dev, seed, cfg)
        else:
            model = P1.build(kind, D, ctx["gidx"]["tax"], cfg, sigma).to(dev)
            best, ep = S.train_nn(model, T, D, dev, seed, cfg, p)
        z = nn_logits(model, T["xn_te"], T["xc_te"])
        pt, prob = z.argmax(1).cpu().numpy(), torch.softmax(z, 1).double().cpu().numpy()
        r = dict(val_f1=best, epochs=ep, sigma=None if kind == "ftt" else sigma, p_swap=p,
                 params=int(sum(q.numel() for q in model.parameters())))
    if binary:
        r.update(metrics_bin(D["y"]["te"], prob, pt, list(D["classes"]).index("Attack")))
    else:
        r.update(P1.metrics(D["y"]["te"], pt, C))
    r["seconds"] = round(time.time() - t0, 1)
    R[key] = r
    save()
    extra = f"{r['epochs']} epoch" if "epochs" in r else f"{r['trees']} cây"
    main_m = f"F1(Attack) {r['test_f1']:.2f} | FAR {r['test_far']:.2f}" if binary else f"test {r['test_f1']:.2f}"
    print(f"  [{ds}] {kind} seed {seed}: val {r['val_f1']:.2f} | {main_m} | {extra} | {r['seconds']}s", flush=True)
    return r


def param_counts(D, cfg, sigma=1.0):
    gidx = S.group_index(D, D["groups"])
    out = {k: int(sum(q.numel() for q in P1.build(k, D, gidx, cfg, sigma).parameters()))
           for k in ("flat", "concat", "attn", "sfg", "sfg_main")}
    out["ftt"] = int(sum(q.numel() for q in FTTransformer(len(D["nums"]), D["cards"], D["n_cls"], cfg).parameters()))
    return out


# ----------------------------------------------------------------------------
# Tổng hợp
# ----------------------------------------------------------------------------
def pair_stats(vecs, top):
    taus, jacs = [], []
    for a, b in itertools.combinations(vecs, 2):
        taus.append(P1.kendall_tau_b(a, b))
        sa, sb = set(np.argsort(-a)[:top].tolist()), set(np.argsort(-b)[:top].tolist())
        jacs.append(len(sa & sb) / len(sa | sb))
    return taus, jacs


def nanmean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else NAN


def f2(v, spec=".2f"):
    a, b = v
    return (format(a, spec) if math.isfinite(a) else "n/a") + " ± " + (format(b, spec) if math.isfinite(b) else "n/a")


def eg_summary(R, cfg, ds, seeds):
    E = [R[f"eg|{ds}|{s}"] for s in seeds if f"eg|{ds}|{s}" in R]
    if not E:
        return None
    classes, feats, groups = R[f"{ds}|classes"], R[f"{ds}|feats"], R[f"{ds}|groups"]
    gname = {f: g for g, fs in groups.items() for f in fs}
    e = dict(n_models=len(E), n_per_class=dict(zip(classes, E[0]["n"])),
             f1_check=[[x["f1_now"], x["f1_p1"]] for x in E],
             hier_err={k: max(x["hier_err"][k] for x in E) for k in E[0]["hier_err"]},
             flat_err={k: max(x["flat_err"][k] for x in E) for k in E[0]["flat_err"]},
             flat_total_gap=ms([x["flat_total_gap"] for x in E]), conv=R.get(f"eg_conv|{ds}"),
             hier_vs_phi={k: ms([x["hier_vs_phi"][k] for x in E]) for k in E[0]["hier_vs_phi"]},
             flat_vs_phi={k: ms([x["flat_vs_phi"][k] for x in E]) for k in E[0]["flat_vs_phi"]},
             deletion={o: {str(k): ms([x["deletion"][o][i] for x in E]) for i, k in enumerate(cfg["del_k"])}
                       for o in E[0]["deletion"]},
             time_per_sample={k: ms([x["time_per_sample"][k] for x in E]) for k in E[0]["time_per_sample"]},
             xgb_status=sorted({x["xgb_status"] for x in E}),
             xgb_check={k: max(x["xgb_check"][k] for x in E) for k in ("sum_err", "np_vs_frame")}
             if all(x.get("xgb_check") for x in E) else None)
    meths = [m for m in ("imp_hier", "imp_flat", "imp_xgb") if all(x.get(m) is not None for x in E)]
    stab, agree = {}, {}
    for m in meths:
        taus, jacs, nc = [], [], 0
        for c in range(len(classes)):
            vecs = [np.asarray(x[m][c], dtype=float) for x in E]
            if len(vecs) < 2 or not all(np.all(np.isfinite(v)) for v in vecs):
                continue
            t, j = pair_stats(vecs, cfg["top_k"])
            taus += t
            jacs += j
            nc += 1
        stab[m[4:]] = dict(kendall=nanmean(taus), jaccard_top=nanmean(jacs), n_classes=nc)
    for a, b in itertools.combinations(meths, 2):
        taus, jacs = [], []
        for x in E:
            for c in range(len(classes)):
                va, vb = np.asarray(x[a][c], dtype=float), np.asarray(x[b][c], dtype=float)
                if np.all(np.isfinite(va)) and np.all(np.isfinite(vb)):
                    t, j = pair_stats([va, vb], cfg["top_k"])
                    taus += t
                    jacs += j
        agree[f"{a[4:]}_vs_{b[4:]}"] = dict(kendall=nanmean(taus), jaccard_top=nanmean(jacs))
    e["stability"], e["agreement"] = stab, agree
    imp = np.nanmean(np.array([x["imp_hier"] for x in E], dtype=float), 0)          # C x F
    top, top_in_group, ttl = {}, {}, {}
    ttl_idx = [feats.index(f) for f in P1.TTL if f in feats]
    for c, cn in enumerate(classes):
        v = imp[c]
        tot = float(np.nansum(v))
        if not math.isfinite(tot) or tot <= 0:
            continue
        top[cn] = [[feats[i], gname[feats[i]], float(v[i] / tot)] for i in np.argsort(-v)[:cfg["top_k"]]]
        top_in_group[cn] = {}
        for g, fs in groups.items():
            ii = [feats.index(f) for f in fs]
            gs = float(np.sum(v[ii]))
            top_in_group[cn][g] = [[feats[i], float(v[i] / gs) if gs > 0 else NAN]
                                   for i in sorted(ii, key=lambda i: -v[i])[:2]]
        if ttl_idx:
            ttl[cn] = {m[4:]: ms([float(np.sum(np.asarray(x[m][c])[ttl_idx]) / np.sum(x[m][c])) for x in E])
                       for m in ("imp_hier", "imp_xgb") if m in meths}
    e["top_features"], e["top_in_group"], e["ttl_share"] = top, top_in_group, ttl
    return e


def summarize(R, R1, cfg, out, multi, bins):
    seeds, Sm = cfg["seeds"], {"config": R.get("config")}
    rows = [dict(key=k, **{a: b for a, b in v.items() if not isinstance(b, (list, dict))})
            for k, v in R.items() if k.count("|") == 3 and isinstance(v, dict) and "test_f1" in v]
    pd.DataFrame(rows).to_csv(out / "runs_p2.csv", index=False)
    Sm["params"] = {ds: R.get(f"{ds}|params") for ds in multi}
    Sm["ftt_multi"] = {}
    for ds in multi:
        g = lambda RR, k, s, m="test_f1": RR.get(f"{ds}|{k}|tax|{s}", {}).get(m)
        ftt, sfg = [g(R, "ftt", s) for s in seeds], [g(R1, "sfg", s) for s in seeds]
        if all(v is None for v in ftt):
            continue
        Sm["ftt_multi"][ds] = dict(
            ftt_f1=ms(ftt), ftt_acc=ms([g(R, "ftt", s, "test_acc") for s in seeds]),
            ftt_epochs=ms([g(R, "ftt", s, "epochs") for s in seeds]),
            ftt_seconds=ms([g(R, "ftt", s, "seconds") for s in seeds]), sfg_f1_phase1=ms(sfg),
            delta_ftt_minus_sfg=ms([a - b for a, b in zip(ftt, sfg) if a is not None and b is not None]))
    MET = ["test_acc", "test_prec", "test_rec", "test_f1", "test_far", "test_macro_f1", "test_auc"]
    Sm["binary"] = {}
    for ds in bins:
        tab = {}
        base = [R.get(f"{ds}|sfg|tax|{s}") for s in seeds]
        for kind in BIN_MODELS:
            rr = [R.get(f"{ds}|{kind}|tax|{s}") for s in seeds]
            if not any(rr):
                continue
            tab[kind] = {m: ms([x[m] for x in rr if x]) for m in MET}
            for m in ("test_macro_f1", "test_f1", "test_far"):
                tab[kind][f"delta_{m}"] = ms([x[m] - b[m] for x, b in zip(rr, base) if x and b])
        Sm["binary"][ds] = tab
    Sm["eg"] = {ds: e for ds in multi if (e := eg_summary(R, cfg, ds, seeds)) is not None}
    (out / "summary_p2.json").write_text(json.dumps(Sm, indent=1, ensure_ascii=False))
    frows = []
    for ds in multi:
        for s in seeds:
            x = R.get(f"eg|{ds}|{s}")
            if not x:
                continue
            gname = {f: g for g, fs in R[f"{ds}|groups"].items() for f in fs}
            for c, cn in enumerate(R[f"{ds}|classes"]):
                for j, f in enumerate(R[f"{ds}|feats"]):
                    frows.append(dict(ds=ds, seed=s, cls=cn, feature=f, group=gname[f], imp_hier=x["imp_hier"][c][j],
                                      imp_hier_signed=x["imp_hier_signed"][c][j], imp_flat=x["imp_flat"][c][j],
                                      imp_xgb=x["imp_xgb"][c][j] if x.get("imp_xgb") else NAN))
    pd.DataFrame(frows).to_csv(out / "eg_features.csv", index=False)
    report(Sm, cfg)
    return Sm


def report(Sm, cfg):
    print("\n=== FT-Transformer, đa lớp (macro-F1 test; SFG-XIDS lấy từ giai đoạn 1) ===")
    for ds, v in Sm["ftt_multi"].items():
        print(f"  {ds}: FT-T {f2(v['ftt_f1'])} | SFG-XIDS {f2(v['sfg_f1_phase1'])} | FT-T - SFG (ghép seed) "
              f"{f2(v['delta_ftt_minus_sfg'])} | {f2(v['ftt_epochs'], '.1f')} epoch, {f2(v['ftt_seconds'], '.0f')} s")
    for ds, prm in Sm["params"].items():
        if prm:
            print(f"  số tham số [{ds}]: " + ", ".join(f"{NAMES.get(k, k)} {v:,}" for k, v in prm.items()))
    print("\n=== Nhị phân (test; F1/Precision/Recall của lớp Attack) ===")
    for ds, tab in Sm["binary"].items():
        print(f"  {ds}\n    {'Mô hình':<16}{'Acc':>15}{'Prec':>15}{'Rec(DR)':>15}{'F1':>15}{'FAR':>15}"
              f"{'macro-F1':>15}{'AUC':>15}{'Δ macro-F1':>15}")
        for kind, v in tab.items():
            print(f"    {NAMES.get(kind, kind):<16}" + "".join(f"{f2(v[m]):>15}" for m in (
                "test_acc", "test_prec", "test_rec", "test_f1", "test_far", "test_macro_f1", "test_auc",
                "delta_test_macro_f1")))
    for ds, e in Sm["eg"].items():
        print(f"\n=== Tầng 2 (Expected Gradients trong nhóm) - {ds}, {e['n_models']} mô hình, mẫu/lớp "
              f"{e['n_per_class']} ===")
        he = e["hier_err"]
        print(f"  |Σ_(k∈g) EG_k - φ_g| / Σ_g|φ_g|: trung vị {he['median']:.1e}, p99 {he['p99']:.1e}, max "
              f"{he['max']:.1e} (tuyệt đối max {he['max_abs']:.1e})")
        if e["conv"]:
            print("  hội tụ theo số nút (trung vị/max): " + ", ".join(
                f"m={m}: {v['median']:.1e}/{v['max']:.1e} ({v['sec_per_sample']:.2f} s/mẫu)"
                for m, v in e["conv"].items()))
        print(f"  EG phẳng: completeness trung vị {e['flat_err']['median']:.1e}; lệch tổng so với Σφ "
              f"{f2(e['flat_total_gap'], '.3f')}")
        for nm in ("flat_vs_phi", "hier_vs_phi"):
            v = e[nm]
            print(f"  {nm}: trái dấu {f2(v['sign_disagree'], '.3f')} | khác nhóm hạng 1 "
                  f"{f2(v['top1_disagree'], '.3f')} | L1 tỷ trọng {f2(v['share_L1'], '.3f')}")
        print("  xóa đặc trưng (giảm TB p lớp thật): " + " | ".join(
            f"k={k}: " + ", ".join(f"{o} {f2(e['deletion'][o][k], '.3f')}" for o in e["deletion"])
            for k in map(str, cfg["del_k"])))
        print("  ổn định qua seed (Kendall τ-b / Jaccard top-k): " + " | ".join(
            f"{m} {v['kendall']:.3f}/{v['jaccard_top']:.3f}" for m, v in e["stability"].items()))
        print("  đồng thuận giữa phương pháp: " + " | ".join(
            f"{m} {v['kendall']:.3f}/{v['jaccard_top']:.3f}" for m, v in e["agreement"].items()))
        print("  thời gian/mẫu (s): " + ", ".join(f"{k} {f2(v, '.4f')}" for k, v in e["time_per_sample"].items())
              + f" | TreeSHAP: {e['xgb_status']}, kiểm tra {e['xgb_check']}")
        for cn, tf in e["top_features"].items():
            ttl = e["ttl_share"].get(cn)
            extra = (" | TTL: " + ", ".join(f"{m} {f2(v, '.2f')}" for m, v in ttl.items())) if ttl else ""
            print(f"    {cn:<15} " + ", ".join(f"{f} ({g[:4]}, {w:.2f})" for f, g, w in tf) + extra)


# ----------------------------------------------------------------------------
# Dữ liệu & chạy
# ----------------------------------------------------------------------------
def load_data(args, out):
    Ds, datasets = {}, [d.strip() for d in args.datasets.split(",") if d.strip()]
    if "unsw" in datasets:
        tr, te, lab = S.load_unsw(args)
        Ds["unsw"] = S.prepare(tr, te, lab, S.UNSW_GROUPS, S.UNSW_CAT, args.quick)     # như giai đoạn 1
        for df in (tr, te):
            df["y_bin"] = np.where(df["attack_cat"] == "Normal", "Normal", "Attack")
        if "label" in tr.columns:
            bad = sum(int(((pd.to_numeric(df["label"], errors="coerce") == 1) != (df["y_bin"] == "Attack")).sum())
                      for df in (tr, te))
            print(f"UNSW-NB15: số dòng có cột label lệch với attack_cat != Normal: {bad}")
        Ds["unsw_bin"] = S.prepare(tr, te, "y_bin", S.UNSW_GROUPS, S.UNSW_CAT, args.quick)
        check_bin_split(Ds["unsw_bin"], te, args.quick, "UNSW-NB15")
    if "nsl" in datasets:
        tr, te, lab = S.load_nsl(args, out)
        pop = S.nsl_population(tr["attack_type"], te["attack_type"])
        for df in (tr, te):
            df["y_bin"] = np.where(df["y_name"] == "Normal", "Normal", "Attack")
        for key, col in (("nsl", lab), ("nsl_bin", "y_bin")):
            D = S.prepare(tr, te, col, S.NSL_GROUPS, S.NSL_CAT, args.quick, keep_col="attack_type")
            D["explain_mask"] = np.isin(D["te_extra"], pop["evaluated"] + ["normal"])    # như giai đoạn 1
            D["profile_population"] = "seen_train_types"
            Ds[key] = D
        check_bin_split(Ds["nsl_bin"], te, args.quick, "NSL-KDD")
    return Ds


def check_bin_split(D, te, quick, name):
    """Lớp dương của bài toán nhị phân (chỉ số của 'Attack') phải đúng tỷ lệ tấn công trong file test gốc."""
    pos = list(D["classes"]).index("Attack")
    got, want = float((D["y"]["te"] == pos).mean()), float((te["y_bin"] == "Attack").mean())
    print(f"{name} nhị phân: lớp {D['classes']}, Attack = chỉ số {pos}; tỷ lệ Attack test {got:.4f} "
          f"(file gốc {want:.4f}); trọng số lớp {np.round(D['cw'], 4).tolist()}")
    if not quick and abs(got - want) > 1e-12:
        sys.exit(f"{name}: nhãn nhị phân sau prepare() không khớp file gốc.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFG-XIDS giai đoạn 2: EG cấp đặc trưng, nhị phân, FT-Transformer")
    ap.add_argument("--unsw_dir", default=None)
    ap.add_argument("--nsl_dir", default=None)
    ap.add_argument("--datasets", default="unsw,nsl")
    ap.add_argument("--p1_dir", default="full_v1", help="Thư mục kết quả giai đoạn 1 (results.json, models/)")
    ap.add_argument("--out_dir", default="phase2_v1")
    ap.add_argument("--parts", default="eg,bin,ftt", help="eg, bin, ftt (phân cách bằng dấu phẩy)")
    ap.add_argument("--eg_rows", type=int, default=None, help="Giảm nếu hết bộ nhớ GPU ở tầng 2 (mặc định 4194304)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_row_check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    if args.selftest:
        selftest()
    parts = {p.strip() for p in args.parts.split(",") if p.strip()}
    if not parts or not parts <= {"eg", "bin", "ftt"}:
        sys.exit("--parts chỉ gồm eg, bin, ftt")
    if "bin" in parts and (xgb is None or lgb is None):
        sys.exit("Thiếu xgboost hoặc lightgbm. Cài bằng: pip install -U xgboost lightgbm")
    cfg = dict(CFG)
    if args.quick:
        cfg.update(QUICK)
    if args.eg_rows:
        cfg["eg_rows"] = args.eg_rows
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.type != "cuda":
        print("CẢNH BÁO: không thấy GPU - sẽ rất chậm. Trên Colab chọn Runtime -> T4 GPU.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "results.json"
    R = json.loads(rpath.read_text()) if rpath.exists() else {}
    conf = json.loads(json.dumps(dict(version=SCRIPT_VERSION, p1_version=P1.SCRIPT_VERSION, quick=args.quick,
                                      **{k: cfg[k] for k in PROTOCOL_KEYS})))
    if R.get("config") not in (None, conf):
        sys.exit(f"{out}: cấu hình khác lần chạy trước; hãy dùng --out_dir khác.")
    R["config"] = conf

    def save():
        tmp = rpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(R))
        tmp.replace(rpath)

    p1 = Path(args.p1_dir)
    R1 = json.loads((p1 / "results.json").read_text()) if (p1 / "results.json").exists() else {}
    if not R1 and not args.quick:
        sys.exit(f"Không thấy {p1 / 'results.json'} của giai đoạn 1. Kiểm tra --p1_dir.")
    Ds = load_data(args, out)
    multi = [d for d in ("unsw", "nsl") if d in Ds]
    bins = [d for d in ("unsw_bin", "nsl_bin") if d in Ds]
    sel = {}
    for ds in multi:
        sg, pv = R1.get(f"{ds}|sigma"), R1.get(f"{ds}|p")
        if sg is None or pv is None:
            if not args.quick:
                sys.exit(f"results.json của giai đoạn 1 thiếu {ds}|sigma hoặc {ds}|p.")
            sg, pv = 1.0, 0.2
        sel[ds] = (sg, pv)
        R[f"{ds}|params"] = param_counts(Ds[ds], cfg, sg)
    print(f"σ, p lấy từ giai đoạn 1: {sel}")
    ctxs = {}

    def ctx_of(ds):
        if ds not in ctxs:
            ctxs[ds] = P1.make_ctx(ds, Ds[ds], cfg, dev)
        return ctxs[ds]

    t0 = time.time()
    if "eg" in parts:
        for ds in multi:
            run_eg(ctx_of(ds), R, R1, cfg, dev, p1, out, save, args.quick)
    if "bin" in parts:
        for ds in bins:
            sg, pv = sel[ds[:-4]]
            for kind in [k for k in BIN_MODELS if k != "ftt"]:
                for seed in cfg["seeds"]:
                    fit_eval(R, save, ctx_of(ds), kind, seed, cfg, dev, sigma=sg, p=pv if kind == "sfg" else 0.0)
    if "ftt" in parts:
        for ds in multi + bins:
            for seed in cfg["seeds"]:
                fit_eval(R, save, ctx_of(ds), "ftt", seed, cfg, dev)
    print(f"\nXong sau {(time.time() - t0) / 60:.1f} phút.")
    summarize(R, R1, cfg, out, multi, bins)


# ----------------------------------------------------------------------------
# Tự kiểm tra (dữ liệu tổng hợp, CPU, vài chục giây)
# ----------------------------------------------------------------------------
def selftest():
    print("Tự kiểm tra phase2_sfg.py (dữ liệu tổng hợp)...")
    torch.manual_seed(0)
    np.random.seed(0)
    dev, bad = torch.device("cpu"), []

    def check(name, cond, info=""):
        print(f"  [{'OK' if cond else 'LỖI'}] {name} {info}".rstrip(), flush=True)
        if not cond:
            bad.append(name)

    def synth(n, seed):
        rng = np.random.RandomState(seed)
        df = pd.DataFrame({"c0": rng.choice(list("abc"), n), "n0": rng.randn(n), "n1": rng.exponential(1.0, n),
                           "n2": rng.randn(n), "n3": rng.randint(0, 5, n).astype(float), "n4": rng.randn(n),
                           "c1": rng.choice(list("wxyz"), n), "n5": rng.randn(n)})
        s = df["n0"] + 1.5 * (df["c0"] == "a") - df["n2"] * df["n4"] + 0.5 * (df["c1"] == "z") + 0.3 * rng.randn(n)
        df["y"] = np.where(s > 1.0, "A", np.where(s > -0.5, "B", "Normal"))
        df["y_bin"] = np.where(df["y"] == "Normal", "Normal", "Attack")
        return df

    groups, cats = {"G1": ["c0", "n0", "n1"], "G2": ["n2", "n3"], "G3": ["n4", "c1"], "G4": ["n5"]}, ["c0", "c1"]
    tr, te = synth(1500, 1), synth(600, 2)
    cfg = dict(CFG, **QUICK)
    cfg.update(d=16, d_emb=4, n_freq=4, n_bg=24, n_per_class=40, max_ep=3, eg_nodes=16, eg_rows=2 ** 16,
               eg_pts=2 ** 12)
    D = S.prepare(tr, te, "y", groups, cats)
    ctx = P1.make_ctx("syn", D, cfg, dev)
    T, C = ctx["T"], D["n_cls"]
    model = P1.build("sfg", D, ctx["gidx"]["tax"], cfg, 1.0)
    S.train_nn(model, T, D, dev, 0, cfg, 0.2)
    model.eval()
    model.requires_grad_(False)
    bt = torch.as_tensor(ctx["ES"][0])
    bn, bc = T["xn_tr"][bt], T["xc_tr"][bt]
    geo = geometry(D, model, dev)

    E = model.encode(T["xn_te"][:30], T["xc_te"][:30])
    err = 0.0
    for k, (i, j) in enumerate(model.pairs):
        for c in range(C):
            pp = pair_parts(model, k, c)
            ref = cz(model.pair(k, E[i], E[j]), c)
            g1 = torch.stack([cz(model.pair(k, E[i], E[j][b:b + 1].expand_as(E[i])), c) for b in range(7)], 1)
            g2 = torch.stack([cz(model.pair(k, E[i][b:b + 1].expand_as(E[j]), E[j]), c) for b in range(7)], 1)
            err = max(err, (pair_rows(pp, E[i], E[j], True) - ref).abs().max().item(),
                      (pair_rows(pp, E[j], E[i], False) - ref).abs().max().item(),
                      (pair_grid_mean(pp, E[i], E[j][:7], True) - g1.mean(1)).abs().max().item(),
                      (pair_grid_mean(pp, E[j], E[i][:7], False) - g2.mean(1)).abs().max().item())
    check("tách lớp đầu của v_k (pair_rows, pair_grid_mean)", err < 1e-5, f"(sai số {err:.1e})")
    ne, ce = model.emb(T["xn_te"][:30], T["xc_te"][:30])
    e2 = (forward_tokens(model, torch.cat([ne, ce], 1), ne.shape[1]) - model(T["xn_te"][:30], T["xc_te"][:30]))
    check("forward theo embedding = forward của mô hình", e2.abs().max().item() < 1e-5)

    c = int(np.argmax([len(i) for i in ctx["ES"][1]]))
    idx = ctx["ES"][1][c][:5]
    xn, xc = T["xn_te"][idx], T["xc_te"][idx]
    phi = P1.phi_g(model, xn, xc, bn, bc)[0][:, :, c].double().numpy()
    errs = {}
    for m in (4, 16, 64, 256):
        H = hier_eg(model, xn, xc, bn, bc, c, m, cfg["eg_rows"], geo["cols"]).double().numpy()
        errs[m] = float((np.abs(H @ geo["onehot"] - phi) / np.abs(phi).sum(1, keepdims=True)).max())
    # tích phân có bước nhảy đạo hàm (LeakyReLU) -> sai số cầu phương giảm cỡ 1/m, không triệt tiêu ở m nhỏ
    check("Σ_(k∈g) EG_k = φ_g (tầng 2), hội tụ theo số nút", errs[256] < 2e-3 and errs[256] < errs[4] / 10,
          "(max sai số tương đối: " + ", ".join(f"m={m}: {v:.1e}" for m, v in errs.items()) + ")")
    H2 = hier_eg(model, xn, xc, bn, bc, c, 16, 2 ** 12, geo["cols"]).double().numpy()     # lô 1 mẫu
    H3 = hier_eg(model, xn, xc, bn, bc, c, 16, 2 ** 20, geo["cols"]).double().numpy()     # cả lô
    check("tầng 2 không phụ thuộc cách chia lô", np.abs(H2 - H3).max() < 1e-5)
    perms = [np.random.RandomState(PERM_SEED + g).permutation(len(bt)) for g in range(model.G)]
    Fl, tgt = flat_eg(model, xn, xc, bn, bc, c, 256, cfg["eg_pts"], geo, perms)
    e4 = float(((Fl.sum(1) - tgt).abs().double().numpy() / np.abs(phi).sum(1)).max())
    check("Σ EG phẳng = z_c(x) - TB_b z_c(nền) (256 nút)", e4 < 2e-3, f"({e4:.1e})")

    cls = torch.full((len(idx),), c, dtype=torch.long)
    allf = torch.arange(len(D["feats"])).repeat(len(idx), 1)
    dd = deletion(model, xn, xc, bn, bc, cls, {"all": allf}, [len(D["feats"])], 4, geo, 7)
    src = np.random.RandomState(7).randint(0, len(bt), size=4)
    pb = torch.softmax(model(bn[src], bc[src]), 1)[:, c].mean()
    p0 = torch.softmax(model(xn, xc), 1)[:, c].mean()
    check("xóa hết đặc trưng = thay bằng mẫu nền", abs(dd["all"][0] - float(p0 - pb)) < 1e-6)

    ftt = FTTransformer(len(D["nums"]), D["cards"], C, cfg)
    check("FT-Transformer: kích thước đầu ra", tuple(ftt(T["xn_te"][:5], T["xc_te"][:5]).shape) == (5, C))
    mb = metrics_bin(np.array([1, 1, 0, 0, 1, 0]), np.array([[.1, .9], [.6, .4], [.8, .2], [.3, .7], [.2, .8],
                                                             [.9, .1]]), np.array([1, 0, 0, 1, 1, 0]), 1)
    check("độ đo nhị phân (TP 2, FP 1, FN 1, TN 2)", abs(mb["test_acc"] - 400 / 6) < 1e-9
          and abs(mb["test_far"] - 100 / 3) < 1e-9 and abs(mb["test_f1"] - 200 / 3) < 1e-9)

    tmp = Path(tempfile.mkdtemp())
    p1, out = tmp / "p1", tmp / "p2"
    (p1 / "models").mkdir(parents=True)
    out.mkdir()
    R1 = {"syn|sigma": 1.0, "syn|p": 0.2}
    for s in cfg["seeds"]:
        mm = P1.build("sfg", D, ctx["gidx"]["tax"], cfg, 1.0)
        torch.manual_seed(s)
        S.train_nn(mm, T, D, dev, s, cfg, 0.2)
        torch.save(mm.state_dict(), p1 / "models" / f"syn_sfg_tax_{s}.pt")
        R1[f"syn|sfg|tax|{s}"] = {"test_f1": S.macro_f1(D["y"]["te"], S.predict(mm, T["xn_te"], T["xc_te"]), C)}
        if xgb is not None:                       # lưu y như do_run của giai đoạn 1 (XGBClassifier.save_model)
            P1.run_tree("xgb", D, s, cfg, dev)[0].save_model(str(p1 / "models" / f"syn_xgb_tax_{s}.ubj"))
    R = {}
    run_eg(ctx, R, R1, cfg, dev, p1, out, lambda: None, False)
    check("tầng 2: nạp lại mô hình khớp F1 giai đoạn 1",
          all(abs(R[f"eg|syn|{s}"]["f1_now"] - R[f"eg|syn|{s}"]["f1_p1"]) < 1e-9 for s in cfg["seeds"]))
    check("tầng 2: completeness trong lần chạy thử (16 nút)",
          max(R[f"eg|syn|{s}"]["hier_err"]["max"] for s in cfg["seeds"]) < 0.1)
    if xgb is not None:
        xc_ = [R[f"eg|syn|{s}"] for s in cfg["seeds"]]
        check("TreeSHAP: Σ contrib = margin; margin từ mảng (đường giai đoạn 1) = margin từ DataFrame",
              all(x["xgb_status"] == "OK" and x["xgb_check"]["np_vs_frame"] < 1e-5 for x in xc_),
              str([(x["xgb_status"], x["xgb_check"]) for x in xc_]))
    else:
        print("  [BỎ QUA] TreeSHAP: chưa cài xgboost")
    Db = S.prepare(tr, te, "y_bin", groups, cats)
    ctxb = P1.make_ctx("syn_bin", Db, cfg, dev)
    check_bin_split(Db, te, False, "syn")
    check("nhị phân: lớp ['Attack', 'Normal'], trọng số 2 lớp của nhãn nhị phân",
          list(Db["classes"]) == ["Attack", "Normal"] and ctxb["T"]["cw"].numel() == 2)
    try:
        check_labels(Db, ctx["T"], True)          # cố ý dùng trọng số 3 lớp của bài toán đa lớp
        caught = False
    except SystemExit:
        caught = True
    check("chặn được trọng số lớp nhầm bài toán", caught)
    kinds = [k for k in BIN_MODELS if k not in P1.TREES or k == "rf" or (k == "xgb" and xgb) or (k == "lgbm" and lgb)]
    for kind in kinds:
        for s in cfg["seeds"]:
            fit_eval(R, lambda: None, ctxb, kind, s, cfg, dev, sigma=1.0, p=0.2 if kind == "sfg" else 0.0)
    for s in cfg["seeds"]:
        fit_eval(R, lambda: None, ctx, "ftt", s, cfg, dev)
    R["syn|params"] = param_counts(D, cfg)
    summarize(R, R1, cfg, out, ["syn"], ["syn_bin"])
    check("tổng hợp ghi đủ file", all((out / f).exists() for f in ("summary_p2.json", "runs_p2.csv",
                                                                   "eg_features.csv", "eg/syn_0.npz")))
    print("\nTỰ KIỂM TRA:", "ĐẠT" if not bad else f"LỖI ở {bad}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
