"""
SFG-XIDS — ứng dụng phát hiện xâm nhập mạng có giải thích 2 cấp.

    streamlit run app_sfg_xids.py

App nạp checkpoint đã train ở giai đoạn 1 và chạy thật trên từng luồng. Hai chế độ:

  * Giám sát theo lô — chạy tầng 1 (dự đoán + đóng góp nhóm φ) cho cả lô luồng, xếp thành
    hàng đợi cảnh báo theo mức nghi vấn. Đây là phần chạy thường trực khi triển khai.
  * Điều tra một luồng — thêm tầng 2: Expected Gradients trong nhóm cho từng đặc trưng.
    Đắt hơn tầng 1 khoảng ba bậc nên chỉ chạy cho luồng chuyên viên chọn.

Không có số liệu dựng sẵn: mọi con số tính trực tiếp từ checkpoint.

Bố cục mặc định (không cần chép mã nguồn vào thư mục app):

    D:\\data_fusion\\                      <- ROOT: mã nguồn nghiên cứu + kết quả
        step0_sfg.py  full_sfg.py  phase2_sfg.py  step0_unsw.py
        sfg_results\\full_v1\\models\\*.pt
        app\\
            app_sfg_xids.py  requirements.txt  README.md
            data\\            <- KDDTrain+.txt, KDDTest+.txt, UNSW_NB15_*-set.csv

Để mã nguồn ngay trong app\\ cũng chạy được: app dò ROOT trước, rồi tới thư mục app.
Dời chỗ khác thì đặt biến môi trường SFG_ROOT, hoặc sửa hai ô đường dẫn ở thanh bên.
"""
import html
import itertools
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

st.set_page_config(page_title="SFG-XIDS", page_icon="🛡", layout="wide")

# ---------------------------------------------------------------- đường dẫn
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SFG_ROOT", HERE.parent))
SRC = ("step0_sfg.py", "full_sfg.py", "phase2_sfg.py", "step0_unsw.py")
MODELS_REL = Path("sfg_results") / "full_v1" / "models"


def first_dir(cands, ok):
    for c in cands:
        if c is not None and ok(c):
            return Path(c)
    return None


# Mã nguồn nghiên cứu: app\modules\ (chép tay vào cho gọn) trước, rồi mới tới thư mục dự án.
CODE_DIR = first_dir([os.environ.get("SFG_CODE"), HERE / "modules", HERE, ROOT / "modules", ROOT],
                     lambda d: all((Path(d) / f).exists() for f in SRC[:3]))
# Checkpoint: ưu tiên app\model\ (chép tay vào cho gọn), sau đó mới tới thư mục kết quả gốc.
MODEL_DIR = first_dir([os.environ.get("SFG_MODELS"), HERE / "model", HERE / "models",
                       ROOT / MODELS_REL, HERE / MODELS_REL],
                      lambda d: any(Path(d).glob("*_sfg_tax_*.pt"))) or ROOT / MODELS_REL
DATA_DIR = first_dir([HERE / "data", ROOT / "data", ROOT / "app" / "data"],
                     lambda d: d.is_dir()) or HERE / "data"


def seeds_on_disk(model_dir, ds):
    """Seed nào thật sự có checkpoint trong thư mục đang trỏ tới."""
    out = []
    for p in Path(model_dir).glob(f"{ds}_sfg_tax_*.pt"):
        tail = p.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            out.append(int(tail))
    return sorted(out)


def find_results(model_dir):
    """results.json (chỉ để hiện đúng σ) — cạnh checkpoint, thư mục cha, hoặc thư mục kết quả gốc."""
    for p in (Path(model_dir) / "results.json", Path(model_dir).parent / "results.json",
              ROOT / MODELS_REL.parent / "results.json"):
        if p.exists():
            return p
    return None

try:
    import xgboost     # noqa: F401
except ImportError:
    # step0_unsw.py import xgboost ở đầu file, nhưng app chỉ dùng phần đọc dữ liệu UNSW-NB15 của
    # nó, không dùng XGBoost. Đặt một module rỗng để import trót lọt; ai chạm tới thì báo rõ.
    class _NoXGB(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            if name[:1].isupper():      # XGBClassifier, DMatrix, Booster… -> phải báo lỗi
                raise ModuleNotFoundError(
                    f"Tính năng này cần XGBoost (xgboost.{name}). Cài bằng: pip install xgboost")
            return lambda *a, **k: None   # set_config(verbosity=0)… -> bỏ qua
    sys.modules["xgboost"] = _NoXGB("xgboost")

if CODE_DIR is not None:
    sys.path.insert(0, str(CODE_DIR))
    import step0_sfg as S      # noqa: E402
    import full_sfg as P1      # noqa: E402
    import phase2_sfg as P2    # noqa: E402

DS_LABEL = {"nsl": "NSL-KDD (5 lớp)", "unsw": "UNSW-NB15 (10 lớp)"}
SIGMA_FALLBACK = {"nsl": 1.0, "unsw": 10.0}
NSL_FILES = ("KDDTrain+.txt", "KDDTest+.txt")
UNSW_FILES = ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
UNSW_ROWS = (175341, 82332)
GC = ["#2563eb", "#e07a00", "#00897b", "#a63fd9"]        # Basic / Content / Time / Host-Additional
GDESC = {"Basic": "thông tin cơ bản của kết nối", "Content": "nội dung phiên",
         "Time": "lưu lượng 2 giây gần nhất", "Host": "100 kết nối gần nhất tới cùng máy đích",
         "Additional": "đặc trưng tạo thêm"}
MODE_BATCH = "Giám sát theo lô"
MODE_ONE = "Điều tra một luồng"


def preflight(ds, seed, data_dir, model_dir):
    """Mọi thứ app cần, kiểm một lần để báo đủ thay vì chết từng lỗi."""
    need = [(f, (CODE_DIR / f) if CODE_DIR else (HERE / "modules" / f)) for f in SRC]
    need += [(f, Path(data_dir) / f) for f in (NSL_FILES if ds == "nsl" else UNSW_FILES)]
    need += [(f"{ds}_sfg_tax_{seed}.pt", Path(model_dir) / f"{ds}_sfg_tax_{seed}.pt")]
    return [(n, p, p.exists()) for n, p in need]


# ---------------------------------------------------------------- nạp mô hình
@st.cache_resource(show_spinner="Đang nạp dữ liệu và mô hình…")
def load_bundle(ds, seed, data_dir, model_dir, n_bg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = types.SimpleNamespace(nsl_dir=data_dir, unsw_dir=data_dir,
                                 quick=False, skip_row_check=(ds == "unsw"))
    warn = None
    if ds == "nsl":
        tr, te, lab = S.load_nsl(args, Path(data_dir))           # tự kiểm 125.973 / 22.544 dòng
        D = S.prepare(tr, te, lab, S.NSL_GROUPS, S.NSL_CAT, keep_col="attack_type")
    else:
        # find_csvs chia vai trò theo TÊN file; vài bản UNSW trên Kaggle đặt tên ngược nhau
        tr, te, lab = S.load_unsw(args)
        exp_tr, exp_te = UNSW_ROWS
        if (len(tr), len(te)) == (exp_te, exp_tr):
            tr, te = te, tr
            warn = (f"Hai file UNSW-NB15 đặt tên ngược nhau (training-set {exp_te:,} dòng, "
                    f"testing-set {exp_tr:,} dòng). App đã hoán lại theo bộ chia chính thức.")
        if (len(tr), len(te)) != (exp_tr, exp_te):
            raise ValueError(f"Số dòng UNSW-NB15 là {len(tr):,}/{len(te):,}, khác bản chuẩn "
                             f"{exp_tr:,}/{exp_te:,} — tiền xử lý sẽ không khớp mô hình đã train.")
        D = S.prepare(tr, te, lab, S.UNSW_GROUPS, S.UNSW_CAT, keep_col="attack_cat")

    cfg = dict(P2.CFG)
    sigma = SIGMA_FALLBACK[ds]
    res = find_results(model_dir)
    if res is not None:
        R = json.loads(res.read_text(encoding="utf-8"))
        sigma = float(R.get(f"{ds}|sfg|tax|{seed}", {}).get("sigma", sigma))

    model = P1.build("sfg", D, S.group_index(D, D["groups"]), cfg, sigma).to(dev)
    model.load_state_dict(torch.load(Path(model_dir) / f"{ds}_sfg_tax_{seed}.pt", map_location=dev))
    model.eval()

    bi = np.random.RandomState(1234).choice(len(D["y"]["tr"]),
                                            min(n_bg, len(D["y"]["tr"])), replace=False)
    bn = torch.as_tensor(D["Z"]["tr"][bi], device=dev)
    bc = torch.as_tensor(D["C"]["tr"][bi], device=dev)
    geo = P2.geometry(D, model, dev)
    prep = fit_preprocess(D)

    # kiểm tra sức khỏe: mã hóa lại tập test phải trùng Z đã dùng khi train
    chk = float(np.abs(encode_num(D["F"]["te"].head(256), D, prep) - D["Z"]["te"][:256]).max())
    f1 = S.macro_f1(D["y"]["te"], S.predict(model, torch.as_tensor(D["Z"]["te"], device=dev),
                                            torch.as_tensor(D["C"]["te"], device=dev)), D["n_cls"])
    info = dict(dev=str(dev), sigma=sigma, macro_f1=f1, encode_err=chk, n_bg=len(bi), warn=warn,
                n_param=sum(p.numel() for p in model.parameters()))
    return D, model, bn, bc, geo, prep, info


def fit_preprocess(D):
    """Dựng lại đúng tham số tiền xử lý của S.prepare -> app nhận được luồng mới, chưa chuẩn hóa."""
    R = D["F"]["tr"][D["nums"]].to_numpy(np.float64, copy=True)
    nonneg = R.min(0) >= 0
    R[:, nonneg] = np.log1p(np.clip(R[:, nonneg], 0, None))
    mu, sd = R.mean(0), R.std(0)
    sd[sd < 1e-6] = 1.0
    vocab = {c: list(D["F"]["tr"][c].cat.categories) for c in D["cats"]}
    return dict(nonneg=nonneg, mu=mu, sd=sd, vocab=vocab)


def encode_num(df, D, prep):
    A = df[D["nums"]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64, copy=True)
    A[~np.isfinite(A)] = 0.0
    A[:, prep["nonneg"]] = np.log1p(np.clip(A[:, prep["nonneg"]], 0, None))
    return np.clip((A - prep["mu"]) / prep["sd"], -10, 10).astype(np.float32)


def encode_cat(df, D, prep):
    cols = []
    for c in D["cats"]:
        v = prep["vocab"][c]
        s = df[c]
        # D["F"] giữ cột phân loại ở dtype category; pandas 2.x báo lỗi khi fillna bằng giá trị
        # ngoài danh mục, nên đưa về object trước rồi mới điền ô trống.
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = s.astype(object)
        s = s.where(s.notna(), "__nan__").astype(str).str.strip()
        k = s.map({x: i for i, x in enumerate(v)})
        cols.append(k.fillna(len(v) - 1).to_numpy(np.int64))      # giá trị lạ -> __unk__
    return np.stack(cols, 1) if cols else np.zeros((len(df), 0), np.int64)


def analyse(model, bn, bc, geo, xn_np, xc_np, nodes=16):
    """Một luồng: tầng 1 (φ, tương tác cặp) + tầng 2 (EG cấp đặc trưng)."""
    dev = bn.device
    xn, xc = torch.as_tensor(xn_np, device=dev), torch.as_tensor(xc_np, device=dev)
    with torch.no_grad():
        prob = torch.softmax(model(xn, xc), 1)[0].cpu().numpy()
    c = int(np.argmax(prob))
    t0 = time.perf_counter()
    phi_all, g_all, err = P1.phi_g(model, xn, xc, bn, bc)
    t1 = time.perf_counter()
    eg = P2.hier_eg(model, xn, xc, bn, bc, c, nodes, 2 ** 22, geo["cols"])[0].cpu().numpy()
    t2 = time.perf_counter()
    return dict(prob=prob, cls=c, sum_err=err, eg=eg, t_l1=t1 - t0, t_l2=t2 - t1,
                phi=phi_all[0, :, c].cpu().numpy(),
                gint=g_all[0, :, c].cpu().numpy() if g_all is not None else None)


def analyse_batch(model, bn, bc, xn_np, xc_np, chunk=128):
    """Cả lô, CHỈ tầng 1: xác suất từng lớp + φ của lớp dự đoán cho mọi luồng."""
    dev = bn.device
    xn, xc = torch.as_tensor(xn_np, device=dev), torch.as_tensor(xc_np, device=dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        prob = torch.softmax(torch.cat([model(xn[s:s + 4096], xc[s:s + 4096])
                                        for s in range(0, len(xn), 4096)]), 1)
    phi, _, err = P1.phi_g(model, xn, xc, bn, bc, chunk=chunk)
    cls = prob.argmax(1)
    phi_c = phi[torch.arange(len(cls), device=dev), :, cls]
    dt = time.perf_counter() - t0
    return prob.cpu().numpy(), cls.cpu().numpy(), phi_c.cpu().numpy(), err, dt


# ---------------------------------------------------------------- vẽ
def bars(items, colors, signed=False):
    mx = max((abs(v) for _, v in items), default=1.0) or 1.0
    out = []
    for (k, v), col in zip(items, colors):
        sign = "+" if signed and v > 0 else ""
        out.append(
            "<div style='display:grid;grid-template-columns:minmax(120px,190px) 1fr 76px;gap:10px;"
            "align-items:center;margin:6px 0;font-size:13px'>"
            f"<div style='color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
            f"{html.escape(str(k))}</div>"
            "<div style='background:rgba(128,128,128,.16);border-radius:4px;height:16px'>"
            f"<div style='width:{max(2, abs(v) / mx * 100):.1f}%;height:100%;background:{col};"
            f"border-radius:0 4px 4px 0;opacity:{0.45 if signed and v < 0 else 1}'></div></div>"
            "<div style='font-family:ui-monospace,monospace;font-size:12px;text-align:right'>"
            f"{sign}{v:.3f}</div></div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def heat(names, vals):
    pairs = list(itertools.combinations(range(len(names)), 2))
    mx = max((abs(v) for v in vals), default=1.0) or 1.0
    m = dict(zip(pairs, vals))
    h = ["<table style='border-collapse:separate;border-spacing:3px;font-size:12px;"
         "font-family:ui-monospace,monospace'><tr><td></td>"]
    h += [f"<td style='padding:2px 6px;color:#888'>{n[:7]}</td>" for n in names] + ["</tr>"]
    for i, a in enumerate(names):
        h.append(f"<tr><td style='padding:2px 6px;color:#888'>{a[:7]}</td>")
        for j in range(len(names)):
            if i == j:
                h.append("<td style='width:60px;height:34px;text-align:center;color:#888;"
                         "background:rgba(128,128,128,.10);border-radius:4px'>–</td>")
                continue
            v = m.get((min(i, j), max(i, j)), 0.0)
            t = abs(v) / mx
            h.append(f"<td style='width:60px;height:34px;text-align:center;border-radius:4px;"
                     f"background:rgba(37,99,235,{0.10 + 0.60 * t});"
                     f"color:{'#fff' if t > 0.55 else 'inherit'}'>{v:+.2f}</td>")
        h.append("</tr>")
    st.markdown("".join(h) + "</table>", unsafe_allow_html=True)


# ---------------------------------------------------------------- lấy luồng vào
def read_csv_flows(D, up):
    """CSV người dùng tải lên -> (DataFrame các luồng, lỗi)."""
    df = pd.read_csv(up)
    miss = [c for c in D["feats"] if c not in df.columns]
    if miss:
        return None, f"Thiếu cột: {miss[:8]}{'…' if len(miss) > 8 else ''}"
    return df.reset_index(drop=True), None


def pick_flow(D):
    """Một luồng. Trả về (dòng thô, mô tả nguồn) hoặc (None, '')."""
    tab_test, tab_edit, tab_csv = st.tabs(["Từ tập test", "Sửa tay một luồng", "Tải file CSV"])
    row, src = None, ""
    n_te = len(D["y"]["te"])

    with tab_test:
        pick = st.selectbox("Lọc theo nhãn thật", ["(tất cả)"] + D["classes"])
        pool = (np.arange(n_te) if pick == "(tất cả)"
                else np.where(D["y"]["te"] == D["classes"].index(pick))[0])
        if len(pool) == 0:
            st.warning("Tập test không có dòng nào thuộc lớp này.")
            pool = np.arange(n_te)
        c1, c2 = st.columns([3, 1])
        with c2:
            if st.button("🎲 Lấy ngẫu nhiên", use_container_width=True):
                st.session_state["idx"] = int(np.random.choice(pool))
        with c1:
            idx = int(st.number_input(f"Chỉ số dòng trong tập test (0–{n_te - 1})", 0, n_te - 1,
                                      int(min(st.session_state.get("idx", pool[0]), n_te - 1))))
        st.session_state["idx"] = idx
        row = D["F"]["te"].iloc[[idx]].reset_index(drop=True)
        extra = D["te_extra"][idx] if D["te_extra"] is not None else ""
        src = (f"dòng {idx} của tập test · nhãn thật **{D['classes'][D['y']['te'][idx]]}**"
               + (f" ({extra})" if extra else ""))

    with tab_edit:
        st.caption("Sửa giá trị rồi tick ô bên dưới — app tiền xử lý lại đúng như khi train.")
        base_i = int(min(st.session_state.get("idx", 0), n_te - 1))
        base = D["F"]["te"].iloc[[base_i]].reset_index(drop=True)
        # astype(str): cột "giá trị" trộn số với chuỗi, để nguyên object thì data_editor kén kiểu.
        # key theo chỉ số dòng: đổi luồng gốc thì bảng nạp lại, không giữ giá trị cũ
        ed = st.data_editor(base.astype(str).T.rename(columns={0: "giá trị"}),
                            use_container_width=True, height=320, key=f"editor_{base_i}")
        if st.checkbox("Dùng luồng đã sửa"):
            row = ed.T.reset_index(drop=True)
            src = f"luồng do người dùng nhập (gốc: dòng {base_i})"

    with tab_csv:
        up = st.file_uploader("CSV có đủ các cột đặc trưng (một dòng = một luồng)", type=["csv"],
                             key="csv_one")
        if up is not None:
            df, err = read_csv_flows(D, up)
            if err:
                st.error(err)
            else:
                k = int(st.number_input("Dòng trong file", 0, len(df) - 1, 0))
                row, src = df.iloc[[k]].reset_index(drop=True), f"dòng {k} của file `{up.name}`"

    return row, src


# ---------------------------------------------------------------- chế độ theo lô
def batch_mode(D, model, bn, bc, prep, gnames):
    st.subheader("1 · Nguồn lưu lượng")
    tab_te, tab_csv = st.tabs(["Lấy mẫu từ tập test", "Tải file CSV"])
    flows, ytrue, ids, src = None, None, None, ""

    with tab_te:
        c1, c2, c3 = st.columns([1, 1, 1])
        n = int(c1.select_slider("Số luồng", [100, 250, 500, 1000, 2000], value=500))
        only = c2.selectbox("Chỉ lấy nhãn thật", ["(tất cả)"] + D["classes"], key="b_cls")
        bseed = int(c3.number_input("Seed lấy mẫu", 0, 9999, 0))
        pool = (np.arange(len(D["y"]["te"])) if only == "(tất cả)"
                else np.where(D["y"]["te"] == D["classes"].index(only))[0])
        take = np.random.RandomState(bseed).choice(pool, min(n, len(pool)), replace=False)
        take = np.sort(take)
        if st.button("▶ Chạy tầng 1 trên lô này", type="primary", key="b_run_te"):
            flows = D["F"]["te"].iloc[take].reset_index(drop=True)
            ytrue, ids = D["y"]["te"][take], take
            src = (f"{len(take):,} luồng lấy ngẫu nhiên từ tập test"
                   + ("" if only == "(tất cả)" else f", nhãn thật {only}") + f" (seed {bseed})")

    with tab_csv:
        up = st.file_uploader("CSV có đủ các cột đặc trưng (mỗi dòng một luồng)", type=["csv"],
                              key="csv_batch")
        if up is not None:
            df, err = read_csv_flows(D, up)
            if err:
                st.error(err)
            elif st.button("▶ Chạy tầng 1 trên file này", type="primary", key="b_run_csv"):
                flows, ids = df, np.arange(len(df))
                src = f"{len(df):,} luồng từ file `{up.name}`"

    if flows is not None:
        with st.spinner(f"Đang chạy tầng 1 trên {len(flows):,} luồng…"):
            prob, cls, phi_c, err, dt = analyse_batch(
                model, bn, bc, encode_num(flows, D, prep), encode_cat(flows, D, prep))
        st.session_state["batch"] = dict(prob=prob, cls=cls, phi=phi_c, err=err, dt=dt,
                                         ids=ids, ytrue=ytrue, src=src, n=len(flows))

    if "batch" not in st.session_state:
        st.info("Chọn nguồn lưu lượng rồi bấm **Chạy tầng 1**.")
        return
    B = st.session_state["batch"]
    show_batch(D, gnames, B)


def show_batch(D, gnames, B):
    cls_names = D["classes"]
    prob, cls, phi = B["prob"], B["cls"], B["phi"]
    normal = cls_names.index("Normal") if "Normal" in cls_names else None
    risk = 1.0 - prob[:, normal] if normal is not None else prob.max(1)
    alert = cls != normal if normal is not None else np.ones(len(cls), bool)

    st.subheader("2 · Toàn cảnh lô")
    st.caption("Nguồn: " + B["src"])
    k = st.columns(4)
    k[0].metric("Luồng đã xử lý", f"{B['n']:,}")
    k[1].metric("Bị gắn cảnh báo", f"{int(alert.sum()):,}", f"{100 * alert.mean():.1f}% của lô")
    k[2].metric("Thông lượng tầng 1", f"{B['n'] / max(B['dt'], 1e-9):,.0f} luồng/s",
                f"{1000 * B['dt'] / B['n']:.2f} ms/luồng", delta_color="off")
    if B["ytrue"] is not None:
        k[3].metric("Đúng nhãn thật", f"{100 * (cls == B['ytrue']).mean():.1f}%",
                    f"macro-F1 lô {S.macro_f1(B['ytrue'], cls, len(cls_names)):.1f}%",
                    delta_color="off")
    else:
        k[3].metric("Sai số Σφ = z − E[z]", f"{B['err']:.1e}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Lô chia theo nhãn dự đoán**")
        cnt = pd.Series(cls).value_counts().sort_values(ascending=False)
        bars([(cls_names[i], float(v)) for i, v in cnt.items()],
             ["#5b6472" if (normal is not None and i == normal) else "#c0392b" for i in cnt.index])
    with c2:
        st.markdown("**Nhóm đặc trưng chi phối quyết định** (argmax |φ|)")
        dom = np.abs(phi).argmax(1)
        dcnt = pd.Series(dom).value_counts()
        bars([(f"{gnames[i]} — {GDESC.get(gnames[i], '')}", float(v)) for i, v in dcnt.items()],
             [GC[i % len(GC)] for i in dcnt.index])

    st.subheader("3 · Hàng đợi cảnh báo")
    st.caption("Xếp theo mức nghi vấn = 1 − P(Normal). Bốn cột φ là đóng góp của từng nhóm vào "
               "logit của lớp được dự đoán — chuyên viên đọc ngay được vì sao luồng bị nêu.")
    only_alert = st.checkbox("Chỉ hiện luồng bị gắn cảnh báo", value=True)
    tbl = pd.DataFrame({
        "luồng": B["ids"],
        "dự đoán": [cls_names[c] for c in cls],
        "xác suất": prob.max(1),
        "mức nghi vấn": risk,
        "nhóm chi phối": [gnames[i] for i in np.abs(phi).argmax(1)],
    })
    if B["ytrue"] is not None:
        tbl.insert(1, "nhãn thật", [cls_names[c] for c in B["ytrue"]])
        tbl.insert(2, "khớp", np.where(cls == B["ytrue"], "✓", "✗"))
    for g, name in enumerate(gnames):
        tbl[f"φ {name}"] = phi[:, g]
    view = tbl[alert] if only_alert else tbl
    view = view.sort_values("mức nghi vấn", ascending=False)
    st.dataframe(
        view, use_container_width=True, hide_index=True, height=420,
        column_config={
            "mức nghi vấn": st.column_config.ProgressColumn(format="%.3f", min_value=0.0,
                                                            max_value=1.0),
            "xác suất": st.column_config.NumberColumn(format="%.3f"),
            **{f"φ {n}": st.column_config.NumberColumn(format="%+.2f") for n in gnames}})
    st.download_button("⬇ Tải hàng đợi (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                       "sfg_xids_alerts.csv", "text/csv")

    st.subheader("4 · Đưa một luồng sang điều tra sâu")
    st.caption("Tầng 2 (EG cấp đặc trưng) đắt hơn tầng 1 khoảng ba bậc nên không chạy cho cả lô. "
               "Chọn một luồng để mở tầng 2.")
    if B["ytrue"] is None:
        st.info("Lô này đến từ file CSV nên không có chỉ số trong tập test; sang tab "
                "**Điều tra một luồng → Tải file CSV** và chọn đúng dòng đó.")
        return
    top = view["luồng"].tolist()[:200]
    pick = st.selectbox("Luồng (theo thứ tự hàng đợi)", top,
                        format_func=lambda i: f"dòng {i} · {tbl.loc[tbl['luồng'] == i, 'dự đoán'].iloc[0]}"
                                              f" · nghi vấn {tbl.loc[tbl['luồng'] == i, 'mức nghi vấn'].iloc[0]:.3f}")
    if st.button("🔎 Mở phân tích 2 cấp cho luồng này", type="primary"):
        st.session_state["idx"] = int(pick)
        # không ghi thẳng vào key "mode": widget đó đã dựng ở lần chạy này. Để lại yêu cầu,
        # main() áp dụng ở đầu lần chạy sau, trước khi dựng radio.
        st.session_state["mode_goto"] = MODE_ONE
        st.session_state.pop("result", None)
        st.rerun()


# ---------------------------------------------------------------- giao diện
def main():
    st.title("SFG-XIDS — phát hiện xâm nhập có giải thích theo nhóm đặc trưng")

    if "mode_goto" in st.session_state:        # yêu cầu đổi chế độ từ lần chạy trước
        st.session_state["mode"] = st.session_state.pop("mode_goto")

    with st.sidebar:
        st.header("Cấu hình")
        st.radio("Chế độ", [MODE_BATCH, MODE_ONE], key="mode")
        ds = st.selectbox("Bộ dữ liệu", ["nsl", "unsw"], format_func=DS_LABEL.get, key="ds")
        model_dir = st.text_input("Thư mục checkpoint", str(MODEL_DIR),
                                  help="Nơi chứa các file {bộ dữ liệu}_sfg_tax_{seed}.pt")
        have = seeds_on_disk(model_dir, ds)
        seed = st.selectbox("Seed mô hình", have or [0], index=0, key="seed",
                            help=("Chỉ liệt kê seed có checkpoint trong thư mục trên."
                                  if have else "Không thấy checkpoint nào trong thư mục trên."))
        data_dir = st.text_input(
            "Thư mục dữ liệu (KDDTrain+.txt / KDDTest+.txt)" if ds == "nsl"
            else "Thư mục dữ liệu (UNSW_NB15_training-set.csv / testing-set.csv)", str(DATA_DIR))
        n_bg = st.select_slider("Số luồng nền (tập B)", [32, 64, 128, 256], value=128,
                                help="Nền lớn thì lời giải thích ổn định hơn nhưng chạy chậm hơn.")
        nodes = st.select_slider("Số nút tích phân (EG)", [4, 8, 16, 32], value=16,
                                 help="16 nút cho sai số khớp 2 cấp khoảng 0,1%.")
    mode = st.session_state.get("mode", MODE_BATCH)

    checks = preflight(ds, seed, data_dir, model_dir)
    if not all(ok for _, _, ok in checks):
        st.error("Thiếu file. Bảng dưới liệt kê những thứ app cần và nơi nó đang tìm.")
        st.dataframe(pd.DataFrame([{"file": n, "trạng thái": "có" if ok else "THIẾU",
                                    "đường dẫn": str(p)} for n, p, ok in checks]),
                     use_container_width=True, hide_index=True)
        st.info(f"Mã nguồn đang tìm ở `{CODE_DIR or ROOT}`. Dời chỗ thì đặt biến môi trường "
                "`SFG_ROOT`, hoặc sửa hai ô đường dẫn ở thanh bên.")
        st.stop()

    try:
        D, model, bn, bc, geo, prep, info = load_bundle(ds, seed, data_dir, model_dir, n_bg)
    except (Exception, SystemExit) as e:                            # noqa: BLE001
        st.error(f"Không nạp được mô hình: {type(e).__name__}: {e}")
        st.stop()

    gnames = list(D["groups"].keys())
    with st.sidebar:
        st.divider()
        st.caption(f"Thiết bị **{info['dev']}** · σ = {info['sigma']:g} · nền {info['n_bg']} luồng")
        st.caption(f"Macro-F1 test **{info['macro_f1']:.2f}%** · {info['n_param']:,} tham số")
        st.caption(f"Kiểm tra tiền xử lý: lệch tối đa {info['encode_err']:.2e}")
    if info.get("warn"):
        st.warning(info["warn"])

    # đổi bộ dữ liệu hoặc seed -> kết quả cũ vô nghĩa
    if st.session_state.get("bundle") != (ds, seed):
        st.session_state["bundle"] = (ds, seed)
        st.session_state.pop("result", None)
        st.session_state.pop("batch", None)
        st.session_state.pop("idx", None)

    if mode == MODE_BATCH:
        batch_mode(D, model, bn, bc, prep, gnames)
        return

    st.subheader("1 · Chọn luồng mạng cần kiểm tra")
    raw_row, source = pick_flow(D)
    if raw_row is None:
        st.stop()
    st.caption("Luồng đang xét: " + source)
    with st.expander("Xem giá trị thô của luồng"):
        # astype(str): cột dọc trộn số với chuỗi, để object thì Arrow không chuyển được
        st.dataframe(raw_row[D["feats"]].astype(str).T.rename(columns={0: "giá trị"}),
                     use_container_width=True, height=280)

    stamp = (ds, seed, n_bg, nodes, source)
    if st.button("▶ Phân tích luồng này", type="primary"):
        with st.spinner("Đang chạy mô hình và tính lời giải thích 2 cấp…"):
            out = analyse(model, bn, bc, geo, encode_num(raw_row, D, prep),
                          encode_cat(raw_row, D, prep), nodes)
        st.session_state["result"] = (stamp, raw_row, out)

    if "result" not in st.session_state:
        st.stop()
    old, raw_row, out = st.session_state["result"]
    if old != stamp:
        st.warning(f"Kết quả bên dưới là của lần phân tích trước ({old[4]}). "
                   "Bấm **Phân tích luồng này** để chạy lại với lựa chọn hiện tại.")

    show_result(D, geo, gnames, raw_row, out)


def show_result(D, geo, gnames, raw_row, out):
    cls, p = D["classes"][out["cls"]], float(out["prob"][out["cls"]])
    order = np.argsort(out["prob"])[::-1]

    st.subheader("2 · Kết luận của mô hình")
    k1, k2, k3 = st.columns(3)
    k1.metric("Nhãn dự đoán", cls, "bình thường" if cls == "Normal" else "CẢNH BÁO",
              delta_color="normal" if cls == "Normal" else "inverse")
    k2.metric("Xác suất", f"{p * 100:.1f}%")
    k3.metric("Lớp khả dĩ thứ hai", D["classes"][order[1]], f"{out['prob'][order[1]] * 100:.1f}%")
    bars([(D["classes"][i], float(out["prob"][i])) for i in order[:5]], ["#5b6472"] * 5)

    st.subheader("3 · Vì sao — đóng góp của từng nhóm đặc trưng")
    st.caption("φᵢ là đóng góp của nhóm i vào logit của lớp được dự đoán, tính bằng công thức đóng "
               "và đúng bằng giá trị Shapley của nhóm đó.")
    bars([(f"{g} — {GDESC.get(g, '')}", float(v)) for g, v in zip(gnames, out["phi"])], GC, True)
    st.caption(f"Σφᵢ = {out['phi'].sum():+.3f} = z_c − E[z_c] · "
               f"sai số đẳng thức {out['sum_err']:.1e} · tầng 1 mất {1000 * out['t_l1']:.1f} ms")

    if out["gint"] is not None:
        st.markdown("**Tương tác thuần giữa các cặp nhóm** — phần chỉ xuất hiện khi cả hai nhóm "
                    "cùng có mặt")
        heat(gnames, [float(v) for v in out["gint"]])

    st.subheader("4 · Mở nhóm xuống từng đặc trưng")
    gsel = st.radio("Nhóm", gnames, horizontal=True, index=int(np.argmax(np.abs(out["phi"]))))
    gi = gnames.index(gsel)
    names = [D["feats"][j] for j in geo["cols"][gi]]
    vals = [float(out["eg"][j]) for j in geo["cols"][gi]]
    top = np.argsort(np.abs(vals))[::-1][:8]
    st.caption("Đóng góp của từng đặc trưng trong nhóm, tính bằng Expected Gradients.")
    bars([(f"{names[i]} = {raw_row[names[i]].iloc[0]}", vals[i]) for i in top],
         [GC[gi % len(GC)]] * len(top), True)
    st.caption(f"Σ EG trong nhóm = {sum(vals):+.3f} · φ_{gsel} = {out['phi'][gi]:+.3f} · lệch "
               f"{abs(sum(vals) - out['phi'][gi]) / (np.abs(out['phi']).sum() + 1e-12) * 100:.2f}% "
               f"tổng độ lớn đóng góp (sai số tính tích phân) · tầng 2 mất {out['t_l2']:.2f} s")

    st.subheader("5 · Tóm tắt cho chuyên viên")
    gt = gnames[int(np.argmax(np.abs(out["phi"])))]
    fn = [D["feats"][j] for j in geo["cols"][gnames.index(gt)]]
    fv = [float(out["eg"][j]) for j in geo["cols"][gnames.index(gt)]]
    best = np.argsort(np.abs(fv))[::-1][:2]
    st.info(f"Mô hình kết luận **{cls}** với xác suất {p * 100:.1f}%. Nhóm **{gt}** "
            f"({GDESC.get(gt, '')}) tác động mạnh nhất, đóng góp "
            f"{out['phi'][gnames.index(gt)]:+.2f} đơn vị logit — chủ yếu do "
            + " và ".join(f"`{fn[i]}` = {raw_row[fn[i]].iloc[0]} ({fv[i]:+.2f})" for i in best) + ".")
    st.caption("Mọi con số được tính trực tiếp từ checkpoint đã train.")


if __name__ == "__main__":
    main()
