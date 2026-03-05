#!/usr/bin/env python3
"""
Save ALL 3 HEATMAPS as PDF + SVG (Illustrator-friendly).

Outputs:
  ./conn_outputs/
    - connection_score_LS_summary.csv
    - heatmap_connection_score.pdf
    - heatmap_connection_score.svg
  ./chaos_outputs/
    - chaos_latent_variable.csv
    - heatmap_chaos_latent_variable.pdf
    - heatmap_chaos_latent_variable.svg
  ./cellsize_outputs_beans/
    - cellsize_beans_LS_summary.csv
    - heatmap_cellsize_beans.pdf
    - heatmap_cellsize_beans.svg

Exclusions:
  L in {1.2, 0.7, 0.3}
  S in {2, 4}

SVG:
  - text stays text (svg.fonttype="none") so Illustrator can edit it.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

# ============================================================
# Pick a font that exists on this machine
# ============================================================
FONT_CANDIDATES = ["Helvetica"]

def pick_font(candidates):
    for name in candidates:
        try:
            fp = font_manager.FontProperties(family=name)
            path = font_manager.findfont(fp, fallback_to_default=False)
            if path and os.path.exists(path):
                return name, path
        except Exception:
            pass
    fp = font_manager.FontProperties()
    path = font_manager.findfont(fp, fallback_to_default=True)
    return fp.get_name(), path

FONT_NAME, FONT_PATH = pick_font(FONT_CANDIDATES)
print(f"[font] Using: {FONT_NAME}")
print(f"[font] Path : {FONT_PATH}")

# ============================================================
# GLOBAL: Illustrator-friendly PDF + SVG settings
# ============================================================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"  # <- critical: keep text editable in SVG
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [FONT_NAME]

# ============================================================
# SHARED CONFIG
# ============================================================
BASE_DIR = "/home/xh300/link/spa/newmethod/SpatialMethod/all_para"
CROPS = ["crop1", "crop2", "crop3", "crop4", "crop5"]
param_re = re.compile(r"L=([\d.]+)S=([\d.]+)")

EXCLUDE_L = {1.2, 0.7, 0.3}
EXCLUDE_S = {2.0, 4.0}

MEAN_FONTSIZE_PT = 12
STD_FONTSIZE_PT  = 8
Y_OFFSET_MEAN = +0.18
Y_OFFSET_STD  = -0.12

COUNT_UNIQUE_XY = False  # beans plot

# ============================================================
# Helpers
# ============================================================
def list_param_folders(base_dir: str):
    items = []
    for folder in sorted(os.listdir(base_dir)):
        m = param_re.match(folder)
        if not m:
            continue
        L = float(m.group(1))
        S = float(m.group(2))
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            items.append((folder_path, L, S))
    return items

def annotate_two_lines(ax, i, j, mval, sval, cmap, norm, fmt_mean="{:.2f}", fmt_std="±{:.2f}"):
    rgba = cmap(norm(mval))
    lum = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
    color = "black" if lum > 0.6 else "white"

    ax.text(j, i + Y_OFFSET_MEAN, fmt_mean.format(mval),
            ha="center", va="center", fontsize=MEAN_FONTSIZE_PT, color=color)
    ax.text(j, i + Y_OFFSET_STD, fmt_std.format(sval),
            ha="center", va="center", fontsize=STD_FONTSIZE_PT, color=color)

def save_pdf_and_svg(fig, out_pdf: str, out_svg: str):
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_svg)  # SVG is vector; dpi irrelevant
    print("[save]", out_pdf)
    print("[save]", out_svg)

# ============================================================
# 1) CONNECTION HEATMAP
# ============================================================
def per_crop_connection_stats(conn_csv: str):
    df = pd.read_csv(conn_csv)
    if "connection_score" not in df.columns:
        raise ValueError(f"Missing 'connection_score' column in {conn_csv}")
    s = pd.to_numeric(df["connection_score"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean()), float(s.std(ddof=0))

def summarize_connection(param_folders):
    records = []
    for folder_path, L, S in param_folders:
        if (L in EXCLUDE_L) or (S in EXCLUDE_S):
            continue
        crop_means = []
        for crop in CROPS:
            conn_csv = os.path.join(folder_path, crop, "connection_score.csv")
            if not os.path.exists(conn_csv):
                continue
            try:
                mean_val, _ = per_crop_connection_stats(conn_csv)
                crop_means.append(mean_val)
            except Exception:
                continue
        if crop_means:
            records.append({
                "L": L, "S": S,
                "mean_conn": float(np.mean(crop_means)),
                "std_conn": float(np.std(crop_means, ddof=0)),
                "n_crops": int(len(crop_means)),
            })
    df = pd.DataFrame(records)
    return df.sort_values(["L", "S"]).reset_index(drop=True) if not df.empty else df

def plot_connection(df: pd.DataFrame, out_pdf: str, out_svg: str):
    if df.empty:
        print("[connection] Empty df, skip.")
        return
    mean_df = df.pivot(index="L", columns="S", values="mean_conn").sort_index(axis=0).sort_index(axis=1)
    std_df  = df.pivot(index="L", columns="S", values="std_conn").loc[mean_df.index, mean_df.columns]

    norm = mcolors.Normalize(vmin=np.nanmin(mean_df.values), vmax=np.nanmax(mean_df.values))
    cmap = plt.cm.RdYlGn  # low=red, high=green

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    im = ax.imshow(mean_df.values, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    im.set_rasterized(True)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Connection Score (higher = better)")

    ax.set_xticks(np.arange(len(mean_df.columns)))
    ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mean_df.index)))
    ax.set_yticklabels(mean_df.index)
    ax.set_xlabel("S parameter")
    ax.set_ylabel("L parameter")
    ax.set_title("Connection Score — mean ± std across crops")

    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            mval = mean_df.values[i, j]
            sval = std_df.values[i, j]
            if np.isnan(mval):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=MEAN_FONTSIZE_PT)
                continue
            annotate_two_lines(ax, i, j, mval, sval, cmap, norm, "{:.2f}", "±{:.2f}")

    save_pdf_and_svg(fig, out_pdf, out_svg)
    plt.show()

# ============================================================
# 2) CHAOS HEATMAP
# ============================================================
import scanpy as sc
LATENT_KEYS_PRIORITY = ["X_scVI", "X_scvi", "X_latent", "X_emb", "X_pca"]

def safe_mode(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]

def load_centroids_and_labels(ct_csv: str):
    ct = pd.read_csv(ct_csv)
    for c in ["cell_id", "x", "y", "leiden_ct"]:
        if c not in ct.columns:
            raise ValueError(f"Missing '{c}' column in {ct_csv}")

    ct["cell_id"] = pd.to_numeric(ct["cell_id"], errors="coerce")
    ct["x"] = pd.to_numeric(ct["x"], errors="coerce")
    ct["y"] = pd.to_numeric(ct["y"], errors="coerce")
    ct_xy = ct.dropna(subset=["cell_id", "x", "y", "leiden_ct"])
    if ct_xy.empty:
        return pd.DataFrame(columns=["cell_id", "leiden_ct"])

    centroid = (
        ct_xy.groupby("cell_id", as_index=False)
             .agg(leiden_ct=("leiden_ct", safe_mode))
    )
    return centroid

def get_latent_matrix(adata: sc.AnnData, n_pcs: int = 30):
    for key in LATENT_KEYS_PRIORITY:
        if key in adata.obsm and adata.obsm[key] is not None:
            X = np.asarray(adata.obsm[key])
            if X.ndim == 2 and X.shape[0] == adata.n_obs and X.shape[1] >= 2:
                return X
    ad = adata.copy()
    sc.pp.pca(ad, n_comps=n_pcs, use_highly_variable=False, svd_solver="arpack")
    return np.asarray(ad.obsm["X_pca"])

def chaos_from_neighbors(coords: np.ndarray, labels: np.ndarray, n_neighbors: int):
    if coords.shape[0] < n_neighbors + 1:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    chaos = [np.mean(labels[idx[i, 1:]] != labels[i]) for i in range(len(labels))]
    return float(np.mean(chaos))

def chaos_latent_for_crop(ct_csv: str, ct_h5ad: str, n_neighbors: int):
    centroid = load_centroids_and_labels(ct_csv)
    if centroid.empty:
        return np.nan

    keep_ids = set(pd.to_numeric(centroid["cell_id"], errors="coerce").dropna().unique())
    adata = sc.read_h5ad(ct_h5ad)
    if "cell_id" not in adata.obs.columns:
        raise ValueError(f"Missing adata.obs['cell_id'] in {ct_h5ad}")
    adata.obs["cell_id"] = pd.to_numeric(adata.obs["cell_id"], errors="coerce")

    adata_sub = adata[adata.obs["cell_id"].isin(keep_ids)].copy()
    if adata_sub.n_obs < n_neighbors + 1:
        return np.nan

    meta = adata_sub.obs[["cell_id"]].merge(centroid[["cell_id", "leiden_ct"]], on="cell_id", how="left")
    valid = meta["leiden_ct"].notna()
    if valid.sum() < n_neighbors + 1:
        return np.nan

    X = get_latent_matrix(adata_sub)[valid.values, :]
    y = meta.loc[valid, "leiden_ct"].values
    return chaos_from_neighbors(X, y, n_neighbors)

def summarize_chaos(param_folders, n_neighbors=6):
    records = []
    for folder_path, L, S in param_folders:
        if (L in EXCLUDE_L) or (S in EXCLUDE_S):
            continue
        vals = []
        for crop in CROPS:
            ct_csv = os.path.join(folder_path, crop, "ct.csv")
            ct_h5ad = os.path.join(folder_path, crop, "ct.h5ad")
            if not (os.path.exists(ct_csv) and os.path.exists(ct_h5ad)):
                continue
            try:
                v = chaos_latent_for_crop(ct_csv, ct_h5ad, n_neighbors)
                if not np.isnan(v):
                    vals.append(v)
            except Exception:
                continue
        if vals:
            records.append({
                "L": L, "S": S,
                "mean_chaos": float(np.mean(vals)),
                "std_chaos": float(np.std(vals, ddof=0)),
                "n_crops": int(len(vals)),
            })
    df = pd.DataFrame(records)
    return df.sort_values(["L", "S"]).reset_index(drop=True) if not df.empty else df

def plot_chaos(df: pd.DataFrame, out_pdf: str, out_svg: str):
    if df.empty:
        print("[chaos] Empty df, skip.")
        return
    mean_df = df.pivot(index="L", columns="S", values="mean_chaos").sort_index(axis=0).sort_index(axis=1)
    std_df  = df.pivot(index="L", columns="S", values="std_chaos").loc[mean_df.index, mean_df.columns]

    norm = mcolors.Normalize(vmin=np.nanmin(mean_df.values), vmax=np.nanmax(mean_df.values))
    cmap = plt.cm.RdYlGn_r  # low=green, high=red

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    im = ax.imshow(mean_df.values, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    im.set_rasterized(True)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean CHAOS (lower = better)")

    ax.set_xticks(np.arange(len(mean_df.columns)))
    ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mean_df.index)))
    ax.set_yticklabels(mean_df.index)
    ax.set_xlabel("S parameter")
    ax.set_ylabel("L parameter")
    ax.set_title("CHAOS (latent kNN) — mean ± std across crops")

    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            mval = mean_df.values[i, j]
            sval = std_df.values[i, j]
            if np.isnan(mval):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=MEAN_FONTSIZE_PT)
                continue
            annotate_two_lines(ax, i, j, mval, sval, cmap, norm, "{:.3f}", "±{:.3f}")

    save_pdf_and_svg(fig, out_pdf, out_svg)
    plt.show()

# ============================================================
# 3) BEANS HEATMAP
# ============================================================
def per_crop_beans_mean(ct_csv: str, count_unique_xy: bool = False):
    ct = pd.read_csv(ct_csv)
    for col in ["cell_id", "x", "y"]:
        if col not in ct.columns:
            raise ValueError(f"Missing '{col}' column in {ct_csv}")

    ct["cell_id"] = pd.to_numeric(ct["cell_id"], errors="coerce")
    ct["x"] = pd.to_numeric(ct["x"], errors="coerce")
    ct["y"] = pd.to_numeric(ct["y"], errors="coerce")
    ct = ct.dropna(subset=["cell_id", "x", "y"])
    if ct.empty:
        return None

    if count_unique_xy:
        ct2 = ct[["cell_id", "x", "y"]].drop_duplicates()
        beans = ct2.groupby("cell_id").size().astype(float)
    else:
        beans = ct.groupby("cell_id").size().astype(float)

    if beans.empty:
        return None
    return float(beans.mean())

def summarize_beans(param_folders):
    records = []
    for folder_path, L, S in param_folders:
        if (L in EXCLUDE_L) or (S in EXCLUDE_S):
            continue
        crop_means = []
        for crop in CROPS:
            ct_csv = os.path.join(folder_path, crop, "ct.csv")
            if not os.path.exists(ct_csv):
                continue
            try:
                v = per_crop_beans_mean(ct_csv, count_unique_xy=COUNT_UNIQUE_XY)
            except Exception:
                continue
            if v is not None and np.isfinite(v):
                crop_means.append(v)
        if crop_means:
            records.append({
                "L": L, "S": S,
                "mean_beans": float(np.mean(crop_means)),
                "std_beans": float(np.std(crop_means, ddof=0)),
                "n_crops": int(len(crop_means)),
            })
    df = pd.DataFrame(records)
    return df.sort_values(["L", "S"]).reset_index(drop=True) if not df.empty else df

def plot_beans(df: pd.DataFrame, out_pdf: str, out_svg: str):
    if df.empty:
        print("[beans] Empty df, skip.")
        return
    mean_df = df.pivot(index="L", columns="S", values="mean_beans").sort_index(axis=0).sort_index(axis=1)
    std_df  = df.pivot(index="L", columns="S", values="std_beans").loc[mean_df.index, mean_df.columns]

    norm = mcolors.Normalize(vmin=np.nanmin(mean_df.values), vmax=np.nanmax(mean_df.values))
    cmap = plt.cm.Blues

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    im = ax.imshow(mean_df.values, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    im.set_rasterized(True)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean beans per cell (higher = larger)")

    ax.set_xticks(np.arange(len(mean_df.columns)))
    ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mean_df.index)))
    ax.set_yticklabels(mean_df.index)
    ax.set_xlabel("S parameter")
    ax.set_ylabel("L parameter")
    suffix = "unique (x,y)" if COUNT_UNIQUE_XY else "all points"
    ax.set_title(f"Cell size (beans per cell) — {suffix}")

    for i in range(mean_df.shape[0]):
        for j in range(mean_df.shape[1]):
            mval = mean_df.values[i, j]
            sval = std_df.values[i, j]
            if np.isnan(mval):
                continue
            annotate_two_lines(ax, i, j, mval, sval, cmap, norm, "{:.2f}", "±{:.2f}")

    save_pdf_and_svg(fig, out_pdf, out_svg)
    plt.show()

# ============================================================
# MAIN
# ============================================================
def main():
    param_folders = list_param_folders(BASE_DIR)
    if not param_folders:
        raise RuntimeError(f"No parameter folders matched pattern in {BASE_DIR}")

    # Connection
    conn_dir = os.path.join(os.getcwd(), "conn_outputs")
    os.makedirs(conn_dir, exist_ok=True)
    df_conn = summarize_connection(param_folders)
    df_conn.to_csv(os.path.join(conn_dir, "connection_score_LS_summary.csv"), index=False)
    plot_connection(
        df_conn,
        os.path.join(conn_dir, "heatmap_connection_score.pdf"),
        os.path.join(conn_dir, "heatmap_connection_score.svg"),
    )

    # CHAOS
    chaos_dir = os.path.join(os.getcwd(), "chaos_outputs")
    os.makedirs(chaos_dir, exist_ok=True)
    df_chaos = summarize_chaos(param_folders, n_neighbors=6)
    df_chaos.to_csv(os.path.join(chaos_dir, "chaos_latent_variable.csv"), index=False)
    plot_chaos(
        df_chaos,
        os.path.join(chaos_dir, "heatmap_chaos_latent_variable.pdf"),
        os.path.join(chaos_dir, "heatmap_chaos_latent_variable.svg"),
    )

    # Beans
    beans_dir = os.path.join(os.getcwd(), "cellsize_outputs_beans")
    os.makedirs(beans_dir, exist_ok=True)
    df_beans = summarize_beans(param_folders)
    df_beans.to_csv(os.path.join(beans_dir, "cellsize_beans_LS_summary.csv"), index=False)
    plot_beans(
        df_beans,
        os.path.join(beans_dir, "heatmap_cellsize_beans.pdf"),
        os.path.join(beans_dir, "heatmap_cellsize_beans.svg"),
    )

    print("[done] Saved PDF + SVG for all three. Font used:", FONT_NAME)

if __name__ == "__main__":
    main()
