#!/usr/bin/env python3
import os, glob, warnings
from collections import Counter
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree
from sklearn.metrics import adjusted_rand_score, mean_squared_error, f1_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
# from sklearn.metrics import silhouette_score  # <- still disabled per request
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, rankdata
from scipy.stats import  linregress

warnings.filterwarnings("ignore")

# ==============================
# CONFIG
# ==============================
GLOBAL_CONFIG = {
    "xenium_spots_path": "/home/lc1418/ST/aligned_filled_cells_integer.csv",
    "gt_labels_path": "/home/xh300/link/spa/newmethod/ResultsForFengwei/gt_results/leiden_ct.csv",
    "gt_adata_path": "/home/xh300/link/spa/newmethod/ResultsForFengwei/gt_results/leiden_ct.h5ad",
    "output_dir": "./para_metrics",
    "sampling_size": None,  # None = all cells
    "gt_x_col": "x_he_px",
    "gt_y_col": "y_he_px",
    "gt_cell_id_col": "cell_id",
    "random_seed": 42,
}
os.makedirs(GLOBAL_CONFIG["output_dir"], exist_ok=True)

TOLERANCE_RADII = [1,5,10]
MIN_MATCHING_SPOTS = 5
LOW_OVERLAP_GENE_THRESHOLD = 30  # applies to ALL methods

# Chaos controls
CHAOS_AVERAGE = "micro"      
CHAOS_NORMALIZE = "diag"     
LEIDEN_COL = "leiden_ct"  

# IMPORTANT: predicted cell type column is "celltypist label" for ALL methods
EVALUATION_RUNS = {}

crop = ['crop1','crop2','crop3','crop4','crop5']
L = ['0','0.05','0.1','0.3','0.5','1.0']
S = ['1','2','3','4','5','7','10']

for i in crop:
    for l in L:
        for s in S:
            EVALUATION_RUNS[f'L:{l}{i}'] = {
            "pred_spots_path": f"/home/xh300/link/spa/newmethod/SpatialMethod/parameter_eva/L/L={l},S=3/gt_{i}/ct.csv",
            "our_adata_path": f"/home/xh300/link/spa/newmethod/SpatialMethod/parameter_eva/L/L={l},S=3/gt_{i}/ct.h5ad",
            "pred_x_col": "x",
            "pred_y_col": "y",
            "gt_cell_type_col": "cell_type",     # from GT CSV
            "pred_cell_id_col": "cell_id",
            "pred_cell_type_col": "leiden_ct",   # <-- unified
            "our_adata_cell_id_col": None,
            "needs_norm": False,}
EXCLUDED_CELL_TYPES = {
    "plasmacytoid dendritic cell",
    "mast cell",
    "neutrophil",
    "megakaryocyte",
}

# ==============================
# HELPERS
# ==============================
def normalize_ids(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def inspect_and_load_df(path, name):
    print(f"--- Loading {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}\n"); return None
    try:
        df = pd.read_csv(path)
        print(f"{name} loaded. Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        print("-"*50); return df
    except Exception as e:
        print(f"ERROR reading {name}: {e}\n"); return None

def ad_n_vars(ad):
    try:
        return ad.n_vars
    except Exception:
        return ad.shape[1] if hasattr(ad, "shape") else "?"

def inspect_and_load_adata(path, name):
    print(f"--- Loading {name} ---")
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}\n"); return None
    try:
        ad = sc.read_h5ad(path)
        print(f"{name} loaded. obs={ad.n_obs}, vars={ad_n_vars(ad)}")
        print("-"*50); return ad
    except Exception as e:
        print(f"ERROR reading {name}: {e}\n"); return None

def _row_to_1d(x):
    """Convert a single row (sparse or dense) to 1D numpy array without modifying values."""
    if hasattr(x, "toarray"):
        return x.toarray().ravel()
    return np.asarray(x).ravel()

def _spearman_dense(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    r, _ = pearsonr(ra, rb); return float(r)

def _cos_jsd(a, b):
    a = a.values.reshape(1, -1); b = b.values.reshape(1, -1)
    return (float(sk_cosine_similarity(a, b)[0, 0]), float(jensenshannon(a.ravel(), b.ravel())))

def compute_chaos_1nn(pred_spots, xcol, ycol, labelcol, average="micro", normalize="diag"):
    """
    CHAOS (1-NN) within same label. Returns (raw_mean_1nn, normalized_percent).
    """
    needed = {xcol, ycol, labelcol}
    if not needed.issubset(pred_spots.columns):
        return None, None

    # normalization scale
    if normalize == "diag":
        dx = float(pred_spots[xcol].max() - pred_spots[xcol].min())
        dy = float(pred_spots[ycol].max() - pred_spots[ycol].min())
        diag = float(np.hypot(dx, dy)) if (dx > 0 or dy > 0) else np.nan
    else:
        diag = np.nan

    per_ct_means, all_nn = [], []
    for _, dfct in pred_spots.groupby(labelcol):
        if len(dfct) < 2: continue
        coords = dfct[[xcol, ycol]].to_numpy(float)
        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)  # 0=self (0), 1=true 1-NN
        nn = dists[:, 1]
        per_ct_means.append(float(np.mean(nn)))
        all_nn.append(nn)

    if not all_nn:
        return None, None

    raw = float(np.mean(per_ct_means)) if str(average).lower() == "macro" else float(np.mean(np.concatenate(all_nn)))
    norm = float((raw / diag) * 100.0) if (normalize == "diag" and np.isfinite(diag) and diag > 0) else None
    return raw, norm

GLOBAL_DISTRIBUTIONS = []
GLOBAL_MAX_RMSE = 0.0

def compute_cell_attribute_metrics(pred_spots: pd.DataFrame, run_name: str, radius: int, outdir: str):
    """
    Cell size = #spots per cell_id (from leiden_ct files).
    Saves per-cell-type mean_cell_size and cell_count; returns overall means for summary_wide.
    """
    req = {"cell_id", "cell_type_pred"}
    if not req.issubset(pred_spots.columns):
        return np.nan, 0

    per_cell = pred_spots.groupby("cell_id").size().rename("n_spots").reset_index()
    ps = pred_spots.merge(per_cell, on="cell_id", how="left")
    cells_table = ps.drop_duplicates(["cell_id", "cell_type_pred"])[["cell_id", "cell_type_pred", "n_spots"]]

    per_ct = (
        cells_table.groupby("cell_type_pred")
        .agg(mean_cell_size=("n_spots", "mean"), cell_count=("cell_id", "nunique"))
        .reset_index()
        .rename(columns={"cell_type_pred": "cell_type"})
    )
    per_ct["method"] = run_name
    per_ct["radius"] = radius

    cellattr_dir = os.path.join(outdir, "per_cellattribute"); os.makedirs(cellattr_dir, exist_ok=True)
    per_ct_csv = os.path.join(cellattr_dir, f"{run_name}_R{radius}_cell_attribute.csv")
    per_ct.to_csv(per_ct_csv, index=False)
    print(f"cell-attribute per-CT -> {per_ct_csv}")

    overall_mean_cell_size = float(cells_table["n_spots"].mean()) if len(cells_table) else np.nan
    overall_cell_count = int(cells_table["cell_id"].nunique()) if len(cells_table) else 0
    return overall_mean_cell_size, overall_cell_count

# ==============================
# CORE
# ==============================
def run_evaluation(config, run_name, tolerance_radius):
    global GLOBAL_DISTRIBUTIONS, GLOBAL_MAX_RMSE
    rng = np.random.default_rng(GLOBAL_CONFIG["random_seed"])

    print("="*60)
    print(f"Starting evaluation for: {run_name} with radius {tolerance_radius}")
    print("="*60)

    xen_spots = inspect_and_load_df(GLOBAL_CONFIG["xenium_spots_path"], "Xenium spots CSV")
    gt_labels = inspect_and_load_df(GLOBAL_CONFIG["gt_labels_path"], "Ground truth labels CSV")
    xen_adata = inspect_and_load_adata(GLOBAL_CONFIG["gt_adata_path"], "Xenium expression H5AD")
    pred_spots = inspect_and_load_df(config["pred_spots_path"], "Predicted spots CSV")
    pred_spots = pred_spots.dropna()
    our_adata  = inspect_and_load_adata(config["our_adata_path"],  "Predicted expression H5AD")
    if any(x is None for x in [xen_spots, gt_labels, xen_adata, pred_spots, our_adata]):
        print(f"Skipping {run_name} due to missing inputs.\n"); return
    pred_spots.columns = pred_spots.columns.str.strip().str.replace('"', '')
    # --- Step 1: Merge GT ---
    print("\n=== Step 1: Merge GT ===")
    # normalize GT id column if needed
    if "Unnamed: 0" in gt_labels.columns and GLOBAL_CONFIG["gt_cell_id_col"] not in gt_labels.columns:
        gt_labels = gt_labels.rename(columns={"Unnamed: 0": GLOBAL_CONFIG["gt_cell_id_col"]})
    if "Unnamed: 0.1" in gt_labels.columns and GLOBAL_CONFIG["gt_cell_id_col"] not in gt_labels.columns:
        gt_labels = gt_labels.rename(columns={"Unnamed: 0.1": GLOBAL_CONFIG["gt_cell_id_col"]})

    merged = pd.merge(
        xen_spots, gt_labels,
        on=GLOBAL_CONFIG["gt_cell_id_col"],
        how="inner", suffixes=("_spot", "_label"),
    )
    rename_map = {}
    if GLOBAL_CONFIG["gt_x_col"] in merged.columns: rename_map[GLOBAL_CONFIG["gt_x_col"]] = "x_gt"
    if GLOBAL_CONFIG["gt_y_col"] in merged.columns: rename_map[GLOBAL_CONFIG["gt_y_col"]] = "y_gt"
    if config["gt_cell_type_col"] in merged.columns: rename_map[config["gt_cell_type_col"]] = "cell_type_gt"
    merged = merged.rename(columns=rename_map)

    for req in ["x_gt", "y_gt", "cell_type_gt"]:
        if req not in merged.columns:
            print(f"Missing required column '{req}' after merge."); return

    before_gt = len(merged)
    merged = merged[~merged["cell_type_gt"].astype(str).isin(EXCLUDED_CELL_TYPES)].copy()
    removed_gt = before_gt - len(merged)
    print(f"Filter GT: removed {removed_gt} rows with excluded CTs {sorted(EXCLUDED_CELL_TYPES)}")

    # --- Step 2: Normalize predicted spot IDs/labels (no expression normalization) ---
    print("\n=== Step 2: Normalize predicted spot IDs/labels ===")
    need_cols = [config["pred_cell_id_col"], config["pred_cell_type_col"]]
    if any(c not in pred_spots.columns for c in need_cols):
        print("Predicted spots missing cell_id or cell_type columns."); return

    pred_spots = pred_spots.dropna(subset=[config["pred_cell_id_col"]]).copy()
    pred_spots["cell_id"] = normalize_ids(pred_spots[config["pred_cell_id_col"]])
    if "10x" in run_name:
        pred_spots["cell_id"] = pred_spots["cell_id"].str.extract(r"(\d+)", expand=False).fillna("")

    # *** THE canonical predicted type we use everywhere ***
    pred_spots["cell_type_pred"] = (
        pred_spots[config["pred_cell_type_col"]].astype(str).str.replace("^p_ct_", "", regex=True)
    )

    before_pred = len(pred_spots)
    pred_spots = pred_spots[~pred_spots["cell_type_pred"].isin(EXCLUDED_CELL_TYPES)].copy()
    removed_pred = before_pred - len(pred_spots)
    print(f"Filter Pred: removed {removed_pred} rows with excluded CTs {sorted(EXCLUDED_CELL_TYPES)}")

    # --- Step 3: Align AnnData IDs ---
    print("\n=== Step 3: Align AnnData IDs ===")
    if "cell_id" in xen_adata.obs.columns:
        xen_adata.obs_names = xen_adata.obs["cell_id"].astype(str)
    else:
        xen_adata.obs_names = xen_adata.obs_names.astype(str)

    if config["our_adata_cell_id_col"] and config["our_adata_cell_id_col"] in our_adata.obs.columns:
        our_adata.obs_names = our_adata.obs[config["our_adata_cell_id_col"]].astype(str)
    elif "assigned_cell_id" in our_adata.obs.columns:
        our_adata.obs_names = our_adata.obs["assigned_cell_id"].astype(str)
    else:
        our_adata.obs_names = our_adata.obs_names.astype(str)
    our_adata.obs_names = normalize_ids(our_adata.obs_names)
    if "10x" in run_name:
        our_adata.obs_names = our_adata.obs_names.str.extract(r"(\d+)", expand=False).fillna("")

    
    # --- Step 4: Gene alignment (force Xenium order exactly) ---
    print("\n=== Step 4: Gene alignment (force xenium order) ===")
    xen_adata.var_names = xen_adata.var_names.astype(str)
    our_adata.var_names = our_adata.var_names.astype(str)

# Find common genes while strictly preserving Xenium order
    common = [g for g in xen_adata.var_names if g in our_adata.var_names]

    if not common:
        print("No common genes. Expression metrics will be skipped.")
        xen_adata = xen_adata[:, []]
        our_adata = our_adata[:, []]
    else:
    # Slice both AnnData objects using the same ordered list
        xen_adata = xen_adata[:, common]
        our_adata = our_adata[:, common]

    # Verify alignment
        assert np.all(xen_adata.var_names == our_adata.var_names), \
            "❌ Gene order mismatch after alignment!"
        print(f"✅ {len(common)} common genes aligned in Xenium order.")

    # Optional sanity print
        print("First 5 genes (xenium):", xen_adata.var_names[:5].tolist())
        print("First 5 genes (ours):  ", our_adata.var_names[:5].tolist())

    # --- Step 5: Proportions & Chaos (1-NN) ---
    print("\n=== Step 5: Proportions & Chaos (1-NN) ===")

# --- Inject Leiden labels from the corresponding .h5ad ---
    if LEIDEN_COL in our_adata.obs.columns:
        print(f"[INFO] Found '{LEIDEN_COL}' in our_adata.obs; merging into pred_spots.")
        leiden_map = our_adata.obs[[LEIDEN_COL]].copy()
        leiden_map.index = leiden_map.index.astype(str)
        pred_spots = pred_spots.merge(
            leiden_map,
            left_on="cell_id",
            right_index=True,
            how="left",
        )
    else:
        print(f"[WARN] '{LEIDEN_COL}' not found in {run_name} H5AD; using fallback 'cell_type_pred'.")
        pred_spots[LEIDEN_COL] = pred_spots["cell_type_pred"]

# --- DEBUG check ---
    print(f"[DEBUG] Unique Leiden labels for {run_name}: {pred_spots[LEIDEN_COL].dropna().unique()[:10]}")
    print(f"[DEBUG] Coordinate ranges (GT vs Pred):")
    print(" GT x:", float(merged['x_gt'].min()), "→", float(merged['x_gt'].max()))
    print(" GT y:", float(merged['y_gt'].min()), "→", float(merged['y_gt'].max()))
    print(" PR x:", float(pred_spots[config['pred_x_col']].min()), "→", float(pred_spots[config['pred_x_col']].max()))
    print(" PR y:", float(pred_spots[config['pred_y_col']].min()), "→", float(pred_spots[config['pred_y_col']].max()))

# --- Cell & spot-level proportions (used for Cosine/JSD metrics) ---
    cell_counts_gt = (
        merged.drop_duplicates([GLOBAL_CONFIG["gt_cell_id_col"], "cell_type_gt"])["cell_type_gt"]
        .value_counts(normalize=True)
        .sort_index()
    )
    cell_counts_pred = (
        pred_spots.drop_duplicates(["cell_id", "cell_type_pred"])["cell_type_pred"]
        .value_counts(normalize=True)
        .sort_index()
    )
    cell_prop_df = pd.concat([cell_counts_gt, cell_counts_pred], axis=1).fillna(0.0)
    cell_prop_df.columns = ["Ground Truth (cells)", "Predicted (cells)"]

    spot_prop_df = pd.concat(
        [
            merged["cell_type_gt"].value_counts(normalize=True).sort_index(),
            pred_spots["cell_type_pred"].value_counts(normalize=True).sort_index(),
        ],
        axis=1,
    ).fillna(0.0)
    spot_prop_df.columns = ["Ground Truth (spots)", "Predicted (spots)"]

    cos_cell, jsd_cell = _cos_jsd(cell_prop_df["Ground Truth (cells)"], cell_prop_df["Predicted (cells)"])
    cos_spot, jsd_spot = _cos_jsd(spot_prop_df["Ground Truth (spots)"], spot_prop_df["Predicted (spots)"])

    print(f"Cell-Level Cosine: {cos_cell:.4f}  JSD: {jsd_cell:.4f}")
    print(f"Spot-Level  Cosine: {cos_spot:.4f}  JSD: {jsd_spot:.4f}")

# --- CHAOS computation using Leiden labels ---
    chaos_raw_1nn, chaos_score_normalized = compute_chaos_1nn(
        pred_spots,
        xcol=config["pred_x_col"],
        ycol=config["pred_y_col"],
        labelcol=LEIDEN_COL,
        average=CHAOS_AVERAGE,
        normalize=CHAOS_NORMALIZE,
    )

    if chaos_score_normalized is not None:
        print(f"Chaos Score (1-NN, % of diag): {chaos_score_normalized:.4f}   [raw mean 1-NN: {chaos_raw_1nn:.4f}]")
    else:
        print("Chaos Score unavailable.")

    # --- Step 6: Matching & Gene Metrics ---
    print("\n=== Step 6: Matching & Gene Metrics (both-nonzero within common genes) ===")
    merged = merged.dropna(subset=["cell_type_gt"]).copy()

    if GLOBAL_CONFIG["sampling_size"] is not None:
        uniq = merged[GLOBAL_CONFIG["gt_cell_id_col"]].unique()
        if GLOBAL_CONFIG["sampling_size"] < len(uniq):
            pick = np.random.choice(uniq, size=GLOBAL_CONFIG["sampling_size"], replace=False)
            merged = merged[merged[GLOBAL_CONFIG["gt_cell_id_col"]].isin(pick)]
            print(f"Sampled {len(pick)} unique cells")
        else:
            print(f"sampling_size >= #unique cells ({len(uniq)}); using all")
    else:
        print("Running on all cells (no sampling).")

    # Access rows on-demand
    xen_X = xen_adata.X
    our_X = our_adata.X
    xen_obs_to_idx = {str(k): i for i, k in enumerate(xen_adata.obs_names.astype(str))}
    our_obs_to_idx = {str(k): i for i, k in enumerate(our_adata.obs_names.astype(str))}
    pred_tree = KDTree(pred_spots[[config["pred_x_col"], config["pred_y_col"]]].to_numpy())

    results = []
    no_match_spot_count = 0
    total_spots = len(merged)

    matched_cells = 0
    expr_pairs = 0
    overlap_pairs = 0

    low_overlap_count = 0
    nuclei_candidate_ids = set()

    for xen_cell_id, grp in merged.groupby(GLOBAL_CONFIG["gt_cell_id_col"]):
        xen_cell_id = str(xen_cell_id)
        xen_ct = grp["cell_type_gt"].iloc[0]

        coords = grp[["x_gt", "y_gt"]].to_numpy(float)
        neighbor_indices = []
        hits = pred_tree.query_radius(coords, r=float(tolerance_radius))
        for arr in hits:
            if len(arr) == 0:
                no_match_spot_count += 1
            else:
                neighbor_indices.extend(arr.tolist())

        if len(neighbor_indices) > 0:
            cand_ids = pred_spots.iloc[list(set(neighbor_indices))]["cell_id"].astype(str).tolist()
            nuclei_candidate_ids.update(cand_ids)

        if len(neighbor_indices) < MIN_MATCHING_SPOTS:
            results.append({
                "xenium_cell_id": xen_cell_id,
                "xenium_cell_type": xen_ct,
                "predicted_cell_type": "no_match",
                "expression_corr_pearson": np.nan,
                "expression_corr_spearman": np.nan,
                "expression_cosine": np.nan,
                "expression_rmse": np.nan,
                "matching_pred_cell_ids": "[]",
                "chosen_pred_cell_id": "",
                "genes_used": 0,
                "nnz_xen": 0,
                "nnz_pred": 0,
                "low_overlap_warning": False
            })
            continue

        matched_cells += 1

        matched_rows = pred_spots.iloc[neighbor_indices].copy()
        matched_rows["cell_id"] = normalize_ids(matched_rows["cell_id"])
        if "10x" in run_name:
            matched_rows["cell_id"] = matched_rows["cell_id"].str.extract(r"(\d+)", expand=False).fillna("")
        matched_ids = matched_rows["cell_id"].tolist()

        counts = Counter(matched_ids)
        max_count = max(counts.values())
        top_cids = [cid for cid, c in counts.items() if c == max_count]
        chosen_cid = top_cids[0] if len(top_cids) == 1 else rng.choice(top_cids)

        row0 = matched_rows.loc[matched_rows["cell_id"] == str(chosen_cid)].iloc[0]
        pred_type = str(row0["cell_type_pred"])  # <-- always from celltypist label

        pcc = spearman_r = cos_sim = rmse = np.nan
        genes_used = nnz_x = nnz_p = 0
        low_flag = False

        if (xen_cell_id in xen_obs_to_idx) and (str(chosen_cid) in our_obs_to_idx) and (our_adata.n_vars > 0):
            expr_pairs += 1
            xi = xen_obs_to_idx[xen_cell_id]
            j  = our_obs_to_idx[str(chosen_cid)]
            xen_vec  = _row_to_1d(xen_X[xi])
            pred_vec = _row_to_1d(our_X[j])

            mask = (xen_vec != 0) & (pred_vec != 0)
            genes_used = int(mask.sum())
            nnz_x = int((xen_vec != 0).sum())
            nnz_p = int((pred_vec != 0).sum())
            # === DEBUG: print gene-level overlap details ===
            if genes_used < 70:  # or any cutoff you like for inspection
                gene_names = xen_adata.var_names.to_numpy()
                mask_x = (xen_vec != 0)
                mask_p = (pred_vec != 0)
                mask_both = mask_x & mask_p

                nnz_x = int(mask_x.sum())
                nnz_p = int(mask_p.sum())
                both_nz = int(mask_both.sum())

                common_gene_names = gene_names[mask_both]
                common_x_values = xen_vec[mask_both]
                common_p_values = pred_vec[mask_both]

                print(f"\n[DEBUG] Cell pair {xen_cell_id} vs {chosen_cid}")
                print(f"  nnz_x={nnz_x}, nnz_p={nnz_p}, both_nonzero={both_nz}")
                print(f"  First 10 common genes:")
                for g, vx, vp in zip(common_gene_names[:10], common_x_values[:10], common_p_values[:10]):
                    print(f"    {g:20s}  Xen={vx:.3g}  Pred={vp:.3g}")

    # Optional: show a few genes that are nonzero in one side only
                only_x = gene_names[mask_x & ~mask_p]
                only_p = gene_names[mask_p & ~mask_x]
                print(f"  Unique-to-Xenium genes (first 5): {only_x[:5].tolist()}")
                print(f"  Unique-to-Pred genes (first 5): {only_p[:5].tolist()}")



            if genes_used >= 2:
                overlap_pairs += 1
                xv = xen_vec[mask]; pv = pred_vec[mask]
                # --- DEBUG: check if either vector is too flat (few unique values) ---
                ux = np.unique(np.round(xv, 6)).size
                up = np.unique(np.round(pv, 6)).size
                #print('xv ', xv)
                #print('pv ', pv)
                if up <= 3 or ux <= 3:
                    print(f"[DEBUG] Flat ranks? cell {xen_cell_id} vs {chosen_cid}: "
                    f"genes_used={genes_used}, unique_x={ux}, unique_p={up}, "
                    f"std_x={np.std(xv):.3g}, std_p={np.std(pv):.3g}")

                try:
                    if (np.std(xv) > 0) and (np.std(pv) > 0):
                        pcc, _ = pearsonr(xv, pv)
                        spearman_r = _spearman_dense(xv, pv)
                        #print('pcc', pcc)
                        #print('spearman', spearman_r)
                    if (np.linalg.norm(xv) * np.linalg.norm(pv)) > 0:
                        cos_sim = float(sk_cosine_similarity(xv.reshape(1, -1), pv.reshape(1, -1))[0, 0])
                    rmse = float(np.sqrt(mean_squared_error(xv, pv)))
                except Exception:
                    pass

            if genes_used < LOW_OVERLAP_GENE_THRESHOLD:
                low_flag = True
                low_overlap_count += 1

        results.append({
            "xenium_cell_id": xen_cell_id,
            "xenium_cell_type": xen_ct,
            "predicted_cell_type": pred_type,  # <-- used for accuracy/F1/ARI/balanced accuracy
            "expression_corr_pearson": pcc,
            "expression_corr_spearman": spearman_r,
            "expression_cosine": cos_sim,
            "expression_rmse": rmse,
            "matching_pred_cell_ids": str(matched_ids),
            "chosen_pred_cell_id": str(chosen_cid),
            "genes_used": genes_used,
            "nnz_xen": nnz_x,
            "nnz_pred": nnz_p,
            "low_overlap_warning": low_flag
        })

    results_df = pd.DataFrame(results)

    base = f"{run_name}_R{tolerance_radius}"
    out_dir = GLOBAL_CONFIG["output_dir"]
    per_cell_dir = os.path.join(out_dir, "per_cell"); os.makedirs(per_cell_dir, exist_ok=True)
    per_ct_dir   = os.path.join(out_dir, "per_ct");   os.makedirs(per_ct_dir,   exist_ok=True)
    plots_cell_dir = os.path.join(out_dir, "plots_per_cell"); os.makedirs(plots_cell_dir, exist_ok=True)

    per_cell_csv = os.path.join(per_cell_dir, f"{base}_per_cell_metrics.csv")
    results_df.to_csv(per_cell_csv, index=False)
    print(f"per-cell metrics -> {per_cell_csv}")

    ct_counts = results_df["xenium_cell_type"].value_counts().rename_axis("cell_type").reset_index(name="n_cells")
    by_ct_csv = os.path.join(per_ct_dir, f"{base}_by_celltype_counts.csv")
    ct_counts.to_csv(by_ct_csv, index=False)
    print(f"per-CT counts -> {by_ct_csv}")

    # --- Cell attribute metrics
    overall_mean_cell_size, overall_cell_count = compute_cell_attribute_metrics(
        pred_spots, run_name, tolerance_radius, out_dir
    )
    # === DEBUG: signal alignment / covariance diagnostic ===

    os.makedirs("checkingput", exist_ok=True)
    p_10x = None
    p_b2c = None
    p_our = None
    x = None


# --- pick one matched cell (any valid index with enough genes) ---
    if 'xen_vec' in locals() and len(xen_vec) > 0:
    # Reuse the last xen_vec/pred_vec from the loop if you didn’t store all
        x = xen_vec
        p_our = pred_vec
    # If you have separate adatas for 10x and b2c, load them here for the same cell:
    # (replace these with actual matched cell lookups if needed)
    # p_10x = tenx_adata.X[tenx_obs_to_idx[cell_id_10x]].toarray().ravel()
    # p_b2c = b2c_adata.X[b2c_obs_to_idx[cell_id_b2c]].toarray().ravel()
    else:
        print("[WARN] xen_vec not found from loop; skipping diagnostic.")
        p_our = p_10x = p_b2c = x = None

    if x is not None and p_our is not None:
    # compute Pearson r and covariance
        def cov(a,b): return np.cov(a,b)[0,1]
        print("\n=== GENE-WISE ALIGNMENT CHECK ===")
        for name, v in [("10x", p_10x), ("ourmethod", p_our), ("b2c", p_b2c)]:
            if v is None: continue
            r, _ = pearsonr(x, v)
            print(f"{name:10s}  r = {r:.4f},  covariance = {cov(x,v):.4f},  var_pred = {np.var(v):.4f},  var_gt = {np.var(x):.4f}")

    # scatter plot with regression lines
        plt.figure(figsize=(6,6))
        colors = {'10x':'tab:blue', 'ourmethod':'tab:orange', 'b2c':'tab:green'}
        for name, v in [("10x", p_10x), ("ourmethod", p_our), ("b2c", p_b2c)]:
            if v is None: continue
            plt.scatter(x, v, label=name, alpha=0.6, s=10, color=colors[name])
            slope, intercept, r, p, stderr = linregress(x, v)
            xx = np.linspace(min(x), max(x), 100)
            plt.plot(xx, intercept + slope*xx, color=colors[name], lw=1.2)
        plt.xlabel("Xenium expression")
        plt.ylabel("Predicted expression")
        plt.title("Gene-wise expression alignment for one cell")
        plt.legend()
        plt.tight_layout()
        plt.savefig("checkingput/gene_alignment_check.png", dpi=200)
        plt.close()
        print("[saved] checkingput/gene_alignment_check.png")


    # === EXTRA DEBUG: sanity check top few gene pairs ===
    print("\nTop 10 gene pairs (GT vs Pred) for inspection:")
    pairs = list(zip(x[:10], p_our[:10]))
    for i, (gx, gp) in enumerate(pairs):
        print(f"{i:02d}: GT={gx:.3f}, Pred={gp:.3f}")

# Check global stats
    print(f"\nMean GT={np.mean(x):.4f}, mean Pred={np.mean(p_our):.4f}")
    print(f"Min/Max GT=({np.min(x):.4f}, {np.max(x):.4f}), Pred=({np.min(p_our):.4f}, {np.max(p_our):.4f})")

    # --- Step 7: Summaries & Save ---
    print("\n=== Step 7: Summaries & Save ===")
    matched_df = results_df[results_df["predicted_cell_type"] != "no_match"].copy()
    num_matched_cells = len(matched_df)

    if not matched_df.empty:
        try:
            ari = adjusted_rand_score(matched_df["xenium_cell_type"], matched_df["predicted_cell_type"])
        except Exception:
            ari = np.nan
        accuracy = float((matched_df["xenium_cell_type"] == matched_df["predicted_cell_type"]).mean())
        try:
            f1_macro = f1_score(matched_df["xenium_cell_type"], matched_df["predicted_cell_type"], average="macro")
        except Exception:
            f1_macro = np.nan
        try:
            balanced_acc = balanced_accuracy_score(
                matched_df["xenium_cell_type"].astype(str), matched_df["predicted_cell_type"].astype(str)
            )
        except Exception:
            balanced_acc = np.nan

        # confusion matrix (labels sorted alphabetically)
        try:
            labels_sorted = sorted(matched_df["xenium_cell_type"].astype(str).unique().tolist())
            cm = confusion_matrix(
                matched_df["xenium_cell_type"].astype(str),
                matched_df["predicted_cell_type"].astype(str),
                labels=labels_sorted
            )
            cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
            cm_csv = os.path.join(per_ct_dir, f"{base}_confusion_matrix.csv")
            cm_df.to_csv(cm_csv)
            print(f"confusion matrix -> {cm_csv}")
        except Exception as e:
            print(f"[warn] confusion matrix failed: {e}")
        
        pcc_vals = matched_df["expression_corr_pearson"].to_numpy(dtype=float)
        spearman_vals = matched_df["expression_corr_spearman"].to_numpy(dtype=float)
        pcc_vals = pcc_vals[np.isfinite(pcc_vals)]
        spearman_vals = spearman_vals[np.isfinite(spearman_vals)]
        plot_df = pd.DataFrame({
            "Pearson r": pcc_vals,
            "Spearman ρ": spearman_vals
        }).melt(var_name="Metric", value_name="Correlation")

        plt.figure(figsize=(6, 4.5))
        sns.histplot(data=plot_df, x="Correlation", hue="Metric", kde=True, bins=30, alpha=0.6)
        plt.title("Distribution of Pearson and Spearman Correlations per Matched Cell")
        plt.xlabel("Correlation value")
        plt.ylabel("Count")
        plt.xlim(-1, 1)
        plt.tight_layout()
        run_name = run_name.replace(':', '_').replace(' ', '_')
        plt.savefig(f'{run_name}correlation_plots.png')
        mean_pcc      = float(np.nanmean(matched_df["expression_corr_pearson"].to_numpy(dtype=float)))
        mean_spearman = float(np.nanmean(matched_df["expression_corr_spearman"].to_numpy(dtype=float)))
        mean_cos      = float(np.nanmean(matched_df["expression_cosine"].to_numpy(dtype=float)))
        mean_rmse     = float(np.nanmean(matched_df["expression_rmse"].to_numpy(dtype=float)))
        no_match_cells = float((results_df["predicted_cell_type"] == "no_match").mean() * 100.0)

        avg_genes_used = float(np.nanmean(matched_df["genes_used"].to_numpy(dtype=float)))
        low_overlap_cells = int(low_overlap_count)
        low_overlap_pct = float(low_overlap_cells / max(1, len(matched_df)) * 100.0)
    else:
        ari = accuracy = f1_macro = mean_pcc = mean_spearman = mean_cos = mean_rmse = np.nan
        balanced_acc = np.nan
        no_match_cells = 100.0
        avg_genes_used = np.nan
        low_overlap_cells = 0
        low_overlap_pct = np.nan
    no_match_spots_pct = float(no_match_spot_count / max(1, total_spots) * 100.0)
    nuclei_candidates = int(len(nuclei_candidate_ids))

    print(f"Matched cells (≥{MIN_MATCHING_SPOTS} neighbor spots): {matched_cells}")
    print(f"Expression pairs: {expr_pairs} | Overlap pairs (≥2 both-nonzero): {overlap_pairs}")
    print(f"Matched cells used for accuracy: {num_matched_cells}")

    summary_df = pd.DataFrame({
        "Metric": [
            "ARI", "Cell Type Accuracy", "Macro F1 Score",
            "Mean PCC", "Mean Spearman R", "Mean Cosine", "Mean RMSE",
            "Cell-Level Cosine Sim", "Cell-Level JSD",
            "Spot-Level Cosine Sim", "Spot-Level JSD",
            "Chaos Score (1-NN, % diag)", "Chaos Raw 1-NN",
            "No Match Spot %", "No Match Cell %",
            "Avg Both-Nonzero Genes (per matched cell)",
            f"Low-Overlap Cells (<{LOW_OVERLAP_GENE_THRESHOLD} both-nonzero)",
            "Low-Overlap % of Matched Cells",
            "Silhouette (Leiden, spatial)",
            "Balanced Accuracy",
            "Overall Mean Cell Size (#spots/cell)",
            "Overall Cell Count",
            "Nuclei Candidates (unique pred cell_id within radius)",
        ],
        "Value": [
            ari, accuracy, f1_macro,
            mean_pcc, mean_spearman, mean_cos, mean_rmse,
            cos_cell, jsd_cell,
            cos_spot, jsd_spot,
            (chaos_score_normalized if chaos_score_normalized is not None else np.nan),
            (chaos_raw_1nn if chaos_raw_1nn is not None else np.nan),
            no_match_spots_pct, no_match_cells,
            avg_genes_used,
            low_overlap_cells,
            low_overlap_pct,
            np.nan,  # silhouette disabled
            balanced_acc,
            overall_mean_cell_size,
            overall_cell_count,
            nuclei_candidates,
        ]
    })

    detailed_csv = os.path.join(out_dir, f"{base}_detailed_results.csv")
    summary_long_csv = os.path.join(out_dir, f"{base}_summary_metrics.csv")
    results_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_long_csv, index=False)

    summary_row = {
        "method": run_name, "radius": tolerance_radius,
        "ARI": ari, "cell_type_accuracy": accuracy, "macro_f1": f1_macro,
        "mean_pcc": mean_pcc, "mean_spearman": mean_spearman, "mean_cosine": mean_cos, "mean_rmse": mean_rmse,
        "cell_level_cosine": cos_cell, "cell_level_jsd": jsd_cell,
        "spot_level_cosine": cos_spot, "spot_level_jsd": jsd_spot,
        "chaos_score": (chaos_score_normalized if chaos_score_normalized is not None else np.nan),
        "chaos_score_raw_1nn": (chaos_raw_1nn if chaos_raw_1nn is not None else np.nan),
        "no_match_spot_pct": no_match_spots_pct, "no_match_cell_pct": no_match_cells,
        "avg_both_nonzero_genes": avg_genes_used,
        "matched_cells": matched_cells, "expr_pairs": expr_pairs, "overlap_pairs": overlap_pairs,
        "matched_cells_used": num_matched_cells,
        "silhouette_score": np.nan,  # disabled
        "balanced_accuracy": balanced_acc,
        "overall_mean_cell_size": overall_mean_cell_size,
        "overall_cell_count": overall_cell_count,
        "nuclei_candidates": nuclei_candidates,
    }
    wide_csv = os.path.join(out_dir, f"{base}_summary_wide.csv")
    pd.DataFrame([summary_row]).to_csv(wide_csv, index=False)

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)

    # store distributions for global violins
    GLOBAL_DISTRIBUTIONS.append({
        "run_name": run_name,
        "radius": tolerance_radius,
        "corr_arrays": {
            "PCC": np.array([x for x in results_df["expression_corr_pearson"].to_numpy(float) if not np.isnan(x)], dtype=float),
            "Spearman": np.array([x for x in results_df["expression_corr_spearman"].to_numpy(float) if not np.isnan(x)], dtype=float),
            "Cosine": np.array([x for x in results_df["expression_cosine"].to_numpy(float) if not np.isnan(x)], dtype=float),
        },
        "rmse_array": np.array([x for x in results_df["expression_rmse"].to_numpy(float) if not np.isnan(x)], dtype=float),
        "base": base,
    })
    if np.isfinite(results_df["expression_rmse"]).any():
        GLOBAL_MAX_RMSE = max(GLOBAL_MAX_RMSE, float(np.nanmax(results_df["expression_rmse"])))

    # Per-cell violin + strip
    plot_df = results_df[results_df["predicted_cell_type"] != "no_match"].copy()
    if not plot_df.empty:
        metrics = [
            ("expression_corr_pearson", "Pearson r", (-1, 1)),
            ("expression_corr_spearman", "Spearman r", (-1, 1)),
            ("expression_cosine", "Cosine", (-1, 1)),
            ("expression_rmse", "RMSE", None),
            ("genes_used", "Both-nonzero genes", None),
        ]
        unique_cts = np.sort(plot_df["xenium_cell_type"].astype(str).unique())
        palette = dict(zip(unique_cts, sns.color_palette(n_colors=len(unique_cts))))
        for col, nice, ylim in metrics:
            sub = plot_df[[col, "xenium_cell_type"]].replace([np.inf, -np.inf], np.nan).dropna()
            if sub.empty: continue
            plt.figure(figsize=(6.2, 5))
            sns.violinplot(
                data=sub.assign(__all__="all"),
                x="__all__", y=col, inner="quartile", cut=0, width=0.8, color="lightgray"
            )
            sns.stripplot(
                data=sub.assign(__all__="all"),
                x="__all__", y=col, hue="xenium_cell_type",
                dodge=False, size=2.0, jitter=0.25, alpha=0.65, palette=palette
            )
            plt.xlabel("")
            plt.title(f"{run_name} — R={tolerance_radius} — {nice} (all cells)")
            if ylim is not None: plt.ylim(ylim)
            plt.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.tight_layout()
            out_png = os.path.join(plots_cell_dir, f"{base}_violin_{col}_all_cells.png")
            plt.savefig(out_png, dpi=220); plt.close()
            print(f"[plot] {out_png}")

# ==============================
# GLOBAL PLOTS
# ==============================
def render_violin_plots(output_dir):
    print("\n=== Rendering violin plots with consistent y-scale across all runs ===")

    def _prep_violin_data(arr):
        arr = np.asarray(arr, dtype=float); arr = arr[~np.isnan(arr)]
        if arr.size == 0: return np.array([np.nan])
        if arr.size == 1: return np.array([arr[0], arr[0]])
        return arr

    def _safe_violin(ax, data_list, labels, title, y_min, y_max):
        processed = [_prep_violin_data(arr) for arr in data_list]
        try:
            _ = ax.violinplot(processed, showmeans=True, showextrema=True, showmedians=False)
        except Exception:
            ax.scatter(np.arange(1, len(processed)+1),
                       [np.nanmean(x) if np.isfinite(np.nanmean(x)) else np.nan for x in processed])
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylim([y_min, y_max])
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    corr_ymin, corr_ymax = -1.0, 1.0
    rmse_ymin, rmse_ymax = 0.0, float(GLOBAL_MAX_RMSE if GLOBAL_MAX_RMSE > 0 else 1.0)

    for entry in GLOBAL_DISTRIBUTIONS:
        run_name = entry["run_name"]; radius = entry["radius"]; base = entry["base"]
        corr_arrays = entry["corr_arrays"]
        corr_data = [corr_arrays["PCC"], corr_arrays["Spearman"], corr_arrays["Cosine"]]
        corr_labels = ["PCC", "Spearman", "Cosine"]

        out_dir = GLOBAL_CONFIG["output_dir"]

        plt.figure(figsize=(6, 5))
        _safe_violin(
            plt.gca(), data_list=corr_data, labels=corr_labels,
            title=f"{run_name} — R={radius} — Gene Metrics (Correlations)",
            y_min=corr_ymin, y_max=corr_ymax
        )
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{base}_gene_metrics_violin_correlations.png")
        plt.savefig(p1, dpi=200); plt.close(); print(f"Saved: {p1}")

        rmse_arr = entry["rmse_array"]
        plt.figure(figsize=(4, 5))
        _safe_violin(
            plt.gca(), data_list=[rmse_arr], labels=["RMSE"],
            title=f"{run_name} — R={radius} — Gene Metrics (RMSE)",
            y_min=rmse_ymin, y_max=rmse_ymax
        )
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{base}_gene_metrics_violin_rmse.png")
        plt.savefig(p2, dpi=200); plt.close(); print(f"Saved: {p2}")

    print("Violin plots generated with identical y-axis scales across all runs.")

def annotate_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if h is None or np.isnan(h): continue
        ax.annotate(f"{h:.3f}",
                    (p.get_x() + p.get_width()/2., h),
                    ha='center', va='bottom', fontsize=9, xytext=(0,3), textcoords='offset points')

def render_global_bar_plots(out_dir):
    print("\n=== Building global bar plots from *_summary_wide.csv ===")
    plot_dir = os.path.join(out_dir, "plots"); os.makedirs(plot_dir, exist_ok=True)

    files = glob.glob(os.path.join(out_dir, "*_summary_wide.csv"))
    if not files:
        print(f"[!] No *_summary_wide.csv files found in {out_dir}"); return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f); dfs.append(df)
        except Exception as e:
            print(f"[skip] {f}: {e}")
    if not dfs:
        print("[!] No readable summary_wide CSVs."); return

    all_df = pd.concat(dfs, ignore_index=True)
    combined_csv = os.path.join(out_dir, "combined_summary_wide.csv")
    all_df.to_csv(combined_csv, index=False)
    print(f"[save] combined tidy summary -> {combined_csv}")

    metrics = [
        ("mean_pcc",               "Mean PCC"),
        ("mean_spearman",          "Mean Spearman R"),
        ("mean_cosine",            "Mean Cosine"),
        ("mean_rmse",              "Mean RMSE"),
        ("cell_level_cosine",      "Cell-Level Cosine Sim"),
        ("cell_level_jsd",         "Cell-Level JSD"),
        ("spot_level_cosine",      "Spot-Level Cosine Sim"),
        ("spot_level_jsd",         "Spot-Level JSD"),
        ("chaos_score",            "Chaos Score"),
        ("no_match_spot_pct",      "No Match Spot %"),
        ("no_match_cell_pct",      "No Match Cell %"),
        ("avg_both_nonzero_genes", "Avg Both-Nonzero Genes (per matched cell)"),
        ("matched_cells",          f"Matched Cells (≥{MIN_MATCHING_SPOTS} neighbor spots)"),
        ("expr_pairs",             "Expression Pairs (both IDs in H5ADs)"),
        ("overlap_pairs",          "Overlap Pairs (≥2 both-nonzero genes)"),
        ("matched_cells_used",     "Matched Cells Used for Accuracy/F1"),
        ("cell_type_accuracy",     "Cell Type Accuracy"),
        ("macro_f1",               "Macro F1 Score"),
        ("ARI",                    "ARI"),
        ("silhouette_score",       "Silhouette (Leiden, spatial) [DISABLED]"),
        ("balanced_accuracy",      "Balanced Accuracy"),
        ("overall_mean_cell_size", "Overall Mean Cell Size"),
        ("overall_cell_count",     "Overall Cell Count"),
        ("nuclei_candidates",      "Nuclei Candidates"),
    ]

    has_radius = "radius" in all_df.columns

    for col, nice in metrics:
        if col not in all_df.columns:
            print(f"[warn] missing metric column: {col} — skipping"); continue

        if has_radius:
            plot_df = all_df[["method", "radius", col]]

            methods = plot_df["method"].unique(); radii = sorted(plot_df["radius"].unique())
            x = np.arange(len(methods), dtype=float); w = 0.8 / max(len(radii), 1)
            plt.figure(figsize=(10, 5)); ax = plt.gca()
            for i, r in enumerate(radii):
                sub = plot_df[plot_df["radius"] == r].set_index("method").reindex(methods)
                heights = sub[col].to_numpy()
                ax.bar(x + i*w - 0.5*(len(radii)-1)*w, heights, width=w, label=f"R={r}")
            ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15)
            ax.set_ylabel(nice); ax.set_title(f"{nice} by method"); ax.legend()
            annotate_bars(ax); plt.tight_layout()
            out_png = os.path.join(plot_dir, f"bar_{col}.png")
            plt.savefig(out_png, dpi=220); plt.close(); print(f"[plot] {out_png}")
        else:
            plot_df = all_df.groupby(["method"], as_index=False)[col].mean()
            plt.figure(figsize=(8.5, 5)); ax = plt.gca()
            ax.bar(plot_df["method"], plot_df[col]); ax.set_ylabel(nice); ax.set_title(f"{nice} by method")
            plt.xticks(rotation=15); annotate_bars(ax); plt.tight_layout()
            out_png = os.path.join(plot_dir, f"bar_{col}.png")
            plt.savefig(out_png, dpi=220); plt.close(); print(f"[plot] {out_png}")

def render_radius_line_plots(out_dir):
    """
    For each metric, plot tolerance radius (x-axis) vs metric value (y-axis),
    with one line per method. Saves to plots_line/.
    """
    print("\n=== Building radius line plots (x = tolerance radius, lines = methods) ===")
    plot_dir = os.path.join(out_dir, "plots_line"); os.makedirs(plot_dir, exist_ok=True)

    files = glob.glob(os.path.join(out_dir, "*_summary_wide.csv"))
    if not files:
        print(f"[!] No *_summary_wide.csv files found in {out_dir}"); return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[skip] {f}: {e}")
    if not dfs:
        print("[!] No readable summary_wide CSVs for line plots."); return

    all_df = pd.concat(dfs, ignore_index=True)
    if "radius" not in all_df.columns:
        print("[!] 'radius' column missing; cannot render radius line plots."); return

    metrics = [
        ("mean_pcc",               "Mean PCC",              (-1, 1)),
        ("mean_spearman",          "Mean Spearman R",       (-1, 1)),
        ("mean_cosine",            "Mean Cosine",           (-1, 1)),
        ("mean_rmse",              "Mean RMSE",             None),
        ("cell_level_cosine",      "Cell-Level Cosine Sim", (-1, 1)),
        ("cell_level_jsd",         "Cell-Level JSD",        None),
        ("spot_level_cosine",      "Spot-Level Cosine Sim", (-1, 1)),
        ("spot_level_jsd",         "Spot-Level JSD",        None),
        ("chaos_score",            "Chaos Score",           None),
        ("no_match_spot_pct",      "No Match Spot %",       None),
        ("no_match_cell_pct",      "No Match Cell %",       None),
        ("avg_both_nonzero_genes", "Avg Both-Nonzero Genes",None),
        ("matched_cells",          f"Matched Cells (≥{MIN_MATCHING_SPOTS})", None),
        ("expr_pairs",             "Expression Pairs",      None),
        ("overlap_pairs",          "Overlap Pairs (≥2 BNZ)",None),
        ("matched_cells_used",     "Matched Cells Used",    None),
        ("cell_type_accuracy",     "Cell Type Accuracy",    (0, 1)),
        ("macro_f1",               "Macro F1 Score",        (0, 1)),
        ("ARI",                    "ARI",                   None),
        ("silhouette_score",       "Silhouette (Leiden, spatial) [DISABLED]", None),
        ("balanced_accuracy",      "Balanced Accuracy",     (0, 1)),
        ("overall_mean_cell_size", "Overall Mean Cell Size",None),
        ("overall_cell_count",     "Overall Cell Count",    None),
        ("nuclei_candidates",      "Nuclei Candidates",     None),
    ]

    # Ensure radii sort numerically and no duplicate index errors
    all_df["radius"] = pd.to_numeric(all_df["radius"], errors="coerce")
    all_df = all_df.dropna(subset=["radius"])
    all_df = all_df.sort_values(["method", "radius"])

    methods = all_df["method"].unique()
    radii_sorted = sorted(all_df["radius"].unique())

    for col, nice, ylim in metrics:
        if col not in all_df.columns:
            print(f"[warn] missing metric column: {col} — skipping"); continue

        plt.figure(figsize=(7.5, 5.2))
        ax = plt.gca()
        for m in methods:
            sub = (all_df[all_df["method"] == m][["radius", col]]
                   .dropna()
                   .drop_duplicates(subset=["radius"], keep="last"))
            sub = sub.set_index("radius").reindex(radii_sorted)
            ax.plot(radii_sorted, sub[col].to_numpy(), marker="o", label=m)
        ax.set_xlabel("Tolerance radius")
        ax.set_ylabel(nice)
        ax.set_title(f"{nice} vs Tolerance Radius (lines = methods)")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        plt.tight_layout()
        out_png = os.path.join(plot_dir, f"line_radius_{col}.png")
        plt.savefig(out_png, dpi=220); plt.close()
        print(f"[plot] {out_png}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("--- Starting evaluation for ourmethod / 10x / STHD / b2c across radii ---")
    for run_name, cfg in EVALUATION_RUNS.items():
        for R in TOLERANCE_RADII:
            run_evaluation(cfg, run_name, R)
    print("--- Evaluation complete across all methods and radii ---")

    render_violin_plots(GLOBAL_CONFIG["output_dir"])
    render_global_bar_plots(GLOBAL_CONFIG["output_dir"])
    render_radius_line_plots(GLOBAL_CONFIG["output_dir"])



#highy var genes
