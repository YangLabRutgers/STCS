#!/usr/bin/env python3
"""
Rebin a raw bin1 Stereo-seq GEM file into several coarser square bin sizes, saving
each resolution as its own AnnData (.h5ad) ready for STCS.

This is the multi-resolution counterpart of Convert_GEM_h5ad.ipynb. That notebook
converts a GEM at one bin size; this script starts from bin1 and produces the whole
series in one pass, so the same tissue can be reconstructed at several resolutions and
the results compared.

Rebinning follows the standard Stereo-seq/SAW convention, i.e. the same operation
Stereopy's read_gef(bin_size=N) performs internally. For bin size N:

    bin_x = (x // N) * N
    bin_y = (y // N) * N

then MIDCount is summed per (bin_x, bin_y, geneID).

Coordinates stay in raw bin1 units at every resolution, which is what makes the
comparison possible: one crop rectangle selects the same physical tissue region in every
output file. Note that the conversion from bin1 units to microns is a property of the
chip (commonly 1 unit = 0.5 um), so confirm your chip's DNB spacing before quoting
micron figures. Only the ratios between bin sizes are guaranteed.

Optionally converts var_names from Ensembl IDs to gene symbols with --gene-symbols.
CellTypist models expect symbols, so this is needed if you intend to annotate; it
requires the `mygene` package and an internet connection. The mapping is queried once
and applied to every resolution, since the gene panel is identical across bin sizes.

Example
-------
    python rebin_gem_to_h5ad.py \
        --gem /path/to/C04042E3.tissue.gem.gz \
        --bin-sizes 1 5 10 20 50 100 200 \
        --out-dir /path/to/results_multiresolution \
        --prefix C04042E3 \
        --gene-symbols --species mouse
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import anndata as ad


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_gem(gem_path):
    log("Reading GEM: %s" % gem_path)
    gem = pd.read_csv(gem_path, sep="\t", compression="infer", comment="#")
    missing = {"x", "y", "geneID", "MIDCount"} - set(gem.columns)
    if missing:
        raise ValueError(
            "GEM is missing column(s) %s. Found: %s. Some GEM files name the count "
            "column 'MIDCounts' or 'UMICount'; rename it to 'MIDCount' first."
            % (sorted(missing), list(gem.columns))
        )
    log("Loaded %s raw rows" % format(len(gem), ","))
    return gem


def gem_to_binned_adata(gem, bin_size):
    """Rebin a raw bin1 GEM dataframe to `bin_size` and build an AnnData.

    Mirrors Convert_GEM_h5ad.ipynb's gem_to_visium_format, with an explicit
    rebinning step first. Spatial coordinates are the bin's lower-left corner in
    raw bin1 units, so they are directly comparable across bin sizes.
    """
    df = gem

    if bin_size > 1:
        df = df.copy()
        df["x"] = (df["x"] // bin_size) * bin_size
        df["y"] = (df["y"] // bin_size) * bin_size
        df = df.groupby(["x", "y", "geneID"], as_index=False)["MIDCount"].sum()

    # Barcodes encode the bin size and position, so files from different
    # resolutions stay distinguishable if they are ever concatenated.
    df = df.assign(barcode=(
        "s_" + "%03dbin_" % bin_size
        + df["x"].astype(int).astype(str).str.zfill(5)
        + "_" + df["y"].astype(int).astype(str).str.zfill(5)
        + "-1"
    ))

    barcodes = df["barcode"].unique()
    genes = df["geneID"].unique()
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    row = df["barcode"].map(barcode_to_idx)
    col = df["geneID"].map(gene_to_idx)
    data = df["MIDCount"].astype(np.int32)
    X = coo_matrix((data, (row, col)), shape=(len(barcodes), len(genes))).tocsr()

    adata = ad.AnnData(X)
    adata.obs_names = barcodes
    adata.var_names = genes

    # Recover x/y from the barcode rather than re-deriving them, so the coordinates
    # can never drift out of step with the obs order.
    coords = pd.DataFrame([bc.split("_")[2:4] for bc in barcodes], columns=["x", "y"])
    coords["x"] = coords["x"].astype(int)
    coords["y"] = coords["y"].str[:-2].astype(int)   # strip the trailing "-1"
    adata.obsm["spatial"] = coords[["x", "y"]].to_numpy()
    adata.obs["bin_size"] = bin_size

    return adata


def ensembl_to_symbol(ens_ids, species):
    """Ensembl gene ID -> symbol, queried once for the whole panel."""
    import mygene   # optional dependency, only needed for --gene-symbols

    log("Querying mygene for %d Ensembl IDs ..." % len(ens_ids))
    mg = mygene.MyGeneInfo()
    res = mg.querymany(ens_ids, scopes="ensembl.gene", fields="symbol",
                       species=species, as_dataframe=True, returnall=True)
    df = res["out"]
    df = df[~df.index.duplicated(keep="first")]
    mapping = df["symbol"].dropna().to_dict()
    log("%d / %d genes mapped to symbols" % (len(mapping), len(ens_ids)))
    return mapping


def apply_symbols(adata, mapping):
    """Rename var_names to symbols, dropping genes with no mapping."""
    keep = [g in mapping for g in adata.var_names]
    adata = adata[:, keep].copy()
    adata.var["ensembl_id"] = adata.var_names
    adata.var_names = [mapping[g] for g in adata.var["ensembl_id"]]
    adata.var_names_make_unique()
    return adata


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gem", required=True, help="raw bin1 .gem or .gem.gz file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bin-sizes", nargs="+", type=int,
                    default=[1, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--prefix", default="sample",
                    help="output name stem: <prefix>_bin<N>.h5ad")
    ap.add_argument("--gene-symbols", action="store_true",
                    help="convert var_names from Ensembl IDs to gene symbols "
                         "(needed for CellTypist; requires the mygene package)")
    ap.add_argument("--species", default="mouse",
                    help="species for the gene-symbol lookup (default: mouse)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild resolutions whose output file already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    suffix = "_symbols" if args.gene_symbols else ""
    paths = {b: os.path.join(args.out_dir, "%s_bin%d%s.h5ad" % (args.prefix, b, suffix))
             for b in args.bin_sizes}
    todo = [b for b in args.bin_sizes if args.overwrite or not os.path.exists(paths[b])]
    for b in args.bin_sizes:
        if b not in todo:
            log("skip bin%d, %s already exists" % (b, paths[b]))
    if not todo:
        return

    gem = load_gem(args.gem)

    mapping = None
    if args.gene_symbols:
        mapping = ensembl_to_symbol(sorted(gem["geneID"].unique()), args.species)

    for b in todo:
        t0 = time.time()
        log("Building bin%d ..." % b)
        adata = gem_to_binned_adata(gem, bin_size=b)
        if mapping is not None:
            adata = apply_symbols(adata, mapping)
        adata.write_h5ad(paths[b])
        log("bin%d: %s bins x %s genes -> %s (%.1fs)"
            % (b, format(adata.shape[0], ","), format(adata.shape[1], ","),
               paths[b], time.time() - t0))


if __name__ == "__main__":
    main()
