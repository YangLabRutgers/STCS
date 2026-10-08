#!/usr/bin/env python3
"""
Run STCS's parameter sweep independently at every Stereo-seq bin resolution.

Takes the series of rebinned .h5ad files produced by rebin_gem_to_h5ad.py and runs
the full crop x search_radius x lambda sweep (STCS.run_parameter_sweep) separately at
each bin size, writing one results directory per resolution.

The point is that the best (S, lambda) is not a property of the tissue alone: it
depends on the bin size, because S is measured in bins. A search radius of 10 covers
five times as much tissue at bin10 as at bin2. Sweeping each resolution on its own
footing is what lets reconstruction quality be compared across bin sizes rather than
confounded with a parameter choice made at one of them.

The same crop rectangles are used at every resolution. Coordinates coming out of
rebin_gem_to_h5ad.py stay in raw bin1 units, so one rectangle selects the same physical
tissue region regardless of bin size. Supply the rectangles with --crops, or omit it and
they will be chosen automatically with pick_dense_crops() on the reference resolution
and reused everywhere, which is the same selection Parameter_Tuning.ipynb performs.

Results are resumable: a resolution whose run_summary.csv already exists is skipped, so
the script can be interrupted and restarted, or run on a subset of bin sizes in
parallel via --bin-sizes.

Example
-------
    python run_multires_sweep.py \
        --data-dir /path/to/results_multiresolution \
        --counts-template '{prefix}_bin{bin}_symbols.h5ad' \
        --prefix C04042E3 \
        --bin-sizes 2 5 10 20 50 \
        --image /path/to/HE_regist.tif \
        --sc-ref /path/to/reference.h5ad \
        --model /path/to/celltypist_model.pkl
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "STCS"))
from STCS_main import STCS


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def load_or_pick_crops(args, bin_size, counts_file):
    """Crop rectangles in raw bin1 units, read from --crops or chosen automatically."""
    if args.crops:
        with open(args.crops) as f:
            rects = [tuple(r) for r in json.load(f)]
        log("Using %d crop(s) from %s" % (len(rects), args.crops))
        return rects

    log("No --crops given; picking crops on bin%d" % bin_size)
    ref = STCS(
        Folder_path=args.data_dir,
        counts_data=counts_file,
        full_res_image_path=args.image,
        Platform="Stereo-seq",
    )
    rects = ref.pick_dense_crops(
        n_crops=args.n_crops,
        target_tissue_fraction=args.target_tissue_fraction,
        density_source="counts",       # no nuclei segmentation at this stage
    )
    out = os.path.join(args.data_dir, "crops.json")
    with open(out, "w") as f:
        json.dump([list(map(int, r)) for r in rects], f, indent=1)
    log("Picked %d crop(s), saved to %s" % (len(rects), out))
    return rects


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="directory holding the rebinned .h5ad files")
    ap.add_argument("--prefix", default="sample")
    ap.add_argument("--counts-template", default="{prefix}_bin{bin}_symbols.h5ad",
                    help="filename pattern of the per-resolution counts files")
    ap.add_argument("--bin-sizes", nargs="+", type=int, default=[2, 5, 10, 20, 50],
                    help="resolutions to sweep; pass a subset to run them in parallel")
    ap.add_argument("--image", default=None, help="registered H&E image (optional)")
    ap.add_argument("--sc-ref", default=None, help="single-cell reference .h5ad")
    ap.add_argument("--model", default=None, help="CellTypist model .pkl")

    ap.add_argument("--crops", default=None,
                    help="JSON list of [x, y, w, h] in raw bin1 units; if omitted, "
                         "crops are picked automatically and written to crops.json")
    ap.add_argument("--crop-reference-bin", type=int, default=None,
                    help="bin size used for automatic crop picking "
                         "(default: the smallest in --bin-sizes)")
    ap.add_argument("--n-crops", type=int, default=5)
    ap.add_argument("--target-tissue-fraction", type=float, default=0.10)

    ap.add_argument("--search-radii", nargs="+", type=int,
                    default=[1, 5, 10, 20, 30, 40, 50, 60])
    ap.add_argument("--lambdas", nargs="+", type=float,
                    default=[0, 0.1, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--no-celltypist", action="store_true")
    ap.add_argument("--prob-thresh", type=float, default=0.1)
    args = ap.parse_args()

    def counts_for(b):
        return args.counts_template.format(prefix=args.prefix, bin=b)

    # Check every input exists before starting, rather than failing hours in.
    missing = [b for b in args.bin_sizes
               if not os.path.exists(os.path.join(args.data_dir, counts_for(b)))]
    if missing:
        raise SystemExit(
            "Missing counts file(s) for bin size(s) %s in %s.\nExpected e.g. %s. Run "
            "rebin_gem_to_h5ad.py first, and check --counts-template matches its output."
            % (missing, args.data_dir, counts_for(missing[0]))
        )

    ref_bin = args.crop_reference_bin or min(args.bin_sizes)
    rects = load_or_pick_crops(args, ref_bin, counts_for(ref_bin))
    n_runs = len(rects) * len(args.search_radii) * len(args.lambdas)
    log("%d crops x %d radii x %d lambdas = %d runs per resolution"
        % (len(rects), len(args.search_radii), len(args.lambdas), n_runs))

    for b in args.bin_sizes:
        results_dir = os.path.join(args.data_dir, "res_bin%d" % b)
        os.makedirs(results_dir, exist_ok=True)
        summary_path = os.path.join(results_dir, "run_summary.csv")
        if os.path.exists(summary_path):
            log("bin%d already complete (%s exists) -- skipping" % (b, summary_path))
            continue

        print("=" * 90)
        log("Resolution bin%d -> %s" % (b, results_dir))
        print("=" * 90)

        try:
            t0 = time.time()
            stcs = STCS(
                Folder_path=args.data_dir,
                counts_data=counts_for(b),
                full_res_image_path=args.image,
                sc_ref=args.sc_ref,
                model_path=args.model,
                Platform="Stereo-seq",
            )
            log("Loaded bin%d in %.1fs" % (b, time.time() - t0))

            summary_df = stcs.run_parameter_sweep(
                rects=rects,
                search_radii=args.search_radii,
                lambdas=args.lambdas,
                results_dir=results_dir,
                run_celltypist=not args.no_celltypist,
                prob_thresh=args.prob_thresh,
            )
            summary_df["bin_size"] = b
            summary_df.to_csv(summary_path, index=False)
            log("bin%d sweep done: %d runs -> %s" % (b, len(summary_df), summary_path))

        except Exception as e:
            # One resolution failing should not take the others down with it.
            log("[ERROR] bin%d failed: %s" % (b, e))
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
