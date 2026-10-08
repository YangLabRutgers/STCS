# STCS: Spatial Transcriptomics Cell Segmentation

<img width="600" alt="STCS pipeline" src="https://github.com/user-attachments/assets/01cac294-32ef-41c2-89ff-7bc2dad2294f">

**STCS (Spatial Transcriptomics Cell Segmentation)** is a platform-agnostic framework that reconstructs **cell-level gene expression profiles** from sequencing-based spatial transcriptomics data by integrating **nuclei segmentation**, **transcriptomic similarity**, and **spatial proximity**.

Sequencing-based spatial transcriptomics technologies such as **Visium HD** and **Stereo-seq** provide transcriptome-wide measurements at very high spatial resolution. However, these platforms measure gene expression from **spatial bins rather than biological cells**, making downstream cell-level analysis challenging.

STCS addresses this problem by reconstructing **coherent cell-level expression profiles** through a joint transcriptomic–spatial assignment model.

---

# Overview of the STCS Pipeline

The STCS pipeline consists of the following steps:

1. **Nuclei Segmentation**  
   H&E images are processed using **StarDist** to detect nuclei.

2. **Initial Bin Assignment**  
   Spatial bins located within detected nuclei are assigned to the corresponding nucleus.

3. **Candidate Nucleus Search**  
   Bins outside nuclei search for nearby nuclei within a specified **search radius (S)**.

4. **Joint Transcriptomic–Spatial Distance Calculation**

   The assignment score between bin *i* and nucleus *c* is computed based on:

   - S (search radius) defines the spatial neighborhood for candidate nuclei.

   - λ (lambda) controls the weight of spatial distance relative to transcriptomic similarity.

5. **Cell Reconstruction**
   
   Bins assigned to each nucleus are aggregated to form cell-level expression profiles.

6. **Cell Type Annotation**
   
   Reconstructed cells can be annotated using CellTypist or other cell-type annotation tools.


# Visium HD Workflow

For Visium HD datasets, we recommend performing parameter tuning before running the full STCS pipeline for new tissue slides.

**Step 1 — Parameter tuning**

Run: **Parameter_Tuning.ipynb**

Parameter selection is **reference-free**: it uses only the slide being processed, and
needs no matched cell boundaries and no external single-cell reference.

*Crop selection.* Tuning runs on a few nucleus-dense crops rather than the whole slide.
`pick_dense_crops()` chooses them automatically and **derives the crop size** from a
coverage target rather than asking you for a number:

```
SIZE = sqrt(target_tissue_fraction * A_tissue / n_crops)
```

so the crops together cover, by default, about 10% of the in-tissue area. Candidate
windows are ranked by density on a grid, filtered to those lying almost entirely within
tissue, and taken greedily so that no two overlap. Density is counted from nuclei when
segmentation has already been run, and from transcripts otherwise.

The notebook then sweeps combinations of

- search radius **S**, in bins
- spatial weight **λ**

running the full assignment on every crop for every combination.

*Scores.* Each combination is judged on two complementary criteria, both computed from
the reconstruction itself:

1. **Spatial connectivity** — the fraction of a cell's bins that fall in its largest
   8-connected component. Cells whose bins break into separate islands score lower.
   Higher is better.
2. **Transcript consistency** — the mean deviation of a cell's UMI count from the
   expected depth, `total UMI / number of nuclei`. This penalises both cells that steal
   transcripts from their neighbours and cells that collect too few. Lower is better.

The two are min-max normalised over the grid, the transcript term is inverted so that
higher is better for both, and they are added to give a **combined score between 0 and
2**. The pair is deliberate: connectivity alone is maximised by a tiny search radius,
since a cell that barely grows is trivially contiguous, and the transcript term is what
stops that. Both are computed before quality-control filtering.

*Reading the heatmaps.* The notebook prints three, in order: connectivity, transcript
deviation, then the combined score. In each, the single best cell is outlined **solid
yellow**, and any cell that is identical to it once rounded to two decimals is outlined
**dashed white**.

Selected settings are slide-specific and also depend on bin size, so re-run this step for
a new tissue, a new platform, or a different binning.

**Step 2 — Run the STCS pipeline**

After selecting parameters, run: **STCS_visium_tutorial.ipynb**

# Stereo-seq Workflow

Stereo-seq datasets are provided as GEM files, which must first be converted into AnnData format before running STCS.

**Step 1 — Convert GEM to AnnData**

Run: **Convert_GEM_h5ad.ipynb**

This notebook converts Stereo-seq GEM files into AnnData (.h5ad) format compatible with the STCS pipeline. 

**Step 2 — Run STCS on Stereo-seq data**

You may run parameter tuning or run the tutorial directly: **STCS_stereo-seq_tutorial.ipynb**

---

# Multi-Resolution Workflow (Stereo-seq)

Bin size is a choice, not a given. Stereo-seq GEM files are delivered at bin1 and binned
to whatever resolution the analysis calls for, and that choice interacts with STCS: the
search radius **S** is measured *in bins*, so S=10 covers five times as much tissue at
bin10 as at bin2. Comparing reconstructions across bin sizes therefore means re-tuning at
each one, otherwise resolution and parameter choice are confounded.

Two scripts support this. They are optional, and sit alongside the single-resolution
route in **Convert_GEM_h5ad.ipynb** rather than replacing it.

**Step 1 — Rebin to a series of resolutions**

```bash
python rebin_gem_to_h5ad.py \
    --gem  /path/to/sample.tissue.gem.gz \
    --out-dir /path/to/results_multiresolution \
    --prefix sample \
    --bin-sizes 1 5 10 20 50 100 200 \
    --gene-symbols --species mouse
```

Starts from a raw bin1 GEM and writes one `.h5ad` per bin size, summing `MIDCount` per
`(x // N * N, y // N * N, geneID)` — the same operation Stereopy's `read_gef(bin_size=N)`
performs internally, so the output matches the standard SAW binning.

Coordinates stay in **raw bin1 units at every resolution**. This is what makes the
comparison possible: one crop rectangle selects the same physical tissue region in every
file. Converting bin1 units to microns is a property of the chip (commonly 1 unit =
0.5 um), so check your chip's DNB spacing before quoting micron figures; only the ratios
between bin sizes are guaranteed.

`--gene-symbols` converts `var_names` from Ensembl IDs to gene symbols, which CellTypist
models require. It needs the `mygene` package and an internet connection, and queries the
mapping once for the whole panel since the gene set is identical at every bin size. Skip
it if you do not intend to annotate.

**Step 2 — Sweep each resolution independently**

```bash
python run_multires_sweep.py \
    --data-dir /path/to/results_multiresolution \
    --prefix sample \
    --counts-template '{prefix}_bin{bin}_symbols.h5ad' \
    --bin-sizes 2 5 10 20 50 \
    --image   /path/to/HE_regist.tif \
    --sc-ref  /path/to/reference.h5ad \
    --model   /path/to/celltypist_model.pkl
```

Runs the full crop x S x lambda sweep separately at each bin size, writing one
`res_bin<N>/run_summary.csv` per resolution. The same crop rectangles are used throughout.
Pass them with `--crops` (a JSON list of `[x, y, w, h]` in bin1 units), or omit it and they
are picked automatically with `pick_dense_crops()` on the finest resolution and saved to
`crops.json` — the same reference-free selection described above.

The run is resumable: a resolution whose `run_summary.csv` already exists is skipped, and a
failure at one bin size does not stop the rest. Pass a subset to `--bin-sizes` to run
resolutions in parallel.

Read each resolution's results with the scoring and heatmap steps from
**Parameter_Tuning.ipynb**, which take a `run_summary.csv` and are not specific to Visium.

--- 


# Example Datasets
**Visium HD Dataset**

Human lung cancer dataset used in the tutorial:

Visium HD Spatial Gene Expression Libraries, Post-Xenium, Human Lung Cancer (FFPE)
https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-lung-cancer-post-xenium-expt

Corresponding histology image:
https://www.10xgenomics.com/datasets/xenium-human-lung-cancer-post-xenium-technote

**Stereo-seq Dataset**

Stereo-seq mouse brain dataset:
https://en.stomics.tech/col1241/index.html


---

# Citation

If you use STCS in your research, please cite:

Chen Wu L*, Hu X*, Zhan F, Sun C, Gonzales J, Ofer R, Tran T,
Verzi MP, Liu L†, Yang J†

STCS: A Platform-Agnostic Framework for Cell-Level Reconstruction
in Sequencing-Based Spatial Transcriptomics
