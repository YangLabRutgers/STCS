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

This notebook searches combinations of:

- Search radius (S)

- Spatial weight (λ)

and evaluates them using several metrics:

- connection score (spatial coherence)

- detected genes per cell

- cell-type stability

These metrics help identify parameters that produce stable cell reconstructions and coherent spatial structures.

**Step 2 — Run the STCS pipeline**

After selecting parameters, run: **STCS_visium_tutorial.ipynb**

# Stereo-seq Workflow

Stereo-seq datasets are provided as GEM files, which must first be converted into AnnData format before running STCS.

**Step 1 — Convert GEM to AnnData**

Run: **Convert_GEM_h5ad.ipynb**

This notebook converts Stereo-seq GEM files into AnnData (.h5ad) format compatible with the STCS pipeline. 

**Step 2 — Run STCS on Stereo-seq data**

Run: **STCS_stereo-seq_tutorial.ipynb**

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

Wu LC*, Hu X*, Zhan F, Sun C, Gonzales J, Ofer R, Tran T,
Verzi MP, Liu L†, Yang J†

STCS: A Platform-Agnostic Framework for Cell-Level Reconstruction
in Sequencing-Based Spatial Transcriptomics
