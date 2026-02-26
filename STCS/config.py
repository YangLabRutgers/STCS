"""
STCS Configuration - Simple storage [FIXME@SILAS:Just put everything here]
"""


# for stardist
mpp = 0.3
prob_thresh = 0.1
max_bin_distance = 4
min_cells = 3
min_counts = 1
sigma = 5
stardist_model = "2D_versatile_he"
n_tiles = (4,4,1)
stardist_mode = 'mean'

# data preprocessing parameters
target_sum = 1e4 
n_top_genes = 5000 
batch_size = 50000

L = 0 # Lambda
search_radius = 7  # radius for search

# for pesudobulk
assignment_mode = 'sum'
empty = False

# for CellTypist
celltypist_target_sum = 10000

# for visualization
figure_size = (50, 50)
dpi = 600
alpha = 0.5
line_width = 0.5
buffer_distance = -0.25

colormap_crc_high_res_ref_ct_group = {
    "Tumor cE01 (Stem/TA-like)": "#ff9999",
    "Tumor cE02 (Stem/TA-like/Immature Goblet)": "#ffa599",
    "Tumor cE03 (Stem/TA-like prolif)": "#ecf4dd",
    "Tumor cE04 (Enterocyte 1)": "#ffbd99",
    "Tumor cE05 (Enterocyte 2)": "#d8a7b1",
    "Tumor cE06 (Immature Goblet)": "#ffd699",
    "Tumor cE07 (Goblet/Enterocyte)": "#ffe299",
    "Tumor cE08 (Goblet)": "#ffee99",
    "Tumor cE09 (Best4)": "#fffa99",
    "Tumor cE10 (Tuft)": "#f6ff99",
    "Tumor cE11 (Enteroendocrine)": "#fe817d",
    "cB1 (B IGD+IgM+)": "#b2e87c",
    "cB2 (B GC-like)": "#aee87c",
    "cB3 (B CD40+ GC-like)": "#aae87c",
    "cE01 (Stem/TA-like)": "#a6e87c",
    "cE02 (Stem/TA-like/Immature Goblet)": "#a2e87c",
    "cE03 (Stem/TA-like prolif)": "#9de87c",
    "cE04 (Enterocyte 1)": "#99e87c",
    "cE05 (Enterocyte 2)": "#95e87c",
    "cE06 (Immature Goblet)": "#91e87c",
    "cE07 (Goblet/Enterocyte)": "#8de87c",
    "cE08 (Goblet)": "#89e87c",
    "cE09 (Best4)": "#85e87c",
    "cE10 (Tuft)": "#81e87c",
    "cE11 (Enteroendocrine)": "#7de87c",
    "cM01 (Monocyte)": "#00a087",
    "cM02 (Macrophage-like)": "#aed185",
    "cM03 (DC1)": "#7ce889",
    "cM04 (DC2)": "#7ce88d",
    "cM05 (DC2 C1Q+)": "#7ce891",
    "cM06 (DC IL22RA2)": "#7ce895",
    "cM07 (pDC)": "#7ce899",
    "cM08 (AS-DC)": "#7ce89d",
    "cM09 (mregDC)": "#7ce8a1",
    "cM10 (Granulocyte)": "#7ce8a5",
    "cMA01 (Mast)": "#7ce8aa",
    "cP1 (Plasma IgA)": "#7ce8ae",
    "cP2 (Plasma IgG)": "#efbb8a",
    "cP3 (Plasma IgG prolif)": "#af6c63",
    "cS01 (Endo arterial)": "#7ce8ba",
    "cS02 (Endo capillary)": "#7ce8be",
    "cS03 (Endo capillary)": "#c1e8ee",
    "cS04 (Endo)": "#92b4c8",
    "cS05 (Endo venous)": "#7ce8ca",
    "cS06 (Endo lymphatic)": "#7ce8cf",
    "cS07 (Endo capillary-like)": "#7ce8d3",
    "cS08 (Endo arterial-like)": "#7ce8d7",
    "cS09 (Endo)": "#7ce8db",
    "cS10 (Endo tip cells)": "#7ce8df",
    "cS11 (Endo proif)": "#7ce8e3",
    "cS12 (Endo)": "#7ce8e7",
    "cS13 (Endo venous-like)": "#7ce4e8",
    "cS14 (Endo)": "#9ec1d4",
    "cS15 (Pericyte)": "#7cdce8",
    "cS16 (Pericyte)": "#7cd7e8",
    "cS17 (Pericyte)": "#7cd3e8",
    "cS18 (Pericyte)": "#7ccfe8",
    "cS19 (Pericyte)": "#7ccbe8",
    "cS20 (Pericyte prolif)": "#7cc7e8",
    "cS21 (Fibro stem cell niche)": "#7cc3e8",
    "cS22 (Fibro stem cell niche)": "#7cbfe8",
    "cS23 (Fibro BMP-producing)": "#7cbbe8",
    "cS24 (Fibro BMP-producing)": "#7cb7e8",
    "cS25 (Fibro CCL8+)": "#7cb2e8",
    "cS26 (Myofibro)": "#7caee8",
    "cS27 (CXCL14+ CAF)": "#7caae8",
    "cS28 (GREM1+ CAF)": "#7ca6e8",
    "cS29 (MMP3+ CAF)": "#7ca2e8",
    "cS30 (CAF CCL8 Fibro-like)": "#7c9ee8",
    "cS31 (CAF stem niche Fibro-like)": "#7c9ae8",
    "cS32 (Smooth Muscle)": "#7c96e8",
    "cS33 (Schwann)": "#7c92e8",
    "cTNI01 (CD4+ IL7R+)": "#7c8ee8",
    "cTNI02 (CD4+ IL7R+SELL+)": "#7c89e8",
    "cTNI03 (CD4+ IL7R+HSP+)": "#7c85e8",
    "cTNI04 (CD4+ IL7R+CCL5+)": "#7c81e8",
    "cTNI05 (CD4+ IL17+)": "#7c7de8",
    "cTNI06 (CD4+ TFH)": "#807ce8",
    "cTNI07 (CD4+ CXCL13+)": "#847ce8",
    "cTNI08 (CD4+ Treg)": "#887ce8",
    "cTNI09 (CD4+ Treg prolif)": "#8c7ce8",
    "cTNI10 (CD8+ IL7R+)": "#907ce8",
    "cTNI11 (CD8+GZMK+)": "#947ce8",
    "cTNI12 (CD8+ IL7R+)": "#997ce8",
    "cTNI13 (CD8+ T IL17+)": "#9d7ce8",
    "cTNI14 (CD8+ CXCL13+)": "#a17ce8",
    "cTNI15 (CD8+ CXCL13+ HSP+)": "#a57ce8",
    "cTNI16 (CD8+ CXCL13+ prolif)": "#a97ce8",
    "cTNI17 (gd-like T)": "#ad7ce8",
    "cTNI18 (gd-like T PDCD1+)": "#b17ce8",
    "cTNI19 (gd-like T prolif)": "#b57ce8",
    "cTNI20 (PLZF+ T)": "#b97ce8",
    "cTNI21 (PLZF+ T prolif)": "#be7ce8",
    "cTNI22 (cTNI22)": "#c27ce8",
    "cTNI23 (NK CD16A+)": "#c67ce8",
    "cTNI24 (NK GZMK+)": "#ca7ce8",
    "cTNI25 (NK XCL1+)": "#ce7ce8",
    "cTNI26 (ILC3)": "#d27ce8",
}



colormap_lung_ct_group = {
    'B cell': '#7A57D1',
    'endothelial cell': '#FF731D',
    'epithelial cell': '#bc8420',
    'fibroblast': '#CF0A0A',
    'malignant cell': '#83FFE6',
    'mast cell': '#0000A1',
    'megakaryocyte': '#fff568',
    'mononuclear phagocyte': '#0080ff',
    'neutrophil': '#81C6E8',
    'plasmacytoid dendritic cell': '#385098',
    'T cell': '#ffb5ba',
    'ambiguous': '#d3d3d3',
    'filtered': '#848884'
}
