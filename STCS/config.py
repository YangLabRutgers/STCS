"""
STCS Configuration 
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

#TODO: input your L and S based on the parameter tuning. 
L = 0.5 # Lambda
search_radius = 5  # radius for search

# for pesudobulk
assignment_mode = 'mean'
empty = False

# for CellTypist
celltypist_target_sum = 10000

# for visualization
figure_size = (50, 50)
dpi = 600
alpha = 0.5
line_width = 0.5
buffer_distance = -0.25

#Example color map:
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
