work_path = '/home/xh300/link/spa/newmethod/Rep2_normalized_all'
visium_path = '/home/xh300/link/spa/sthd/STHD/notebooks/data/rachel/rep2'
b2c_img = '/home/xh300/link/spa/sthd/STHD/notebooks/data/rachel/rep2/Jej1_IB5_20241106_resized_50_quality_90.jpg'
model_path = '/home/xh300/link/spa/sthd/STHD/notebooks/data/rachel/rep2/rep2.pkl'
sc_ref = '/home/xh300/link/spa/sthd/STHD/notebooks/data/rachel/GSE214821PaperNerve_pp.h5ad'

crop = False
use_sc = True
normalize = True
feature_name = False
skipb2c = True
L = 0.5
n_top_genes = 5000

# cmap = {
#     'B cell': '#7A57D1',
#     'endothelial cell': '#FF731D',
#     'epithelial cell': '#bc8420',
#     'fibroblast': '#CF0A0A',
#     'malignant cell': '#83FFE6',
#     'mast cell': '#0000A1',
#     'megakaryocyte': '#fff568',
#     'mononuclear phagocyte': '#0080ff',
#     'neutrophil': '#81C6E8',
#     'plasmacytoid dendritic cell': '#385098',
#     'T cell': '#ffb5ba',
#     'ambiguous': '#d3d3d3',
#     'filtered': '#848884'
# }


cmap = {
    'EC -villi base': '#1f77b4',
    'EC': '#ff7f0e',
    'EC-TA': '#2ca02c',
    'Progenitor': '#d62728',
    'Tuft': '#00FF7F',
    'B cell': '#2E8B57',
    'Endothelial': '#e377c2',
    'Goblet': '#7f7f7f',
    'Villi tip': '#bcbd22',
    'T cell': '#17becf',
    'TA': '#98FB98',
    'Paneth cell': "#730fec",
    'Mesenchyme': '#7570b3',
    'EEC':"#984ea3",
    'SC':"#708090",
    'Myeloid-macrophage':"#ffff33",
    'Microglial-macrophage':"#a65628",
}



def simple_scale(adata):
    X = adata.X
    if sp.issparse(X):
        std = np.sqrt(X.multiply(X).mean(axis=0).A1 - np.square(X.mean(axis=0).A1))
        std[std == 0] = 1
        scaled = X.multiply(1 / std)
        adata.X = scaled.tocsr()
    else:
        std = np.std(X, axis=0)
        std[std == 0] = 1
        adata.X = X / std

def round(x0, y0, r):
    points = []
    for x in range(x0 - r, x0 + r + 1):
        for y in range(y0 - r, y0 + r + 1):
            if (x - x0)**2 + (y - y0)**2 <= r**2:
                points.append((x, y))
    return points

import bin2cell as b2c
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
import cv2
import tifffile as tf
from tqdm import tqdm
import scipy.sparse as sp

import pandas as pd
from collections import Counter
from scipy.spatial.distance import cdist
import decoupler as dc
from sklearn.metrics.pairwise import cosine_similarity

from shapely.geometry import box
from shapely.ops import unary_union
import random
if skipb2c != True:
#b2c pipeline
    path = visium_path
    source_image_path = b2c_img
    spaceranger_image_path = visium_path+"/spatial"
    os.makedirs(work_path+'/stardist', exist_ok=True)
    os.chdir(work_path+'/stardist')

    adata = b2c.read_visium(path, source_image_path = source_image_path,spaceranger_image_path = spaceranger_image_path)
    adata.var_names_make_unique()
    adata

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=1)
    adata

    mpp = 0.3

    thr = 10

    b2c.destripe(adata)
    b2c.scaled_he_image(adata, mpp=mpp, save_path= "scaled.tiff")

    b2c.stardist(image_path="scaled.tiff", 
                labels_npz_path="he_labels_prime.npz", 
                stardist_model="2D_versatile_he", 
                prob_thresh=0.1
                )
    b2c.insert_labels(adata, 
                    labels_npz_path="he_labels_prime.npz", 
                    basis="spatial", 
                    spatial_key="spatial_cropped_150_buffer",
                    mpp=mpp, 
                    labels_key="labels_he"
                    )
    b2c.expand_labels(adata, 
                    labels_key='labels_he', 
                    expanded_labels_key="labels_he_expanded",
                    max_bin_distance=4,
                    ) #bins expanded at a distance of 4 bins out
    img = b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=5)
    cv2.imwrite("gex_labeling_prime.tiff", img)
    b2c.insert_labels(adata, 
                    labels_npz_path="he_labels_prime.npz", 
                    basis="array", 
                    mpp=mpp, 
                    labels_key="labels_gex"
                    )
    b2c.salvage_secondary_labels(adata, 
                                primary_label="labels_he_expanded", 
                                secondary_label="labels_gex", 
                                labels_key="labels_joint"
                                )
    cdata = b2c.bin_to_cell(adata, labels_key="labels_joint", spatial_keys=["spatial", "spatial_cropped_150_buffer"])
    cdata.write_h5ad('prime_cell_outputs.h5ad')
    adata.write_h5ad('prime_barcode_outputs.h5ad')

#newmethod pipeline

os.chdir(work_path)
b = sc.read_h5ad(work_path+'/stardist/prime_barcode_outputs.h5ad')
if crop:
    b = b[(b.obs['array_row'] >= 1000) & (b.obs['array_row'] <= 2000) & (b.obs['array_row'] <= 2000) & (b.obs['array_row'] >= 1000)] #small region
b = b[b.obs['labels_he'] != 0]
cb = {}
for i in set(b.obs['labels_he']):
    p = list(b[b.obs['labels_he'] == int(i)].obs_names)
    cb[i] = p

f = open("bar_cell.csv", "w")
f.write('cell\twhy\n')
for i in cb:
    for k in cb[i]:
        f.write(k+'\t'+str(i)+'\t')
        f.write('\n')
f.close()

b_c = pd.read_csv('bar_cell.csv',sep='\t')
adata = sc.read_10x_h5(visium_path+'/filtered_feature_bc_matrix.h5')
adata.obs['array_col'] = [int(i.split('_')[3][:-2]) for i in adata.obs_names]
adata.obs['array_row'] = [int(i.split('_')[2]) for i in adata.obs_names]
if crop:
    adata = adata[(adata.obs['array_row'] >= 1000) & (adata.obs['array_row'] <= 2000) & (adata.obs['array_row'] <= 2000) & (adata.obs['array_row'] >= 1000)] #small region
adata.obs['cell'] = b.obs['labels_he']

bar = {}
for i in adata.obs_names:
    cl = []
    x = i.split('_')[2]
    y = i.split('_')[3][:-2]
    if i in b_c.index:
        cl.append(b_c.loc[i]['cell'])
    else:
        for m,n in round(int(x),int(y),3):
            t = 's_002um_'+str(m).zfill(5)+'_' +str(n).zfill(5)+'-1'
            if t in b_c.index and b_c.loc[t]['cell'] not in cl:
                cl.append(b_c.loc[t]['cell'])
    bar[i] = cl

celldata = adata[~np.isnan(adata.obs['cell'])]
f = open("cell_bar.csv", "w")
f.write('\tbarcodes\n')
for k in b_c['cell'].unique():
    f.write(str(k))
    f.write('\t'+'-'.join(str([int(i.split('_')[2]),int(i.split('_')[3][:-2])]) for i in b_c[b_c['cell']==k].index)+'\n')
f.close()

c_b = pd.read_csv("cell_bar.csv",sep='\t')
c_b.index = c_b['Unnamed: 0']

tmp = dc.get_pseudobulk(
    celldata,
    sample_col='cell',
    groups_col='cell',
    mode='mean',
    min_cells=0, min_counts=0,remove_empty=False

)
tmp.obs['cell'] = tmp.obs_names



#sc ref
if use_sc:
    scdata = sc.read_h5ad(sc_ref)
    if feature_name:
        scdata.var.index = scdata.var['feature_name'].astype(str)

    #sc.pp.filter_cells(tmp, min_genes=10)
    sc.pp.filter_genes(tmp, min_cells=3)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, subset=True)
    simple_scale(tmp)
    sc.tl.pca(tmp)
    genes = list(set(scdata.var_names) & set(tmp.var_names[tmp.var["highly_variable"]]))
    tmp.var_names_make_unique()
    tmp = tmp[:, genes].copy()
    tmp.write_h5ad('processed_nuclie.h5ad')

    loadings = tmp.varm['PCs']


    adata.var_names_make_unique()
    adata = adata[:,tmp.var_names].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    simple_scale(adata)
    batch_size = 50000
    X_pca_list = []
    for i in range(0, adata.shape[0], batch_size):
        X_batch = adata.X[i:i+batch_size].dot(loadings)
        X_pca_list.append(X_batch)
    adata.obsm['X_pca'] = np.vstack(X_pca_list)
    adata.write_h5ad('processed_adata.h5ad')

    scdata.var_names_make_unique()
    scdata = scdata[:,tmp.var_names].copy()

    sc.pp.normalize_total(scdata, target_sum=1e4)
    sc.pp.log1p(scdata)
    simple_scale(scdata)

    scdata.obsm['X_pca'] = scdata.X @ loadings
    scdata.write_h5ad('processed_scdata.h5ad')

    gene_cos_sim = cosine_similarity(scdata.obsm['X_pca'].T)

    
#spa ref
else:
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, subset=True)
    simple_scale(tmp)
    sc.tl.pca(tmp)
    tmp.write_h5ad('nuclie.h5ad')
    tmp = sc.read_h5ad('nuclie.h5ad')
    tmp.var_names_make_unique()
    adata.var_names_make_unique()
    adata = adata[:, tmp.var_names]

    loadings = tmp.varm['PCs']
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    simple_scale(adata)

    batch_size = 50000
    X_pca_list = []
    for i in range(0, adata.shape[0], batch_size):
        X_batch = adata.X[i:i+batch_size].dot(loadings)
        X_pca_list.append(X_batch)

    adata.obsm['X_pca'] = np.vstack(X_pca_list)
    gene_cos_sim = cosine_similarity(adata.obsm['X_pca'].T)


#normalize
if normalize:
    one = []
    for i in tqdm(bar.keys()):
        if len(bar[i])>1:
            dis = []
            loc= [int(i.split('_')[2]),int(i.split('_')[3][:-2])]
            for k in bar[i]:
                loc_dis = []
                mtx = pd.DataFrame(list(adata[i].obsm['X_pca'][0])) - pd.DataFrame(list(tmp[str(int(float(k)))].obsm['X_pca'][0]))
                t = mtx.T@gene_cos_sim@mtx
                for q in c_b.loc[int(float(k))]['barcodes'].split('-'):
                    loc_dis.append(np.linalg.norm(np.array([int(q[1:-1].split(',')[0]),int(q[1:-1].split(',')[1])])-np.array(loc)))
                nearest_index = np.argmin(loc_dis)
                one.append([float(t.iloc[0]),loc_dis[nearest_index]])
    one = pd.DataFrame(one)
    one.to_csv('dist.csv')

    os.chdir(work_path)
    dis_array = pd.read_csv('dist.csv')
    x = dis_array['0']
    log_data = np.log(x)
    normalized = (log_data - log_data.min()) / (log_data.max() - log_data.min())
    all_dis = [normalized[i] + L*dis_array['1'][i] for i in range(len(normalized))]
    l=0
    one = {}
    for i in tqdm(bar.keys()):
        if len(bar[i])>1:
            dis=[]
            for k in bar[i]:
                dis.append(all_dis[l])
                l+=1
            if len(dis) == 0: 
                one[i] = None
            else:
                nearest_index = np.argmin(dis)
                one[i] = str(int(bar[i][nearest_index]))
        elif len(bar[i])==1:
            one[i] = str(int(bar[i][0]))
        else:
            one[i] = None

else:
    one = {}
    for i in tqdm(bar.keys()):
        if len(bar[i])>1:
            dis = []
            loc= [int(i.split('_')[2]),int(i.split('_')[3][:-2])]
            for k in bar[i]:
                loc_dis = []
                mtx = pd.DataFrame((adata[i].obsm['X_pca']).toarray().tolist()[0]) - pd.DataFrame(list(tmp[str(int(float(k)))].obsm['X_pca'][0]))
                t = mtx.T@gene_cos_sim@mtx
                for q in c_b.loc[int(k)]['barcodes'].split('-'):
                    loc_dis.append(np.linalg.norm(np.array([int(q[1:-1].split(',')[0]),int(q[1:-1].split(',')[1])])-np.array(loc)))
                nearest_index = np.argmin(loc_dis)
                dis.append(t + L*loc_dis[nearest_index])
            if len(dis) == 0: 
                one[i] = None
            else:
                nearest_index = np.argmin(dis)
                one[i] = str(int(bar[i][nearest_index]))
        elif len(bar[i])==1:
            one[i] = str(int(bar[i][0]))
        else:
            one[i] = None

os.chdir(work_path)
f = open("near_onecell.csv", "w")
f.write('\tcells\n')
for i in one:
    if type(one[i]) == str:
        f.write(i+'\t'+one[i].replace(' ','')+'\n')
    else:
        f.write(i+'\t'+'None'+'\n')
f.close()

onecell = pd.read_csv('near_onecell.csv',sep='\t',index_col='Unnamed: 0')
f = open("near_output.tsv", "w")
f.write('\tbarcodes\n')
for i in c_b.index:
    f.write(str(i)+'\t'+','.join(onecell[onecell['cells'] == i].index)+'\n')
f.close()

cells = pd.read_csv('near_output.tsv',sep='\t',index_col=0)
t = onecell.astype(str)
new_adata = sc.read_10x_h5(visium_path+'/filtered_feature_bc_matrix.h5')
new_adata.obs['cell'] = t
new_adata = new_adata[new_adata.obs['cell'].notna()] #small region
new_adata.write_h5ad('before_psedo.h5ad')
new_adata = sc.read_h5ad('before_psedo.h5ad')
pdata = dc.get_pseudobulk(
    new_adata,
    sample_col='cell',
    groups_col='cell',
    mode='sum',
    min_cells=0, min_counts=0,remove_empty=False)
pdata.write_h5ad('near_output.h5ad')

#smooth


#celltypist
import scanpy as sc
import celltypist
from celltypist import models

test = sc.read('near_output.h5ad')
sc.pp.normalize_total(test,target_sum=10000)
sc.pp.log1p(test)
predictions = celltypist.annotate(test, model = model_path, majority_voting = True)
predict = predictions.to_adata()
barcode_to_coords = {i:[int(i.split('_')[2]),int(i.split('_')[3][:-2])] for i in adata.obs_names}
whole = cells[[str(i) in predict.obs_names for i in cells.index]]
predict = predict[predict.obs_names != 'nan']
predict.obs_names = [str(float(i)) for i in predict.obs_names]
predict.write_h5ad('final_output.h5ad')
cells.index = [str(float(i)) for i in cells.index]
cells['STHD_pred_ct'] = predict.obs['predicted_labels']
cells.to_csv('final_output.csv')
barcode = pd.DataFrame(adata.obs_names)
barcode.to_csv('barcodes.csv')

results = sc.read_h5ad(work_path+'/final_output.h5ad')
out = pd.read_csv(work_path+'/final_output.csv')
f = open("bar_cor.csv", "w")
f.write('barcode'+'\t'+'cell'+'\t'+'x'+'\t'+'y'+'\n')
for i in tqdm(out.index):
    for k in out.loc[i]['barcodes'].split(','):
        x=int(k.split('_')[2])
        y=int(k.split('_')[3][:-2])
        c=int(out.loc[i]['Unnamed: 0'])
        f.write(k+'\t'+str(c)+'\t'+str(x)+'\t'+str(y)+'\n')

f.close()

#plotting


# Load the data
os.chdir(work_path)
print("Loading data...")
bindata_path = work_path+"/final_output.csv"
newpdata_path = work_path+"/barcodes.csv"
bindata = pd.read_csv(bindata_path)
newpdata = pd.read_csv(newpdata_path)
print("Data loaded successfully.")

# Precompute mappings for faster lookups
print("Precomputing mappings...")
barcode_to_coords = {i:[int(i.split('_')[2]),int(i.split('_')[3][:-2])] for i in newpdata['0']}
valid_barcodes = set(barcode_to_coords.keys())  # Faster lookups
print(f"Precomputed mappings for {len(valid_barcodes)} barcodes.")

# Assign colors to cell typess
print("Assigning colors to cell types...")
cell_types = bindata['STHD_pred_ct'].unique()
cell_type_colors = {cell_type: cmap.get(cell_type, "#000000") for cell_type in cell_types}  # Default to black if not found
print(f"Assigned colors to {len(cell_types)} cell types.")

# Preprocess data for boundary computation
print("Processing data for boundary computation...")
cell_boundaries = []  # Store boundary polygons for each cell

for idx, row in bindata.iterrows():
    if pd.isnull(row['barcodes']):
        continue  # Skip rows with no barcodes
    cell_type = row['STHD_pred_ct']
    color = cell_type_colors[cell_type]
    barcodes = row['barcodes'].split(',')

    # Get coordinates for the current cell's barcodes
    points = [
        (barcode_to_coords[barcode][0], barcode_to_coords[barcode][1])
        for barcode in barcodes if barcode in valid_barcodes
    ]

    # Create 1x1 squares where each point is itself represented as a square
    squares = [box(x - 0.5, y - 0.5, x + 0.5, y + 0.5) for x, y in points]

    # Compute the union of all squares (to handle connected regions)
    if squares:
        boundary = unary_union(squares)  # Combine all squares
        cell_boundaries.append((boundary, color))

print(f"Computed boundaries for {len(cell_boundaries)} cells.")

# Plot the boundaries
print("Plotting boundaries...")
plt.figure(figsize=(50, 50))
for idx, (boundary, color) in enumerate(cell_boundaries, start=1):
    if idx % 100 == 0 or idx == len(cell_boundaries):
        print(f"Plotting boundary {idx}/{len(cell_boundaries)}...")
    if boundary.geom_type == 'Polygon':
        # Shrink the area slightly to avoid overlapping the boundary
        inner_boundary = boundary.buffer(-0.25)  # Shrink the filled area slightly
        if not inner_boundary.is_empty:  # Only fill if the buffer operation succeeded
            x, y = inner_boundary.exterior.xy
            plt.fill(x, y, color=color, alpha=0.5)  # Fill the shrunken boundary
        # Draw the original boundary line
        x, y = boundary.exterior.xy
        plt.plot(x, y, color=color, linewidth=0.5)
    elif boundary.geom_type == 'MultiPolygon':
        for polygon in boundary.geoms:  # Access the individual polygons
            # Shrink the area slightly to avoid overlapping the boundary
            inner_polygon = polygon.buffer(-0.25)  # Shrink the filled area slightly
            if not inner_polygon.is_empty:  # Only fill if the buffer operation succeeded
                x, y = inner_polygon.exterior.xy
                plt.fill(x, y, color=color, alpha=0.5)  # Fill the shrunken boundary
            # Draw the original boundary line
            x, y = polygon.exterior.xy
            plt.plot(x, y, color=color, linewidth=0.5)

# Remove coordinate lines, ticks, and labels
plt.axis('off')
plt.title("Cell Boundaries for 1x1 Squares")
plt.tight_layout()

# Save the plot
output_path = work_path+"/dist.png"
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Plot saved as '{output_path}'.")

# Show plot
plt.show()
print("Plot displayed successfully.")
