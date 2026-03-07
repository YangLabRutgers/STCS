"""
Core STCS class for spatial transcriptomics cell segmentation (STCS)
Merged all functionality: IO, stardist, PCA processing, pseudobulk, and annotation
Updated with multi-platform support and improved data loading
"""

import os
import pandas as pd
import scanpy as sc
import numpy as np
import squidpy as sq
import copy
import json
import tifffile
import cv2
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import decoupler as dc
import celltypist
from pathlib import Path
from h5py import File
from imageio.v2 import imread

from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from stardist.models import StarDist2D
import scipy

from scipy.sparse import issparse
import anndata as ad
from anndata import AnnData

import shutil
import random
import itertools
import matplotlib.patches as patches
import glob
import re
from collections import deque
import matplotlib.colors as mcolors
import scipy.sparse as sp
import matplotlib as mpl
from matplotlib import font_manager
import warnings

from config import (
    prob_thresh,stardist_model,n_tiles, min_cells,
    target_sum, n_top_genes,L, batch_size, search_radius as default_search_radius, assignment_mode, empty,
     stardist_mode, celltypist_target_sum
)

class STCS:
    """
    stcs = STCS(
            Folder_path= "/lab01/silas.sun/S05_data/spatial-omics/VisiumHD/Human_colon_10X/P1_CRC/binned_outputs/square_002um",
            counts_data= "filtered_feature_bc_matrix.h5",
            full_res_image_path = "/lab01/silas.sun/S05_data/spatial-omics/VisiumHD/Human_colon_10X/P1_CRC/binned_outputs/Visium_HD_Human_Colon_Cancer_P1_tissue_image.btf",
            Platform ="Visium"  # or "Stereo-seq", "Slide-seq"
    """
    
    def __init__(self, Folder_path, counts_data='filtered_feature_bc_matrix.h5', 
             full_res_image_path=None, sc_ref=None, model_path=None,
             cropped=False, crop_coords=None, Platform="Visium"):
        """
        Initialize STCS object and load data directly
        
        Parameters
        ----------
        Folder_path : str or Path
            Path to the main dataset folder. 
            - For Visium, it must contain:
                - 'filtered_feature_bc_matrix.h5'
                - 'spatial/' folder with:
                    - 'tissue_positions.csv' or '.parquet'
                    - 'scalefactors_json.json'
                    - high-res image (optional)
            - For Stereo-seq or Slide-seq, it should contain a .h5ad file.

        counts_data : str, default='filtered_feature_bc_matrix.h5'
            Name of the counts file.
            - Visium: usually 'filtered_feature_bc_matrix.h5'
            - Stereo-seq: typically a binned .h5ad expression matrix.

        full_res_image_path : str or Path, optional
            Path to the full-resolution H&E image (e.g., 'tissue_hires_image.png').
            Used for visualization and mapping, stored in `.uns['spatial']`.

        sc_ref : str, optional
            Path to a reference single-cell RNA-seq `.h5ad` file. Used in downstream pseudobulk projection or cell type annotation.

        model_path : str, optional
            Path to a pretrained model (e.g., CellTypist model `.pkl`). Used later for annotation.

        cropped : bool, default=False
            Whether the image and dataset are already cropped to a region of interest.

        crop_coords : tuple, optional
            Cropping coordinates in the form `(x1, x2, y1, y2)`. Used when `cropped=True`.

        Platform : str, default='Visium'
            Platform type. Must be one of:
                - 'Visium'
                - 'Stereo-seq'
                - 'Slide-seq'
            Determines how the data and spatial information is loaded and parsed.

        Usage Examples
        --------------
        # Visium example:
        stcs = STCS(
            Folder_path="/path/to/visium_sample",
            counts_data="filtered_feature_bc_matrix.h5",
            full_res_image_path="/path/to/visium_sample/spatial/tissue_hires_image.png",
            Platform="Visium"
        )

        # Stereo-seq example:
        stcs = STCS(
            Folder_path="/path/to/stereo_seq_sample",
            counts_data="binned_outputs.h5ad",
            Platform="Stereo-seq"
        )

        """
        
        # Store parameters
        self.Folder_path = Folder_path
        self.full_res_image_path = full_res_image_path  
        self.sc_ref = sc_ref
        self.model_path = model_path
        self.cropped = cropped
        self.crop_coords = crop_coords
        
        if Platform.lower() == "visium":
            Folder_path = Path(Folder_path)
            counts_path = Folder_path / counts_data
            self.raw_adata = sc.read_10x_h5(counts_path)
            self.raw_adata.uns["spatial"] = {}

            # Get library ID
            with File(counts_path, mode="r") as f:
                attrs = dict(f.attrs)
            library_id = str(attrs.pop("library_ids")[0], "utf-8")
            self.raw_adata.uns["spatial"][library_id] = {}

            # Load tissue positions
            spatial_path = Folder_path / "spatial"
            csv_path = spatial_path / "tissue_positions.csv"
            parquet_path = spatial_path / "tissue_positions.parquet"

            if csv_path.exists():
                tissue_positions_file = csv_path
            elif parquet_path.exists():
                tissue_positions_file = parquet_path
            else:
                raise FileNotFoundError("No tissue_positions.csv or .parquet found")

            if tissue_positions_file.suffix == ".csv":
                positions = pd.read_csv(tissue_positions_file, index_col=0)
            else:
                positions = pd.read_parquet(tissue_positions_file)
                positions.set_index("barcode", inplace=True)

            positions.columns = [
                "in_tissue", "array_row", "array_col", 
                "pxl_col_in_fullres", "pxl_row_in_fullres"
            ]
            self.raw_adata.obs = self.raw_adata.obs.join(positions, how="left")

            self.raw_adata.obsm["spatial"] = self.raw_adata.obs[
                ["pxl_row_in_fullres", "pxl_col_in_fullres"]
            ].to_numpy()
            self.raw_adata.obs.drop(columns=["pxl_row_in_fullres", "pxl_col_in_fullres"], inplace=True)

            # Load scale factors
            scale_json_path = spatial_path / "scalefactors_json.json"
            if scale_json_path.exists():
                scalefactors = json.loads(scale_json_path.read_bytes())
                self.raw_adata.uns["spatial"][library_id]["scalefactors"] = scalefactors
            else:
                raise FileNotFoundError("Missing scalefactors_json.json")

            # Add metadata
            self.raw_adata.uns["spatial"][library_id]["metadata"] = {
                k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
                for k in ("chemistry_description", "software_version")
                if k in attrs
            }

            # Set source image path
            if full_res_image_path is not None:
                full_res_image_path = str(Path(full_res_image_path).resolve())
                self.raw_adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = full_res_image_path
                
            # Try to load image for squidpy compatibility
            if full_res_image_path is not None:
                try:
                    hires_img = imread(full_res_image_path)
                    self.raw_adata.uns["spatial"][library_id]["images"] = {
                        "hires": hires_img
                    }
                except Exception as e:
                    print(f"[Warning]: Failed to load hires image for squidpy compatibility: {e}")

            # Filter to in-tissue spots
            self.adata = self.raw_adata[self.raw_adata.obs["in_tissue"] == 1].copy()
            
            print('[Log]: Loaded Visium data')
            
            if full_res_image_path is not None:
                spatial_keys = list(self.adata.uns['spatial'].keys())
                if spatial_keys:
                    library = spatial_keys[0]
                    set_path = self.adata.uns['spatial'][library]['metadata'].get('source_image_path')
                    print(f'[Log]: source_image_path set in metadata: {set_path}')
        
        elif Platform.lower() in ["stereo-seq", "slide-seq"]:
            file_path = os.path.join(Folder_path, counts_data)
            self.adata = sc.read_h5ad(file_path)
            self.raw_adata = self.adata.copy()

            print(f'[Log]: Loaded {Platform} data with {self.adata.n_obs} spots')
            
        else:
            raise ValueError(f"Unsupported platform '{Platform}'. Choose from 'Visium', 'Stereo-seq', or 'Slide-seq'.")
        
        # Make variable names unique and set data types
        self.adata.var_names_make_unique() 
        if 'array_row' in self.adata.obs.columns and 'array_col' in self.adata.obs.columns:
            self.adata.obs['array_row'] = self.adata.obs['array_row'].astype('int')
            self.adata.obs['array_col'] = self.adata.obs['array_col'].astype('int')

        # Check spatial coordinates
        if 'spatial' in self.adata.obsm:
            print('[Log]: Found spatial info')
            self.adata.obsm['spatial'] = self.adata.obsm['spatial'].astype(np.float64)
        else:
            print('[Log]: No spatial info provided')
            
        # Handle cropping if requested
        if cropped:
            if crop_coords is not None and all(k in crop_coords for k in ['x1', 'x2', 'y1', 'y2']):
                print("[Log]: Cropping using provided coordinates")
                cropped_obj = self.crop(
                    crop_coords['x1'], crop_coords['x2'], crop_coords['y1'], crop_coords['y2']
                )
            else:
                print("[Log]: Cropping automatically (no coordinates provided)")
                cropped_obj = self.crop()

            # Replace self with cropped version
            if cropped_obj is not None:
                self.raw_adata = cropped_obj.raw_adata
                self.adata = cropped_obj.adata
                self.crop_coords = cropped_obj.crop_coords
                self.cropped = True

        # Initialize data paths generated by downstream analysis
        self._barcode_data_path = None      # barcode data generated by stardist
        self._cell_data_path = None         # cell data generated by stardist
        self._dc_pseudobulk_data_path = None     # pseudobulk data generated by decoupler based on stardist
        self._dc_assignment_pseudobulk_data_path = None  # pseudobulk data generated by decoupler based on assignment
        self._celltypist_results_path = None
        
        print(f"[Log]: STCS loaded: {self.adata.n_obs} spots, {self.adata.n_vars} genes")
    
    def get_sequencing_data_region(self, adata=None):
        """Get the bounds of the sequencing data region"""
        if adata is None:
            adata = self.adata
        coords = adata.obsm["spatial"]

        x1, y1 = np.nanmin(coords, axis=0)
        x2, y2 = np.nanmax(coords, axis=0)

        return int(x1), int(x2), int(y1), int(y2) # in crop(x1, x2, y1, y2) order
    

    def crop(self, x1, x2, y1, y2, factor=None):
        """
            Crrrrrrrrrrop
            
            in original full res image's pixel coordinates.

            (x1, y1) ... (x2, y1)
            .                   .
            .                   .
            (x1, y2) ... (x2, y2)

            factor: This is the scaling factor. by default we should use the scale factor for 'hires' image loaded in raw data

        s"""

        if factor is None:
            spatial_key = list(self.adata.uns['spatial'].keys())[0]
            factor = self.adata.uns['spatial'][spatial_key]['scalefactors']['tissue_hires_scalef']
        else:
            factor = factor

        try:
            # Load original image
            img = imread(self.full_res_image_path)

            # Crop image using pixel-space coordinates
            cropped_img = img[y1:y2, x1:x2]  # Y first, X second

            # Extract coordinates
            coords = self.adata.obsm["spatial"]
            x_coords = coords[:, 0]  # X
            y_coords = coords[:, 1]  # Y

            # Mask for points inside crop box
            mask = (
                (x_coords >= x1) & (x_coords < x2) &
                (y_coords >= y1) & (y_coords < y2)
            )

            adata_subset = self.adata[mask].copy()
            raw_adata_subset = self.raw_adata[mask].copy()

            if adata_subset.shape[0] == 0:
                print("[Error]: No spots found in the crop region.")
                print(f"Crop box: x=({x1}, {x2}), y=({y1}, {y2})")
                return None

            # Shift coordinates so cropped region starts at (0,0)
            adata_subset.obsm["spatial"][:, 0] -= x1
            adata_subset.obsm["spatial"][:, 1] -= y1

            raw_adata_subset.obsm["spatial"][:, 0] -= x1
            raw_adata_subset.obsm["spatial"][:, 1] -= y1

            # Get corrected bounds
            x1c, x2c, y1c, y2c = self.get_sequencing_data_region(adata_subset)
            print(f"[Log]: Cropped region aligned to {x1c}, {y1c}, {x2c}, {y2c}")

            # Create new STCS instance
            new_stcs = STCS.__new__(STCS)
            new_stcs.raw_adata = raw_adata_subset
            new_stcs.adata = adata_subset
            new_stcs.cropped = True
            new_stcs.Folder_path = self.Folder_path
            new_stcs.full_res_image_path = self.full_res_image_path
            new_stcs.cropped_img = cropped_img 
            new_stcs.sc_ref = self.sc_ref
            new_stcs.model_path = self.model_path
            new_stcs.cropped = True
            new_stcs.crop_coords = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'factor': 1,
                'aligned_x1': x1c, 'aligned_y1': y1c,
                'aligned_x2': x2c, 'aligned_y2': y2c
            }

            # Reset paths
            new_stcs._barcode_data_path = None
            new_stcs._cell_data_path = None
            new_stcs._dc_pseudobulk_data_path = None
            new_stcs._dc_assignment_pseudobulk_data_path = None
            new_stcs._celltypist_results_path = None

            print(f"[Log]: Cropped from {self.adata.n_obs} to {adata_subset.n_obs} spots")
            return new_stcs
        except Exception as e:
            print(f"[Error]: Cropping failed: {e}")
            return None

        
    
    def copy(self):
        """Create a deep copy of the STCS object"""
        new = copy.deepcopy(self)
        return new
    
    def save(self, path):
        """Save the STCS object to disk"""
        if not os.path.exists(path):
            print(f"[Log]: Creating folder to save data: {path}")
            os.makedirs(path)
                
        metadata = {
            "Folder_path": str(self.Folder_path) if self.Folder_path else None,
            "full_res_image_path": self.full_res_image_path,
            "sc_ref": self.sc_ref,
            "model_path": self.model_path,
            "cropped": self.cropped,
            "crop_coords": self.crop_coords,
            "_barcode_data_path": self._barcode_data_path,
            "_cell_data_path": self._cell_data_path,
            "_dc_pseudobulk_data_path": self._dc_pseudobulk_data_path,
            "_dc_assignment_pseudobulk_data_path": self._dc_assignment_pseudobulk_data_path,
            "_celltypist_results_path": self._celltypist_results_path,
             "cropped_img_path": None 
        }
        
        if getattr(self, "cropped_img", None) is not None:
            cropped_path = os.path.join(path, "cropped_image.tif")
            tifffile.imwrite(cropped_path, self.cropped_img)
            metadata["cropped_img_path"] = "cropped_image.tif"

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        # save adata
        self.raw_adata.write_h5ad(os.path.join(path, "raw_adata.h5ad.gz"))
        self.adata.write_h5ad(os.path.join(path, "adata.h5ad.gz"))

        print(f"[Log]: STCS data saved to {path}")
    
    def load_img(self):
        """
        Load the associated image for the STCS object.
        If a cropped image is present, return that instead.
        """
        if hasattr(self, "cropped_img") and self.cropped_img is not None:
            print("[Log]: Returning cropped image from memory.")
            return self.cropped_img

        try:
            img = imread(self.full_res_image_path)
            print(f"[Log]: Loaded full image | Shape: {img.shape} | Dtype: {img.dtype}")
            return img
        except Exception as e:
            print(f"[Error]: Failed to load image from {self.full_res_image_path}: {e}")
            return None

            

    def load_stardist_barcode_data(self):
        """Load stardist barcode data"""
        if self._barcode_data_path and os.path.exists(self._barcode_data_path):
            return sc.read_h5ad(self._barcode_data_path)
        else:
            print("[Warning]: No stardist barcode data path available")
            return None

    def load_stardist_cell_data(self):
        """Load stardist cell data"""
        if self._cell_data_path and os.path.exists(self._cell_data_path):
            return sc.read_h5ad(self._cell_data_path)
        else:
            print("[Warning]: No stardist cell data path available")
            return None

    def load_stardist_pseudobulk_data(self):
        """Load stardist pseudobulk data"""
        if self._dc_pseudobulk_data_path and os.path.exists(self._dc_pseudobulk_data_path):
            return sc.read_h5ad(self._dc_pseudobulk_data_path)
        else:
            print("[Warning]: No decoupler pseudobulk data path available")
            return None

    def load_assignment_pseudobulk_data(self):
        """Load assignment pseudobulk data"""
        if self._dc_assignment_pseudobulk_data_path and os.path.exists(self._dc_assignment_pseudobulk_data_path):
            return sc.read_h5ad(self._dc_assignment_pseudobulk_data_path)
        else:
            print("[Warning]: No decoupler pseudobulk data path available")
            return None

    @classmethod
    def from_saved(cls, path):
        """Load STCS object from saved data"""
        
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        raw_adata = sc.read_h5ad(os.path.join(path, "raw_adata.h5ad.gz"))
        adata = sc.read_h5ad(os.path.join(path, "adata.h5ad.gz"))

        stcs = cls.__new__(cls)
        stcs.raw_adata = raw_adata
        stcs.adata = adata
        
        stcs.Folder_path = metadata.get("Folder_path")
        stcs.full_res_image_path = metadata.get("full_res_image_path")
        stcs.sc_ref = metadata.get("sc_ref")
        stcs.model_path = metadata.get("model_path")
        stcs.cropped = metadata.get("cropped", False)
        stcs.crop_coords = metadata.get("crop_coords", None)
        
        # Restore data paths
        stcs._barcode_data_path = metadata.get("_barcode_data_path")  
        stcs._cell_data_path = metadata.get("_cell_data_path")        
        stcs._dc_pseudobulk_data_path = metadata.get("_dc_pseudobulk_data_path")
        stcs._dc_assignment_pseudobulk_data_path = metadata.get("_dc_assignment_pseudobulk_data_path")
        stcs._celltypist_results_path =  metadata.get("_celltypist_results_path")
        cropped_img_path = metadata.get("cropped_img_path")
        if cropped_img_path:
            cropped_img_file = os.path.join(path, cropped_img_path)
            if os.path.exists(cropped_img_file):
                try:
                    stcs.cropped_img = tifffile.imread(cropped_img_file)
                    print(f"[Log]: Loaded cropped image from {cropped_img_file}")
                except Exception as e:
                    print(f"[Warning]: Failed to load cropped image: {e}")
            else:
                print(f"[Warning]: Cropped image file not found: {cropped_img_file}")
        else:
            stcs.cropped_img = None 
        print(f"[Log]: STCS loaded raw data from {path}: {raw_adata.n_obs} spots, {raw_adata.n_vars} genes")
        return stcs

    # ========== STARDIST PIPELINE ==========
    
    def run_stardist_pipeline(self, path, prob_thresh=prob_thresh,stardist_model=stardist_model,n_tiles=n_tiles, factor=None):
        """Run complete stardist pipeline"""
        if self.full_res_image_path is None:
            print("[Error]: No H&E image path provided in STCS object")
            return self
        
        print("[Log]: Starting stardist pipeline")
        if self.raw_adata is None:
            print("[Error]: No data provided in STCS object")
            return self
        
        if factor is None:
            spatial_key = list(self.adata.uns['spatial'].keys())[0]
            factor = self.adata.uns['spatial'][spatial_key]['scalefactors']['tissue_hires_scalef']
        else:
            factor = factor

        adata = self.raw_adata.copy()
        adata.var_names_make_unique()

        adata.obs['array_row'] = adata.obs['array_row'].astype('int')
        adata.obs['array_col'] = adata.obs['array_col'].astype('int')
        
        print("[Debug]: has cropped_img?", hasattr(self, "cropped_img"))
        if hasattr(self, "cropped_img"):
            print("[Debug]: cropped_img is None?", self.cropped_img is None)
            print("[Debug]: cropped_img shape:", getattr(self.cropped_img, "shape", "None"))
        else:
            print("[Warning]: Attribute 'cropped_img' is missing")


        if hasattr(self, "cropped_img") and self.cropped_img is not None:
            img = self.cropped_img
            print("[Log]: Using cropped image for stardist pipeline.")
        else:
            img = imread(self.full_res_image_path)
            print(f"[Log]: Loaded full image | Shape: {img.shape} | Dtype: {img.dtype}")

        """
        library = list(adata.uns['spatial'].keys())[0]
        mpp_source = adata.uns['spatial'][library]['scalefactors']['microns_per_pixel']
        scalef = mpp_source/mpp
        dim = (np.array(img.shape[:2])*scalef).astype(int)[::-1]
        scaled_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
"""
        model = StarDist2D.from_pretrained(stardist_model)
        scaled_img = normalize(img)
        labels, _ = model.predict_instances(scaled_img,prob_thresh=prob_thresh,n_tiles=n_tiles)
        
        # Save the full label image (optional - only needed if you want to keep the full segmentation map)
        stardist_path = os.path.join(path, 'stardist')
        os.makedirs(stardist_path, exist_ok=True)
        labels_path = os.path.join(stardist_path, 'stardist_labels.tif')
        tifffile.imwrite(labels_path, labels)


         # Assign labels to spots
        labels_sparse = scipy.sparse.csr_matrix(labels)
        coords = (self.raw_adata.obsm["spatial"]*factor).astype(int)[:,::-1]
        
         
        # Initialize labels in both raw and processed adata
        self.raw_adata.obs['labels_he'] = 0
        self.adata.obs['labels_he'] = 0
    
        mask = ((coords[:,0] >= 0) & 
                (coords[:,0] < labels_sparse.shape[0]) & 
                (coords[:,1] >= 0) & 
                (coords[:,1] < labels_sparse.shape[1])
                )
        cell_labels = np.asarray(labels_sparse[coords[mask,0], coords[mask,1]]).flatten()
            
        self.raw_adata.obs.loc[mask, 'labels_he'] = cell_labels
        self.adata.obs.loc[mask, 'labels_he'] = cell_labels  # Ensure both versions have labels
        
        barcode_path = os.path.join(stardist_path, "stardist_barcode_outputs.h5ad")
        self.adata.write_h5ad(barcode_path)
        # Update paths
        self._barcode_data_path = barcode_path
        self._stardist_labels_path = labels_path
        return self

    # ========== PSEUDOBULK CREATION ==========
    
    def create_pseudobulk_from_stardist(self, output_path, mode=stardist_mode, remove_empty=empty):
        """Create pseudobulk directly from stardist detection results"""
        
        print("[Log]: Starting pseudobulk creation from stardist results")
        
        # Check prerequisites
        if self._barcode_data_path is None:
            print("[Error]: No stardist barcode data available. Run stardist pipeline first.")
            return self
        
        # Create output directory
        pseudobulk_path = os.path.join(output_path, 'pseudobulk')
        os.makedirs(pseudobulk_path, exist_ok=True)
        
        # Load stardist barcode results
        print("[Log]: Loading stardist barcode data")
        barcode_data = self.load_stardist_barcode_data()
        if barcode_data is None:
            print("[Error]: Failed to load stardist barcode data")
            return self
        
        # Create pseudobulk following original logic exactly
        # First add labels_he to adata.obs from barcode_data        
        adata_with_labels = self.adata.copy()        
        # Follow the exact original logic but vectorized
        common_barcodes = adata_with_labels.obs.index.intersection(barcode_data.obs.index)
        adata_with_labels.obs['labels_he'] = np.nan
        
        # Vectorized assignment with progress tracking
        print(f"[Log]: Processing {len(common_barcodes)} common barcodes")
        
        # For very large datasets, chunk the assignment to show progress
        chunk_size = 100000  # Process 100k barcodes at a time
        if len(common_barcodes) > chunk_size:
            for i in tqdm(range(0, len(common_barcodes), chunk_size), desc="Assigning labels_he", unit="chunk"):
                chunk_barcodes = common_barcodes[i:i+chunk_size]
                adata_with_labels.obs.loc[chunk_barcodes, 'labels_he'] = barcode_data.obs.loc[chunk_barcodes, 'labels_he']
        else:
            # For smaller datasets, just do it all at once
            adata_with_labels.obs.loc[common_barcodes, 'labels_he'] = barcode_data.obs.loc[common_barcodes, 'labels_he']
        
        print(f"[Log]: Assigned labels to {adata_with_labels.obs['labels_he'].notna().sum()} cells")


        pseudobulk_data = self._create_pseudobulk(
            adata=adata_with_labels,
            mode=mode,
            cell_key='labels_he'
        )

        
        pseudobulk_file_path = os.path.join(pseudobulk_path, 'direct_pseudobulk.h5ad')
        pseudobulk_data.write_h5ad(pseudobulk_file_path)
        
        # Update STCS with pseudobulk path
        self._dc_pseudobulk_data_path = pseudobulk_file_path
        
        print(f"[Log]: Pseudobulk creation completed")
        print(f"[Log]: Results saved to {pseudobulk_path}")
        
        return self

    def create_pseudobulk_from_assignments(self, output_path, cell_key='assigned_cell_id', mode=assignment_mode, remove_empty=empty):
        """Create pseudobulk based on assignment results"""
        
        print("[Log]: Starting pseudobulk creation from assignment results")
        
        # Check prerequisites
        if not hasattr(self.adata, 'obs') or cell_key not in self.adata.obs.columns:
            print("[Error]: No assignment results available. Run assignment pipeline first.")
            return self
        
        # Create output directory
        pseudobulk_path = os.path.join(output_path, 'pseudobulk')
        os.makedirs(pseudobulk_path, exist_ok=True)
        
        # Create pseudobulk following original logic exactly
        pseudobulk_data = self._create_pseudobulk(
            self.adata, mode=mode, cell_key=cell_key
        )
        
        pseudobulk_file_path = os.path.join(pseudobulk_path, 'assignment_pseudobulk.h5ad')
        pseudobulk_data.write_h5ad(pseudobulk_file_path)
        
        # Update STCS with pseudobulk path
        self._dc_assignment_pseudobulk_data_path = pseudobulk_file_path
        
        print(f"[Log]: Pseudobulk creation completed")
        print(f"[Log]: Results saved to {pseudobulk_path}")
        
        return self

    def _create_pseudobulk(self, adata, mode, cell_key):
        """Unified pseudobulk creation function"""
        
        print(f"[Log]: Creating pseudobulk based on {cell_key}")
        
        adata = adata.copy()
        adata.obs['cell'] = adata.obs[cell_key].astype(str)
        celldata = adata[adata.obs['cell'].notna()]
        
        if not sp.isspmatrix_csr(celldata.X):
            celldata.X = celldata.X.tocsr()
            
        try:
            groups = celldata.obs['cell'].values
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            n_genes = celldata.n_vars
            
            print(f"[Log]: Found {n_groups} unique groups")
            
            # Create mapping from group name to index
            group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
            
            # Convert group names to indices for vectorized operations
            group_indices = np.array([group_to_idx[group] for group in groups])
            
            if mode == 'sum':
                print("[Log]: Starting vectorized sum aggregation...")
                # Use scipy.sparse matrix multiplication for aggregation
                # Create a binary matrix: rows=groups, cols=cells
                aggregation_matrix = sp.csr_matrix(
                    (np.ones(len(group_indices)), 
                    (group_indices, np.arange(len(group_indices)))),
                    shape=(n_groups, celldata.n_obs)
                )
                
                # Matrix multiplication: groups x cells @ cells x genes = groups x genes
                result = aggregation_matrix @ celldata.X
                
            elif mode == 'mean':
                print("[Log]: Starting vectorized mean aggregation...")

                # groups x cells (binary membership)
                aggregation_matrix = sp.csr_matrix(
                    (np.ones(len(group_indices), dtype=np.float32),
                    (group_indices, np.arange(len(group_indices)))),
                    shape=(n_groups, celldata.n_obs),
                    dtype=np.float32
                )

                # sum per group: (groups x cells) @ (cells x genes) = (groups x genes)
                result_sum = aggregation_matrix @ celldata.X

                # counts per group
                group_counts = np.asarray(aggregation_matrix.sum(axis=1)).ravel().astype(np.float32)
                group_counts[group_counts == 0] = 1.0  # avoid divide-by-zero

                # IMPORTANT: cast to float before dividing (prevents integer truncation)
                result_sum = result_sum.astype(np.float32)

                # mean: divide each row by its group count
                inv = sp.diags(1.0 / group_counts, format="csr", dtype=np.float32)
                result = (inv @ result_sum).tocsr() 

            print("[Log]: Creating final pseudobulk object...")
            adata_pseudo = ad.AnnData(
                X=result.tocsr(),  # Convert to efficient CSR format
                obs=pd.DataFrame(index=unique_groups),
                var=pd.DataFrame(index=celldata.var_names)
            )
            print(f"[Log]: Successfully created pseudobulk with {adata_pseudo.n_obs} cells and {adata_pseudo.n_vars} genes")
            
        except Exception as e:
            print(f"[Error]: Failed to create pseudobulk: {e}")
            raise

        return adata_pseudo

    # ========== PCA PROCESSING AND ASSIGNMENT ==========
    
    def run_assignment(self, output_path, min_cells=min_cells, L=L, target_sum=target_sum,
                   top_genes=n_top_genes, search_radius=default_search_radius,
                   use_sc_ref=True, normalize_distances=True, feature_name=False):
        """
        Integrated assignment pipeline: candidate search + PCA processing + gene space assignment
        Optimized for speed, identical outputs.
        """
        print("[Log]: Starting assignment pipeline based on candidate search + PCA processing + gene space assignment ")

        if self._dc_pseudobulk_data_path is None:
            print("[Error]: No pseudobulk data available. Run stardist pseudobulk creation first.")
            return self
        if self._barcode_data_path is None:
            print("[Error]: No stardist barcode data available. Run stardist pipeline first.")
            return self

        assignment_path = os.path.join(output_path, 'integrated_assignment')
        os.makedirs(assignment_path, exist_ok=True)

        # 1) Candidates (uses precomputed circular offsets)
        print("[Log]: Creating barcode to candidate cells mapping")
        barcode_candidates = self._create_barcode_candidates_mapping_fast(assignment_path, search_radius)

        # 2) Load pseudobulk once
        print("[Log]: Loading pseudobulk data")
        pseudobulk_data = self.load_stardist_pseudobulk_data()
        if pseudobulk_data is None:
            print("[Error]: Failed to load pseudobulk data")
            return self

        # 3) Process data (unchanged logic; just faster numerics later)
        if use_sc_ref and self.sc_ref is not None:
            print("[Log]: Processing with single-cell reference")
            processed_data = self._process_with_sc_reference(pseudobulk_data, assignment_path, feature_name)
        else:
            print("[Log]: Processing with spatial reference")
            processed_data = self._process_with_spatial_reference(pseudobulk_data, assignment_path)

        # 4) Cell→barcodes (keep same structure but cache np arrays)
        print("[Log]: Creating cell-to-barcodes mapping for spatial distances")
        cell_to_barcodes = self._create_cell_to_barcodes_mapping_from_stardist_fast(assignment_path)

        # 5) Assignment (vectorized math, same formulas)
        print("[Log]: Performing gene space assignment")
        final_assignments = self._perform_gene_space_assignment_fast(
            barcode_candidates, cell_to_barcodes, processed_data, assignment_path,
            normalize_distances, L
        )

        self._update_stcs_with_results(final_assignments, assignment_path)
        print(f"[Log]: Results saved to {assignment_path}")
        return self


    def _simple_scale(self, adata):
        """Simple scaling function (from original code)"""
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

    def _batch_pca_projection(self, X, loadings):
        """Project data to PCA space in batches"""
        print(f"[Log]: Projecting {X.shape[0]} samples to PCA space")
        
        X_pca_list = []
        for i in range(0, X.shape[0], batch_size):
            if i % (batch_size * 5) == 0:
                print(f"[Log]: Processing batch {i//batch_size + 1}/{(X.shape[0]-1)//batch_size + 1}")
            
            X_batch = X[i:i+batch_size].dot(loadings)
            X_pca_list.append(X_batch)
        
        return np.vstack(X_pca_list)

    def _create_barcode_candidates_mapping_fast(self, output_path, search_radius):
        """
        Same logic as _create_barcode_candidates_mapping but ~10–50x faster on large slides.

        - Precomputes circle offsets once
        - Uses dict lookup for exact (row,col)→cell_id
        - Keeps candidate order deterministic: [own_label(if any)] + neighbors in offsets order
        """
        print("[Log]: Creating barcode candidates mapping with precomputed circle offsets")

        barcode_data = self.load_stardist_barcode_data()
        if barcode_data is None:
            raise ValueError("Failed to load stardist barcode data")

        detected = barcode_data[barcode_data.obs['labels_he'] != 0]
        det_coords = detected.obs[['array_row', 'array_col']].to_numpy(dtype=int, copy=False)
        det_cells  = detected.obs['labels_he'].to_numpy(copy=False)

        # (row,col) -> cell_id dict for O(1) lookup
        coord_to_cell = { (int(r), int(c)): int(x) for (r, c), x in zip(det_coords, det_cells) }

        # own barcode -> own label (if detected)
        barcode_to_own_label = { b: int(l) for b, l in zip(detected.obs_names, det_cells) }

        adata = self.adata
        all_coords = adata.obs[['array_row', 'array_col']].to_numpy(dtype=int, copy=False)
        all_barcodes = adata.obs_names.to_numpy(copy=False)

        offsets = self._circle_offsets(search_radius)

        barcode_candidates = {}
        
        for i, barcode in enumerate(all_barcodes):
            if i % 500000 == 0:
                print(f"[Log]: Processed {i}/{len(all_barcodes)} barcodes")
            
            # Grid search for candidate cells (following original round function logic)
            row, col = int(all_coords[i][0]), int(all_coords[i][1])
            candidate_cells = []
            
            if barcode in barcode_to_own_label:
                candidate_cells.append(barcode_to_own_label[barcode])
            
            # Search in circular neighborhood
            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    neighbor_coord = (int(row + dr), int(col + dc))
                    
                    # Check if within circular radius and has detected cell
                    distance_sq = dr*dr + dc*dc
                    if distance_sq <= search_radius*search_radius and neighbor_coord in coord_to_cell:
                        cell_id = coord_to_cell[neighbor_coord]
                        if cell_id not in candidate_cells:
                            candidate_cells.append(int(cell_id))
            
            barcode_candidates[barcode] = candidate_cells
        
        # Save for reference
        candidates_path = os.path.join(output_path, 'barcode_candidates.json')
        with open(candidates_path, 'w') as f:
            json.dump({k: v for k, v in barcode_candidates.items()}, f)

        print(f"[Log]: Created candidates mapping for {len(barcode_candidates)} barcodes")
        print(f"[Log]: Candidates mapping saved to: {candidates_path}")
        return barcode_candidates


    def _process_with_sc_reference(self, pseudobulk_data, output_path, feature_name=False):
        """Process data using scRNA-seq reference (following original logic)"""
        
        print("[Log]: Loading scRNA-seq reference data")
        if not os.path.exists(self.sc_ref):
            raise ValueError(f"scRNA-seq reference not found: {self.sc_ref}")
        
        scdata = sc.read_h5ad(self.sc_ref)
        
        # Handle feature names if needed
        if feature_name and 'feature_name' in scdata.var.columns:
            scdata.var.index = scdata.var['feature_name'].astype(str)

        # Process pseudobulk data (pseudobulk_data in original code)
        print("[Log]: Processing pseudobulk data")
        pseudobulk_data = pseudobulk_data.copy()
        print("[Debug] pseudobulk shape before filtering:", pseudobulk_data.shape)

        
        # Following original preprocessing exactly
        sc.pp.filter_genes(pseudobulk_data, min_cells=min_cells)
        sc.pp.normalize_total(pseudobulk_data, target_sum=target_sum)
        sc.pp.log1p(pseudobulk_data)
        sc.pp.highly_variable_genes(pseudobulk_data, n_top_genes=n_top_genes, subset=True)
        self._simple_scale(pseudobulk_data)
        sc.tl.pca(pseudobulk_data)
        
        # Find common genes between scdata and pseudobulk_data
        genes = list(set(scdata.var_names) & set(pseudobulk_data.var_names[pseudobulk_data.var["highly_variable"]]))
        print(f"[Log]: Found {len(genes)} common genes")
        
        pseudobulk_data.var_names_make_unique()
        pseudobulk_data = pseudobulk_data[:, genes].copy()
        pseudobulk_data.write_h5ad(os.path.join(output_path, 'processed_nuclei.h5ad'))
        
        # Get PCA loadings
        loadings = pseudobulk_data.varm['PCs']
        
        # Process original spatial adata
        print("[Log]: Processing spatial adata") 
        adata = self.adata.copy()
        adata.var_names_make_unique()
        adata = adata[:, pseudobulk_data.var_names].copy()
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        self._simple_scale(adata)
        
        # Project to PCA space in batches
        adata.obsm['X_pca'] = self._batch_pca_projection(adata.X, loadings)
        adata.write_h5ad(os.path.join(output_path, 'processed_spatial_adata.h5ad'))
        
        # Process scRNA-seq reference
        print("[Log]: Processing scRNA-seq reference")
        scdata.var_names_make_unique()  
        scdata = scdata[:, pseudobulk_data.var_names].copy()
        sc.pp.normalize_total(scdata, target_sum=target_sum)
        sc.pp.log1p(scdata)
        self._simple_scale(scdata)
        
        scdata.obsm['X_pca'] = scdata.X @ loadings
        scdata.var.index.name = None
        scdata.write_h5ad(os.path.join(output_path, 'processed_scdata.h5ad'))
        
        # Compute gene similarity matrix
        print("[Log]: Computing gene similarity matrix")
        gene_cos_sim = cosine_similarity(scdata.obsm['X_pca'].T)
        
        return {
            'pseudobulk_data': pseudobulk_data,
            'spatial_adata': adata,
            'scdata': scdata,
            'gene_cos_sim': gene_cos_sim,
            'loadings_from_pseudobulk': loadings
        }

    def _process_with_spatial_reference(self, pseudobulk_data, output_path):
        """Process data using spatial reference (following original logic)"""
        
        print("[Log]: Processing with spatial reference")
        pseudobulk_data = pseudobulk_data.copy()
        
        # Following original preprocessing for spatial reference
        sc.pp.filter_genes(pseudobulk_data, min_cells=min_cells)
        sc.pp.normalize_total(pseudobulk_data, target_sum=target_sum)
        sc.pp.log1p(pseudobulk_data)
        sc.pp.highly_variable_genes(pseudobulk_data, n_top_genes=n_top_genes, subset=True)
        self._simple_scale(pseudobulk_data)
        sc.tl.pca(pseudobulk_data)
        pseudobulk_data.write_h5ad(os.path.join(output_path, 'nuclei.h5ad'))
        
        pseudobulk_data.var_names_make_unique()
        
        # Process original adata
        adata = self.adata.copy()
        adata.var_names_make_unique()
        adata = adata[:, pseudobulk_data.var_names]
        
        loadings = pseudobulk_data.varm['PCs']
        adata.raw = adata.copy()
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        self._simple_scale(adata)
        
        # Project to PCA space in batches
        adata.obsm['X_pca'] = self._batch_pca_projection(adata.X, loadings)
        
        # Compute gene similarity matrix using adata
        gene_cos_sim = cosine_similarity(adata.obsm['X_pca'].T)
        
        return {
            'pseudobulk_data': pseudobulk_data,
            'spatial_adata': adata,
            'scdata': None,
            'gene_cos_sim': gene_cos_sim,
            'loadings_from_pseudobulk': loadings
        }

    def _create_cell_to_barcodes_mapping_from_stardist_fast(self, output_path):
        """
        Same output keys as before, but also caches a numpy array of coordinates per cell
        to avoid repeatedly parsing strings in the assignment loop.
        """
        print("[Log]: Creating cell-to-barcodes mapping from stardist results (fast)")
        barcode_data = self.load_stardist_barcode_data()
        detected = barcode_data[barcode_data.obs['labels_he'] != 0]

        # group by cell id
        cell_to_barcodes = {}
        # We’ll also compute coordinates directly from barcode strings once
        # Expected format "..._<x>_<y>..." (matching your current splitter)
        for barcode_name, row in zip(detected.obs_names, detected.obs[['labels_he']].itertuples(index=False, name=None)):
            (cell_id_raw,) = row
            cid = str(int(cell_id_raw))
            cell_to_barcodes.setdefault(cid, []).append(barcode_name)

        # Build final info dict with both legacy 'barcodes' string and fast numpy coords
        info = {}
        for cid, blist in cell_to_barcodes.items():
            coords = []
            coord_strings = []
            for b in blist:
                try:
                    x = int(b.split('_')[2])    # as in your original
                    y = int(b.split('_')[3][:-2])
                    coords.append([x, y])
                    coord_strings.append(f'[{x},{y}]')
                except (IndexError, ValueError):
                    # skip malformed
                    continue
            arr = np.asarray(coords, dtype=int) if coords else np.zeros((0, 2), dtype=int)
            info[cid] = {
                'barcodes': '-'.join(coord_strings),   # unchanged
                'barcode_list': blist,
                'coordinates': coords,                 # unchanged
                'coords_np': arr                       # NEW: for fast min-distance
            }

        # Save CSV (unchanged)
        df = pd.DataFrame.from_dict({k: v['barcodes'] for k, v in info.items()},
                                    orient='index', columns=['barcodes'])
        df.index.name = 'cell_id'
        df.to_csv(os.path.join(output_path, 'cell_bar_mapping.csv'))

        print(f"[Log]: Created cell mapping for {len(info)} cells")
        return info


    def _perform_gene_space_assignment_fast(self, bar, c_b, processed_data, output_path,
                                        normalize_distances: bool, L: float):
        adata = processed_data['spatial_adata']
        pseudobulk = processed_data['pseudobulk_data']
        S = processed_data['gene_cos_sim']  # (PC x PC)

        # --- Pre-extract PCA matrices to NumPy; build index maps ---
        # adata.obsm['X_pca']: per-barcode PCA row vector
        A = np.asarray(adata.obsm['X_pca'])     # (NB, PC)
        # pseudobulk.obsm['X_pca']: per-cell PCA row vector
        P = np.asarray(pseudobulk.obsm['X_pca'])  # (NC, PC)

        # Map pseudobulk obs_names (strings like "123.0") to row indices
        pb_index = { name: i for i, name in enumerate(pseudobulk.obs_names) }
        # adata barcodes to row index
        ad_index = { name: i for i, name in enumerate(adata.obs_names) }

        # helper to get "id.0" form and "id" form once
        def _id_strings(k):
            cell_id_int = int(float(k))
            return f"{cell_id_int}.0", str(cell_id_int)

        # transcriptomic distance v^T S v where v = a - p
        # returns scalar
        def _quad_form(v):
            # v and S are small-ish; do (S @ v) then dot
            Sv = S @ v
            return float(v @ Sv)

        # Pre-extract barcode integer coords from their names once (same parsing you used)
        # This only impacts the "normalize=True" path when loc calculated from barcode name
        # and the direct path; we keep it consistent.
        def _xy_from_barcode(bname: str):
            try:
                return int(bname.split('_')[2]), int(bname.split('_')[3][:-2])
            except Exception:
                # fallback: if parsing fails, return something harmless (will force inf distance)
                return None

        assignments = {}
        distances_rows = []  # only used to save distances.csv like before

        if normalize_distances:
            # First pass: compute all pairs’ (t_raw, s_raw) to normalize transcriptomic term
            pair_keys = []
            t_list = []
            s_list = []

            for bname, cand in tqdm(bar.items(), desc="Computing distances", disable=False):
                if len(cand) <= 1:
                    continue
                loc = _xy_from_barcode(bname)
                if loc is None:
                    continue
                ai = ad_index.get(bname)
                if ai is None:
                    continue
                a = A[ai]  # (PC,)

                for k in cand:
                    pid_str, cid_str = _id_strings(k)
                    pi = pb_index.get(pid_str)
                    cell_info = c_b.get(cid_str)
                    if pi is None or cell_info is None:
                        continue

                    p = P[pi]  # (PC,)
                    v = a - p
                    t_raw = _quad_form(v)

                    coords_np = cell_info.get('coords_np')
                    if coords_np is None or coords_np.shape[0] == 0:
                        continue
                    # nearest neighbor distance to any barcode belonging to the cell
                    d = coords_np - np.array(loc, dtype=int)
                    s_raw = float(np.sqrt((d * d).sum(axis=1)).min())

                    pair_keys.append((bname, k))
                    t_list.append(t_raw)
                    s_list.append(s_raw)

            if not t_list:
                print("[Error]: No valid distance calculations found")
                return {}

            t_arr = np.asarray(t_list, dtype=float)
            s_arr = np.asarray(s_list, dtype=float)

            # same normalization as original: log -> min/max scale
            t_log = np.log(t_arr)
            t_norm = (t_log - t_log.min()) / (t_log.max() - t_log.min())
            s_w = L * s_arr
            comb = t_norm + s_w

            # Save distances.csv (same columns)
            dist_df = pd.DataFrame({
                'transcriptomic_raw': t_arr,
                'spatial_raw': s_arr,
                'transcriptomic_log': t_log,
                'transcriptomic_normalized': t_norm,
                'spatial_weighted': s_w,
                'combined_distance': comb
            })
            dist_df.to_csv(os.path.join(output_path, 'distances.csv'), index=False)

            # make lookup: barcode -> {cell_id_string: combined_distance}
            d_lookup = {}
            for (bname, k), val in zip(pair_keys, comb):
                d_lookup.setdefault(bname, {})[k] = val

            # Second pass: choose argmin per barcode (stable wrt your original ordering)
            for bname, cand in tqdm(bar.items(), desc="Final assignment", disable=False):
                if len(cand) == 0:
                    assignments[bname] = None
                elif len(cand) == 1:
                    assignments[bname] = str(int(cand[0]))
                else:
                    dl = d_lookup.get(bname, {})
                    # keep candidate order; pick min where present
                    best = None
                    best_val = float('inf')
                    for k in cand:
                        v = dl.get(k)
                        if v is not None and v < best_val:
                            best_val = v
                            best = k
                    assignments[bname] = str(int(best)) if best is not None else None

        else:
            # Direct (unnormalized) path; write distances.csv like before
            for bname, cand in tqdm(bar.items(), desc="Direct assignment", disable=False):
                if len(cand) == 0:
                    assignments[bname] = None
                    continue
                if len(cand) == 1:
                    assignments[bname] = str(int(cand[0]))
                    continue

                loc = _xy_from_barcode(bname)
                ai = ad_index.get(bname)
                if loc is None or ai is None:
                    assignments[bname] = None
                    continue
                a = A[ai]

                best = None
                best_val = float('inf')

                for k in cand:
                    pid_str, cid_str = _id_strings(k)
                    pi = pb_index.get(pid_str)
                    cell_info = c_b.get(cid_str)

                    if pi is None or cell_info is None:
                        continue
                    p = P[pi]
                    v = a - p
                    t_raw = _quad_form(v)

                    coords_np = cell_info.get('coords_np')
                    if coords_np is None or coords_np.shape[0] == 0:
                        continue
                    d = coords_np - np.array(loc, dtype=int)
                    s_raw = float(np.sqrt((d * d).sum(axis=1)).min())

                    s_w = L * s_raw
                    tot = t_raw + s_w
                    distances_rows.append([t_raw, s_raw, s_w, tot])

                    if tot < best_val:
                        best_val = tot
                        best = k

                assignments[bname] = str(int(best)) if np.isfinite(best_val) else None

            if distances_rows:
                pd.DataFrame(distances_rows,
                            columns=['transcriptomic_raw', 'spatial_raw', 'spatial_weighted', 'combined_distance']
                            ).to_csv(os.path.join(output_path, 'distances.csv'), index=False)

        print(f"[Log]: Assigned {sum(v is not None for v in assignments.values())} barcodes")
        pd.DataFrame.from_dict(assignments, orient='index', columns=['assigned_cell']).to_csv(
            os.path.join(output_path, 'assignments.csv'))
        return assignments


    def _update_stcs_with_results(self, assignments, output_path):
        """Update STCS object with final assignment results"""
        
        print("[Log]: Saving integrated assignment results")
        
        # Save assignment as CSV
        assignment_df = pd.DataFrame.from_dict(assignments, orient='index', columns=['cells'])
        assignment_df.index.name = 'barcode'
        assignment_df.to_csv(os.path.join(output_path, 'final_assignments.csv'))
        
        # Add assignments to adata.obs
        assignment_series = pd.Series(assignments, name='assigned_cell_id')
        self.adata.obs['assigned_cell_id'] = assignment_series
        
        print(f"[Log]: Assignment is done")
        
    def _circle_offsets(self, radius: int):
        """Inclusive integer circle offsets: {(dr, dc) | dr^2+dc^2 <= r^2}"""
        # cache on the instance so repeated calls are O(1)
        key = f"_circle_offsets_r{radius}"
        if hasattr(self, key):
            return getattr(self, key)
        offs = []
        r2 = radius * radius
        for dr in range(-radius, radius + 1):
            # tighten dc range using circle equation
            max_dc = int((r2 - dr*dr)**0.5)
            for dc in range(-max_dc, max_dc + 1):
                offs.append((dr, dc))
        setattr(self, key, offs)
        return offs

    # ========== CELLTYPIST ANNOTATION ==========
    
    def run_celltypist_annotation(self, output_path, train_celltypist_model=False):
        """
        Final annotated results:
        1. Create pseudobulk from assignments
        2. Run CellTypist annotation
        3. Save final annotated results
        """
        
        print("[Log]: Starting CellTypist annotation pipeline")
        
        # Check prerequisites
        if not hasattr(self.adata, 'obs') or 'assigned_cell_id' not in self.adata.obs.columns:
            print("[Error]: No assignment results found. Run assignment pipeline first.")
            return self
        
        # Create output directory
        output_dir = os.path.join(output_path, 'celltypist_annotation')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create pseudobulk for CellTypist
        print("[Log]: Creating pseudobulk for CellTypist")
        self.create_pseudobulk_from_assignments(
            output_path, cell_key='assigned_cell_id', 
            mode=assignment_mode, remove_empty=empty
        )
        
        # Load pseudobulk data
        pseudobulk_data = self.load_assignment_pseudobulk_data()
        if pseudobulk_data is None:
            print("[Error]: No pseudobulk data available")
            return self
        
        if 'nan' in pseudobulk_data.obs_names:
            pseudobulk_data = pseudobulk_data[pseudobulk_data.obs_names != 'nan']
            print(f"[Log]: Removed 'unassigned' cell from pseudobulk data")

        # Train CellTypist model if needed
        if train_celltypist_model and self.sc_ref is not None:
            print('[FIXME@SILAS: idk how to train celltypist left it here :DD ]')

        # Run CellTypist annotation
        print("[Log]: Running CellTypist annotation")
        
        # Prepare data for CellTypist
        test_data = pseudobulk_data.copy()
        sc.pp.normalize_total(test_data, target_sum=celltypist_target_sum)
        sc.pp.log1p(test_data)
        
        if self.model_path and os.path.exists(self.model_path):
            if os.path.exists(self.model_path):
                print(f"[Log]: Using model: {self.model_path}")
                predictions = celltypist.annotate(
                    test_data, 
                    model=self.model_path, 
                    majority_voting=True
                )
            else:
                print(f"[Error]: Model not found: {self.model_path}")
                return self
        else:
            print(f"[Error]: model_path not specified or doesn't exist: {self.model_path}")
            return self
        
        # Get prediction results
        predict_adata = predictions.to_adata()
        
        # Save final annotated results
        print("[Log]: Saving annotated results")
        
        # Save the annotated pseudobulk data
        predict_adata.write_h5ad(os.path.join(output_dir, 'annotated_cells.h5ad'))
        
        results_df = predict_adata.obs.copy()
        results_df.index.name = 'cell_id'
        results_df.to_csv(os.path.join(output_dir, 'cell_type_annotations.csv'))

        for col in results_df.columns:
            print(f"[Log]: Merging back column: {col}")
            self.adata.obs[f'celltypist_{col}'] = self.adata.obs['assigned_cell_id'].apply(
                lambda x: 'undefined' if pd.isna(x) else results_df.loc[str(x), col] if str(x) in results_df.index else 'undefined'
            )
        
        for col in self.adata.obs.columns:
            if self.adata.obs[col].dtype == 'object':
                self.adata.obs[col] = self.adata.obs[col].astype(str)

        # Save updated adata with annotations
        self.adata.write_h5ad(os.path.join(output_dir, 'annotated_spatial_adata.h5ad'))
        
        # Store results path
        self._celltypist_results_path = output_dir
        
        print(f"[Log]: CellTypist annotation completed")
        print(f"[Log]: Results saved to {output_dir}")
        
        return self
    
    def fill_isolated_holes(self):
        """
        For each spot with assigned_cell_id == None, look at its 8 neighbors.
        If all non-None neighbors share the same cell_id, fill this spot with that ID.
        Does not expand beyond the original x/y coordinate bounds.
        """
        import pandas as pd

        # integer coords and current assignments
        coords = self.adata.obsm['spatial'].astype(int)
        assign = self.adata.obs['assigned_cell_id'].copy()  # may contain None / NaN

        # determine original bounds
        xs, ys = coords[:, 0], coords[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        # build a fast lookup from (x,y) to AnnData index
        coord2idx = { (int(x), int(y)): i for i, (x, y) in enumerate(coords) }

        # offsets for 8-neighbors
        neigh_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        new_assign = assign.copy()

        for i, (x, y) in enumerate(coords):
            if pd.isna(assign.iloc[i]):
                nbr_ids = []
                for dx, dy in neigh_offsets:
                    nx, ny = x + dx, y + dy

                    # skip neighbors outside original bounds
                    if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                        continue

                    j = coord2idx.get((nx, ny))
                    if j is not None:
                        cid = assign.iloc[j]
                        if pd.notna(cid):
                            nbr_ids.append(cid)

                # fill only if all non-NaN neighbors agree on the same cell_id
                if nbr_ids and len(set(nbr_ids)) == 1:
                    new_assign.iloc[i] = nbr_ids[0]

        # overwrite the obs column
        self.adata.obs['assigned_cell_id'] = new_assign
        print("[Log]: Filled isolated holes within original x/y bounds based on neighbor consensus.")


    def _rect_to_xyxy(self, rect):
        """
        Convert (x, y, w, h) -> (x1, x2, y1, y2)
        """
        x, y, w, h = rect
        return x, x + w, y, y + h
    
    
    
    def count_spatial_points_in_rect(self, rect, adata=None):
        """
        Count how many spatial coordinates fall inside the crop rectangle.

        Parameters
        ----------
        rect : tuple
            (x, y, w, h) in image pixel coordinates
        adata : AnnData or None
            If None, use self.adata

        Returns
        -------
        int
            Number of spatial coordinates inside the rectangle
        """
        if adata is None:
            adata = self.adata

        if "spatial" not in adata.obsm:
            raise ValueError("No spatial coordinates found in adata.obsm['spatial'].")

        x1, x2, y1, y2 = self._rect_to_xyxy(rect)

        coords = adata.obsm["spatial"]
        xs = coords[:, 0]
        ys = coords[:, 1]

        mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        return int(np.sum(mask))
    
    
    def generate_random_crops(
        self,
        n_crops=5,
        crop_w=3000,
        crop_h=3000,
        min_spots=500,
        seed=123,
        max_tries_per_crop=200,
        use_image_boundary=True,
        verbose=True,
        allow_overlap=False,
        min_gap=0
    ):
        """
        Generate random crop rectangles, keeping only those that contain
        at least `min_spots` spatial coordinates.

        Parameters
        ----------
        n_crops : int
            Number of crops to generate
        crop_w, crop_h : int
            Crop width and height in image pixel coordinates
        min_spots : int
            Minimum number of spatial coordinates required inside crop
        seed : int
            Random seed
        max_tries_per_crop : int
            Maximum tries per requested crop
        use_image_boundary : bool
            If True, sample only inside loaded image boundary
        verbose : bool
            Print debug information
        allow_overlap : bool
            If False, accepted crops cannot overlap
        min_gap : int
            Minimum gap between accepted crops in pixels

        Returns
        -------
        rects : list of tuples
            List of (x, y, w, h)
        """
        rng = np.random.default_rng(seed)

        coords = self.adata.obsm["spatial"]
        x_min = int(np.floor(np.nanmin(coords[:, 0])))
        x_max = int(np.ceil(np.nanmax(coords[:, 0])))
        y_min = int(np.floor(np.nanmin(coords[:, 1])))
        y_max = int(np.ceil(np.nanmax(coords[:, 1])))

        if use_image_boundary:
            img = self.load_img()
            if img is None:
                raise RuntimeError("Failed to load image.")
            img_h, img_w = img.shape[:2]

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

        if verbose:
            print(f"[Debug] valid region: x=({x_min}, {x_max}), y=({y_min}, {y_max})")

        if (x_max - x_min) < crop_w or (y_max - y_min) < crop_h:
            raise ValueError(
                f"Crop size ({crop_w}, {crop_h}) is larger than valid region "
                f"({x_max - x_min}, {y_max - y_min})."
            )

        rects = []
        tries = 0
        max_total_tries = max_tries_per_crop * max(1, n_crops)

        while len(rects) < n_crops and tries < max_total_tries:
            tries += 1

            x = int(rng.integers(x_min, x_max - crop_w + 1))
            y = int(rng.integers(y_min, y_max - crop_h + 1))
            rect = (x, y, crop_w, crop_h)

            n_inside = self.count_spatial_points_in_rect(rect)

            if n_inside < min_spots:
                if verbose:
                    print(f"[Debug] try={tries}, rect={rect}, rejected: only {n_inside} spots")
                continue

            if not allow_overlap:
                overlaps_existing = any(
                    self._rectangles_overlap(rect, existing_rect, min_gap=min_gap)
                    for existing_rect in rects
                )
                if overlaps_existing:
                    if verbose:
                        print(f"[Debug] try={tries}, rect={rect}, rejected: overlaps existing crop")
                    continue

            rects.append(rect)
            if verbose:
                print(f"[Debug] try={tries}, rect={rect}, accepted: {n_inside} spots")

        if len(rects) < n_crops:
            print(
                f"[Warning]: Only found {len(rects)} valid crops out of requested {n_crops}. "
                f"Try reducing n_crops, crop size, min_spots, or min_gap."
            )

        return rects
    
    def plot_crop_rectangles(self, rects, save_file=None, figsize=(12, 12)):
        """
        Plot crop rectangles on the full-resolution image.
        rects: list of (x, y, w, h)
        """
        img = self.load_img()
        if img is None:
            raise RuntimeError("Failed to load full image for plotting.")

        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        ax = plt.gca()

        cmap = plt.cm.get_cmap("tab10", max(1, len(rects)))

        for i, (x, y, w, h) in enumerate(rects):
            color = cmap(i)
            rect_patch = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor="none"
            )
            ax.add_patch(rect_patch)
            ax.text(
                x, y - 10,
                f"crop_{i+1}",
                color=color,
                fontsize=10,
                weight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
            )

        if save_file is not None:
            plt.savefig(save_file, dpi=200, bbox_inches="tight")

        plt.show()
        
    def run_single_crop_parameter_set(
        self,
        rect,
        crop_id,
        search_radius,
        lam,
        results_dir,
        tmp_root,
        run_celltypist=True,
        prob_thresh=0.1,
        factor=1,
        pseudobulk_mode="mean",
        use_sc_ref=True,
        normalize_distances=True,
        feature_name=True
    ):
        """
        Run one crop + one (search_radius, lambda) combination.
        Keeps only the final h5ad file.
        """
        x, y, w, h = rect
        x1, x2, y1, y2 = self._rect_to_xyxy(rect)

        run_name = f"crop{crop_id:02d}_x{x}_y{y}_w{w}_h{h}_S{search_radius}_L{lam}"
        tmp_dir = os.path.join(tmp_root, run_name)
        os.makedirs(tmp_dir, exist_ok=True)

        final_h5ad = os.path.join(results_dir, f"{run_name}.h5ad")

        try:
            crop_data = self.crop(x1, x2, y1, y2, factor=factor)
            if crop_data is None:
                raise RuntimeError(f"Cropping failed for {run_name}")

            crop_data = crop_data.run_stardist_pipeline(
                path=tmp_dir,
                prob_thresh=prob_thresh,
                factor=factor
            )

            crop_data = crop_data.create_pseudobulk_from_stardist(
                output_path=tmp_dir,
                mode=pseudobulk_mode
            )

            crop_data = crop_data.run_assignment(
                output_path=tmp_dir,
                use_sc_ref=use_sc_ref,
                search_radius=search_radius,
                L=lam,
                normalize_distances=normalize_distances,
                feature_name=feature_name
            )

            if run_celltypist:
                crop_data = crop_data.run_celltypist_annotation(
                    output_path=tmp_dir,
                    train_celltypist_model=False
                )

            crop_data.adata.write_h5ad(final_h5ad)

            out = {
                "run_name": run_name,
                "crop_id": crop_id,
                "x": x, "y": y, "w": w, "h": h,
                "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                "search_radius": search_radius,
                "lambda": lam,
                "n_spots": int(crop_data.adata.n_obs),
                "n_genes": int(crop_data.adata.n_vars),
                "output_h5ad": final_h5ad,
                "status": "success",
            }

        except Exception as e:
            out = {
                "run_name": run_name,
                "crop_id": crop_id,
                "x": x, "y": y, "w": w, "h": h,
                "x1": x1, "x2": x2, "y1": y1, "y2": y2,
                "search_radius": search_radius,
                "lambda": lam,
                "n_spots": None,
                "n_genes": None,
                "output_h5ad": None,
                "status": f"failed: {repr(e)}",
            }

        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return out

    def run_parameter_sweep(
        self,
        rects,
        search_radii,
        lambdas,
        results_dir,
        tmp_root=None,
        run_celltypist=True,
        prob_thresh=0.1,
        factor=1,
        pseudobulk_mode="mean",
        use_sc_ref=True,
        normalize_distances=True,
        feature_name=True,
        summary_csv_name="run_summary.csv"
    ):
        """
        Run all crop x search_radius x lambda combinations.

        Parameters
        ----------
        rects : list
            List of (x, y, w, h)
        search_radii : list
            List of search radius values
        lambdas : list
            List of lambda values
        """
        os.makedirs(results_dir, exist_ok=True)

        if tmp_root is None:
            tmp_root = os.path.join(results_dir, "_tmp_runs")
        os.makedirs(tmp_root, exist_ok=True)

        all_results = []
        param_grid = list(itertools.product(search_radii, lambdas))

        print(f"[Log]: Number of crops = {len(rects)}")
        print(f"[Log]: Number of parameter combinations per crop = {len(param_grid)}")
        print(f"[Log]: Total runs = {len(rects) * len(param_grid)}")

        for crop_idx, rect in enumerate(rects, start=1):
            print("=" * 80)
            print(f"[Log]: Running crop {crop_idx}/{len(rects)} | rect={rect}")
            print("=" * 80)

            for s, lam in param_grid:
                print(f"[Log]: crop={crop_idx}, S={s}, L={lam}")

                result = self.run_single_crop_parameter_set(
                    rect=rect,
                    crop_id=crop_idx,
                    search_radius=s,
                    lam=lam,
                    results_dir=results_dir,
                    tmp_root=tmp_root,
                    run_celltypist=run_celltypist,
                    prob_thresh=prob_thresh,
                    factor=factor,
                    pseudobulk_mode=pseudobulk_mode,
                    use_sc_ref=use_sc_ref,
                    normalize_distances=normalize_distances,
                    feature_name=feature_name
                )
                all_results.append(result)
                print(f"[Log]: {result['status']}")

        summary_df = pd.DataFrame(all_results)
        summary_csv = os.path.join(results_dir, summary_csv_name)
        summary_df.to_csv(summary_csv, index=False)

        print(f"[Log]: Sweep finished. Summary saved to {summary_csv}")
        return summary_df
    
    def _rectangles_overlap(self, rect1, rect2, min_gap=0):
        """
        Check whether two rectangles overlap.

        Parameters
        ----------
        rect1, rect2 : tuple
            (x, y, w, h)
        min_gap : int
            Minimum required gap between rectangles.
            min_gap=0 means rectangles may touch edges but not overlap.
            min_gap>0 enforces separation.

        Returns
        -------
        bool
            True if overlapping (or too close), False otherwise.
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1

        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2

        # expand both boxes by min_gap rule
        if right1 + min_gap <= left2:
            return False
        if right2 + min_gap <= left1:
            return False
        if bottom1 + min_gap <= top2:
            return False
        if bottom2 + min_gap <= top1:
            return False

        return True
    
    def _largest_connected_component_size(self, coords):
        """
        coords: iterable of (row, col) integer coordinates for one cell

        Returns
        -------
        int
            Size of the largest 8-connected component
        """
        coords = set((int(r), int(c)) for r, c in coords)
        if len(coords) == 0:
            return 0

        visited = set()
        largest = 0

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        for start in coords:
            if start in visited:
                continue

            q = deque([start])
            visited.add(start)
            comp_size = 0

            while q:
                r, c = q.popleft()
                comp_size += 1

                for dr, dc in neighbors:
                    nxt = (r + dr, c + dc)
                    if nxt in coords and nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)

            if comp_size > largest:
                largest = comp_size

        return largest
    
    
    def compute_connection_scores_from_adata(
        self,
        adata,
        cell_col="assigned_cell_id",
        row_col="array_row",
        col_col="array_col",
        exclude_unassigned=True
    ):
        """
        Compute per-cell connection scores:
            CS(c) = |largest connected component| / |all spots assigned to cell|
        """
        if cell_col not in adata.obs.columns:
            raise ValueError(f"Missing '{cell_col}' in adata.obs")

        if row_col not in adata.obs.columns or col_col not in adata.obs.columns:
            raise ValueError(f"Need '{row_col}' and '{col_col}' in adata.obs")

        df = adata.obs[[cell_col, row_col, col_col]].copy()
        df[row_col] = pd.to_numeric(df[row_col], errors="coerce")
        df[col_col] = pd.to_numeric(df[col_col], errors="coerce")
        df = df.dropna(subset=[row_col, col_col])

        if exclude_unassigned:
            bad_vals = {"nan", "None", "none", "undefined", ""}
            df = df[df[cell_col].notna()].copy()
            df = df[~df[cell_col].astype(str).isin(bad_vals)].copy()

        results = []

        for cell_id, sub in df.groupby(cell_col):
            coords = list(zip(sub[row_col].astype(int), sub[col_col].astype(int)))
            n_total = len(coords)
            if n_total == 0:
                continue

            lcc = self._largest_connected_component_size(coords)
            conn = lcc / n_total

            results.append({
                "cell_id": str(cell_id),
                "n_spots": int(n_total),
                "largest_component_size": int(lcc),
                "connection_score": float(conn),
            })

        out = pd.DataFrame(results)
        if not out.empty:
            out = out.sort_values(
                ["connection_score", "n_spots"],
                ascending=[False, False]
            ).reset_index(drop=True)

        return out
    
    def _parse_run_filename(self, path):
        """
        Parse filenames like:
        crop01_x500_y500_w3000_h3000_S3_L2.h5ad
        """
        pattern = re.compile(
            r"crop(?P<crop_id>\d+)_x(?P<x>-?\d+)_y(?P<y>-?\d+)_w(?P<w>\d+)_h(?P<h>\d+)_S(?P<S>[-+]?\d*\.?\d+)_L(?P<L>[-+]?\d*\.?\d+)\.h5ad$"
        )

        name = os.path.basename(path)
        m = pattern.match(name)
        if m is None:
            return None

        d = m.groupdict()
        return {
            "file": path,
            "crop_id": int(d["crop_id"]),
            "x": int(d["x"]),
            "y": int(d["y"]),
            "w": int(d["w"]),
            "h": int(d["h"]),
            "S": float(d["S"]),
            "L": float(d["L"]),
        }
        
    def compute_connection_scores_for_saved_runs(
        self,
        results_path,
        output_dir_name="connection_scores",
        cell_col="assigned_cell_id",
        row_col="array_row",
        col_col="array_col"
    ):
        """
        Compute per-cell connection score CSV for each saved run (.h5ad),
        and also save a per-run summary table.
        """
        h5ad_files = sorted(glob.glob(os.path.join(results_path, "*.h5ad")))
        out_dir = os.path.join(results_path, output_dir_name)
        os.makedirs(out_dir, exist_ok=True)

        records = []

        for f in h5ad_files:
            meta = self._parse_run_filename(f)
            if meta is None:
                print(f"[Skip]: filename not matched: {os.path.basename(f)}")
                continue

            print(f"[Log]: Processing {os.path.basename(f)}")
            adata = sc.read_h5ad(f)

            conn_df = self.compute_connection_scores_from_adata(
                adata=adata,
                cell_col=cell_col,
                row_col=row_col,
                col_col=col_col,
                exclude_unassigned=True
            )

            out_csv = os.path.join(
                out_dir,
                os.path.basename(f).replace(".h5ad", "_connection_score.csv")
            )
            conn_df.to_csv(out_csv, index=False)

            if conn_df.empty:
                mean_conn = np.nan
                std_conn = np.nan
                n_cells = 0
            else:
                mean_conn = float(conn_df["connection_score"].mean())
                std_conn = float(conn_df["connection_score"].std(ddof=0))
                n_cells = int(conn_df.shape[0])

            records.append({
                **meta,
                "connection_csv": out_csv,
                "mean_connection_score": mean_conn,
                "std_connection_score": std_conn,
                "n_cells": n_cells,
            })

        summary = pd.DataFrame(records)
        summary_csv = os.path.join(out_dir, "connection_score_summary_per_run.csv")
        summary.to_csv(summary_csv, index=False)

        print(f"[Log]: Saved per-run summary to {summary_csv}")
        return summary
    

    def summarize_connection_across_crops(self, conn_run_summary):
        """
        Aggregate per-run mean connection score across crops
        for each (L, S) combination.
        """
        df = conn_run_summary.copy()

        grouped = (
            df.groupby(["L", "S"], as_index=False)
              .agg(
                  mean_conn=("mean_connection_score", "mean"),
                  std_conn=("mean_connection_score", lambda x: np.std(x, ddof=0)),
                  n_crops=("crop_id", "nunique"),
              )
              .sort_values(["L", "S"])
              .reset_index(drop=True)
        )
        return grouped
    
    def _annotate_heatmap_two_lines(
        self,
        ax,
        i,
        j,
        mval,
        sval,
        cmap,
        norm,
        fmt_mean="{:.2f}",
        fmt_std="±{:.2f}",
        mean_fontsize=11,
        std_fontsize=9,
        y_offset_mean=-0.12,
        y_offset_std=0.18
    ):
        rgba = cmap(norm(mval))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = "black" if lum > 0.6 else "white"

        ax.text(
            j, i + y_offset_mean, fmt_mean.format(mval),
            ha="center", va="center",
            fontsize=mean_fontsize, color=color
        )
        ax.text(
            j, i + y_offset_std, fmt_std.format(sval),
            ha="center", va="center",
            fontsize=std_fontsize, color=color
        )
        
    def plot_connection_heatmap(
        self,
        conn_summary_df,
        out_pdf=None,
        out_svg=None,
        title="Connection Score — mean ± std across crops"
    ):
        """
        Plot heatmap of connection score summarized across crops.
        """
        if conn_summary_df.empty:
            print("[connection] Empty dataframe, skip.")
            return

        mean_df = (
            conn_summary_df
            .pivot(index="L", columns="S", values="mean_conn")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        std_df = (
            conn_summary_df
            .pivot(index="L", columns="S", values="std_conn")
            .loc[mean_df.index, mean_df.columns]
        )

        norm = mcolors.Normalize(
            vmin=np.nanmin(mean_df.values),
            vmax=np.nanmax(mean_df.values)
        )
        cmap = plt.cm.RdYlGn

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(
            mean_df.values,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )
        im.set_rasterized(True)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Connection Score (higher = better)")

        ax.set_xticks(np.arange(len(mean_df.columns)))
        ax.set_xticklabels(mean_df.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(mean_df.index)))
        ax.set_yticklabels(mean_df.index)

        ax.set_xlabel("S parameter")
        ax.set_ylabel("L parameter")
        ax.set_title(title)

        for i in range(mean_df.shape[0]):
            for j in range(mean_df.shape[1]):
                mval = mean_df.values[i, j]
                sval = std_df.values[i, j]

                if np.isnan(mval):
                    ax.text(j, i, "NA", ha="center", va="center", fontsize=11)
                    continue

                self._annotate_heatmap_two_lines(
                    ax=ax,
                    i=i,
                    j=j,
                    mval=mval,
                    sval=sval,
                    cmap=cmap,
                    norm=norm
                )

        fig.tight_layout()

        if out_pdf is not None:
            fig.savefig(out_pdf, dpi=300)
            print("[save]", out_pdf)
        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save]", out_svg)

        plt.show()
        
    # ========== DETECTED GENES / CELL OVER PARAMETER SWEEP ==========

    def _clean_cell_ids(self, series):
        """
        Normalize assigned cell IDs and convert invalid strings to NA.
        """
        s = pd.Series(series).astype("string").str.strip()
        bad = {"nan", "None", "none", "undefined", ""}
        s = s.where(~s.isin(bad), pd.NA)
        return s

    def detected_genes_per_reconstructed_cell(self, adata, cell_col="assigned_cell_id"):
        """
        From a spot-level AnnData, reconstruct per-cell counts by summing spots
        assigned to the same cell, then count nonzero genes per cell.

        Parameters
        ----------
        adata : AnnData
            Spot-level AnnData from one saved run
        cell_col : str
            Column in adata.obs containing reconstructed cell assignments

        Returns
        -------
        DataFrame
            Columns:
                - cell_id
                - n_detected_genes
                - n_spots
        """
        if cell_col not in adata.obs.columns:
            raise ValueError(f"Missing '{cell_col}' in adata.obs")

        cell_ids = self._clean_cell_ids(adata.obs[cell_col])
        valid_mask = cell_ids.notna().to_numpy()

        if valid_mask.sum() == 0:
            return pd.DataFrame(columns=["cell_id", "n_detected_genes", "n_spots"])

        X = adata.X[valid_mask]
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        groups = cell_ids[valid_mask].astype(str).to_numpy()
        unique_groups, group_indices = np.unique(groups, return_inverse=True)

        n_groups = len(unique_groups)
        n_obs = len(groups)

        # groups x spots membership matrix
        M = sp.csr_matrix(
            (np.ones(n_obs, dtype=np.float32), (group_indices, np.arange(n_obs))),
            shape=(n_groups, n_obs),
            dtype=np.float32
        )

        # summed expression per reconstructed cell
        X_sum = (M @ X).tocsr()

        # number of detected genes after summing all assigned spots
        n_detected = np.asarray(X_sum.getnnz(axis=1)).ravel()

        # number of spots per reconstructed cell
        n_spots = np.asarray(M.sum(axis=1)).ravel().astype(int)

        out = pd.DataFrame({
            "cell_id": unique_groups.astype(str),
            "n_detected_genes": n_detected.astype(int),
            "n_spots": n_spots
        })

        return out.sort_values(
            ["n_detected_genes", "n_spots"],
            ascending=[False, False]
        ).reset_index(drop=True)

    def collect_shared_cell_ids_per_crop_from_success_df(
        self,
        success_df,
        cell_col="assigned_cell_id"
    ):
        """
        For each crop_id, find the intersection of reconstructed cell IDs
        across all successful runs for that crop.

        Parameters
        ----------
        success_df : DataFrame
            Must contain:
                - crop_id
                - output_h5ad
        cell_col : str
            Assignment column inside each saved h5ad

        Returns
        -------
        dict
            {crop_id: set(shared_cell_ids)}
        """
        required = {"crop_id", "output_h5ad"}
        missing = required - set(success_df.columns)
        if missing:
            raise ValueError(f"success_df missing required columns: {missing}")

        shared = {}

        for crop_id, sub in success_df.groupby("crop_id"):
            sets = []
            print(f"[shared] crop_id={crop_id}, runs={len(sub)}")

            for _, row in sub.iterrows():
                h5 = row["output_h5ad"]
                if not os.path.exists(h5):
                    continue

                try:
                    adata = sc.read_h5ad(h5)
                    cell_df = self.detected_genes_per_reconstructed_cell(
                        adata,
                        cell_col=cell_col
                    )
                    ids = set(cell_df["cell_id"].astype(str).tolist())
                    sets.append(ids)
                except Exception as e:
                    print(f"[Warning]: failed reading {h5}: {e}")
                    continue

            if len(sets) == 0:
                shared[crop_id] = set()
            else:
                inter = sets[0]
                for s in sets[1:]:
                    inter = inter.intersection(s)
                shared[crop_id] = inter

            print(f"[shared] crop_id={crop_id}: {len(shared[crop_id])} shared cell_ids")

        return shared

    def mean_detected_genes_on_shared_cells(
        self,
        adata,
        shared_ids_set,
        cell_col="assigned_cell_id"
    ):
        """
        Mean detected genes per reconstructed cell, restricted to shared cell IDs.
        """
        if not shared_ids_set:
            return np.nan

        cell_df = self.detected_genes_per_reconstructed_cell(
            adata,
            cell_col=cell_col
        )
        if cell_df.empty:
            return np.nan

        sub = cell_df[cell_df["cell_id"].isin(shared_ids_set)].copy()
        if sub.empty:
            return np.nan

        return float(sub["n_detected_genes"].mean())

    def summarize_detected_genes_shared_over_success_df(
        self,
        success_df,
        shared_ids_per_crop,
        cell_col="assigned_cell_id"
    ):
        """
        For each (lambda, search_radius):
          - for each crop: mean detected genes per reconstructed cell on shared cells
          - aggregate across crops: mean ± std

        Parameters
        ----------
        success_df : DataFrame
            Must contain:
                - run_name
                - crop_id
                - lambda
                - search_radius
                - output_h5ad
        shared_ids_per_crop : dict
            Output of collect_shared_cell_ids_per_crop_from_success_df()
        cell_col : str
            Assignment column in saved h5ad

        Returns
        -------
        df_summary : DataFrame
            Columns:
                - L
                - S
                - mean_value
                - std_value
                - n_crops
        df_per_run : DataFrame
            Per-run values before crop aggregation
        """
        required = {"run_name", "crop_id", "lambda", "search_radius", "output_h5ad"}
        missing = required - set(success_df.columns)
        if missing:
            raise ValueError(f"success_df missing required columns: {missing}")

        records = []
        per_run_records = []

        for (L, S), sub in success_df.groupby(["lambda", "search_radius"]):
            crop_vals = []

            for _, row in sub.iterrows():
                crop_id = row["crop_id"]
                shared_ids = shared_ids_per_crop.get(crop_id, set())
                h5 = row["output_h5ad"]

                if not shared_ids or not os.path.exists(h5):
                    continue

                try:
                    adata = sc.read_h5ad(h5)
                    v = self.mean_detected_genes_on_shared_cells(
                        adata,
                        shared_ids_set=shared_ids,
                        cell_col=cell_col
                    )

                    if np.isfinite(v):
                        crop_vals.append(v)
                        per_run_records.append({
                            "run_name": row["run_name"],
                            "crop_id": crop_id,
                            "L": float(L),
                            "S": float(S),
                            "mean_detected_genes_shared_cells": float(v),
                            "n_shared_cells": int(len(shared_ids)),
                            "output_h5ad": h5,
                        })

                except Exception as e:
                    print(f"[Warning]: failed summarizing {h5}: {e}")
                    continue

            if crop_vals:
                records.append({
                    "L": float(L),
                    "S": float(S),
                    "mean_value": float(np.mean(crop_vals)),
                    "std_value": float(np.std(crop_vals, ddof=0)),
                    "n_crops": int(len(crop_vals)),
                })

        df_summary = pd.DataFrame(records)
        if not df_summary.empty:
            df_summary = df_summary.sort_values(["L", "S"]).reset_index(drop=True)

        df_per_run = pd.DataFrame(per_run_records)
        if not df_per_run.empty:
            df_per_run = df_per_run.sort_values(["L", "S", "crop_id"]).reset_index(drop=True)

        return df_summary, df_per_run

    def _setup_editable_vector_fonts(self, font_candidates=None):
        """
        Configure matplotlib for editable SVG/PDF text.
        """
        if font_candidates is None:
            font_candidates = ["Helvetica"]

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

        font_name, font_path = pick_font(font_candidates)

        mpl.rcParams["svg.fonttype"] = "none"
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [font_name]
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42

        print(f"[font] Using: {font_name}")
        print(f"[font] Path : {font_path}")

    def plot_detected_genes_across_S_by_L(
        self,
        df_summary,
        out_svg=None,
        title="Shared-cell mean #detected genes per reconstructed cell — across S"
    ):
        """
        Plot x=S, one line per L, errorbar=std across crops.
        """
        if df_summary.empty:
            print("[plot] Empty df, skipping.")
            return

        d = df_summary.copy()
        d["S"] = d["S"].astype(float)
        d["L"] = d["L"].astype(float)

        Ls = sorted(d["L"].unique())

        fig, ax = plt.subplots(figsize=(9, 5))

        for l in Ls:
            sub = d[d["L"] == l].sort_values("S")
            ax.errorbar(
                sub["S"].values,
                sub["mean_value"].values,
                yerr=sub["std_value"].values,
                marker="o",
                linewidth=1.4,
                capsize=2.8,
                label=f"L={l:g}",
            )

        ax.set_title(title)
        ax.set_xlabel("S")
        ax.set_ylabel("Mean #detected genes / cell (mean ± std across crops)")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend(title="L", bbox_to_anchor=(1.02, 1), loc="upper left")

        fig.tight_layout()

        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save] wrote", out_svg)

        plt.show()

    def run_detected_genes_shared_analysis(
        self,
        success_df,
        out_dir,
        cell_col="assigned_cell_id",
        summary_csv_name="gene_counts_shared_LS_summary.csv",
        per_run_csv_name="gene_counts_per_run_summary.csv",
        out_svg_name="lines_gene_across_S_by_L.svg",
        font_candidates=None
    ):
        """
        Full pipeline:
          1) collect shared reconstructed cell IDs per crop
          2) summarize mean detected genes per reconstructed cell on shared cells
          3) save CSVs
          4) plot line plot across S by L

        Returns
        -------
        dict with:
            - shared_ids_per_crop
            - df_summary
            - df_per_run
            - summary_csv
            - per_run_csv
            - out_svg
        """
        os.makedirs(out_dir, exist_ok=True)
        warnings.filterwarnings("ignore")

        self._setup_editable_vector_fonts(font_candidates=font_candidates)

        shared_ids_per_crop = self.collect_shared_cell_ids_per_crop_from_success_df(
            success_df=success_df,
            cell_col=cell_col
        )

        df_summary, df_per_run = self.summarize_detected_genes_shared_over_success_df(
            success_df=success_df,
            shared_ids_per_crop=shared_ids_per_crop,
            cell_col=cell_col
        )

        summary_csv = os.path.join(out_dir, summary_csv_name)
        per_run_csv = os.path.join(out_dir, per_run_csv_name)
        out_svg = os.path.join(out_dir, out_svg_name)

        df_summary.to_csv(summary_csv, index=False)
        df_per_run.to_csv(per_run_csv, index=False)

        print("[save] wrote", summary_csv)
        print("[save] wrote", per_run_csv)

        self.plot_detected_genes_across_S_by_L(
            df_summary=df_summary,
            out_svg=out_svg
        )

        return {
            "shared_ids_per_crop": shared_ids_per_crop,
            "df_summary": df_summary,
            "df_per_run": df_per_run,
            "summary_csv": summary_csv,
            "per_run_csv": per_run_csv,
            "out_svg": out_svg,
        }
        
        
    def _find_celltype_column(self, adata, preferred=None):
        """
        Find a usable cell type column in adata.obs.
        """
        if preferred is not None:
            if preferred not in adata.obs.columns:
                raise ValueError(f"Requested cell type column not found: {preferred}")
            return preferred

        candidates = [
            "celltypist_majority_voting",
            "celltypist_predicted_labels",
            "celltypist_over_clustering",
            "leiden_ct",
        ]

        for c in candidates:
            if c in adata.obs.columns:
                return c

        # fallback: first column starting with celltypist_
        for c in adata.obs.columns:
            if str(c).startswith("celltypist_"):
                return c

        raise ValueError("No cell type column found in adata.obs")
    
    
    def get_celltype_per_reconstructed_cell(
        self,
        adata,
        cell_col="assigned_cell_id",
        celltype_col=None,
        exclude_unassigned=True
    ):
        """
        Collapse spot-level annotations into one cell type per reconstructed cell.

        Returns
        -------
        DataFrame with columns:
            - cell_id
            - cell_type
            - n_spots
        """
        if cell_col not in adata.obs.columns:
            raise ValueError(f"Missing '{cell_col}' in adata.obs")

        celltype_col = self._find_celltype_column(adata, preferred=celltype_col)

        df = adata.obs[[cell_col, celltype_col]].copy()
        df[cell_col] = self._clean_cell_ids(df[cell_col])

        if exclude_unassigned:
            df = df[df[cell_col].notna()].copy()

        # clean cell type values
        df[celltype_col] = pd.Series(df[celltype_col]).astype("string").str.strip()
        bad_ct = {"nan", "None", "none", "undefined", ""}
        df.loc[df[celltype_col].isin(bad_ct), celltype_col] = pd.NA

        records = []

        for cell_id, sub in df.groupby(cell_col):
            ct_series = sub[celltype_col].dropna().astype(str)

            if len(ct_series) == 0:
                cell_type = "undefined"
            else:
                # most frequent label among spots assigned to that cell
                cell_type = ct_series.value_counts().idxmax()

            records.append({
                "cell_id": str(cell_id),
                "cell_type": str(cell_type),
                "n_spots": int(sub.shape[0]),
            })

        out = pd.DataFrame(records)
        if not out.empty:
            out = out.sort_values(["cell_type", "n_spots"], ascending=[True, False]).reset_index(drop=True)

        return out
    
    
    def get_celltype_counts_from_h5ad(
        self,
        h5ad_path,
        cell_col="assigned_cell_id",
        celltype_col=None,
        exclude_unassigned=True
    ):
        """
        Read one saved run h5ad and return cell type counts.
        """
        if not os.path.exists(h5ad_path):
            raise FileNotFoundError(f"h5ad not found: {h5ad_path}")

        adata = sc.read_h5ad(h5ad_path)

        cell_df = self.get_celltype_per_reconstructed_cell(
            adata=adata,
            cell_col=cell_col,
            celltype_col=celltype_col,
            exclude_unassigned=exclude_unassigned
        )

        if cell_df.empty:
            return {}

        counts = cell_df["cell_type"].value_counts().to_dict()
        return counts
    
    def summarize_celltype_counts_for_fixed_S(
        self,
        counts_dict,
        fixed_S
    ):
        """
        Prepare dataframe for plotting cell type counts vs L at fixed S.
        """
        rows = []
        all_ct = set()

        for (L, S), counts in counts_dict.items():
            if float(S) != float(fixed_S):
                continue
            all_ct.update(counts.keys())

        all_ct = sorted(all_ct)

        L_values = sorted({float(L) for (L, S) in counts_dict.keys() if float(S) == float(fixed_S)})

        for ct in all_ct:
            for L in L_values:
                count = counts_dict.get((L, float(fixed_S)), {}).get(ct, 0)
                rows.append({
                    "cell_type": ct,
                    "L": float(L),
                    "S": float(fixed_S),
                    "count": int(count),
                })

        return pd.DataFrame(rows), all_ct, L_values

    def summarize_celltype_counts_for_fixed_L(
        self,
        counts_dict,
        fixed_L
    ):
        """
        Prepare dataframe for plotting cell type counts vs S at fixed L.
        """
        rows = []
        all_ct = set()

        for (L, S), counts in counts_dict.items():
            if float(L) != float(fixed_L):
                continue
            all_ct.update(counts.keys())

        all_ct = sorted(all_ct)

        S_values = sorted({float(S) for (L, S) in counts_dict.keys() if float(L) == float(fixed_L)})

        for ct in all_ct:
            for S in S_values:
                count = counts_dict.get((float(fixed_L), S), {}).get(ct, 0)
                rows.append({
                    "cell_type": ct,
                    "L": float(fixed_L),
                    "S": float(S),
                    "count": int(count),
                })

        return pd.DataFrame(rows), all_ct, S_values
    
    
    def _make_celltype_color_map(self, all_ct):
        """
        Make a stable color map for cell types.
        """
        all_ct = sorted(all_ct)

        if len(all_ct) <= 20:
            cmap = plt.cm.tab20
        elif len(all_ct) <= 40:
            cmap = plt.cm.gist_ncar
        else:
            cmap = plt.cm.turbo

        colors = cmap(np.linspace(0, 1, len(all_ct)))
        return dict(zip(all_ct, colors))
    
    def plot_celltype_counts_vs_L(
        self,
        df_plot,
        all_ct,
        L_values,
        fixed_S,
        out_svg=None,
        title=None
    ):
        """
        Plot cell type counts vs L at fixed S.
        """
        if df_plot.empty:
            print("[plot] Empty dataframe, skipping.")
            return

        color_map = self._make_celltype_color_map(all_ct)

        fig, ax = plt.subplots(figsize=(10, 6))

        for ct in all_ct:
            sub = df_plot[df_plot["cell_type"] == ct].sort_values("L")
            counts = sub["count"].values
            if len(counts) == 0 or np.max(counts) == 0:
                continue

            ax.plot(
                sub["L"].values,
                counts,
                marker="o",
                label=ct,
                color=color_map[ct],
                linewidth=2
            )

        ax.set_xlabel("L", fontsize=12)
        ax.set_ylabel("Number of Cells", fontsize=12)

        if title is None:
            title = f"Cell Type Counts vs L (S={fixed_S})"
        ax.set_title(title, fontsize=14)

        ax.set_xticks(L_values)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", title="cell type")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save] wrote", out_svg)

        plt.show()
        
        
    def plot_celltype_counts_vs_S(
        self,
        df_plot,
        all_ct,
        S_values,
        fixed_L,
        out_svg=None,
        title=None
    ):
        """
        Plot cell type counts vs S at fixed L.
        """
        if df_plot.empty:
            print("[plot] Empty dataframe, skipping.")
            return

        color_map = self._make_celltype_color_map(all_ct)

        fig, ax = plt.subplots(figsize=(10, 6))

        for ct in all_ct:
            sub = df_plot[df_plot["cell_type"] == ct].sort_values("S")
            counts = sub["count"].values
            if len(counts) == 0 or np.max(counts) == 0:
                continue

            ax.plot(
                sub["S"].values,
                counts,
                marker="o",
                label=ct,
                color=color_map[ct],
                linewidth=2
            )

        ax.set_xlabel("S", fontsize=12)
        ax.set_ylabel("Number of Cells", fontsize=12)

        if title is None:
            title = f"Cell Type Counts vs S (L={fixed_L})"
        ax.set_title(title, fontsize=14)

        ax.set_xticks(S_values)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", title="cell type")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save] wrote", out_svg)

        plt.show()
        
    def run_celltype_count_lineplots_from_results(
        self,
        results_path,
        out_dir,
        summary_csv_name="run_summary.csv",
        crop_id=1,
        fixed_S=5,
        fixed_L=0.05,
        cell_col="assigned_cell_id",
        celltype_col=None,
        exclude_unassigned=True,
        save_tables=True
    ):
        """
        Full pipeline:
          1) load success runs from summary CSV
          2) collect cell type counts from saved h5ad files for one crop
          3) plot:
                - counts vs L at fixed S
                - counts vs S at fixed L

        Returns
        -------
        dict with summary tables and plot paths
        """
        os.makedirs(out_dir, exist_ok=True)
        self._setup_editable_vector_fonts(font_candidates=["Helvetica"])

        counts_dict = self.collect_celltype_counts_from_results(
            results_path=results_path,
            summary_csv_name=summary_csv_name,
            crop_id=crop_id,
            cell_col=cell_col,
            celltype_col=celltype_col,
            exclude_unassigned=exclude_unassigned
        )

        df_L, all_ct_L, L_values = self.summarize_celltype_counts_for_fixed_S(
            counts_dict=counts_dict,
            fixed_S=fixed_S
        )

        df_S, all_ct_S, S_values = self.summarize_celltype_counts_for_fixed_L(
            counts_dict=counts_dict,
            fixed_L=fixed_L
        )

        svg_L = os.path.join(out_dir, f"celltype_counts_vs_L_S{fixed_S}_crop{crop_id}.svg")
        svg_S = os.path.join(out_dir, f"celltype_counts_vs_S_L{fixed_L}_crop{crop_id}.svg")

        self.plot_celltype_counts_vs_L(
            df_plot=df_L,
            all_ct=all_ct_L,
            L_values=L_values,
            fixed_S=fixed_S,
            out_svg=svg_L,
            title=f"Cell Type Counts vs L (S={fixed_S}, crop{crop_id})"
        )

        self.plot_celltype_counts_vs_S(
            df_plot=df_S,
            all_ct=all_ct_S,
            S_values=S_values,
            fixed_L=fixed_L,
            out_svg=svg_S,
            title=f"Cell Type Counts vs S (L={fixed_L}, crop{crop_id})"
        )

        csv_L = os.path.join(out_dir, f"celltype_counts_vs_L_S{fixed_S}_crop{crop_id}.csv")
        csv_S = os.path.join(out_dir, f"celltype_counts_vs_S_L{fixed_L}_crop{crop_id}.csv")

        if save_tables:
            df_L.to_csv(csv_L, index=False)
            df_S.to_csv(csv_S, index=False)
            print("[save] wrote", csv_L)
            print("[save] wrote", csv_S)

        return {
            "counts_dict": counts_dict,
            "df_vs_L": df_L,
            "df_vs_S": df_S,
            "svg_vs_L": svg_L,
            "svg_vs_S": svg_S,
            "csv_vs_L": csv_L,
            "csv_vs_S": csv_S,
        }
        

    def _find_reference_celltype_column(self, adata, preferred=None):
        """
        Find a usable cell type column in scRNA reference adata.obs.
        """
        if preferred is not None:
            if preferred not in adata.obs.columns:
                raise ValueError(f"Requested reference cell type column not found: {preferred}")
            return preferred

        candidates = [
            "cell_type",
            "celltype",
            "cell_type_level1",
            "cell_type_level2",
            "celltype_major",
            "celltype_minor",
            "annotation",
            "annot",
            "label",
            "labels",
            "celltypist_predicted_labels",
            "leiden_ct",
        ]

        for c in candidates:
            if c in adata.obs.columns:
                return c

        raise ValueError("No usable reference cell type column found in sc_ref adata.obs")


    def get_single_cell_reference_type_counts(
        self,
        ref_path=None,
        celltype_col=None
    ):
        """
        Read the single-cell reference h5ad and return cell type counts.

        Returns
        -------
        counts : dict
            {cell_type: count}
        """
        if ref_path is None:
            ref_path = self.sc_ref

        if ref_path is None or not os.path.exists(ref_path):
            raise FileNotFoundError(f"Single-cell reference not found: {ref_path}")

        adata = sc.read_h5ad(ref_path)
        ct_col = self._find_reference_celltype_column(adata, preferred=celltype_col)

        s = pd.Series(adata.obs[ct_col]).astype("string").str.strip()
        bad = {"nan", "None", "none", "undefined", ""}
        s = s[~s.isin(bad)].dropna()

        return s.value_counts().to_dict()


    def build_single_cell_reference_count_tables(
        self,
        L_values,
        S_values,
        fixed_S=5,
        fixed_L=0.05,
        ref_path=None,
        celltype_col=None
    ):
        """
        Build plotting tables for single-cell reference cell type counts.

        Since the reference itself does not depend on L or S, the same counts are
        repeated across the requested x-values so you can compare against your STCS plots.
        """
        counts = self.get_single_cell_reference_type_counts(
            ref_path=ref_path,
            celltype_col=celltype_col
        )

        all_ct = sorted(counts.keys())

        rows_L = []
        for ct in all_ct:
            for L in sorted([float(x) for x in L_values]):
                rows_L.append({
                    "cell_type": ct,
                    "L": float(L),
                    "S": float(fixed_S),
                    "count": int(counts.get(ct, 0)),
                })

        rows_S = []
        for ct in all_ct:
            for S in sorted([float(x) for x in S_values]):
                rows_S.append({
                    "cell_type": ct,
                    "L": float(fixed_L),
                    "S": float(S),
                    "count": int(counts.get(ct, 0)),
                })

        df_L = pd.DataFrame(rows_L)
        df_S = pd.DataFrame(rows_S)

        return df_L, df_S, all_ct


    def run_single_cell_reference_count_lineplots(
        self,
        out_dir,
        L_values,
        S_values,
        fixed_S=5,
        fixed_L=0.05,
        ref_path=None,
        celltype_col=None,
        save_tables=True
    ):
        """
        Plot single-cell reference cell type counts in the same line-plot format:
          - counts vs L at fixed S
          - counts vs S at fixed L

        This is useful as a reference/baseline figure alongside STCS results.
        """
        os.makedirs(out_dir, exist_ok=True)
        self._setup_editable_vector_fonts(font_candidates=["Helvetica"])

        df_L, df_S, all_ct = self.build_single_cell_reference_count_tables(
            L_values=L_values,
            S_values=S_values,
            fixed_S=fixed_S,
            fixed_L=fixed_L,
            ref_path=ref_path,
            celltype_col=celltype_col
        )

        svg_L = os.path.join(out_dir, f"singlecell_celltype_counts_vs_L_S{fixed_S}.svg")
        svg_S = os.path.join(out_dir, f"singlecell_celltype_counts_vs_S_L{fixed_L}.svg")
        csv_L = os.path.join(out_dir, f"singlecell_celltype_counts_vs_L_S{fixed_S}.csv")
        csv_S = os.path.join(out_dir, f"singlecell_celltype_counts_vs_S_L{fixed_L}.csv")

        self.plot_celltype_counts_vs_L(
            df_plot=df_L,
            all_ct=all_ct,
            L_values=sorted([float(x) for x in L_values]),
            fixed_S=fixed_S,
            out_svg=svg_L,
            title=f"Single-cell Reference Cell Type Counts vs L (S={fixed_S})"
        )

        self.plot_celltype_counts_vs_S(
            df_plot=df_S,
            all_ct=all_ct,
            S_values=sorted([float(x) for x in S_values]),
            fixed_L=fixed_L,
            out_svg=svg_S,
            title=f"Single-cell Reference Cell Type Counts vs S (L={fixed_L})"
        )

        if save_tables:
            df_L.to_csv(csv_L, index=False)
            df_S.to_csv(csv_S, index=False)
            print("[save] wrote", csv_L)
            print("[save] wrote", csv_S)

        return {
            "df_vs_L": df_L,
            "df_vs_S": df_S,
            "svg_vs_L": svg_L,
            "svg_vs_S": svg_S,
            "csv_vs_L": csv_L,
            "csv_vs_S": csv_S,
        }
        
    def get_single_cell_reference_detected_genes(
        self,
        ref_path=None
    ):
        """
        Read the single-cell reference h5ad and return per-cell detected gene counts.

        Returns
        -------
        DataFrame with columns:
            - cell_id
            - n_detected_genes
        """
        if ref_path is None:
            ref_path = self.sc_ref

        if ref_path is None or not os.path.exists(ref_path):
            raise FileNotFoundError(f"Single-cell reference not found: {ref_path}")

        adata = sc.read_h5ad(ref_path)

        if "n_genes_by_counts" not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True)

        cell_ids = pd.Series(adata.obs_names).astype(str)

        out = pd.DataFrame({
            "cell_id": cell_ids.values,
            "n_detected_genes": pd.to_numeric(adata.obs["n_genes_by_counts"], errors="coerce").values
        }).dropna(subset=["n_detected_genes"])

        return out.reset_index(drop=True)


    def build_single_cell_reference_detected_gene_tables(
        self,
        L_values,
        S_values,
        fixed_S=5,
        fixed_L=0.05,
        ref_path=None
    ):
        """
        Build plotting tables for single-cell reference detected genes.

        Since the reference does not depend on L or S, the same mean/std are
        repeated across the requested x-values.
        """
        df_ref = self.get_single_cell_reference_detected_genes(ref_path=ref_path)

        if df_ref.empty:
            raise ValueError("Single-cell reference detected-gene table is empty.")

        mean_val = float(df_ref["n_detected_genes"].mean())
        std_val = float(df_ref["n_detected_genes"].std(ddof=0))
        n_cells = int(df_ref.shape[0])

        rows_L = []
        for L in sorted([float(x) for x in L_values]):
            rows_L.append({
                "L": float(L),
                "S": float(fixed_S),
                "mean_value": mean_val,
                "std_value": std_val,
                "n_cells": n_cells,
            })

        rows_S = []
        for S in sorted([float(x) for x in S_values]):
            rows_S.append({
                "L": float(fixed_L),
                "S": float(S),
                "mean_value": mean_val,
                "std_value": std_val,
                "n_cells": n_cells,
            })

        df_L = pd.DataFrame(rows_L)
        df_S = pd.DataFrame(rows_S)

        return df_L, df_S, df_ref


    def plot_single_series_detected_genes_vs_L(
        self,
        df_plot,
        fixed_S,
        out_svg=None,
        title=None
    ):
        """
        Plot single-cell reference mean detected genes vs L at fixed S.
        """
        if df_plot.empty:
            print("[plot] Empty dataframe, skipping.")
            return

        sub = df_plot.sort_values("L")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            sub["L"].values,
            sub["mean_value"].values,
            yerr=sub["std_value"].values,
            marker="o",
            linewidth=1.8,
            capsize=3
        )

        ax.set_xlabel("L", fontsize=12)
        ax.set_ylabel("Mean detected genes / single cell", fontsize=12)

        if title is None:
            title = f"Single-cell Reference Detected Genes vs L (S={fixed_S})"
        ax.set_title(title, fontsize=14)

        ax.set_xticks(sub["L"].values)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save] wrote", out_svg)

        plt.show()


    def plot_single_series_detected_genes_vs_S(
        self,
        df_plot,
        fixed_L,
        out_svg=None,
        title=None
    ):
        """
        Plot single-cell reference mean detected genes vs S at fixed L.
        """
        if df_plot.empty:
            print("[plot] Empty dataframe, skipping.")
            return

        sub = df_plot.sort_values("S")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            sub["S"].values,
            sub["mean_value"].values,
            yerr=sub["std_value"].values,
            marker="o",
            linewidth=1.8,
            capsize=3
        )

        ax.set_xlabel("S", fontsize=12)
        ax.set_ylabel("Mean detected genes / single cell", fontsize=12)

        if title is None:
            title = f"Single-cell Reference Detected Genes vs S (L={fixed_L})"
        ax.set_title(title, fontsize=14)

        ax.set_xticks(sub["S"].values)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if out_svg is not None:
            fig.savefig(out_svg)
            print("[save] wrote", out_svg)

        plt.show()


    def run_single_cell_reference_detected_gene_lineplots(
        self,
        out_dir,
        L_values,
        S_values,
        fixed_S=5,
        fixed_L=0.05,
        ref_path=None,
        save_tables=True
    ):
        """
        Full pipeline for single-cell reference detected-gene baseline plots:
          - mean detected genes vs L at fixed S
          - mean detected genes vs S at fixed L
        """
        os.makedirs(out_dir, exist_ok=True)
        self._setup_editable_vector_fonts(font_candidates=["Helvetica"])

        df_L, df_S, df_ref = self.build_single_cell_reference_detected_gene_tables(
            L_values=L_values,
            S_values=S_values,
            fixed_S=fixed_S,
            fixed_L=fixed_L,
            ref_path=ref_path
        )

        svg_L = os.path.join(out_dir, f"singlecell_detected_genes_vs_L_S{fixed_S}.svg")
        svg_S = os.path.join(out_dir, f"singlecell_detected_genes_vs_S_L{fixed_L}.svg")
        csv_L = os.path.join(out_dir, f"singlecell_detected_genes_vs_L_S{fixed_S}.csv")
        csv_S = os.path.join(out_dir, f"singlecell_detected_genes_vs_S_L{fixed_L}.csv")
        csv_ref = os.path.join(out_dir, "singlecell_detected_genes_per_cell.csv")

        self.plot_single_series_detected_genes_vs_L(
            df_plot=df_L,
            fixed_S=fixed_S,
            out_svg=svg_L
        )

        self.plot_single_series_detected_genes_vs_S(
            df_plot=df_S,
            fixed_L=fixed_L,
            out_svg=svg_S
        )

        if save_tables:
            df_L.to_csv(csv_L, index=False)
            df_S.to_csv(csv_S, index=False)
            df_ref.to_csv(csv_ref, index=False)
            print("[save] wrote", csv_L)
            print("[save] wrote", csv_S)
            print("[save] wrote", csv_ref)

        return {
            "df_vs_L": df_L,
            "df_vs_S": df_S,
            "df_ref": df_ref,
            "svg_vs_L": svg_L,
            "svg_vs_S": svg_S,
            "csv_vs_L": csv_L,
            "csv_vs_S": csv_S,
            "csv_ref": csv_ref,
        }
        
    def load_success_df_from_summary_csv(
        self,
        results_path,
        summary_csv_name="run_summary.csv"
    ):
        """
        Load run summary CSV and keep only successful runs.
        """
        summary_csv = os.path.join(results_path, summary_csv_name)
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

        summary_df = pd.read_csv(summary_csv)
        if "status" not in summary_df.columns:
            raise ValueError("summary CSV must contain a 'status' column")

        success_df = summary_df[summary_df["status"] == "success"].copy()
        if success_df.empty:
            print("[Warning]: No successful runs found in summary CSV.")

        return success_df


    def collect_celltype_counts_from_results(
        self,
        results_path,
        summary_csv_name="run_summary.csv",
        crop_id=1,
        cell_col="assigned_cell_id",
        celltype_col=None,
        exclude_unassigned=True
    ):
        """
        Load successful runs from summary CSV, restricted to one crop_id,
        and collect reconstructed-cell cell type counts for each (L, S).

        Returns
        -------
        dict keyed by (L, S)
        """
        success_df = self.load_success_df_from_summary_csv(
            results_path=results_path,
            summary_csv_name=summary_csv_name
        )

        required = {"crop_id", "lambda", "search_radius", "output_h5ad"}
        missing = required - set(success_df.columns)
        if missing:
            raise ValueError(f"summary CSV missing required columns: {missing}")

        sub = success_df[success_df["crop_id"] == crop_id].copy()
        if sub.empty:
            print(f"[Warning]: No successful runs found for crop_id={crop_id}")
            return {}

        out = {}

        for _, row in sub.iterrows():
            L = float(row["lambda"])
            S = float(row["search_radius"])
            h5 = row["output_h5ad"]

            try:
                counts = self.get_celltype_counts_from_h5ad(
                    h5ad_path=h5,
                    cell_col=cell_col,
                    celltype_col=celltype_col,
                    exclude_unassigned=exclude_unassigned
                )
                out[(L, S)] = counts
            except Exception as e:
                print(f"[Warning]: Failed for {h5}: {e}")
                out[(L, S)] = {}

        return out
    
    
    def annotate_saved_runs_with_celltypist(
        self,
        results_path,
        annotated_dir_name="annotated_runs",
        cell_col="assigned_cell_id",
        model_path=None
    ):
        """
        Run CellTypist annotation on already-saved sweep .h5ad files.

        Parameters
        ----------
        results_path : str
            Folder containing sweep .h5ad outputs
        annotated_dir_name : str
            Output folder for annotated runs
        cell_col : str
            Column containing reconstructed cell IDs
        model_path : str
            CellTypist model path (defaults to self.model_path)
        """

        import glob
        import scanpy as sc
        import pandas as pd
        import celltypist
        import scipy.sparse as sp

        if model_path is None:
            model_path = self.model_path

        if model_path is None or not os.path.exists(model_path):
            raise ValueError("Valid CellTypist model_path is required")

        annotated_dir = os.path.join(results_path, annotated_dir_name)
        os.makedirs(annotated_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(results_path, "*.h5ad")))

        print(f"[Annotate]: Found {len(files)} runs")

        for f in files:

            print(f"[Annotate]: {os.path.basename(f)}")

            try:

                adata = sc.read_h5ad(f)

                if cell_col not in adata.obs.columns:
                    print("[Skip]: no assigned cells")
                    continue

                # remove unassigned spots
                cell_ids = pd.Series(adata.obs[cell_col]).astype("string")
                mask = cell_ids.notna()
                ad = adata[mask].copy()

                if ad.n_obs == 0:
                    print("[Skip]: no valid spots")
                    continue

                # build pseudobulk
                groups = cell_ids[mask].astype(str).values
                unique_cells, group_idx = np.unique(groups, return_inverse=True)

                X = ad.X
                if not sp.issparse(X):
                    X = sp.csr_matrix(X)

                M = sp.csr_matrix(
                    (np.ones(len(groups)), (group_idx, np.arange(len(groups)))),
                    shape=(len(unique_cells), len(groups))
                )

                X_sum = (M @ X).tocsr()

                pseudo = sc.AnnData(
                    X=X_sum,
                    obs=pd.DataFrame(index=unique_cells),
                    var=ad.var.copy()
                )

                # CellTypist preprocessing
                sc.pp.normalize_total(pseudo, target_sum=1e4)
                sc.pp.log1p(pseudo)

                preds = celltypist.annotate(
                    pseudo,
                    model=model_path,
                    majority_voting=True
                )

                pred_adata = preds.to_adata()

                ct_labels = pred_adata.obs["majority_voting"].astype(str)

                # map back to spots
                adata.obs["celltypist_predicted_labels"] = adata.obs[cell_col].map(ct_labels)

                out_file = os.path.join(
                    annotated_dir,
                    os.path.basename(f)
                )

                adata.write_h5ad(out_file)

                print("[Saved]", out_file)

            except Exception as e:
                print("[Error]:", e)

        print("[Done] Annotation complete")

        return annotated_dir