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

            # Assume array_row and array_col already reflect pixel coordinates
            if "array_row" in self.adata.obs.columns and "array_col" in self.adata.obs.columns:
                self.adata.obsm["spatial"] = self.adata.obs[["array_row", "array_col"]].to_numpy()
                print("[Log]: Found pixel-aligned array_row/array_col; assigned to obsm['spatial']")
            else:
                raise ValueError("Expected columns 'array_row' and 'array_col' in .obs for pixel coordinates")

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
        common_barcodes = adata_with_labels.obs.index.intersection(barcode_data.obs.index)
        adata_with_labels.obs['labels_he'] = np.nan
        
        for barcode in common_barcodes:
            adata_with_labels.obs.loc[barcode, 'labels_he'] = barcode_data.obs.loc[barcode, 'labels_he']
        
        pseudobulk_data = self._create_pseudobulk(
            adata=adata_with_labels, 
            mode = mode, 
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
        
        try:
            grouped = celldata.obs['cell']

            if mode == 'sum':
                pseudobulk = pd.DataFrame(
                    celldata.X.toarray() if hasattr(celldata.X, "toarray") else celldata.X,
                    index=celldata.obs_names
                ).groupby(grouped).sum()
            elif mode == 'mean':
                pseudobulk = pd.DataFrame(
                    celldata.X.toarray() if hasattr(celldata.X, "toarray") else celldata.X,
                    index=celldata.obs_names
                ).groupby(grouped).mean()
            else:
                print(f"[Error]: Wrong mode name: {mode}")

            pseudobulk.columns = celldata.var_names

            adata_pseudo = ad.AnnData(
                X=pseudobulk.values,
                obs=pd.DataFrame(index=pseudobulk.index),
                var=pd.DataFrame(index=pseudobulk.columns)
            )
            print(f"[Log]: Successfully created pseudobulk with {adata_pseudo.n_obs} cells and {adata_pseudo.n_vars} genes")
            
        except Exception as e:
            print(f"[Error]: Failed to create pseudobulk: {e}")
            raise

        return adata_pseudo

    # ========== PCA PROCESSING AND ASSIGNMENT ==========
    
    def run_assignment(self, output_path, min_cells=min_cells, L = L, target_sum=target_sum, top_genes=n_top_genes, search_radius=default_search_radius, use_sc_ref=True, normalize_distances=True, feature_name=False):
        """
        Integrated assignment pipeline: candidate search + PCA processing + gene space assignment
        """
        
        print("[Log]: Starting assignment pipeline based on candidate search + PCA processing + gene space assignment ")
        
        # Check data
        if self._dc_pseudobulk_data_path is None:
            print("[Error]: No pseudobulk data available. Run stardist pseudobulk creation first.")
            return self
        
        if self._barcode_data_path is None:
            print("[Error]: No stardist barcode data available. Run stardist pipeline first.")
            return self
        
        # Create output directory
        assignment_path = os.path.join(output_path, 'integrated_assignment')
        os.makedirs(assignment_path, exist_ok=True)
        
        # Create candidate mappings (integrated from stardist_postprocessing)
        print("[Log]: Creating barcode to candidate cells mapping")
        barcode_candidates = self._create_barcode_candidates_mapping(assignment_path, search_radius)
        
        # Load and process data for PCA
        print("[Log]: Loading pseudobulk data")
        pseudobulk_data = self.load_stardist_pseudobulk_data()
        if pseudobulk_data is None:
            print("[Error]: Failed to load pseudobulk data")
            return self
        
        # Process reference data and compute PCA
        if use_sc_ref and self.sc_ref is not None:
            print("[Log]: Processing with single-cell reference")
            processed_data = self._process_with_sc_reference(pseudobulk_data, assignment_path, feature_name)
        else:
            print("[Log]: Processing with spatial reference")
            processed_data = self._process_with_spatial_reference(pseudobulk_data, assignment_path)
        
        # Create cell-to-barcodes mapping for spatial distance calculation
        print("[Log]: Creating cell-to-barcodes mapping for spatial distances")
        cell_to_barcodes = self._create_cell_to_barcodes_mapping_from_stardist(assignment_path)
        
        # Perform integrated gene space assignment
        print("[Log]: Performing gene space assignment")
        final_assignments = self._perform_gene_space_assignment(
            barcode_candidates, cell_to_barcodes, processed_data, assignment_path, normalize_distances, L
        )
        
        # Update STCS with final results
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

    def _create_barcode_candidates_mapping(self, output_path, search_radius):
        """Create barcode to candidate cells mapping"""
        
        print("[Log]: Creating barcode candidates mapping with grid search")
        
        # Load stardist barcode data
        barcode_data = self.load_stardist_barcode_data()
        if barcode_data is None:
            raise ValueError("Failed to load stardist barcode data")
        
        # Get detected cells and their coordinates
        detected = barcode_data[barcode_data.obs['labels_he'] != 0]
        detected_coords = detected.obs[['array_row', 'array_col']].values
        coord_1 = detected.obs['array_row'].min()
        coord_2 = detected.obs['array_row'].max()
        coord_3 = detected.obs['array_col'].min()
        coord_4 = detected.obs['array_col'].max()

        print(f'Stardist - Row: {coord_1} - {coord_2}')
        print(f'Stardist - Col: {coord_3} - {coord_4}')

        detected_cells = detected.obs['labels_he'].values
        detected_barcodes = detected.obs.index.values
        
        # Create coordinate to cell mapping for fast lookup
        coord_to_cell = {}
        for i, barcode in enumerate(detected_barcodes):
            coord = tuple(detected_coords[i])
            coord_to_cell[coord] = detected_cells[i]
        
        barcode_to_own_label = {}
        for barcode in detected_barcodes:
            label = detected.obs.loc[barcode, 'labels_he']
            barcode_to_own_label[barcode] = int(label)
        
        print(f"[Log]: Found {len(barcode_to_own_label)} barcodes with their own labels")

        # Get all barcode info from spatial data
        adata = self.adata.copy()

        all_coords = adata.obs[['array_row', 'array_col']].values
        coord_1 = adata.obs['array_row'].min()
        coord_2 = adata.obs['array_row'].max()
        coord_3 = adata.obs['array_col'].min()
        coord_4 = adata.obs['array_col'].max()
        print(f'Spatial - Row: {coord_1} - {coord_2}')
        print(f'Spatial - Col: {coord_3} - {coord_4}')
            
        all_barcodes = adata.obs.index.values
        
        print(f"[Log]: Processing grid search (radius={search_radius})")
        
        # Grid searching (equivalent to original 'bar' creation)
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

    def _create_cell_to_barcodes_mapping_from_stardist(self, output_path):
        """Create cell-to-barcodes mapping from stardist results"""
        
        print("[Log]: Creating cell-to-barcodes mapping from stardist results")
        
        # Load stardist barcode data
        barcode_data = self.load_stardist_barcode_data()
        detected = barcode_data[barcode_data.obs['labels_he'] != 0]
        
        # Create true cell to barcodes mapping (based on stardist detection)
        true_cell_to_barcodes = {}
        for barcode_name in detected.obs_names:
            cell_id = detected.obs.loc[barcode_name, 'labels_he']
            cell_id_str = str(int(cell_id))
            
            if cell_id_str not in true_cell_to_barcodes:
                true_cell_to_barcodes[cell_id_str] = []
            true_cell_to_barcodes[cell_id_str].append(barcode_name)
        
        # Create coordinate information (compatible with original c_b format)
        cell_barcodes_info = {}
        
        for cell_id, barcodes in true_cell_to_barcodes.items():
            # Create coordinate string like original format: '[x,y]-[x,y]-...'
            coord_strings = []
            coordinates = []
            
            for barcode in barcodes:
                try:
                    x = int(barcode.split('_')[2])
                    y = int(barcode.split('_')[3][:-2])
                    coord_strings.append(f'[{x},{y}]')
                    coordinates.append([x, y])
                except (IndexError, ValueError):
                    continue
            
            cell_barcodes_info[cell_id] = {
                'barcodes': '-'.join(coord_strings),  # Original c_b format
                'barcode_list': barcodes,
                'coordinates': coordinates
            }
        
        # Save for reference
        cell_bar_df = pd.DataFrame.from_dict(
            {k: v['barcodes'] for k, v in cell_barcodes_info.items()}, 
            orient='index', 
            columns=['barcodes']
        )
        cell_bar_df.index.name = 'cell_id'
        cell_bar_df.to_csv(os.path.join(output_path, 'cell_bar_mapping.csv'))
        
        print(f"[Log]: Created cell mapping for {len(cell_barcodes_info)} cells")
        
        return cell_barcodes_info

    def _perform_gene_space_assignment(self, bar, c_b, processed_data, output_path, normalize_distances, L):
        """Perform gene space assignment following the EXACT original algorithm"""
        
        adata = processed_data['spatial_adata']
        pseudobulk_data = processed_data['pseudobulk_data']  # pseudobulk data
        gene_cos_sim = processed_data['gene_cos_sim']
        
        if normalize_distances:
            return self._compute_assignment_unified(adata, pseudobulk_data, L, gene_cos_sim, bar, c_b, output_path, normalize=True)
        else:
            return self._compute_assignment_unified(adata, pseudobulk_data, L, gene_cos_sim, bar, c_b, output_path, normalize=False)

    def _compute_assignment_unified(self, adata, pseudobulk_data, L, gene_cos_sim, bar, c_b, output_path, normalize=True):
        """Unified assignment computation with normalization parameter"""
        
        if normalize:
            print("[Log]: Computing distances with normalization")
            # First pass: collect all distances for normalization
            one = []  # Store [transcriptomic_distance, spatial_distance] pairs
            valid_pairs = []
            
            for i in tqdm(bar.keys(), desc="Computing distances"):
                if len(bar[i]) > 1:
                    loc = [int(i.split('_')[2]), int(i.split('_')[3][:-2])]
                    
                    for k in bar[i]:
                        cell_id_int = int(float(k))
                        cell_id_str_float = f"{cell_id_int}.0"
                        cell_id_str_int = str(cell_id_int)
                        if cell_id_str_float not in pseudobulk_data.obs_names:
                            continue
                        if cell_id_str_int not in c_b:
                            continue
                            
                        try:
                            barcode_pca = list(adata[i].obsm['X_pca'][0])
                            cell_pca = list(pseudobulk_data[cell_id_str_float].obsm['X_pca'][0])
                            
                            mtx = pd.DataFrame(barcode_pca) - pd.DataFrame(cell_pca)
                            t = mtx.T @ gene_cos_sim @ mtx
                            
                            loc_dis = []
                            if cell_id_str_int in c_b:
                                for q in c_b[cell_id_str_int]['barcodes'].split('-'):
                                    try:
                                        coord_str = q[1:-1]
                                        x, y = coord_str.split(',')
                                        cell_coord = np.array([int(x), int(y)])
                                        barcode_coord = np.array(loc)
                                        distance = np.linalg.norm(cell_coord - barcode_coord)
                                        loc_dis.append(distance)
                                    except:
                                        continue
                            else:
                                print(f"[DEBUG]: Cell {cell_id_str_int} not in c_b")
                            
                            if loc_dis:
                                nearest_index = np.argmin(loc_dis)
                                one.append([float(t.iloc[0,0]), loc_dis[nearest_index]])
                                valid_pairs.append([i, k])

                        except Exception as e:
                            print(f"[Warning]: Error processing barcode {i}, cell {k}: {e}")
                            continue
                            
            if not one:
                print("[Error]: No valid distance calculations found")
                return {}
            
            # Normalize distances and create unified output
            one_df = pd.DataFrame(one, columns=['transcriptomic_raw', 'spatial_raw'])
            log_transcriptomic = np.log(one_df['transcriptomic_raw'])
            normalized_transcriptomic = (log_transcriptomic - log_transcriptomic.min()) / (log_transcriptomic.max() - log_transcriptomic.min())
            weighted_spatial = L * one_df['spatial_raw']
            all_dis = normalized_transcriptomic + weighted_spatial
            
            # Create unified distance DataFrame
            distance_df = pd.DataFrame({
                'transcriptomic_raw': one_df['transcriptomic_raw'],
                'spatial_raw': one_df['spatial_raw'],
                'transcriptomic_log': log_transcriptomic,
                'transcriptomic_normalized': normalized_transcriptomic,
                'spatial_weighted': weighted_spatial,
                'combined_distance': all_dis
            })
            distance_df.to_csv(os.path.join(output_path, 'distances.csv'), index=False)
            
            distance_lookup = {}
            for idx, (barcode, cell) in enumerate(valid_pairs):
                if barcode not in distance_lookup:
                    distance_lookup[barcode] = {}
                distance_lookup[barcode][cell] = all_dis.iloc[idx]
            
            # Second pass: assignment
            assignments = {}
            
            for i in tqdm(bar.keys(), desc="Final assignment"):
                if len(bar[i]) == 0:
                    assignments[i] = None
                elif len(bar[i]) == 1:
                    assignments[i] = str(int(bar[i][0]))
                else:
                    candidate_distances = []
                    for k in bar[i]:
                       if i in distance_lookup and k in distance_lookup[i]:
                           candidate_distances.append((k, distance_lookup[i][k]))
                    if len(candidate_distances) == 0:
                       assignments[i] = None
                    else:
                       # select candidate with minimum distance
                        best_candidate, _ = min(candidate_distances, key=lambda x: x[1])
                        assignments[i] = str(int(best_candidate))
                        
        else:
            print("[Log]: Computing direct assignment")
            assignments = {}
            all_distances = []  # Store all distance components
            
            for i in tqdm(bar.keys(), desc="Direct assignment"):
                if len(bar[i]) == 0:
                   assignments[i] = None
                elif len(bar[i]) == 1:
                    assignments[i] = str(int(bar[i][0]))
                else:
                   loc = [int(i.split('_')[2]), int(i.split('_')[3][:-2])]
                   candidate_distances = []
                   
                   for k in bar[i]:
                       cell_id_int = int(float(k))
                       cell_id_str_float = f"{cell_id_int}.0"
                       cell_id_str_int = str(cell_id_int)
                       
                       if cell_id_str_float not in pseudobulk_data.obs_names:
                           candidate_distances.append((k, float('inf')))
                           continue
                       
                       try:
                           if sp.issparse(adata[i].obsm['X_pca']):
                               barcode_pca = adata[i].obsm['X_pca'].toarray().tolist()[0]
                           else:
                               barcode_pca = list(adata[i].obsm['X_pca'][0])
                           
                           cell_pca = list(pseudobulk_data[cell_id_str_float].obsm['X_pca'][0])
                           mtx = pd.DataFrame(barcode_pca) - pd.DataFrame(cell_pca)
                           transcriptomic_dist = float(mtx.T @ gene_cos_sim @ mtx)
                           
                           loc_dis = []
                           if cell_id_str_int in c_b:
                               for q in c_b[cell_id_str_int]['barcodes'].split('-'):
                                   try:
                                       coord_str = q[1:-1]
                                       x, y = coord_str.split(',')
                                       distance = np.linalg.norm(np.array([int(x), int(y)]) - np.array(loc))
                                       loc_dis.append(distance)
                                   except:
                                       continue
                           
                           if loc_dis:
                               nearest_index = np.argmin(loc_dis)
                               spatial_dist = loc_dis[nearest_index]
                               weighted_spatial = L * spatial_dist
                               combined_distance = transcriptomic_dist + weighted_spatial
                               candidate_distances.append((k, combined_distance))
                               all_distances.append([transcriptomic_dist, spatial_dist, weighted_spatial, combined_distance])
                           else:
                               candidate_distances.append((k, float('inf')))
                               
                       except Exception as e:
                           print(f"[Warning]: Error processing barcode {i}, cell {k}: {e}")
                           candidate_distances.append((k, float('inf')))
                   
                   # select candidate with minimum distance
                   if candidate_distances:
                       best_candidate, best_distance = min(candidate_distances, key=lambda x: x[1])
                       if best_distance != float('inf'):
                           assignments[i] = str(int(best_candidate))
                       else:
                           assignments[i] = None
                   else:
                       assignments[i] = None
            
            # Create unified distance DataFrame
            if all_distances:
                distance_df = pd.DataFrame(all_distances, columns=['transcriptomic_raw', 'spatial_raw', 'spatial_weighted', 'combined_distance'])
                distance_df.to_csv(os.path.join(output_path, 'distances.csv'), index=False)
        
        # Save assignments
        pd.DataFrame.from_dict(assignments, orient='index', columns=['assigned_cell']).to_csv(
            os.path.join(output_path, 'assignments.csv'))
        
        print(f"[Log]: Assigned {len([v for v in assignments.values() if v is not None])} barcodes")
        
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