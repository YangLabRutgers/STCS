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
        stcs._dc_assignment_pseudobulk_data_path = metadata.get("_dc_assignment_pseudobulk_dFnata_path")
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
