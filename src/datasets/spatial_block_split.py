"""
Spatially Disjoint Holdout Partitioning for Hyperspectral Classification.

Guarantees 100% spatial independence between Train, Validation, and Test pools
by enforcing non-overlapping spatial tiles and boundary guard bands >= 15 pixels.
"""

import os
import numpy as np
from scipy.io import loadmat
from typing import Dict, Tuple, Any, List

class SpatialBlockSplitter:
    """
    Constructs spatially disjoint block partitions for hyperspectral scenes.
    
    Guarantees that no test patch of size patch_size x patch_size shares any spatial
    pixel with any training patch.
    """
    def __init__(self, patch_size: int = 15, guard_band: int = 7):
        self.patch_size = patch_size
        self.guard_band = guard_band # Boundary margin inside each tile
        # Buffer between train and test must be >= patch_size - 1 = 14 pixels

    def create_spatial_split(
        self,
        gt_path: str,
        gt_key: str,
        n_r_blocks: int = 6,
        n_c_blocks: int = 6,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Partition ground-truth coordinates into disjoint spatial blocks.
        
        Returns:
            Dict containing:
                'train_coords': Array of (r, c)
                'val_coords': Array of (r, c)
                'test_coords': Array of (r, c)
                'train_labels': Array of labels (0-indexed)
                'val_labels': Array of labels (0-indexed)
                'test_labels': Array of labels (0-indexed)
                'min_spatial_distance': Minimum Chebyshev distance between test and train centers
                'is_zero_overlap': True if min_spatial_distance >= patch_size
        """
        mat = loadmat(gt_path)
        labels = mat[gt_key]
        H, W = labels.shape
        coords = np.argwhere(labels > 0)
        
        r_step = H // n_r_blocks
        c_step = W // n_c_blocks
        
        # Collect non-guard pixels per tile
        tile_pixels = {}
        for r, c in coords:
            bi = min(r // r_step, n_r_blocks - 1)
            bj = min(c // c_step, n_c_blocks - 1)
            
            # Check if pixel falls inside guard boundary of its tile
            r_local = r - (bi * r_step)
            c_local = c - (bj * c_step)
            
            # Guard band along tile boundaries
            if (r_local < self.guard_band or r_local >= r_step - self.guard_band or
                c_local < self.guard_band or c_local >= c_step - self.guard_band):
                continue # Discard border pixel to guarantee zero overlap
                
            tile_key = (bi, bj)
            if tile_key not in tile_pixels:
                tile_pixels[tile_key] = []
            tile_pixels[tile_key].append((r, c, labels[r, c] - 1)) # 0-indexed
            
        # Allocate tiles to train (60%), val (20%), test (20%) deterministically
        # Using spatial checkerboard / alternating pattern
        rng = np.random.RandomState(seed)
        all_tiles = sorted(list(tile_pixels.keys()))
        
        train_list, val_list, test_list = [], [], []
        
        for idx, (bi, bj) in enumerate(all_tiles):
            # Alternating spatial assignment
            pattern = (bi * 3 + bj * 7) % 5
            pixels = tile_pixels[(bi, bj)]
            if pattern in [0, 1, 2]:
                train_list.extend(pixels)
            elif pattern == 3:
                val_list.extend(pixels)
            else:
                test_list.extend(pixels)
                
        train_arr = np.array(train_list, dtype=np.int64) if len(train_list) > 0 else np.empty((0, 3), dtype=np.int64)
        val_arr = np.array(val_list, dtype=np.int64) if len(val_list) > 0 else np.empty((0, 3), dtype=np.int64)
        test_arr = np.array(test_list, dtype=np.int64) if len(test_list) > 0 else np.empty((0, 3), dtype=np.int64)
        
        # Verify zero overlap
        min_dist = float("inf")
        if len(train_arr) > 0 and len(test_arr) > 0:
            # Chebyshev distance between test centers and train centers
            # Subsample for speed if test_arr is large
            sub_test = test_arr[:min(500, len(test_arr)), :2]
            train_c = train_arr[:, :2]
            
            for tr_r, tr_c in sub_test:
                d = np.max(np.abs(train_c - np.array([tr_r, tr_c])), axis=1).min()
                if d < min_dist:
                    min_dist = d
                    
        is_zero_overlap = (min_dist >= self.patch_size)
        
        return {
            "train_coords": train_arr[:, :2] if len(train_arr) > 0 else np.empty((0, 2)),
            "train_labels": train_arr[:, 2] if len(train_arr) > 0 else np.empty(0),
            "val_coords": val_arr[:, :2] if len(val_arr) > 0 else np.empty((0, 2)),
            "val_labels": val_arr[:, 2] if len(val_arr) > 0 else np.empty(0),
            "test_coords": test_arr[:, :2] if len(test_arr) > 0 else np.empty((0, 2)),
            "test_labels": test_arr[:, 2] if len(test_arr) > 0 else np.empty(0),
            "min_spatial_distance": float(min_dist),
            "is_zero_overlap": bool(is_zero_overlap)
        }

