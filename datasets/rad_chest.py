import torch
import torch.nn as nn
import torch.utils.data
import os
import numpy as np
import logging
from tqdm import tqdm
from datasets import ColoredFormatter
import nibabel as nib

# Global constants for LIDC-style preprocessing
LOW_HU, HIGH_HU = -1000.0, 1000.0
TARGET_SHAPE = (256, 256, 256)
CACHE_VER = "v3_lidcstyle_fixedwin_cc"

class RADChestCTDataset(torch.utils.data.Dataset):
    def __init__(self, directory, mode='train', img_size=256,
                 cache_dir=None, preprocess="cc", 
                 val_ratio=0.1, test_ratio=0.1, train_ratio=0.8,
                 normalize=None,
                 seed=42, 
                 augment=False,
                 train_size=1000, 
                 val_size=200, 
                 test_size=200, 
                 rank=0):
        """
        RAD-ChestCT Dataset Loader with LIDC-IDRI-style preprocessing.
        
        Args:
            directory: Path to data files
                - For 'train'/'val'/'test': Path to .npz files (each must have 'ct' key in HU)  
                - For 'fake': Path to directory tree containing .nii.gz files
                - For 'real': Path to directory containing cc_cache/b6/ subdirectory
            mode: Split type ('train' | 'val' | 'test' | 'fake' | 'real')
                - 'train'/'val'/'test': Standard splits from cached .npz files
                - 'fake': Load .nii.gz files recursively from directory tree
                - 'real': Load concatenated val+test splits from cc_cache/b6/
            img_size: Target image size (256, 128, or 64)
            cache_dir: Custom cache directory path
            preprocess: Preprocessing type ('cc' or 'rs')
            val_ratio, test_ratio, train_ratio: Split ratios
            seed: Random seed for reproducible splits
            train_size, val_size, test_size: Maximum samples per split
            rank: Process rank for distributed training
        """
        super().__init__()
        assert preprocess in ("cc", "rs"), "preprocess must be 'cc' or 'rs'"
        assert img_size in (256, 128, 64), "img_size must be 256, 128, or 64"
        assert mode in ("train", "val", "test", "fake", "real"), "mode must be 'train', 'val', 'test', 'fake', or 'real'"
        
        # Setup proper logging without duplicates
        self.rank = rank
        if rank == 0:
            # Create a unique logger name to avoid conflicts
            logger_name = f'RADChestCT'
            self.logger = logging.getLogger(logger_name)
            
            # Prevent propagation to root logger (this stops duplicate output)
            self.logger.propagate = False
            
            # Only add handler if none exists
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                # Use colored formatter
                formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        else:
            # For non-rank-0, create a silent logger
            self.logger = logging.getLogger('silent')
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
        
        self.directory = os.path.expanduser(directory)
        self.mode = mode
        self.img_size = img_size
        self.augment = augment
        self.preprocess = preprocess
        self.rank = rank
        self.normalize = normalize
        self.train_size, self.val_size, self.test_size = train_size, val_size, test_size

        # Cache setup
        self.cache_dir = cache_dir or os.path.join(self.directory, f'splits_{CACHE_VER}')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.train_split_path = os.path.join(self.cache_dir, f'train_split_{img_size}.npz')
        self.val_split_path = os.path.join(self.cache_dir, f'val_split_{img_size}.npz')
        self.test_split_path = os.path.join(self.cache_dir, f'test_split_{img_size}.npz')

        if rank == 0:
            self.logger.info(f"Initializing RAD-ChestCT dataset in {mode} mode")
            self.logger.info(f"Data directory: {directory}")
            self.logger.info(f"Cache directory: {self.cache_dir}")
            self.logger.info(f"Image size: {img_size}x{img_size}x{img_size}")
            self.logger.info(f"Preprocess: {preprocess}")
            self.logger.info(f"Train size: {train_size}, Val size: {val_size}")
            self.logger.info(f"Using custom normalization: {normalize is not None}")
            self.logger.info(f"Using augmentation: {augment}")

        # Handle different modes
        if mode == "fake":
            if rank == 0:
                self.logger.info(f"📁 Loading fake data from {self.directory}...")
            all_files = []
            for root, dirs, files in tqdm(os.walk(self.directory), desc="Loading fake data", disable=(self.rank != 0)):
                nii_files = [f for f in files if f.endswith('.nii.gz')]
                for f in nii_files:
                    all_files.append(os.path.join(root, f))
            self.file_paths = all_files
            if rank == 0:
                self.logger.info(f"✅ Found {len(self.file_paths)} fake data files")
            # For fake mode, we'll load data on-demand in __getitem__
            self.dataitems = [(None, fp) for fp in self.file_paths]
            
        elif mode == "real":
            if rank == 0:
                self.logger.info(f"📁 Loading real data from {self.directory}...")
            train_path = self.train_split_path
            val_path = self.val_split_path
            test_path = self.test_split_path
            
            # Load all splits
            train_data = np.load(train_path)
            val_data = np.load(val_path)
            test_data = np.load(test_path)
            
            # Concatenate volumes and filenames from all splits (excluding train as commented)
            all_volumes = np.concatenate([train_data['volumes'], val_data['volumes'], test_data['volumes']
                                         ], axis=0)
            
            all_filenames = np.concatenate([train_data['filenames'], val_data['filenames'], test_data['filenames']
                                         ], axis=0)
            
            # Create items list for real mode
            items = list(zip(all_volumes, all_filenames))
            self.dataitems = items
            if rank == 0:
                self.logger.info(f"✅ Loaded {len(self.dataitems)} real data samples")
        else:
            # Handle standard modes: train, val, test
            # Check if cache exists
            have_cache = all(os.path.exists(p) for p in
                             [self.train_split_path, 
                              self.val_split_path, 
                              self.test_split_path])
            
            # Special handling for img_size=256 without cache - load directly from .npz files
            if not have_cache and img_size == 256:
                if rank == 0:
                    self.logger.info("No cache found for img_size=256, loading directly from .npz files...")
                self._setup_direct_loading(mode, seed)
            elif have_cache:
                if rank == 0:
                    self.logger.info("Found cached splits")
                self.dataitems = np.load(
                    self.train_split_path if mode == 'train' else
                    self.val_split_path   if mode == 'val'   else
                    self.test_split_path
                )
                # Finalize items
                items = list(zip(self.dataitems["volumes"], self.dataitems["filenames"]))
                if mode == 'train':
                    rng = np.random.default_rng(seed)
                    rng.shuffle(items)
                    items = items[:self.train_size]
                self.dataitems = items
            else:
                if rank == 0:
                    self.logger.warning("No cached splits found. Creating new splits...")
                self.dataitems = self.create_train_val_test_splits(seed, mode)
                # Finalize items
                items = list(zip(self.dataitems["volumes"], self.dataitems["filenames"]))
                if mode == 'train':
                    rng = np.random.default_rng(seed)
                    rng.shuffle(items)
                    items = items[:self.train_size]
                self.dataitems = items
        
        # Show data statistics for the first sample (skip for on-demand loading modes)
        if rank == 0 and len(self.dataitems) > 0 and mode not in ["fake"] and not (hasattr(self, '_direct_loading') and self._direct_loading):
            first_sample = self.dataitems[0][0]  # Get first volume
            if first_sample is not None:  # Additional safety check
                self.logger.info("Data Statistics (first sample):")
                self.logger.info("--- After LIDC preprocessing [0,1] range ---")
                self.logger.info(f"Min:  {np.min(first_sample):.6f}")
                self.logger.info(f"Max:  {np.max(first_sample):.6f}")
                self.logger.info(f"Mean: {np.mean(first_sample):.6f}")
                self.logger.info(f"Std:  {np.std(first_sample):.6f}")
                
                # Check data range
                if np.min(first_sample) >= 0 and np.max(first_sample) <= 1.01:
                    self.logger.info("✅ Data is in expected [0,1] range after preprocessing")
                else:
                    self.logger.warning("⚠️ Data outside [0,1] range - check preprocessing")
                
                # Show normalized data if normalization is applied
                if self.normalize:
                    normalized_sample = self.normalize(first_sample)
                    self.logger.info("--- After custom normalization (2*x-1) to [-1,1] range ---")
                    self.logger.info(f"Min:  {np.min(normalized_sample):.6f}")
                    self.logger.info(f"Max:  {np.max(normalized_sample):.6f}")
                    self.logger.info(f"Mean: {np.mean(normalized_sample):.6f}")
                    self.logger.info(f"Std:  {np.std(normalized_sample):.6f}")
                    
                    # Check normalized range
                    if np.min(normalized_sample) >= -1.01 and np.max(normalized_sample) <= 1.01:
                        self.logger.info("✅ Data is in expected [-1,1] range after normalization")
                    else:
                        self.logger.warning("⚠️ Data outside [-1,1] range after normalization")
        
        if rank == 0:
            self.logger.info(f"📊 Final {mode} size: {len(self.dataitems)}")

    def _setup_direct_loading(self, mode: str, seed: int) -> None:
        """
        Setup direct loading from .npz files for img_size=256 without cache.
        This avoids creating cache files and loads data on-demand during training.
        """
        # Get all .npz files in directory
        all_files = [f for f in os.listdir(self.directory) if f.endswith('.npz')]
        if len(all_files) == 0:
            raise ValueError(f"No .npz files found in {self.directory}")
        
        if self.rank == 0:
            self.logger.info(f"Found {len(all_files)} .npz files for direct loading")
        
        # Create reproducible splits
        rng = np.random.default_rng(seed)
        rng.shuffle(all_files)
        
        # Calculate split indices
        total_files = len(all_files)
        train_end = int(total_files * 0.8)
        val_end = int(total_files * 0.9)
        
        # Split files
        if mode == 'train':
            files = all_files[:train_end]
            # Limit to train_size if specified
            if self.train_size and len(files) > self.train_size:
                files = files[:self.train_size]
        elif mode == 'val':
            files = all_files[train_end:val_end]
            # Limit to val_size if specified
            if self.val_size and len(files) > self.val_size:
                files = files[:self.val_size]
        elif mode == 'test':
            files = all_files[val_end:]
            # Limit to test_size if specified
            if self.test_size and len(files) > self.test_size:
                files = files[:self.test_size]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if self.rank == 0:
            self.logger.info(f"Selected {len(files)} files for {mode} mode")
        
        # Store file paths for on-demand loading
        self.dataitems = [(None, os.path.join(self.directory, f)) for f in files]
        
        # Set flag to indicate direct loading mode
        self._direct_loading = True

    def create_train_val_test_splits(self, seed, mode):
        all_filenames = np.array([f for f in os.listdir(self.directory) if f.endswith('.npz')])
        if len(all_filenames) == 0:
            raise ValueError(f"No .npz files found in {self.directory}")
        rng = np.random.default_rng(seed)
        rng.shuffle(all_filenames)

        train_files, val_files, test_files = [], [], []
        cur = 0

        def need_counts():
            return (len(train_files) < self.train_size or
                    len(val_files) < self.val_size or
                    len(test_files) < self.test_size)

        while need_counts() and cur < len(all_filenames):
            batch = all_filenames[cur:cur + 100]
            cur += 100
            vols, names = self._create_dataitems_cc(batch)
            pairs = list(zip(vols, names))
            if not pairs: 
                continue

            def take(dst, n):
                k = min(n, len(pairs))
                dst.extend(pairs[:k])
                return pairs[k:]

            if len(train_files) < self.train_size:
                pairs = take(train_files, self.train_size - len(train_files))
            if len(val_files) < self.val_size:
                pairs = take(val_files, self.val_size - len(val_files))
            if len(test_files) < self.test_size:
                pairs = take(test_files, self.test_size - len(test_files))

        # Save splits
        if train_files:
            tv, tn = zip(*train_files)
            np.savez_compressed(self.train_split_path, volumes=np.stack(tv), filenames=np.array(tn))
        if val_files:
            vv, vn = zip(*val_files)
            np.savez_compressed(self.val_split_path, volumes=np.stack(vv), filenames=np.array(vn))
        if test_files:
            sv, sn = zip(*test_files)
            np.savez_compressed(self.test_split_path, volumes=np.stack(sv), filenames=np.array(sn))

        # Return split
        if self.mode == 'train': return np.load(self.train_split_path)
        if self.mode == 'val':   return np.load(self.val_split_path)
        if self.mode == 'test':  return np.load(self.test_split_path)
        raise ValueError(f"Invalid mode: {self.mode}")

    def downsample(self, image: torch.Tensor) -> torch.Tensor:
        if self.img_size == 128:
            return nn.AvgPool3d(2, 2)(image)
        if self.img_size == 64:
            return nn.AvgPool3d(2, 2)(nn.AvgPool3d(2, 2)(image))
        return image

    def _preprocess_like_lidc(self, image: np.ndarray) -> np.ndarray:
        """
        Apply exact LIDC-IDRI preprocessing: fixed HU clip, center crop 256³,
        quantile clip, normalize [0,1].
        """
        # 1) Clip HU to [-1000, 1000]
        image = np.clip(image, LOW_HU, HIGH_HU)

        # 2) Center crop directly to (256,256,256)
        d, h, w = image.shape
        d_start = (d - TARGET_SHAPE[0]) // 2
        h_start = (h - TARGET_SHAPE[1]) // 2
        w_start = (w - TARGET_SHAPE[2]) // 2
        cropped = image[
            d_start:d_start + TARGET_SHAPE[0],
            h_start:h_start + TARGET_SHAPE[1],
            w_start:w_start + TARGET_SHAPE[2]
        ]
        if cropped.shape != TARGET_SHAPE:
            raise ValueError(f"Unexpected shape {cropped.shape}, expected {TARGET_SHAPE}")

        # 3) Clip upper 0.999 quantile
        upper_clip = np.quantile(cropped, 0.995)
        cropped = np.clip(cropped, LOW_HU, upper_clip)

        # 4) Normalize to [0,1]
        normalized = (cropped - np.min(cropped)) / (np.max(cropped) - np.min(cropped))

        return normalized.astype(np.float32)

    def _create_dataitems_cc(self, files, min_size=100):
        """Process files using LIDC-style preprocessing."""
        volumes, filenames, skipped = [], [], []

        for file in tqdm(files, desc="Processing RAD-ChestCT", ncols=100, disable=(self.rank != 0)):
            try:
                path = os.path.join(self.directory, file)
                image = np.load(path)['ct'].astype(np.float32)

                if any(s < min_size for s in image.shape):
                    if self.rank == 0:
                        print(f"Skipping {file} because small dimensions")
                    skipped.append((file, "Small dimensions"))
                    continue
                
                # Preprocess and convert to tensor
                image = self._preprocess_like_lidc(image)
                image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
                image = self.downsample(image)

                # Store as numpy for efficient batching
                volumes.append(image.numpy())
                filenames.append(file)

            except Exception as e:
                skipped.append((file, f"Error: {e}"))
                continue

        # Log skipped files (only on rank 0)
        if skipped and self.rank == 0:
            log_file = os.path.join(os.path.dirname(self.directory), f'skipped_{CACHE_VER}.txt')
            with open(log_file, 'a') as f:
                for fp, rsn in skipped:
                    f.write(f"{fp}: {rsn}\n")

        if volumes:
            return np.stack(volumes, axis=0), np.array(filenames, dtype="U")
        else:
            return np.array([]), np.array([], dtype="U")

    def __len__(self): 
        return len(self.dataitems)

    def _load_nifti_file(self, file_path):
        """Load and preprocess a NIfTI file for fake mode."""
        # Load NIfTI file
        nii_img = nib.load(file_path)
        image = nii_img.get_fdata().astype(np.float32)
        
        # Convert to tensor and downsample
        image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
        image = self.downsample(image)
        
        return image.numpy()

    def _load_npz_file(self, file_path):
        """Load and preprocess a .npz file for direct loading mode."""
        try:
            # Load .npz file
            data = np.load(file_path)
            image = data['ct'].astype(np.float32)
            
            # Apply LIDC-style preprocessing
            image = self._preprocess_like_lidc(image)
            
            # Convert to tensor and downsample
            image = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dim
            image = self.downsample(image)
            
            return image.numpy()
            
        except Exception as e:
            if self.rank == 0:
                self.logger.warning(f"Failed to load {file_path}: {e}")
            # Return a dummy tensor with correct shape
            dummy_shape = (1, self.img_size, self.img_size, self.img_size)
            return np.zeros(dummy_shape, dtype=np.float32)

    def __getitem__(self, idx):
        """Get a single data item."""
        image, fname = self.dataitems[idx]
        
        # Handle on-demand loading for different modes
        if image is None:
            if self.mode == "fake":
                # Load .nii.gz files on-demand
                image = self._load_nifti_file(fname)
            elif hasattr(self, '_direct_loading') and self._direct_loading:
                # Load .npz files on-demand for direct loading mode
                image = self._load_npz_file(fname)
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()  # Shape: [C,D,H,W]
        
        # Apply normalization if specified
        if self.normalize:
            image = self.normalize(image)
            
        # Apply augmentation if specified
        if self.augment:
            prob = np.random.rand()
            if prob > 0.5:
                image = torch.flip(image, dims=[-1])
                
        return {"image": image}
