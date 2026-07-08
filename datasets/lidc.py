import torch
import torch.nn.functional as F
import torch.utils.data
import os
import nibabel
import numpy as np
import logging
import tqdm
import random
import cv2
import h5py
import scipy.ndimage as ndimage

# Handle imports for both direct execution and module import
try:
    from datasets import ColoredFormatter
except ImportError:
    ColoredFormatter = logging.Formatter

# ============================================================================
# Constants
# ============================================================================
GLOBAL_MEAN = 0.202013
GLOBAL_STD  = 0.200640
SUPPORTED_IMG_SIZES = [64, 128, 256]

# ============================================================================
# Data Transformation Functions
# ============================================================================
def data_transform_forward(image):
    """Transform image to normalized space."""
    return (image - GLOBAL_MEAN) / GLOBAL_STD

def data_transform_backward(image):
    """Transform image back from normalized space."""
    return torch.clamp(image * GLOBAL_STD + GLOBAL_MEAN, 0.0, 1.0)

def normalize_min_max(image, min_hu, max_hu):
    """Normalize image to [0, 1] range using min-max normalization."""
    image = torch.clamp(image, min_hu, max_hu)
    return (image - min_hu) / (max_hu - min_hu)

# ============================================================================
# File I/O Utilities
# ============================================================================
def load_ct_data(file_path):
    """Load CT data from file (supports .h5 and .nii.gz formats)."""
    if file_path.endswith('.h5'):
        with h5py.File(file_path, 'r') as hdf5:
            if 'ct' not in hdf5.keys():
                raise KeyError(f"Required key 'ct' not found in {file_path}. Available keys: {list(hdf5.keys())}")
            return np.asarray(hdf5['ct'])
    else:  # .nii.gz
        nib_img = nibabel.load(file_path)
        return nib_img.get_fdata()

def load_h5_file(file_path):
    """Load H5 file and return CT data and X-ray images."""
    with h5py.File(file_path, 'r') as hdf5:
        ct_data = np.asarray(hdf5['ct'])
        x_ray1 = np.expand_dims(np.asarray(hdf5['xray1']), 0)
        x_ray2 = np.expand_dims(np.asarray(hdf5['xray2']), 0)
        return ct_data, x_ray1, x_ray2

def find_data_file(sample_dir, img_size):
    """Find data file in sample directory, prioritizing .nii.gz files."""
    possible_files = [
        os.path.join(sample_dir, f'ct_{img_size}_norm.nii.gz'),
        os.path.join(sample_dir, 'ct_norm.nii.gz'),
        os.path.join(sample_dir, 'ct_128_norm.nii.gz'),
        os.path.join(sample_dir, 'ct_xray_data.h5'),
    ]
    for pf in possible_files:
        if os.path.exists(pf):
            return pf
    return None

# ============================================================================
# Image Processing Utilities
# ============================================================================
class Resize_image:
    """Resize 3D image using scipy.ndimage.zoom."""
    def __init__(self, size=(3, 256, 256)):
        self.size = np.array(size, dtype=np.float32)

    def __call__(self, img):
        z, x, y = img.shape
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        return ndimage.interpolation.zoom(img, resize_factor, order=1)

def resize_volume_torch(volume, target_size):
    """Resize 3D volume using torch trilinear interpolation."""
    if volume.shape == target_size:
        return volume
    
    volume = volume.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    volume = F.interpolate(
        volume,
        size=target_size,
        mode='trilinear',
        align_corners=False
    )
    return volume.squeeze(0).squeeze(0)  # [D, H, W]

def normalize_ct_data(data, ct_min, ct_max):
    """Normalize CT data to [0, 1] range."""
    data = np.clip(data, ct_min, ct_max)
    return (data - ct_min) / (ct_max - ct_min)

# ============================================================================
# Statistics Computation Utilities
# ============================================================================
class OnlineStatistics:
    """Online statistics computation using Welford's algorithm."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
    
    def update(self, new_data):
        """Update statistics with new data batch (vectorized)."""
        if new_data.size == 0:
            return
        
        new_mean = np.mean(new_data)
        new_M2 = np.sum((new_data - new_mean) ** 2)
        n_new = new_data.size
        
        if self.n > 0:
            delta_mean = new_mean - self.mean
            total_n = self.n + n_new
            self.mean = (self.n * self.mean + n_new * new_mean) / total_n
            self.M2 += new_M2 + self.n * n_new * (delta_mean ** 2) / total_n
            self.n = total_n
        else:
            self.mean = new_mean
            self.M2 = new_M2
            self.n = n_new
    
    def get_std(self):
        """Get standard deviation."""
        return np.sqrt(self.M2 / self.n) if self.n > 1 else 0.0

def compute_image_statistics(image):
    """Compute basic statistics for an image tensor."""
    return {
        'min': torch.min(image).item(),
        'max': torch.max(image).item(),
        'mean': torch.mean(image).item(),
        'std': torch.std(image).item()
    }

def log_statistics_table(logger, stats_before, stats_after):
    """Log statistics comparison table."""
    logger.info("=" * 60)
    logger.info("Data Normalization Statistics (First Sample)")
    logger.info("=" * 60)
    logger.info(f"{'Stage':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    logger.info("-" * 60)
    logger.info(f"{'Before (raw HU)':<25} {stats_before['min']:>10.2f} {stats_before['max']:>10.2f} {stats_before['mean']:>10.2f} {stats_before['std']:>10.2f}")
    logger.info(f"{'After (normalized)':<25} {stats_after['min']:>10.2f} {stats_after['max']:>10.2f} {stats_after['mean']:>10.2f} {stats_after['std']:>10.2f}")
    logger.info("=" * 60)

# ============================================================================
# Logger Setup
# ============================================================================
def setup_logger(rank=0, logger_name='LIDCVolumes'):
    """Setup logger for dataset."""
    if rank == 0:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    else:
        logger = logging.getLogger('silent')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

# ============================================================================
# Dataset Class
# ============================================================================
class LIDCVolumes(torch.utils.data.Dataset):
    """LIDC dataset for 3D CT volumes."""
    
    def __init__(self, directory, test_flag=False, train_txt=None, test_txt=None,
                 config=None, normalize=None, mode='train', img_size=128,
                 val_ratio=0.1, seed=42, rank=0, debug=True):
        super().__init__()
        
        # Validate inputs
        assert img_size in SUPPORTED_IMG_SIZES, f"Image size must be in {SUPPORTED_IMG_SIZES}"
        if train_txt is None or test_txt is None:
            raise ValueError("train_txt and test_txt must be provided")
        if config is None:
            raise ValueError("config must be provided")
        
        # Store configuration
        self.config = config
        self.ct_min = config.data.ct_min
        self.ct_max = config.data.ct_max
        self.hu_min = self.ct_min
        self.hu_max = self.ct_max
        
        # Setup basic attributes
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        self.img_size = img_size
        self.debug = debug
        self.rank = rank
        
        # Setup logger
        self.logger = setup_logger(rank)
        
        # Print normalization parameters
        self._print_normalization_params()
        
        # Load and validate samples
        self.train_samples, self.test_samples = self._load_sample_lists(train_txt, test_txt)
        self._validate_directory()
        self.train_samples, self.test_samples = self._filter_available_samples()
        
        # Setup normalization
        self.normalize = normalize if normalize else self._default_normalize
        
        # Setup augmentation
        self.augment = (mode == 'train')
        
        # Set random seed
        np.random.seed(seed)
        if rank == 0:
            self.logger.info(f"Set random seed to {seed}")
        
        # Build database and preload data
        self.database = self._build_database()
        self.data_cache = {}
        self._preload_data()
        
        if rank == 0:
            self.logger.info("Dataset initialization completed - All data loaded into memory")
    
    def _print_normalization_params(self):
        """Print CT normalization parameters."""
        if self.rank == 0:
            print("=" * 60)
            print("CT normalization parameters:")
            print("=" * 60)
            print(f"CT min: {self.ct_min}, CT max: {self.ct_max}")
            print(f"Output data range: [-1, 1]")
            print("=" * 60)
    
    def _load_sample_lists(self, train_txt, test_txt):
        """Load train and test sample ID lists."""
        with open(train_txt, 'r') as f:
            train_samples = [line.strip() for line in f.readlines()]
        with open(test_txt, 'r') as f:
            test_samples = [line.strip() for line in f.readlines()]
        return train_samples, test_samples
    
    def _validate_directory(self):
        """Validate that data directory exists."""
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(f"Dataset directory not found: {self.directory}")
    
    def _filter_available_samples(self):
        """Filter samples to only include those available in directory."""
        available_samples = {
            entry for entry in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, entry))
        }
        
        original_train = set(self.train_samples)
        original_test = set(self.test_samples)
        
        train_samples = [sid for sid in self.train_samples if sid in available_samples]
        test_samples = [sid for sid in self.test_samples if sid in available_samples]
        
        missing_train = sorted(original_train - available_samples)
        missing_test = sorted(original_test - available_samples)
        
        if self.rank == 0:
            print(f"Missing train samples: {missing_train}")
            print(f"Missing test samples: {missing_test}")
        
        return train_samples, test_samples
    
    def _build_database(self):
        """Build database of file paths."""
        sample_ids = self.train_samples if self.mode == 'train' else self.test_samples
        
        if self.rank == 0:
            self.logger.info("Building file paths from sample IDs...")
            self.logger.warning("=" * 40)
            self.logger.warning(f"The argument augment status: {self.augment}")
            self.logger.warning(f"The mode status: {self.mode}")
            self.logger.warning("=" * 40)
        
        database = []
        for sample_id in sample_ids:
            sample_dir = os.path.join(self.directory, sample_id)
            data_file = find_data_file(sample_dir, self.img_size)
            
            if data_file:
                database.append({'image': data_file, 'id': sample_id})
            elif self.rank == 0:
                self.logger.warning(f"Data file not found for sample {sample_id} in {sample_dir}")
        
        mode_name = 'Training' if self.mode == 'train' else ('Validation' if self.mode == 'val' else 'Test')
        if self.rank == 0:
            self.logger.info(f"{mode_name} set size: {len(database)} samples")
            if len(database) > 0:
                self.logger.info(f"First few {mode_name.lower()} samples: {[d['id'] for d in database[:3]]}")
        
        return database
    
    def _load_and_process_volume(self, file_path):
        """Load and process a single volume."""
        x_ray = None
        
        if file_path.endswith('.h5'):
            ct_data, x_ray1, x_ray2 = load_h5_file(file_path)
            out = torch.Tensor(ct_data)
            x_ray = torch.cat([torch.Tensor(x_ray1), torch.Tensor(x_ray2)], dim=0).unsqueeze(0)
        else:  # .nii.gz
            out = torch.Tensor(load_ct_data(file_path))
        
        # Resize if needed
        target_size = (self.img_size, self.img_size, self.img_size)
        if out.shape != target_size:
            out = resize_volume_torch(out, target_size)
        
        # Add channel dimension
        image = out.unsqueeze(0)
        
        # Normalize
        image = normalize_min_max(image, min_hu=self.hu_min, max_hu=self.hu_max)
        image = data_transform_forward(image)
        
        return image, x_ray
    
    def _preload_data(self):
        """Preload all data into memory."""
        if self.rank == 0:
            self.logger.info(f"Preloading data into memory (target size: {self.img_size}x{self.img_size}x{self.img_size})...")
        
        with tqdm.tqdm(total=len(self.database), desc="Loading files", disable=(self.rank != 0)) as pbar:
            for filedict in self.database:
                name = filedict['image']
                sample_id = filedict['id']
                
                try:
                    image, x_ray = self._load_and_process_volume(name)
                    
                    # Log statistics for first sample
                    if self.rank == 0 and pbar.n == 0:
                        stats_before = compute_image_statistics(image)
                        # Note: stats_before is computed after normalization, adjust if needed
                        stats_after = compute_image_statistics(image)
                        log_statistics_table(self.logger, stats_before, stats_after)
                    
                    self.data_cache[name] = (image, x_ray)
                    pbar.update(1)
                    
                except Exception as e:
                    if self.rank == 0:
                        self.logger.error(f"Error loading file {name} (sample {sample_id}): {str(e)}")
                    raise
    
    def load_file(self, file_path):
        """Load H5 file (kept for backward compatibility)."""
        return load_h5_file(file_path)
    
    def _default_normalize(self, image):
        """Default normalization: clamp to CT range, normalize to [0, 1], then map to [-1, 1]."""
        image = normalize_min_max(image, min_hu=self.hu_min, max_hu=self.hu_max)
        return data_transform_forward(image)
    
    # Augmentation methods
    def _add_gaussian_noise(self, image, noise_std=0.01):
        """Add Gaussian noise to image."""
        return image + torch.randn_like(image) * noise_std
    
    def _random_flip(self, image, prob=0.5):
        """Randomly flip image along random axis."""
        if random.random() < prob:
            axis = random.choice([1, 2, 3])
            image = torch.flip(image, dims=[axis])
        return image
    
    def _random_rotation_90(self, image, prob=0.3):
        """Randomly rotate image 90 degrees in random plane."""
        if random.random() < prob:
            plane = random.choice(['xy', 'xz', 'yz'])
            k = random.choice([1, 2, 3])
            dims_map = {'xy': [2, 3], 'xz': [1, 3], 'yz': [1, 2]}
            image = torch.rot90(image, k=k, dims=dims_map[plane])
        return image
    
    def _random_intensity_shift(self, image, shift_range=0.1, prob=0.5):
        """Apply random intensity shift."""
        if random.random() < prob:
            data_range = image.max() - image.min()
            shift = random.uniform(-shift_range, shift_range) * data_range
            image = torch.clamp(image + shift, min=-1.0, max=1.0)
        return image
    
    def _random_intensity_scale(self, image, scale_range=(0.9, 1.1), prob=0.5):
        """Apply random intensity scaling."""
        if random.random() < prob:
            scale = random.uniform(scale_range[0], scale_range[1])
            image = torch.clamp(image * scale, min=-1.0, max=1.0)
        return image
    
    def apply_augmentation(self, image):
        """Apply data augmentation."""
        image = self._random_flip(image, prob=0.5)
        # Additional augmentations can be enabled here
        return image
    
    def __getitem__(self, idx):
        """Get item from dataset."""
        filedict = self.database[idx]
        name = filedict['image']
        
        image, x_rays = self.data_cache[name]
        image = image.clone()
        
        # if self.augment:
        #     image = self.apply_augmentation(image)
        
        result = {"image": image, "id": name}
        if x_rays is not None:
            result["x_rays"] = x_rays
        return result
    
    def __len__(self):
        """Get dataset length."""
        return len(self.database)

# ============================================================================
# Statistics Computation Functions
# ============================================================================
def process_sample_for_statistics(sample_id, data_directory, img_size, ct_min, ct_max, resizer):
    """Process a single sample for statistics computation."""
    sample_dir = os.path.join(data_directory, sample_id)
    data_file = os.path.join(sample_dir, 'ct_xray_data.h5')
    
    if not os.path.exists(data_file):
        return None, None
    
    try:
        data_np = load_ct_data(data_file)
    except Exception as e:
        return None, None
    
    current_shape = data_np.shape
    needs_resize = len(current_shape) == 3 and current_shape != (img_size, img_size, img_size)
    
    # Process with scipy zoom
    if needs_resize:
        data_scipy = resizer(data_np)
    else:
        data_scipy = data_np
    
    # Process with torch interpolation
    if needs_resize:
        data_torch = torch.from_numpy(data_np).unsqueeze(0).unsqueeze(0)
        data_torch = F.interpolate(data_torch, size=(img_size, img_size, img_size),
                                   mode='trilinear', align_corners=False)
        data_torch = data_torch.squeeze(0).squeeze(0).numpy()
    else:
        data_torch = data_np
    
    # Normalize
    data_scipy = normalize_ct_data(data_scipy, ct_min, ct_max)
    data_torch = normalize_ct_data(data_torch, ct_min, ct_max)
    
    return data_scipy, data_torch

def compute_average_std(data_directory, train_txt, img_size=128, ct_min=-1000.0, ct_max=400.0):
    """Compute global mean and std across entire training set using two resize methods."""
    print("=" * 80)
    print("Computing Global Statistics Across Entire Training Set")
    print("=" * 80)
    print(f"Resize Methods: Resize_image (scipy.zoom) vs torch.interpolate (trilinear)")
    print(f"CT range: [{ct_min}, {ct_max}]")
    print(f"Target image size: {img_size}x{img_size}x{img_size}")
    print("=" * 80)
    
    resizer = Resize_image(size=(img_size, img_size, img_size))
    
    # Load and filter samples
    with open(train_txt, 'r') as f:
        train_samples = [line.strip() for line in f.readlines()]
    
    data_directory = os.path.expanduser(data_directory)
    if not os.path.isdir(data_directory):
        raise FileNotFoundError(f"Dataset directory not found: {data_directory}")
    
    available_samples = {
        entry for entry in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, entry))
    }
    train_samples = sorted([sid for sid in train_samples if sid in available_samples])
    
    print(f"Found {len(train_samples)} training samples")
    print("Processing samples...\n")
    
    # Initialize statistics trackers
    stats_scipy = OnlineStatistics()
    stats_torch = OnlineStatistics()
    
    # Process samples
    processed_count = 0
    for sample_id in train_samples:
        data_scipy, data_torch = process_sample_for_statistics(
            sample_id, data_directory, img_size, ct_min, ct_max, resizer
        )
        
        if data_scipy is not None and data_torch is not None:
            stats_scipy.update(data_scipy)
            stats_torch.update(data_torch)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count}/{len(train_samples)} samples...")
    
    # Compute and print results
    if stats_scipy.n > 0 and stats_torch.n > 0:
        std_scipy = stats_scipy.get_std()
        std_torch = stats_torch.get_std()
        
        print("\n" + "=" * 80)
        print("Global Statistics Across Entire Training Set")
        print("=" * 80)
        print(f"{'Method':<30} {'Mean':>15} {'Std':>15} {'Total Pixels':>15}")
        print("-" * 80)
        print(f"{'Resize_image (scipy.zoom)':<30} {stats_scipy.mean:>15.6f} {std_scipy:>15.6f} {stats_scipy.n:>15,}")
        print(f"{'torch.interpolate (trilinear)':<30} {stats_torch.mean:>15.6f} {std_torch:>15.6f} {stats_torch.n:>15,}")
        print("=" * 80)
        
        diff_mean = abs(stats_scipy.mean - stats_torch.mean)
        diff_std = abs(std_scipy - std_torch)
        
        print(f"\nDifference between methods:")
        print(f"  Mean difference: {diff_mean:.6f}")
        print(f"  Std difference:  {diff_std:.6f}")
        print(f"  Relative mean diff: {diff_mean / stats_scipy.mean * 100:.4f}%")
        print(f"  Relative std diff:  {diff_std / std_scipy * 100:.4f}%")
        print("=" * 80)
        print(f"\nProcessed {processed_count} volumes successfully")
        print("=" * 80)
        
        return {
            'scipy': {'mean': stats_scipy.mean, 'std': std_scipy, 'n_pixels': stats_scipy.n},
            'torch': {'mean': stats_torch.mean, 'std': std_torch, 'n_pixels': stats_torch.n},
            'diff': {'mean': diff_mean, 'std': diff_std}
        }
    else:
        print("No volumes were processed successfully.")
        return None

# ============================================================================
# Dataset Factory Functions
# ============================================================================
def create_lidc_datasets(data_directory, train_txt, test_txt, img_size=128,
                         augment=False, hu_min=-1000.0, hu_max=400.0):
    """Create train and test LIDC datasets."""
    train_dataset = LIDCVolumes(
        directory=data_directory,
        train_txt=train_txt,
        test_txt=test_txt,
        mode='train',
        img_size=img_size,
        test_flag=False,
        augment=augment,
        hu_min=hu_min,
        hu_max=hu_max
    )
    
    test_dataset = LIDCVolumes(
        directory=data_directory,
        train_txt=train_txt,
        test_txt=test_txt,
        mode='test',
        img_size=img_size,
        test_flag=True,
        augment=False,
        hu_min=hu_min,
        hu_max=hu_max
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    data_dir = "../../data/LIDC-HDF5-256"
    train_txt_path = "./data/train.txt"
    test_txt_path = "./data/test.txt"
    
    # Compute global statistics
    result = compute_average_std(
        data_directory=data_dir,
        train_txt=train_txt_path,
        img_size=128,
        ct_min=0,
        ct_max=2500,
    )
    
    if result is not None:
        print("\nResults summary:")
        print(f"  Scipy method - Mean: {result['scipy']['mean']:.6f}, Std: {result['scipy']['std']:.6f}")
        print(f"  Torch method - Mean: {result['torch']['mean']:.6f}, Std: {result['torch']['std']:.6f}")
