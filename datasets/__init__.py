import logging
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_colors = self._supports_color()

    def _supports_color(self):
        try:
            import sys

            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                term = os.environ.get('TERM', '')
                return 'color' in term or term in ['xterm', 'xterm-256color', 'screen', 'tmux']
            return False
        except Exception:
            return False

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        base_message = f'{timestamp} - {record.name} - {record.levelname} - {record.getMessage()}'
        if not self.use_colors:
            return base_message

        color = self.COLORS.get(record.levelname, '')
        colored_timestamp = f'\033[90m{timestamp}\033[0m'
        colored_level = f'{color}{self.BOLD}{record.levelname}\033[0m'
        logger_name = f'\033[94m{record.name}\033[0m'

        if record.levelname == 'INFO':
            if any(label in record.getMessage() for label in ('Min:', 'Max:', 'Mean:', 'Std:')):
                message = f'\033[96m{record.getMessage()}\033[0m'
            elif 'expected [0,1] range' in record.getMessage():
                message = f'\033[92m{record.getMessage()}\033[0m'
            elif 'Final' in record.getMessage() and 'size:' in record.getMessage():
                message = f'\033[93m{record.getMessage()}\033[0m'
            else:
                message = f'{color}{record.getMessage()}\033[0m'
        elif record.levelname == 'WARNING':
            message = f'{color}{record.getMessage()}\033[0m'
        else:
            message = f'{color}{record.getMessage()}\033[0m'

        return f'{colored_timestamp} - {logger_name} - {colored_level} - {message}'


from datasets.rad_chest import RADChestCTDataset
from datasets.lidc import LIDCVolumes

RAD_CHEST_TASKS = {"rad_chestCT", "rad_chest"}

def get_voxel_dataset(dataroot, 
                      task="rad_chestCT", 
                      roi_size=(128, 128, 128), 
                      config=None,
                      data_type="train", 
                      train_txt="./data/train.txt",
                      test_txt="./data/test.txt",
                      test_frac=0.1,
                      val_frac=0.1,
                      seed=42,
                      cache_dir=None,
                      preprocess=None,
                      normalize=False,
                      augment=False,
                      rank=0):
    if rank == 0:
        print(f"Loading {task} dataset...")
    
    if task == "lidc":
        img_size = roi_size[0]
        assert img_size in [64, 128, 256], "LIDC dataset only supports image sizes: 128, 256"
        
        # Note: normalize parameter is ignored for LIDC dataset
        # Normalization is handled internally using config.data.ct_min, ct_max
        # Data is normalized to [-1, 1] range for training
        return LIDCVolumes(
            directory=os.path.join(dataroot, config.data.target_path),
            train_txt=train_txt,
            test_txt=test_txt,
            config=config,
            test_flag=False,
            normalize=None,  # Always use config-based normalization
            mode='train' if data_type == 'train' else 'test',
            img_size=img_size,
            val_ratio=val_frac,
            seed=seed,
            rank=rank
        )
    elif task in RAD_CHEST_TASKS:
        dataset_name = "rad_chestCT" if task == "rad_chestCT" else "rad_chest"
        preprocess = preprocess if preprocess in ("cc", "rs") else "cc"
        return RADChestCTDataset(
            directory=os.path.join(dataroot, dataset_name),
            mode=data_type,
            img_size=roi_size[0],
            cache_dir=cache_dir,
            preprocess=preprocess,
            normalize=(lambda x: 2*x - 1) if normalize else None,
            val_ratio=val_frac,
            test_ratio=test_frac,
            seed=seed,
            augment=augment if task == "rad_chestCT" else False,
            rank=rank,
        )
    else:
        raise ValueError(f"Invalid task: {task}. Supported tasks: 'lidc', 'rad_chestCT', 'rad_chest'")
