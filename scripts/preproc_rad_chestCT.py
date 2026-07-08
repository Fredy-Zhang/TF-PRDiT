import os
import numpy as np
import json
from tqdm import tqdm
from scipy.ndimage import zoom
import nibabel as nib
import random

def resize_volume(volume: np.ndarray, target_shape: tuple = (256, 256, 256)) -> np.ndarray:
    """
    Resize a volume to target shape using trilinear interpolation.
    
    Args:
        volume: Input 3D volume
        target_shape: Desired output shape (default: 256x256x256)
    """
    factors = [float(t) / float(s) for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1, mode='constant', cval=-1000)

def save_batch(volumes: list, output_path: str, batch_idx: int):
    """
    Save a batch of volumes to a single NPZ file.
    
    Args:
        volumes: List of processed volumes
        output_path: Base path for output file
        batch_idx: Batch index
    """
    # Stack volumes into a single array
    volumes_array = np.stack(volumes, axis=0)
    
    # Save volumes
    output_file = f"{output_path}_batch{batch_idx:03d}.npz"
    np.savez_compressed(output_file, volumes=volumes_array)
    return output_file

def save_debug_nifti(volume: np.ndarray, output_path: str, filename: str):
    """
    Save a single volume as NIfTI file for debugging.
    
    Args:
        volume: 3D volume to save
        output_path: Directory to save the file
        filename: Name of the file
    """
    debug_dir = os.path.join(output_path, 'debug_nifti')
    os.makedirs(debug_dir, exist_ok=True)
    
    nii_path = os.path.join(debug_dir, f"{filename}.nii.gz")
    nii_img = nib.Nifti1Image(volume, affine=np.eye(4))
    nib.save(nii_img, nii_path)
    return nii_path

def preprocess_npz_to_npz(input_path: str, output_dir: str, min_size: int = 110, 
                         target_shape: tuple = (256, 256, 256),
                         save_debug: bool = False) -> dict:
    """
    Preprocess CT volume with size filtering and resizing.
    
    Args:
        input_path: Path to input .npz file
        output_dir: Directory to save output files
        min_size: Minimum allowed dimension size
        target_shape: Final shape after resizing (default: 256x256x256)
        save_debug: Whether to save debug NIfTI file
    
    Returns:
        dict: Metadata about the processed volume
    """
    try:
        # Load and check data
        ct = np.load(input_path)['ct'].astype(np.float32)
        original_shape = ct.shape
        
        # Check if any dimension is too small
        if any(s < min_size for s in original_shape):
            print(f"⚠️  {os.path.basename(input_path)}: {original_shape} → SKIPPED (dimension < {min_size})")
            return None
        
        # Resize to target shape
        ct = resize_volume(ct, target_shape)
        
        # Save debug NIfTI file if requested
        debug_path = None
        if save_debug:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            debug_path = save_debug_nifti(ct, output_dir, base_name)
            print(f"    Debug NIfTI saved to: {debug_path}")
        
        print(f"✅  {os.path.basename(input_path)}: {original_shape} → {ct.shape}")
        
        # Return metadata
        return {
            'original_file': input_path,
            'original_shape': original_shape,
            'processed_shape': ct.shape,
            'min_value': float(np.min(ct)),
            'max_value': float(np.max(ct)),
            'mean_value': float(np.mean(ct)),
            'std_value': float(np.std(ct)),
            'debug_nifti': debug_path,
            'volume': ct  # Include the processed volume in metadata
        }
        
    except Exception as e:
        print(f"❌  {os.path.basename(input_path)}: Failed - {str(e)}")
        return None

def batch_convert(input_dir: str, output_dir: str, min_size: int = 110,
                 target_shape: tuple = (256, 256, 256)):
    """
    Convert NPZ files to processed NPZ format with size filtering and resizing.
    
    Args:
        input_dir: Input directory containing .npz files
        output_dir: Output directory for processed .npz files
        min_size: Minimum allowed dimension size
        target_shape: Final shape after resizing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    print(f"\nProcessing {len(files)} files...")
    print(f"Minimum dimension size: {min_size}")
    print(f"Processing pipeline:")
    print(f"1. Filter volumes < {min_size}")
    print(f"2. Resize to final shape {target_shape}")
    print(f"3. Save 10 random debug NIfTI files")
    print(f"4. Save in batches of 1000")
    print("-" * 80)
    
    successful = 0
    skipped = 0
    failed = 0
    metadata = {}
    current_batch = []
    batch_idx = 0
    saved_files = []
    
    # Select 10 random files for debug NIfTI
    debug_files = set(random.sample(files, min(10, len(files))))
    
    for file in tqdm(files, desc="Converting", leave=False):
        input_path = os.path.join(input_dir, file)
        save_debug = file in debug_files
        result = preprocess_npz_to_npz(input_path, output_dir, min_size, target_shape, save_debug)
        
        if result is not None:
            successful += 1
            metadata[file] = result
            current_batch.append(result['volume'])
            
            # Save batch if it reaches 1000 samples
            if len(current_batch) >= 1000:
                output_path = os.path.join(output_dir, "processed")
                saved_file = save_batch(current_batch, output_path, batch_idx)
                saved_files.append(saved_file)
                current_batch = []
                batch_idx += 1
        elif "dimension <" in str(result):
            skipped += 1
        else:
            failed += 1
    
    # Save remaining volumes
    if current_batch:
        output_path = os.path.join(output_dir, "processed")
        saved_file = save_batch(current_batch, output_path, batch_idx)
        saved_files.append(saved_file)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'metadata': metadata,
            'saved_files': saved_files,
            'total_samples': successful,
            'skipped_samples': skipped,
            'failed_samples': failed,
            'debug_files': list(debug_files)
        }, f, indent=2)
    
    print("-" * 80)
    print(f"Results:")
    print(f"- Processed successfully: {successful}")
    print(f"- Skipped (too small): {skipped}")
    print(f"- Failed: {failed}")
    print(f"- Total files: {len(files)}")
    print(f"- Number of batches: {len(saved_files)}")
    print(f"- Debug NIfTI files saved in: {os.path.join(output_dir, 'debug_nifti')}")
    print(f"- Number of debug files: {len(debug_files)}")
    print(f"- Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert NPZ files to processed NPZ format with size filtering and resizing")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing .npz files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed .npz files")
    parser.add_argument('--min_size', type=int, default=110, 
                      help="Minimum allowed dimension size (default: 110)")
    parser.add_argument('--target_shape', type=int, nargs=3, default=(256, 256, 256),
                      help="Final shape after resizing (default: 256 256 256)")
    args = parser.parse_args()

    batch_convert(
        args.input_dir,
        args.output_dir,
        args.min_size,
        tuple(args.target_shape)
    )
