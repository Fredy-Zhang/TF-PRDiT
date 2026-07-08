import os
import sys
import time
import math
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
from torchio import ScalarImage, Subject as TorchioSubject
from diffdrr.drr import DRR

from conds.utils import construct_subject_from_ct, save_xrays

#============================================================================
# CONSTANTS
#============================================================================
SAD = 700.0
SID = 820.0
PHYSICAL_SIZE = 256.0  # Keep the same physical size to maintain anatomical scale
# PIXEL_PITCH = PHYSICAL_SIZE / IMG_SIZE
PARAMETERIZATION = "euler_angles"
CONVENTION = "XYZ"
#============================================================================

def get_xrays_from_ct(volume: str, 
                      idx: int, 
                      device: str = "cpu", 
                      rotations: int = 2):
    subject = construct_subject_from_ct(volume)
    
    IMG_SIZE = volume.shape[-1]
    PIXEL_PITCH = PHYSICAL_SIZE / IMG_SIZE
    
    drr_module = DRR(
                subject=subject,
                sdd=SID,           
                height=IMG_SIZE,
                width=IMG_SIZE,
                delx=PIXEL_PITCH,
                dely=PIXEL_PITCH,
                renderer="siddon",
            ).to(device)
    
    translations = torch.tensor([[0.0, 0.0, SAD]], device=device, dtype=torch.float32)
    base = math.pi
    if rotations == 2:
        angles_deg = [0.0, 90.0]
        angles_rad = [float(math.radians(base - angle)) for angle in angles_deg]
        rotations = torch.tensor([[0.0, angle, 0.0] for angle in angles_rad], 
                                 device=device, dtype=torch.float32)
    elif rotations == 1:
        angles_deg = [0.0]
        angles_rad = [float(math.radians(base - angle)) for angle in angles_deg]
        rotations = torch.tensor([[0.0, angle, 0.0] for angle in angles_rad], 
                                 device=device, dtype=torch.float32)
    elif rotations > 2:
        # 1. Start with the priority anchors
        anchors = {0.0, 90.0}
        
        # 2. Generate a uniform distribution for the requested count
        # This ensures that as num_rotations increases, the "view" broadens
        uniform_spacing = [round((360.0 / rotations) * i, 2) for i in range(rotations)]
        
        # 3. Combine them. The set handles duplicates automatically.
        combined = anchors.union(set(uniform_spacing))
        
        # 4. Sort and trim to the desired number of rotations
        # If the union created too many, we sort by 'importance' or just take the spread
        angles_deg = sorted(list(combined))
        
        # If we have more than requested due to the union, 
        # we prioritize keeping the anchors and thinning the rest.
        while len(angles_deg) > rotations:
            # Remove the element that is closest to its neighbor (minimizing info loss)
            # but never remove 0.0 or 90.0
            idx_to_remove = -1
            min_diff = float('inf')
            for i in range(len(angles_deg)):
                if angles_deg[i] in [0.0, 90.0]:
                    continue
                # Calculate distance to neighbors
                prev_dist = abs(angles_deg[i] - angles_deg[i-1])
                next_dist = abs(angles_deg[(i+1)%len(angles_deg)] - angles_deg[i])
                if min(prev_dist, next_dist) < min_diff:
                    min_diff = min(prev_dist, next_dist)
                    idx_to_remove = i
            angles_deg.pop(idx_to_remove)
        angles_rad = [math.radians(base - d) for d in angles_deg]
        rotations = torch.tensor([[0.0, angle, 0.0] for angle in angles_rad], 
                                 device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Invalid number of rotations: {rotations}")
    
    imgs = drr_module(rotations, translations, 
                      parameterization=PARAMETERIZATION, 
                      convention=CONVENTION)
    
    # for i in range(imgs.shape[0]):
    #     print(f"X-rays ID: {i}, min: {imgs[i].min()}, max: {imgs[i].max()}")
    # save the conditions x-rays.add()
    # save_xrays(imgs, "conditions", idx)
    
    return imgs
