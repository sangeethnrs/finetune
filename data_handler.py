import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
import logging
from pathlib import Path
from typing import Tuple, Optional
import cv2
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafePointCloudProcessor:
    """Thread-safe point cloud processor with memory management"""
    
    def __init__(self, max_points: int = 50000, image_size: Tuple[int, int] = (1024, 1024)):
        self.max_points = max_points
        self.image_size = image_size
        self._lock = torch.multiprocessing.Lock()
    
    def process_point_cloud(self, pcd_path: str) -> Optional[np.ndarray]:
        """Process point cloud with proper error handling and memory management"""
        with self._lock:
            try:
                # Validate file
                pcd_path = Path(pcd_path)
                if not pcd_path.exists():
                    raise FileNotFoundError(f"PCD file not found: {pcd_path}")
                
                # Load point cloud
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                if not pcd or len(pcd.points) == 0:
                    raise ValueError(f"Empty point cloud: {pcd_path}")
                
                # Convert to numpy with memory management
                points = np.asarray(pcd.points, dtype=np.float32)
                del pcd
                
                # Validate dimensions
                if points.shape[1] != 3:
                    raise ValueError(f"Expected 3D points, got {points.shape[1]}D")
                
                # Sample points if necessary
                if len(points) > self.max_points:
                    indices = np.random.choice(len(points), self.max_points, replace=False)
                    points = points[indices]
                
                # Remove outliers
                mean = np.mean(points, axis=0)
                std = np.std(points, axis=0)
                mask = np.all(np.abs(points - mean) <= 3 * std, axis=1)
                points = points[mask]
                
                gc.collect()
                return points
                
            except Exception as e:
                logger.error(f"Error processing {pcd_path}: {str(e)}")
                return None
    
    def create_bev_image(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Create bird's eye view image with memory efficiency"""
        try:
            if points is None or len(points) == 0:
                raise ValueError("Empty point cloud")
            
            H, W = self.image_size
            bev = np.zeros((H, W), dtype=np.float32)
            
            # Process in chunks
            chunk_size = min(10000, len(points))
            for i in range(0, len(points), chunk_size):
                chunk = points[i:i + chunk_size]
                x, y, z = chunk[:, 0], chunk[:, 1], chunk[:, 2]
                
                # Remove ground plane
                z_min, z_max = np.percentile(z, [5, 95])
                mask = (z >= z_min) & (z <= z_max)
                x, y, z = x[mask], y[mask], z[mask]
                
                # Normalize coordinates
                x_norm = np.clip((x - np.min(x)) / (np.ptp(x) + 1e-6), 0, 1)
                y_norm = np.clip((y - np.min(y)) / (np.ptp(y) + 1e-6), 0, 1)
                
                # Convert to image coordinates
                x_img = (x_norm * (W - 1)).astype(np.int32)
                y_img = (y_norm * (H - 1)).astype(np.int32)
                
                # Update height map
                np.maximum.at(bev, (y_img, x_img), z)
                
                del chunk
            
            # Normalize and create RGB image
            bev_norm = cv2.normalize(bev, None, 0, 255, cv2.NORM_MINMAX)
            bev_rgb = np.stack([bev_norm.astype(np.uint8)] * 3, axis=-1)
            
            gc.collect()
            return bev_rgb
            
        except Exception as e:
            logger.error(f"Error creating BEV image: {str(e)}")
            return None

class LiDARDataset(Dataset):
    """Memory-efficient LiDAR dataset"""
    
    def __init__(self, data_dir: str, processor: SafePointCloudProcessor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.pcd_files = sorted(list(self.data_dir.glob("*.pcd")))
        logger.info(f"Found {len(self.pcd_files)} PCD files in {data_dir}")
    
    def __len__(self):
        return len(self.pcd_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        try:
            pcd_path = self.pcd_files[idx]
            
            # Process point cloud
            points = self.processor.process_point_cloud(pcd_path)
            if points is None:
                return self._get_empty_sample(), pcd_path.name
            
            # Create BEV image
            bev_image = self.processor.create_bev_image(points)
            if bev_image is None:
                return self._get_empty_sample(), pcd_path.name
            
            # Convert to tensor
            image_tensor = torch.from_numpy(bev_image).float().permute(2, 0, 1) / 255.0
            
            del points, bev_image
            gc.collect()
            
            return image_tensor, pcd_path.name
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            return self._get_empty_sample(), self.pcd_files[idx].name
    
    def _get_empty_sample(self) -> torch.Tensor:
        return torch.zeros((3, 1024, 1024), dtype=torch.float32)
