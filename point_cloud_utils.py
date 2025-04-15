import numpy as np
import open3d as o3d
from typing import Tuple, Optional
import torch
import cv2
import logging

logger = logging.getLogger(__name__)

class PointCloudProcessor:
    def __init__(self, max_points: int = 50000, image_size: Tuple[int, int] = (1024, 1024)):
        self.max_points = max_points
        self.image_size = image_size
        
    def load_and_preprocess(self, pcd_path: str) -> Optional[np.ndarray]:
        """Load and preprocess point cloud data."""
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(pcd_path)
            if not pcd or len(pcd.points) == 0:
                raise ValueError(f"Empty point cloud: {pcd_path}")
                
            # Convert to numpy array
            points = np.asarray(pcd.points)
            
            # Validate dimensions
            if points.shape[1] != 3:
                raise ValueError(f"Expected 3D points, got {points.shape[1]}D")
                
            # Subsample if too many points
            if len(points) > self.max_points:
                indices = np.random.choice(
                    len(points), 
                    self.max_points, 
                    replace=False
                )
                points = points[indices]
            
            # Remove statistical outliers
            pcd_clean = o3d.geometry.PointCloud()
            pcd_clean.points = o3d.utility.Vector3dVector(points)
            cl, ind = pcd_clean.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )
            points = np.asarray(cl.points)
            
            return points.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading point cloud {pcd_path}: {str(e)}")
            return None
            
    def create_bev_image(self, points: np.ndarray) -> np.ndarray:
        """Convert point cloud to bird's eye view image."""
        try:
            if points is None or len(points) == 0:
                raise ValueError("Empty point cloud")
                
            H, W = self.image_size
            bev = np.zeros((H, W), dtype=np.float32)
            
            # Process points
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            
            # Remove ground and ceiling points
            z_min, z_max = np.percentile(z, [5, 95])
            mask = (z >= z_min) & (z <= z_max)
            x, y, z = x[mask], y[mask], z[mask]
            
            # Normalize coordinates
            x_norm = np.clip((x - x.min()) / (x.ptp() + 1e-6), 0, 1)
            y_norm = np.clip((y - y.min()) / (y.ptp() + 1e-6), 0, 1)
            
            # Convert to image coordinates
            x_img = (x_norm * (W - 1)).astype(np.int32)
            y_img = (y_norm * (H - 1)).astype(np.int32)
            
            # Create height map
            for i in range(len(x_img)):
                bev[y_img[i], x_img[i]] = max(bev[y_img[i], x_img[i]], z[i])
            
            # Normalize and enhance contrast
            bev_norm = (bev - bev.min()) / (bev.ptp() + 1e-6)
            bev_img = (bev_norm * 255).astype(np.uint8)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            bev_img = clahe.apply(bev_img)
            
            # Create RGB image
            bev_rgb = np.stack([bev_img] * 3, axis=-1)
            
            return bev_rgb
            
        except Exception as e:
            logger.error(f"Error creating BEV image: {str(e)}")
            return np.zeros((1024, 1024, 3), dtype=np.uint8)
