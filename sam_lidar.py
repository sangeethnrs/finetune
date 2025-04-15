import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SAMLiDAR(nn.Module):
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h", device: str = "cuda"):
        super().__init__()
        self.device = device
        logger.info(f"Loading SAM model type {model_type} from {checkpoint_path}")
        
        # Load SAM model without weights_only parameter
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.input_size = (1024, 1024)
    
    def forward(self, image: torch.Tensor) -> Dict[str, Any]:
        # Ensure input is properly formatted
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # Verify input dimensions
        if H != self.input_size[0] or W != self.input_size[1]:
            raise ValueError(f"Expected input size {self.input_size}, got {(H, W)}")
        
        # Generate image embeddings
        with torch.no_grad():
            image_embedding = self.sam.image_encoder(image)
        
        # Create single point prompt for each image in batch
        input_point = torch.tensor([[[H//2, W//2]]], device=self.device).expand(B, 1, 2)
        input_label = torch.ones((B, 1), device=self.device)
        
        # Get prompt embeddings with fixed sizes
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(input_point, input_label),
            boxes=None,
            masks=None
        )
        
        # Predict masks
        mask_predictions, _ = self.sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        return {"masks": mask_predictions}
