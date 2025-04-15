import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import gc
from sam_lidar import SAMLiDAR
from data_handler import SafePointCloudProcessor, LiDARDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_sam_on_lidar(config_path: str):
    try:
        # Load configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize processor
        processor = SafePointCloudProcessor(
            max_points=config['preprocessing']['max_points'],
            image_size=(1024, 1024)
        )
        
        # Setup dataset
        dataset = LiDARDataset(config['data']['pcd_dir'], processor)
        
        # Create dataloader with fixed batch size
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=1,  # Reduced to 1 for debugging
            pin_memory=True,
            drop_last=True,
        )
        
        # Initialize model
        model = SAMLiDAR(
            checkpoint_path=config['model']['checkpoint_path'],
            model_type=config['model']['type'],
            device=str(device)
        ).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Training loop
        model.train()
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"\n--- Epoch {epoch+1}/{config['training']['num_epochs']} ---")
            
            for batch_idx, (images, filenames) in enumerate(tqdm(dataloader)):
                try:
                    # Move to device
                    images = images.to(device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(images)
                    mask_pred = outputs['masks']
                    
                    # Create target masks matching batch size
                    target_masks = torch.zeros_like(mask_pred)
                    
                    # Verify tensor sizes
                    assert mask_pred.size(0) == target_masks.size(0), \
                        f"Batch size mismatch: {mask_pred.size(0)} vs {target_masks.size(0)}"
                    
                    # Calculate loss
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        mask_pred,
                        target_masks
                    )
                    
                    # Backward pass
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    
                    # Log progress
                    if batch_idx % config['training']['log_interval'] == 0:
                        logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
                    # Clean up
                    del outputs, mask_pred, target_masks
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
            
            # Save checkpoint
            if (epoch + 1) % config['training']['save_interval'] == 0:
                save_path = Path(config['training']['checkpoint_dir'])
                save_path.mkdir(exist_ok=True)
                
                checkpoint_path = save_path / f"sam_lidar_epoch_{epoch+1}.pth"
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    },
                    checkpoint_path,
                    _use_new_zipfile_serialization=True
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Set environment variables for better debugging
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    train_sam_on_lidar("config.yaml")
