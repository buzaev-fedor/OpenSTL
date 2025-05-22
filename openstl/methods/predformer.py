import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm

from openstl.models import PredFormer_Model
from openstl.utils import print_log
from openstl.core import metric
from .base_method import Base_method

class PredFormer(Base_method):
    """PredFormer: A Transformer model with MoE support for spatiotemporal prediction."""
    
    def __init__(self, **args):
        Base_method.__init__(self, **args)
        
        # MoE related parameters
        if 'model_config' in args and 'use_moe' in args['model_config']:
            self.use_moe = args['model_config']['use_moe']
            self.moe_loss_weight = args['model_config'].get('moe_loss_weight', 0.01)
        else:
            self.use_moe = args.get('use_moe', False)
            self.moe_loss_weight = args.get('moe_loss_weight', 0.01)
    
    def _build_model(self, **args):
        """Build the PredFormer model."""
        # Get input shape information
        in_shape = args.get('in_shape', None)
        if in_shape is not None:
            args['height'] = in_shape[2]  # Height from in_shape [seq_len, channels, height, width]
            args['width'] = in_shape[3]   # Width from in_shape
            args['num_channels'] = in_shape[1]  # Channels from in_shape
        
        # Get or set defaults for dimensions
        height = args.get('height', 64)
        width = args.get('width', 64)
        num_channels = args.get('num_channels', 1)
        
        # Get sequence lengths
        pre_seq_length = args.get('pre_seq_length', 10)
        aft_seq_length = args.get('aft_seq_length', 10)
        
        # Extract model_config if available (from config file)
        model_params = {}
        if 'model_config' in args:
            model_params = args['model_config']
            
        # Build model config
        model_config = {
            'height': height,
            'width': width,
            'num_channels': num_channels,
            'pre_seq': pre_seq_length,
            'after_seq': aft_seq_length,
            'patch_size': model_params.get('patch_size', 8),
            'dim': model_params.get('dim', 256),
            'heads': model_params.get('heads', 8),
            'dim_head': model_params.get('dim_head', 32),
            'dropout': model_params.get('dropout', 0.0),
            'attn_dropout': model_params.get('attn_dropout', 0.0),
            'drop_path': model_params.get('drop_path', 0.0),
            'scale_dim': model_params.get('scale_dim', 4),
            'depth': model_params.get('depth', 1),
            'Ndepth': model_params.get('Ndepth', 6),
            'use_moe': model_params.get('use_moe', False),
            'num_experts': model_params.get('num_experts', 8),
            'top_k': model_params.get('top_k', 2),
            'noisy_gate': model_params.get('noisy_gate', True),
            'gate_noise': model_params.get('gate_noise', 0.1),
            'moe_loss_weight': model_params.get('moe_loss_weight', 0.01)
        }
        
        return PredFormer_Model(**model_config)
    
    def forward(self, batch_x, batch_y=None):
        """Forward the model for prediction.
        
        Args:
            batch_x (torch.Tensor): Input tensor of shape [batch_size, in_len, channel, height, width]
            batch_y (torch.Tensor, optional): Target tensor for teacher forcing. Defaults to None.
        """
        # For autoregressive prediction when aft_seq_length > pre_seq_length
        if hasattr(self.hparams, 'aft_seq_length') and hasattr(self.hparams, 'pre_seq_length'):
            if self.hparams.aft_seq_length == self.hparams.pre_seq_length:
                pred_y = self.model(batch_x)
            elif self.hparams.aft_seq_length < self.hparams.pre_seq_length:
                pred_y = self.model(batch_x)
                pred_y = pred_y[:, :self.hparams.aft_seq_length]
            elif self.hparams.aft_seq_length > self.hparams.pre_seq_length:
                pred_y = []
                d = self.hparams.aft_seq_length // self.hparams.pre_seq_length
                m = self.hparams.aft_seq_length % self.hparams.pre_seq_length
                
                cur_seq = batch_x.clone()
                for i in range(d):
                    cur_seq = self.model(cur_seq)
                    pred_y.append(cur_seq)

                if m != 0:
                    cur_seq = self.model(cur_seq)
                    pred_y.append(cur_seq[:, :m])
                
                pred_y = torch.cat(pred_y, dim=1)
        else:
            pred_y = self.model(batch_x)
        
        return pred_y
    
    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        start_time = time.time()
        batch_x, batch_y = batch
        
        # Forward pass
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        
        # Add MoE auxiliary loss if using MoE
        moe_loss = torch.tensor(0.0, device=pred_y.device)
        if self.use_moe and hasattr(self.model, 'get_moe_loss'):
            moe_loss = self.model.get_moe_loss()
            if not torch.isnan(moe_loss) and not torch.isinf(moe_loss):
                # Scale and add MoE loss to the main loss
                loss = loss + self.moe_loss_weight * moe_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        if self.use_moe:
            self.log('train_moe_loss', moe_loss, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        batch_x, batch_y = batch
        
        # Forward pass
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        
        # Calculate MoE loss but don't add it to the main loss during validation
        if self.use_moe and hasattr(self.model, 'get_moe_loss'):
            moe_loss = self.model.get_moe_loss()
            self.log('val_moe_loss', moe_loss, prog_bar=False)
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Lightning test step."""
        batch_x, batch_y = batch
        
        # Forward pass
        pred_y = self(batch_x, batch_y)
        
        # Store results for metric calculation in on_test_epoch_end
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy()
        }
        
        return outputs 