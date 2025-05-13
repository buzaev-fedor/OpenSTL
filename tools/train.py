# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)


class MetricLogger(BaseExperiment):
    """Extension of BaseExperiment to log metrics at specified intervals"""
    
    def __init__(self, args):
        super().__init__(args)
        self.eval_interval = args.eval_interval
        # Ensure required attributes exist
        if not hasattr(self, 'start_epoch'):
            self.start_epoch = 0
        if not hasattr(self, 'epochs'):
            self.epochs = args.epoch
        
    def train(self):
        """Override train method to evaluate and log metrics periodically"""
        rank, _ = get_dist_info()
        
        # Get original training method to ensure we have access to all necessary variables
        # This helps us maintain compatibility with the parent class
        orig_train_method = super().train
        
        # If parent has train_one_epoch, use it; otherwise define our own loop
        if hasattr(self, 'train_one_epoch'):
            for epoch in range(self.start_epoch, self.epochs):
                # Run regular training for this epoch
                self.train_one_epoch(epoch)
                
                # Check if we should evaluate at this epoch
                if self.eval_interval > 0 and (epoch + 1) % self.eval_interval == 0:
                    if rank == 0:
                        print(f"Evaluating at epoch {epoch+1}...")
                    if hasattr(self, 'validate'):
                        metrics = self.validate(epoch)
                        
                        # Log to Comet if enabled
                        if hasattr(self, 'logger') and self.args.use_comet:
                            for metric_name, metric_value in metrics.items():
                                self.logger.log_metric(f'val_{metric_name}', metric_value, step=epoch+1)
            
            # Finalize training
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            if hasattr(self, 'logger'):
                self.logger.finalize()
        else:
            # If no train_one_epoch method exists, use the parent's train method
            # but we'll lose the periodic evaluation capability
            print("Warning: Falling back to parent's train method; periodic eval disabled.")
            orig_train_method()


if __name__ == '__main__':
    args = create_parser().parse_args()
    # Add evaluation interval parameter
    args.eval_interval = getattr(args, 'eval_interval', 5)  # Default to every 5 epochs
    config = args.__dict__

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    print('>'*35 + ' training ' + '<'*35)
    # Use the new MetricLogger class instead of BaseExperiment
    exp = MetricLogger(args) if not args.test else BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()