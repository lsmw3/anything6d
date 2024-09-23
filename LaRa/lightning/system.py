
import torch
import numpy as np
from lightning.loss import Losses
import pytorch_lightning as L
import wandb

import torchvision
from torchvision import transforms

import torch.nn as nn
from lightning.vis import vis_images
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.utils import CosineWarmupScheduler

from lightning.network import Network

class system(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.loss = Losses()
        self.net = Network(cfg)

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):  
        output = self.net(batch, with_fine=self.global_step>self.cfg.train.start_fine)
        loss, scalar_stats = self.loss(batch, output, self.global_step)
        for key, value in scalar_stats.items():
            prog_bar = True if key in ['psnr','mask','depth'] else False
            self.log(f'train/{key}', value, prog_bar=prog_bar)
        self.log('lr',self.trainer.optimizers[0].param_groups[0]['lr'])
        
        self.log('train loss', loss)
        
        # log the gt rgb and output
        if batch_idx % 5 == 0:
            output_rgb = output['image_fine'].detach().cpu().numpy() if 'image_fine' in output else output['image'].detach().cpu().numpy()
            gt_rgb = batch['suv_rgb'].detach().cpu().numpy() if 'suv_rgb' in batch else batch['tar_rgb'].detach().cpu().numpy()
            
            B,V,H,W,C = gt_rgb.shape
            output_rgb = output_rgb.reshape(B, H, V, W, C).transpose(0, 2, 1, 3, 4)
            
            # gt_grid = torchvision.utils.make_grid(gt_rgb, normalize=True)
            # output_grid = torchvision.utils.make_grid(output_rgb, normalize=True)

            # self.logger.experiment.add_image('Ground Truth', gt_grid, self.global_step)
            # self.logger.experiment.add_image('Output', output_grid, self.global_step)
            
            for idx in range(B):
                log_dict = {
                    "Ground Truth": [wandb.Image(img, caption=f"Ground Truth {idx}") for img in gt_rgb[idx]],
                    "Model Output": [wandb.Image(img, caption=f"Model Output {idx}") for img in output_rgb[idx]]
                }
                self.logger.experiment.log(log_dict)

        if 0 == self.trainer.global_step % 3000  and (self.trainer.local_rank == 0):
            self.vis_results(output, batch, prex='train')
            
        torch.cuda.empty_cache()

        return loss

    def validation_step(self, batch, batch_idx):
        self.net.eval()
        output = self.net(batch, with_fine=self.global_step>self.cfg.train.start_fine)
        loss, scalar_stats = self.loss(batch, output, self.global_step)
        if batch_idx == 0 and (self.trainer.local_rank == 0):
            self.vis_results(output, batch, prex='val')
        self.validation_step_outputs.append(scalar_stats)
        
        # self.log('val loss', loss)
        
        return loss

    def on_validation_epoch_end(self):
        keys = self.validation_step_outputs[0]
        for key in keys:
            prog_bar = True if key in ['psnr','mask','depth'] else False
            metric_mean = torch.stack([x[key] for x in self.validation_step_outputs]).mean()
            self.log(f'val/{key}', metric_mean, prog_bar=prog_bar, sync_dist=True)

        self.validation_step_outputs.clear()  # free memory
        torch.cuda.empty_cache()

    def vis_results(self, output, batch, prex):
        output_vis = vis_images(output, batch)
        for key, value in output_vis.items():
            if isinstance(self.logger, TensorBoardLogger):
                B,h,w = value.shape[:3]
                value = value.reshape(1,B*h,w,3).transpose(0,3,1,2)
                self.logger.experiment.add_images(f'{prex}/{key}', value, self.global_step)
            else:
                imgs = [np.concatenate([img for img in value],axis=0)]
                self.logger.log_image(f'{prex}/{key}', imgs, step=self.global_step)
        self.net.train()

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs * self.cfg.train.limit_train_batches // (self.trainer.accumulate_grad_batches * num_devices)
        return int(num_steps)

    def configure_optimizers(self):
        decay_params, no_decay_params = [], []

        # add all bias and LayerNorm params to no_decay_params
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)

        # add remaining parameters to decay_params
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in self.parameters() if id(p) not in _no_decay_ids]

        # filter out parameters with no grad
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))

        # Optimizer
        opt_groups = [
            {'params': decay_params, 'weight_decay': self.cfg.train.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=self.cfg.train.lr,
            betas=(self.cfg.train.beta1, self.cfg.train.beta2),
        )

        total_global_batches = self.num_steps()
        scheduler = CosineWarmupScheduler(
                        optimizer=optimizer,
                        warmup_iters=self.cfg.train.warmup_iters,
                        max_iters=2 * total_global_batches,
                    )

        return {"optimizer": optimizer, 
                "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step'  # or 'epoch' for epoch-level updates
            }}