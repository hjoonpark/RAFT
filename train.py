from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import core.datasets as datasets
import core.loss as loss_fn

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def flow_warp(x, flow, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW
    base_grid = torch.stack([x_base, y_base], 1).to(x.device)  # B2HW
    _, _, H, W = base_grid.size()

    # scale grid to [-1,1]
    v_grid = base_grid + flow
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    v_grid_norm = v_grid_norm.permute(0, 2, 3, 1)  # BHW2

    x_warped = F.grid_sample(x, v_grid_norm, mode=mode, padding_mode=pad)
    return x_warped


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: {:,}".format(count_parameters(model)))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print("Loaded model: {}".format(args.restore_ckpt))

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 1000
    add_noise = False

    plot_dir = f"logs/{args.stage}/plots"
    os.makedirs(plot_dir, exist_ok=1)

    w_smooth_reg = 0.01
    should_keep_training = True
    losses_all = {}
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow_true, _ = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)

            # warp frame 1 using flows
            loss = 0
            image2_preds, flow_preds = [], []
            for i in range(len(flow_predictions)):
                flow_pred = flow_predictions[i]
                image2_pred = flow_warp(image1, flow_pred)
                losses = loss_fn.compute_losses(image1, image2, flow_pred, image2_pred)
                
                image2_preds.append(image2_pred)
                flow_preds.append(flow_pred)
                l_photometric = losses["photometric"]
                l_smooth_reg = losses["smooth_reg"]

                loss += l_photometric + w_smooth_reg*l_smooth_reg

                if i == len(flow_predictions)-1:
                    for k, l in losses.items():
                        if k not in losses_all:
                            losses_all[k] = []
                        losses_all[k].append(l.item())

            # loss, metrics = loss_fn.compute_losses(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            if total_steps % 1000 == 0:
                plt.figure(figsize=(10,4))
                for k, l in losses_all.items():
                    plt.plot(np.arange(len(l)), l, label=k)
                plt.legend()
                plt.title("Iteration: {}".format(total_steps))
                plt.tight_layout()
                save_path = os.path.join(plot_dir, "losses.jpg")
                plt.savefig(save_path, dpi=150)
                print("[{}] {}".format(total_steps, save_path))
                plt.close()

                B = 0
                R = 3
                C = 3 + len(flow_predictions)
                I1 = image1[B].cpu().squeeze().permute(1,2,0).numpy()
                I2 = image2[B].cpu().squeeze().permute(1,2,0).numpy()

                fig = plt.figure(figsize=(60, 10))
                ax = fig.add_subplot(R, C, 1)
                ax.set_title("image 1 ({:.2f}, {:.2f})".format(I1.min(), I1.max()))
                I1b = (I1-I1.min())/(I1.max()-I1.min())
                ax.imshow(I1b)

                ax = fig.add_subplot(R, C, 2)
                ax.set_title("image 2 ({:.2f}, {:.2f})".format(I2.min(), I2.max()))
                I2b = (I2-I2.min())/(I2.max()-I2.min())
                ax.imshow(I2b)

                for i in range(len(flow_predictions)):
                    ax = fig.add_subplot(R, C, i + 3)
                    f = flow_predictions[i][B].detach().cpu().permute(1,2,0).numpy()
                    f = np.concatenate([f, np.zeros((f.shape[0], f.shape[1], 1))], -1)
                    ax.set_title("flow pred [{}]\n ({:.2f}, {:.2f})".format(i, f.min(), f.max()))
                    f = (f-f.min())/(f.max()-f.min())
                    ax.imshow(f)
                
                for i in range(len(image2_preds)):
                    ax = fig.add_subplot(R, C, C + i + 3)
                    f = image2_preds[i][B].detach().cpu().permute(1,2,0).numpy()
                    err = np.abs(I1-f)

                    ax.set_title("image2 pred [{}]\n ({:.2f}, {:.2f})".format(i, f.min(), f.max()))
                    f = (f-f.min())/(f.max()-f.min())
                    ax.imshow(f)

                    ax = fig.add_subplot(R, C, 2*C + i + 3)
                    ax.set_title("error: ({:.2f}, {:.2f})".format(err.min(), err.max()))
                    err = (err-err.min())/(err.max()-err.min())
                    ax.imshow(err)

                ax = fig.add_subplot(R, C, C)
                f = flow_true[B].detach().cpu().squeeze().permute(1,2,0).numpy()
                f = np.concatenate([f, np.zeros((f.shape[0], f.shape[1], 1))], -1)
                ax.set_title("flow true\n({:.2f}, {:.2f})".format(f.min(), f.max()))
                f = (f-f.min())/(f.max()-f.min())
                ax.imshow(f)
                plt.tight_layout()
                save_path = os.path.join(plot_dir, "plot_{:05d}.jpg".format(total_steps))
                plt.savefig(save_path, dpi=150)
                plt.close()
                print("[{}] {}".format(total_steps, save_path))

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%s.pth' % args.name
                torch.save(model.state_dict(), PATH)

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)