# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# common_utils.py からクラスと定数をインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, device, UNKNOWN_SNR_VALUE

def save_checkpoint(model, optimizer, scheduler, step, loss_history, filepath):
    """チェックポイント保存用関数（スケジューラも保存）"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(), 
        'loss_history': loss_history
    }
    torch.save(checkpoint, filepath)

def train():
    parser = argparse.ArgumentParser(description="Train DDPM for Channel Estimation with Resume & Blind SNR")
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    BATCH_SIZE = 256  
    TOTAL_STEPS = 100000 
    LR = 1e-4
    MIN_LR = 1e-6           
    SNR_DROPOUT_PROB = 0.15 
    
    os.makedirs("checkpoints", exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    
    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    model = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=TOTAL_STEPS, 
        eta_min=MIN_LR
    )
    
    T = 100
    beta = torch.linspace(1e-4, 0.02, T).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    loss_history = []
    start_step = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            start_step = checkpoint['step']
            loss_history = checkpoint.get('loss_history', [])
            print(f"Resumed from step {start_step}.")
        else:
            print(f"Error: Checkpoint file {args.resume} not found.")
            return

    print(f"Starting Training on {device} from step {start_step} to {TOTAL_STEPS}...")
    model.train()
    
    pbar = tqdm(range(start_step, TOTAL_STEPS), initial=start_step, total=TOTAL_STEPS)
    
    for step in pbar:
        current_snr_val = np.random.uniform(-5.0, 30.0)
        
        x_gt, x_cond = generator.get_batch(snr_db=current_snr_val)
        
        t = torch.randint(0, T, (BATCH_SIZE,)).to(device)
        epsilon = torch.randn_like(x_gt)
        
        sqrt_ab = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_onem_ab = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
        
        x_noisy = sqrt_ab * x_gt + sqrt_onem_ab * epsilon
        
        snr_tensor = torch.full((BATCH_SIZE,), current_snr_val, device=device)
        if np.random.rand() < SNR_DROPOUT_PROB:
            snr_tensor.fill_(UNKNOWN_SNR_VALUE)

        pred_noise = model(x_noisy, x_cond, t, snr_tensor)
        loss = mse(pred_noise, epsilon)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Loss: {loss_val:.5f} | LR: {current_lr:.2e}")
        
        if (step + 1) % 5000 == 0:
            save_path = f"checkpoints/ckpt_step_{step+1}.pth"
            save_checkpoint(model, optimizer, scheduler, step + 1, loss_history, save_path)
            
            plt.figure()
            plt.plot(loss_history)
            plt.title("Training Loss")
            plt.yscale('log')
            plt.savefig("loss_curve.png")
            plt.close()

    save_checkpoint(model, optimizer, scheduler, TOTAL_STEPS, loss_history, "diff_model_final.pth")
    print("Training Finished. Model saved.")
    
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig("loss_curve.png")

if __name__ == "__main__":
    train()