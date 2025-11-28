# evaluate_snr.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# PILOT_FREQ_STEP を追加でインポート
from common_utils import SionnaChannelGeneratorGPU, ConditionalUNet, device, UNKNOWN_SNR_VALUE, PILOT_FREQ_STEP
from inference import sample_ddpm, compute_nmse

def estimate_channel_covariance(generator, num_batches=50):
    """チャネル統計量(R_hh, mu_h)の推定"""
    print(f"Estimating Channel Statistics (R_hh) using {num_batches} batches...")
    h_samples = []
    with torch.no_grad():
        for _ in range(num_batches):
            x_gt, _ = generator.get_batch(snr_db=100.0)
            B, C_combined, T, F = x_gt.shape
            C = C_combined // 2
            h_complex = x_gt[:, :C, :, :] + 1j * x_gt[:, C:, :, :]
            h_flat = h_complex.permute(0, 1, 2, 3).reshape(-1, T * F)
            h_samples.append(h_flat)
        
    h_all = torch.cat(h_samples, dim=0)
    mu_h = h_all.mean(dim=0)
    h_centered = h_all - mu_h.unsqueeze(0)
    N_samples = h_centered.shape[0]
    R_hh = (h_centered.T.conj() @ h_centered) / N_samples
    print("Channel statistics estimated.")
    return R_hh, mu_h

def compute_lmmse(x_cond, generator, R_hh, mu_h, snr_db):
    """
    間引きパイロット対応版 LMMSE推定
    """
    pilot_indices_time = generator.pilot_indices # [2, 11]
    B, C_combined, T, F = x_cond.shape
    C = C_combined // 2
    
    # x_cond は既に補間されているが、パイロット位置の値は観測値に近い
    y_full = x_cond[:, :C, :, :] + 1j * x_cond[:, C:, :, :] 
    y_flat = y_full.permute(0, 1, 2, 3).reshape(-1, T * F)
    
    # ★変更点: 周波数方向の間引きを考慮したインデックス作成
    pilot_flat_indices = []
    
    # 0, 4, 8... のサブキャリアインデックス
    freq_indices = np.arange(0, F, PILOT_FREQ_STEP)
    
    for t_idx in pilot_indices_time:
        # 時間 t_idx の中で、指定された周波数インデックスのみを選択
        indices = freq_indices + t_idx * F
        pilot_flat_indices.extend(indices)
    
    pilot_idx_tensor = torch.tensor(pilot_flat_indices, device=device).long()
    
    # 観測ベクトル y_p抽出
    y_p = y_flat[:, pilot_idx_tensor]
    
    # LMMSE行列構築
    R_pp = R_hh[pilot_idx_tensor][:, pilot_idx_tensor]
    R_hp = R_hh[:, pilot_idx_tensor]
    
    sig_pwr = torch.real(torch.diagonal(R_hh).mean())
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / snr_linear
    
    eye = torch.eye(R_pp.shape[0], device=device, dtype=R_pp.dtype)
    # ノイズ項を加算して逆行列
    inv_term = torch.linalg.inv(R_pp + noise_pwr * eye)
    W = R_hp @ inv_term
    
    # 推定
    mu_p = mu_h[pilot_idx_tensor]
    y_p_centered = y_p - mu_p.unsqueeze(0)
    h_est_centered = W @ y_p_centered.T
    h_est_flat = h_est_centered.T + mu_h.unsqueeze(0)
    
    h_est_complex = h_est_flat.reshape(B, C, T, F)
    x_est = torch.cat([h_est_complex.real, h_est_complex.imag], dim=1).float()
    
    return x_est

def evaluate_snr_vs_nmse():
    # --- 設定 ---
    SNR_LIST = np.arange(-5, 35, 5) 
    BATCH_SIZE = 16 
    # ★重要: 再学習したモデルのパスを指定してください
    MODEL_PATH = "checkpoints/ckpt_step_200000.pth" 
    
    print(f"Loading model from {MODEL_PATH}...")
    
    model = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found.")
        return

    generator = SionnaChannelGeneratorGPU(batch_size=BATCH_SIZE)
    
    # LMMSE用統計情報
    R_hh, mu_h = estimate_channel_covariance(generator, num_batches=50)
    
    nmse_linear_history = []
    nmse_lmmse_history = []
    nmse_diff_history = []
    
    print(f"Starting Evaluation over SNRs: {SNR_LIST} dB (Pilot Freq Step={PILOT_FREQ_STEP})")
    
    for snr_db in SNR_LIST:
        print(f"\nEvaluating at True SNR = {snr_db} dB...")
        
        # 1. データ生成 (間引き済み)
        x_gt, x_cond = generator.get_batch(snr_db=float(snr_db))
        
        # 2. Diffusion (Blind)
        blind_snr_tensor = torch.full((BATCH_SIZE,), UNKNOWN_SNR_VALUE, device=device)
        x_diff = sample_ddpm(model, x_cond, x_gt.shape, device, blind_snr_tensor)
        
        # 3. LMMSE (Ideal)
        x_lmmse = compute_lmmse(x_cond, generator, R_hh, mu_h, float(snr_db))
        
        nmse_linear = compute_nmse(x_gt, x_cond)
        nmse_lmmse = compute_nmse(x_gt, x_lmmse)
        nmse_diff = compute_nmse(x_gt, x_diff)
        
        nmse_linear_history.append(nmse_linear)
        nmse_lmmse_history.append(nmse_lmmse)
        nmse_diff_history.append(nmse_diff)
        
        print(f"  -> Linear NMSE : {nmse_linear:.4f} dB")
        print(f"  -> LMMSE NMSE  : {nmse_lmmse:.4f} dB")
        print(f"  -> DiffModel   : {nmse_diff:.4f} dB")

    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_LIST, nmse_linear_history, marker='o', linestyle=':', label='Linear Interpolation (Sparse)', color='tab:orange', alpha=0.8)
    plt.plot(SNR_LIST, nmse_lmmse_history, marker='^', linestyle='--', label='LMMSE (Ideal)', color='tab:green')
    plt.plot(SNR_LIST, nmse_diff_history, marker='s', linestyle='-', label='Diffusion Model (Blind)', color='tab:blue', linewidth=2)
    
    plt.title(f"NMSE vs SNR (Freq Sparsity: 1/{PILOT_FREQ_STEP})")
    plt.xlabel("SNR [dB]")
    plt.ylabel("NMSE [dB]")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(SNR_LIST)
    
    save_filename = "snr_vs_nmse_sparse.png"
    plt.savefig(save_filename)
    print(f"\nEvaluation finished. Plot saved to {save_filename}")
    plt.show()

if __name__ == "__main__":
    evaluate_snr_vs_nmse()