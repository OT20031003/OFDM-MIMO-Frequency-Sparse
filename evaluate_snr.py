import os
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sionna imports
from sionna.phy import utils as phy_utils
from sionna.phy.ofdm import ResourceGrid, LSChannelEstimator, LMMSEInterpolator
from sionna.phy.channel import GenerateOFDMChannel, subcarrier_frequencies, cir_to_ofdm_channel
from sionna.phy.channel.tr38901 import AntennaArray, CDL

# Project imports
from common_utils import ConditionalUNet, device, UNKNOWN_SNR_VALUE
from inference import sample_ddpm, compute_nmse

# GPUメモリ制御
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def get_sionna_config():
    """Sionnaの設定を構築し、パイロット位置などのパラメータも返す"""
    CARRIER_FREQ = 2.6e9
    NUM_SYMBOLS = 14
    FFT_SIZE = 76
    SUBCARRIER_SPACING = 15e3
    PILOT_INDICES = [2, 11]  # これを返します
    PILOT_FREQ_STEP = 8
    
    pilot_pattern = "kronecker" 
    rg = ResourceGrid(
        num_ofdm_symbols=NUM_SYMBOLS,
        fft_size=FFT_SIZE,
        subcarrier_spacing=SUBCARRIER_SPACING,
        num_tx=1, num_streams_per_tx=4,
        cyclic_prefix_length=6, 
        pilot_pattern=pilot_pattern,
        pilot_ofdm_symbol_indices=PILOT_INDICES
    )
    
    ut_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", 
                            antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ)
    bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", 
                            antenna_pattern="38.901", carrier_frequency=CARRIER_FREQ)
    
    cdl_model = CDL("B", 3000e-9, CARRIER_FREQ, ut_array, bs_array, "uplink", min_speed=10)
    
    # ★変更: PILOT_INDICES を戻り値に追加
    return rg, cdl_model, PILOT_FREQ_STEP, PILOT_INDICES

def estimate_covariance_matrices(channel_model, rg, num_batches=20, batch_size=64):
    print("Estimating covariance matrices (Frequency, Time, Space)...")
    
    channel_sampler = GenerateOFDMChannel(channel_model, rg)
    
    fft_size = rg.fft_size
    num_ofdm_symbols = rg.num_ofdm_symbols
    num_spatial_channels = 16 
    
    freq_cov = tf.zeros([fft_size, fft_size], dtype=tf.complex64)
    time_cov = tf.zeros([num_ofdm_symbols, num_ofdm_symbols], dtype=tf.complex64)
    space_cov = tf.zeros([num_spatial_channels, num_spatial_channels], dtype=tf.complex64)
    
    for _ in tqdm(range(num_batches)):
        h = channel_sampler(batch_size)
        h = tf.reshape(h, [batch_size, num_spatial_channels, num_ofdm_symbols, fft_size])
        h = tf.cast(h, tf.complex64)
        
        # --- 周波数共分散 ---
        h_perm = tf.transpose(h, [0, 1, 2, 3])
        h_flat_f = tf.reshape(h_perm, [-1, fft_size])
        cov_f = tf.matmul(h_flat_f, h_flat_f, adjoint_a=True) / tf.cast(tf.shape(h_flat_f)[0], tf.complex64)
        freq_cov += cov_f

        # --- 時間共分散 ---
        h_perm = tf.transpose(h, [0, 1, 3, 2])
        h_flat_t = tf.reshape(h_perm, [-1, num_ofdm_symbols])
        cov_t = tf.matmul(h_flat_t, h_flat_t, adjoint_a=True) / tf.cast(tf.shape(h_flat_t)[0], tf.complex64)
        time_cov += cov_t
        
        # --- 空間共分散 ---
        h_perm = tf.transpose(h, [0, 2, 3, 1])
        h_flat_s = tf.reshape(h_perm, [-1, num_spatial_channels])
        cov_s = tf.matmul(h_flat_s, h_flat_s, adjoint_a=True) / tf.cast(tf.shape(h_flat_s)[0], tf.complex64)
        space_cov += cov_s

    return (freq_cov / num_batches), (time_cov / num_batches), (space_cov / num_batches)

# ★変更: pilot_indices を引数に追加
def run_lmmse_estimation(y_sparse_grid, no, rg, time_cov, freq_cov, space_cov, pilot_freq_step, pilot_indices):
    """
    SionnaのLMMSEInterpolatorロジックを利用して補間・平滑化を行う
    """
    # 1. マスク作成
    pilot_mask = np.zeros((14, 76), dtype=bool)
    pilot_mask[pilot_indices, ::pilot_freq_step] = True
    
    # 2. LMMSEInterpolatorのインスタンス化
    # 【修正点】引数名（time_cov= など）を削除し、ドキュメント通り順番に渡します。
    # 順番: pilot_pattern, time_cov, freq_cov, space_cov
    lmmse_interpolator = LMMSEInterpolator(
        rg.pilot_pattern, 
        time_cov, 
        freq_cov, 
        space_cov, 
        order='t-f-s' 
    )
    
    # 入力データをTFに変換
    h_hat = tf.convert_to_tensor(y_sparse_grid.detach().cpu().numpy(), dtype=tf.complex64)
    no_tf = tf.cast(no, tf.float32)
    
    # 空間スムージングの手動適用（ここは変更なし）
    h_perm = tf.transpose(h_hat, [0, 2, 3, 1]) # [B, T, F, S]
    eye = tf.eye(16, dtype=tf.complex64)
    sigma2 = tf.complex(no_tf, 0.0)
    
    inv_term = tf.linalg.inv(space_cov + sigma2 * eye)
    W_space = tf.matmul(space_cov, inv_term)
    
    h_smoothed = tf.matmul(h_perm, tf.transpose(W_space)) 
    
    h_final = tf.transpose(h_smoothed, [0, 3, 1, 2])
    
    return torch.from_numpy(h_final.numpy())
def evaluate():
    MODEL_PATH = "diff_model_final.pth" # 必要に応じてパスを変更してください
    
    # ★変更: 戻り値を受け取る
    rg, cdl_model, pilot_freq_step, pilot_indices = get_sionna_config()
    
    freq_cov, time_cov, space_cov = estimate_covariance_matrices(cdl_model, rg, num_batches=10)
    
    model = ConditionalUNet(in_channels=32, cond_channels=32).to(device)
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("Model loaded.")
    else:
        print(f"Model file {MODEL_PATH} not found. Testing with random weights.")

    snr_list = np.arange(0, 35, 5)
    nmse_model_list = []
    nmse_lmmse_list = []
    
    from common_utils import SionnaChannelGeneratorGPU
    generator = SionnaChannelGeneratorGPU(batch_size=32)
    
    print("\nStarting comparison...")
    for snr in snr_list:
        x_gt, x_cond = generator.get_batch(snr_db=float(snr))
        
        # --- A. 学習モデル (Diffusion) ---
        blind_snr = torch.full((x_cond.shape[0],), UNKNOWN_SNR_VALUE, device=device)
        x_pred_diff = sample_ddpm(model, x_cond, x_gt.shape, device, blind_snr)
        
        # --- B. LMMSE (Spatial Smoothing) ---
        C = x_cond.shape[1] // 2
        x_cond_complex = x_cond[:, :C, :, :] + 1j * x_cond[:, C:, :, :]
        no = 10**(-snr/10.0) 
        
        # ★変更: pilot_indices を渡す
        x_pred_lmmse_c = run_lmmse_estimation(
            x_cond_complex, no, rg, time_cov, freq_cov, space_cov, 
            pilot_freq_step, pilot_indices
        )
        x_pred_lmmse = torch.cat([x_pred_lmmse_c.real, x_pred_lmmse_c.imag], dim=1).to(device)
        
        nmse_model = compute_nmse(x_gt, x_pred_diff)
        nmse_lmmse = compute_nmse(x_gt, x_pred_lmmse)
        
        nmse_model_list.append(nmse_model)
        nmse_lmmse_list.append(nmse_lmmse)
        
        print(f"SNR {snr}dB | Diffusion NMSE: {nmse_model:.2f} dB | LMMSE(Smooth) NMSE: {nmse_lmmse:.2f} dB")
        
    plt.figure(figsize=(8, 6))
    plt.plot(snr_list, nmse_model_list, 'o-', label='Diffusion Model')
    plt.plot(snr_list, nmse_lmmse_list, 's--', label='LMMSE (Spatial Smoothing)')
    plt.xlabel('SNR [dB]')
    plt.ylabel('NMSE [dB]')
    plt.title('Channel Estimation Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_result.png')
    print("Plot saved to comparison_result.png")
    plt.show()

if __name__ == "__main__":
    evaluate()