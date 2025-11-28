# common_utils.py
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

# 高速化: DLPackを使ってGPUメモリ間でデータを渡すためのインポート
from torch.utils.dlpack import from_dlpack

# Sionna関連のインポート
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# --- 定数設定 ---
# SNRが不明であることを示す値
UNKNOWN_SNR_VALUE = -100.0

# ★変更点: パイロットの周波数方向の間引き間隔
# 8サブキャリアごとに1つだけパイロットを配置（かなりスカスカにする）
PILOT_FREQ_STEP = 8 

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorFlowのGPUメモリ確保設定
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # TFがメモリを食いつぶさないようにGrowthモードにする
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# ==========================================
# 1. データ生成クラス (Sparse Pilot対応版)
# ==========================================
class SionnaChannelGeneratorGPU:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.pilot_indices = [2, 11] # 時間方向のパイロット位置
        self.num_symbols = 14
        
        # --- Sionna設定 ---
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_symbols,
            fft_size=76, subcarrier_spacing=15e3,
            num_tx=1, num_streams_per_tx=4,
            cyclic_prefix_length=6, pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.pilot_indices
        )
        carrier_freq = 2.6e9
        ut_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        
        # ★変更点: 遅延広がり(Delay Spread)を 300ns -> 3000ns に拡大
        # これにより周波数選択性が強くなり、単純な線形補間が難しくなる
        self.cdl = CDL("B", 3000e-9, carrier_freq, ut_array, bs_array, "uplink", min_speed=10)
        
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)
        
        # 線形補間用の時間グリッド (GPU)
        self.t_grid = torch.arange(self.num_symbols, device=device).view(1, 1, -1, 1).float()

    def get_batch(self, snr_db=10.0):
        # 1. Sionnaで真のチャネル生成 (全周波数・全時間)
        a, tau = self.cdl(batch_size=self.batch_size, 
                          num_time_steps=self.rg.num_ofdm_symbols, 
                          sampling_frequency=1/self.rg.ofdm_symbol_duration)
        
        h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        
        # DLPackによるゼロコピー転送
        try:
            h_torch_complex = from_dlpack(tf.experimental.dlpack.to_dlpack(h_freq))
            h_torch_complex = h_torch_complex.cfloat() # 型変換
        except Exception:
            h_torch_complex = torch.from_numpy(h_freq.numpy()).cfloat().to(device)

        if h_torch_complex.device != device:
             h_torch_complex = h_torch_complex.to(device)
        
        # Reshape: [Batch, Rx, RxAnt, Tx, TxAnt, T, F] -> [Batch, Channels, T, F]
        b, rx, rx_ant, tx, tx_ant, t, f = h_torch_complex.shape
        h_gt_complex = h_torch_complex.view(b, rx*rx_ant*tx*tx_ant, t, f)

        # 2. パイロット抽出とノイズ付加
        t1, t2 = self.pilot_indices[0], self.pilot_indices[1]
        val_t1_clean = h_gt_complex[:, :, t1:t1+1, :]
        val_t2_clean = h_gt_complex[:, :, t2:t2+1, :]
        
        sig_pwr = torch.mean(torch.abs(h_gt_complex)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / snr_linear
        noise_std = torch.sqrt(noise_pwr / 2.0)
        
        noise_n1 = (torch.randn_like(val_t1_clean) + 1j * torch.randn_like(val_t1_clean)) * noise_std
        noise_n2 = (torch.randn_like(val_t2_clean) + 1j * torch.randn_like(val_t2_clean)) * noise_std
        
        val_t1_noisy_full = val_t1_clean + noise_n1
        val_t2_noisy_full = val_t2_clean + noise_n2

        # ==========================================
        # ★変更点: 周波数方向の間引き (Sparsification)
        # ==========================================
        # 0, 8, 16... のインデックスのみを取得
        f_indices = torch.arange(0, self.rg.fft_size, PILOT_FREQ_STEP, device=device)
        
        # 間引かれたパイロットデータ
        val_t1_sparse = val_t1_noisy_full[:, :, :, f_indices] # [B, C, 1, F_sparse]
        val_t2_sparse = val_t2_noisy_full[:, :, :, f_indices]

        # 3. 補間 (x_condの作成)
        # Step A: 時間方向の線形補間 (間引かれた周波数位置のみ)
        slope_sparse = (val_t2_sparse - val_t1_sparse) / (t2 - t1)
        h_sparse_time_interp = val_t1_sparse + slope_sparse * (self.t_grid - t1) # [B, C, T, F_sparse]
        
        # Step B: 周波数方向の引き伸ばし (Bilinear Interpolation)
        # 間引かれたデータを元のサイズ(76)に引き伸ばして入力ヒントとする
        h_real = h_sparse_time_interp.real
        h_imag = h_sparse_time_interp.imag
        
        target_size = (self.rg.num_ofdm_symbols, self.rg.fft_size)
        
        # align_corners=True で位置ズレを最小限に抑えつつリサイズ
        cond_real = torch.nn.functional.interpolate(h_real, size=target_size, mode='bilinear', align_corners=True)
        cond_imag = torch.nn.functional.interpolate(h_imag, size=target_size, mode='bilinear', align_corners=True)
        
        h_cond_complex = torch.complex(cond_real, cond_imag)

        # 4. 実部・虚部の結合
        x_gt = torch.cat([h_gt_complex.real, h_gt_complex.imag], dim=1).float()
        x_cond = torch.cat([h_cond_complex.real, h_cond_complex.imag], dim=1).float()
        
        return x_gt, x_cond

# ==========================================
# 2. モデル定義 (変更なし)
# ==========================================
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=64, cond_channels=64, time_emb_dim=32):
        super().__init__()
        total_in_channels = in_channels + cond_channels
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, 512)

        # SNR Embedding
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.snr_proj = nn.Linear(time_emb_dim, 512)
        
        self.down1 = nn.Conv2d(total_in_channels, 128, 3, padding=1)
        self.down2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        self.bot1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = nn.Conv2d(256 + 128, 256, 3, padding=1) 
        self.conv2 = nn.Conv2d(256, in_channels, 3, padding=1)
        
        self.act = nn.GELU()

    def forward(self, x, cond, t, snr): 
        x_in = torch.cat([x, cond], dim=1) 
        
        # Time Embedding
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).view(-1, 512, 1, 1)

        # SNR Embedding
        s = snr.float().view(-1, 1) / 30.0 
        s_emb = self.snr_mlp(s)
        s_emb = self.snr_proj(s_emb).view(-1, 512, 1, 1)
        
        x1 = self.act(self.down1(x_in))
        x2 = self.act(self.down2(self.pool(x1)))
        
        x_bot = self.act(self.bot1(x2) + t_emb + s_emb)
        x_bot = self.act(self.bot2(x_bot))
        
        x_up = self.up1(x_bot)
        if x_up.shape != x1.shape:
             x_up = torch.nn.functional.interpolate(x_up, size=x1.shape[2:])
        
        x_dec = torch.cat([x_up, x1], dim=1)
        x_dec = self.act(self.conv1(x_dec))
        out = self.conv2(x_dec)
        
        return out