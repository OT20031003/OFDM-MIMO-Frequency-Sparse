# common_utils.py
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

# 高速化: DLPackを使ってGPUメモリ間でデータを渡すためのインポート
from torch.utils.dlpack import from_dlpack

# Sionna関連のインポート
from sionna.phy.ofdm import ResourceGrid, PilotPattern, LSChannelEstimator
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel

# --- 定数設定 ---
UNKNOWN_SNR_VALUE = -100.0
PILOT_FREQ_STEP = 8 

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorFlowのGPUメモリ確保設定
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# ==========================================
# 1. データ生成クラス (Sionna標準準拠版)
# ==========================================
class SionnaChannelGeneratorGPU:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.pilot_symbol_indices = [2, 11] # 時間方向のパイロット位置
        self.num_symbols = 14
        self.fft_size = 76
        self.num_tx = 1
        self.num_streams = 4 # num_streams_per_tx
        
        # --- Sionna設定: パイロットパターンの定義 ---
        # "kronecker"ではなく、明示的にマスクを作成してSionnaに認識させる
        
        # マスク形状: [num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        pilot_mask = np.zeros((self.num_tx, self.num_streams, self.num_symbols, self.fft_size), dtype=bool)
        
        # 指定されたシンボル(2, 11)の、指定された周波数間隔(8)の位置をTrueにする
        pilot_mask[:, :, self.pilot_symbol_indices, ::PILOT_FREQ_STEP] = True
        
        # カスタムパイロットパターンの作成
        self.pilot_pattern = PilotPattern(mask=pilot_mask, pilot_sequence="kronecker")
        
        # ResourceGridにカスタムパターンを渡す
        self.rg = ResourceGrid(
            num_ofdm_symbols=self.num_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=15e3,
            num_tx=self.num_tx, 
            num_streams_per_tx=self.num_streams,
            cyclic_prefix_length=6, 
            pilot_pattern=self.pilot_pattern # <--- ここが重要
        )
        
        # LS推定器 (線形補間モード) を準備
        # これにより、手動計算ではなくSionnaの機能で補間を行う
        self.ls_estimator = LSChannelEstimator(self.rg, interpolation_type="lin")

        carrier_freq = 2.6e9
        ut_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual", polarization_type="cross", antenna_pattern="38.901", carrier_frequency=carrier_freq)
        
        self.cdl = CDL("B", 3000e-9, carrier_freq, ut_array, bs_array, "uplink", min_speed=10)
        
        self.frequencies = subcarrier_frequencies(self.rg.fft_size, self.rg.subcarrier_spacing)

    def get_batch(self, snr_db=10.0):
        # 1. Sionnaで真のチャネル生成
        # Output: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
        a, tau = self.cdl(batch_size=self.batch_size, 
                          num_time_steps=self.rg.num_ofdm_symbols, 
                          sampling_frequency=1/self.rg.ofdm_symbol_duration)
        
        h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=True)
        
        # 2. LS推定用の受信信号(y)をシミュレート
        # y = h * x + n ですが、チャネル推定タスクではパイロット位置の x=1 と仮定して
        # y_pilot = h_pilot + n を作れば良い。
        
        # ノイズ生成
        sig_pwr = tf.reduce_mean(tf.abs(h_freq)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / tf.cast(snr_linear, h_freq.dtype)
        no = noise_pwr # ノイズ分散
        
        noise = tf.complex(
            tf.random.normal(tf.shape(h_freq), stddev=tf.sqrt(noise_pwr/2.0)),
            tf.random.normal(tf.shape(h_freq), stddev=tf.sqrt(noise_pwr/2.0))
        )
        
        # 観測信号 y (全リソースエレメントにノイズを乗せる)
        # ※SionnaのLSChannelEstimatorはパイロット位置の値だけをピックアップして使うため、
        # データ部分に何が入っていても（あるいはhそのままでも）推定には影響しません。
        y_obs = h_freq + noise

        # 3. Sionna標準のLS推定 + 線形補間を実行
        # これにより、手動でスライスや補間を書く必要がなくなります
        # shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_symbols, fft_size]
        h_hat, _ = self.ls_estimator(y_obs, no)
        
        # 4. PyTorch形式に変換 (学習用データの整形)
        # h_freq (Ground Truth) と h_hat (Conditional Input)
        
        # 形状を [Batch, Channels, T, F] に合わせる
        # Channels = Rx * RxAnt * Tx * TxAnt = 1 * 2 * 1 * 2 = 4 ?
        # 実際のCDL出力形状に依存しますが、ここではフラット化します
        
        # テンソルをGPU上のPyTorchへ変換
        x_gt = self._tf_to_torch(h_freq)
        x_cond = self._tf_to_torch(h_hat)
        
        return x_gt, x_cond

    def _tf_to_torch(self, tf_tensor):
        """TensorFlowテンソルを[B, C_real+imag, T, F]のPyTorchテンソルに変換"""
        # 複素数 -> 実部・虚部結合
        # 元の形状: [B, 1, RxAnt, 1, TxAnt, T, F] などを想定
        # ここでは全アンテナペアをチャンネル次元にまとめる
        
        # [B, ..., T, F] -> [B, -1, T, F]
        b = tf_tensor.shape[0]
        t = tf_tensor.shape[-2]
        f = tf_tensor.shape[-1]
        
        # 軸を入れ替えてフラット化の準備 [B, ..., T, F]
        # アンテナ次元を統合
        flat_tensor = tf.reshape(tf_tensor, [b, -1, t, f])
        
        try:
            # DLPackでゼロコピー変換
            torch_tensor = from_dlpack(tf.experimental.dlpack.to_dlpack(flat_tensor))
            torch_tensor = torch_tensor.cfloat()
        except Exception:
            torch_tensor = torch.from_numpy(flat_tensor.numpy()).cfloat().to(device)
            
        if torch_tensor.device != device:
             torch_tensor = torch_tensor.to(device)
             
        # 実部と虚部をチャンネル方向に結合 [B, C*2, T, F]
        x_out = torch.cat([torch_tensor.real, torch_tensor.imag], dim=1).float()
        return x_out

# ==========================================
# 2. モデル定義 (変更なし)
# ==========================================
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=64, cond_channels=64, time_emb_dim=32):
        super().__init__()
        total_in_channels = in_channels + cond_channels
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_proj = nn.Linear(time_emb_dim, 512)

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
        
        t = t.float().view(-1, 1) / 1000.0
        t_emb = self.time_mlp(t)
        t_emb = self.time_proj(t_emb).view(-1, 512, 1, 1)

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