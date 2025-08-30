import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class WaveletDecomposition(nn.Module):
    def __init__(self, wavelet='db4', level=2, use_amp = False, mode='symmetric' ,device=None):
        super(WaveletDecomposition, self).__init__()
        self.level = level
        self.wavelet = wavelet
        self.wavelet_name = wavelet
        self.use_amp = use_amp
        self.device = device
        self.mode = mode

        self.dwt = DWT1DForward(wave=self.wavelet_name, J=self.level, use_amp=self.use_amp, mode=self.mode).cuda() if self.device == 'cuda' else DWT1DForward(wave=self.wavelet_name, J=self.level, use_amp=self.use_amp, mode=self.mode)
        self.idwt = DWT1DInverse(wave=self.wavelet_name, use_amp=self.use_amp, mode=self.mode).cuda() if self.device == 'cuda' else DWT1DInverse(wave=self.wavelet_name, use_amp=self.use_amp, mode=self.mode)
        
        if device:
            self.to(device)

    def _wavelet_decompose(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # permute to (batch, channel, seq_len) for pytorch_wavelets
        x_permuted = x.permute(0, 2, 1)
        yl, yh = self.dwt(x_permuted)
        
        return yl, yh

    def _wavelet_reconstruct(self, coeffs, original_length):
        # (yl, yh) -> (batch, channel, seq_len)
        recon_signal_permuted = self.idwt(coeffs)
        
        recon_signal_cropped = recon_signal_permuted[..., :original_length]

        # permute back to (batch, seq_len, channel)
        recon_signal = recon_signal_cropped.permute(0, 2, 1)
        
        return recon_signal

    def reconstruct_freq_bands(self, coeffs, original_length):
        all_coeffs = [coeffs[0]] + coeffs[1]
        
        reconstructed_bands = []
        
        for i in range(self.level + 1):
            temp_coeffs_list = [torch.zeros_like(c) for c in all_coeffs]
            temp_coeffs_list[i] = all_coeffs[i]
            yl_recon, yh_recon = temp_coeffs_list[0], temp_coeffs_list[1:]
            band_signal = self._wavelet_reconstruct((yl_recon, yh_recon), original_length)
            reconstructed_bands.append(band_signal)
            
        return reconstructed_bands

    def forward(self, x):
        original_length = x.shape[1]

        coeffs = self._wavelet_decompose(x)
        
        reconstructed_bands = self.reconstruct_freq_bands(coeffs, original_length)
        return reconstructed_bands
