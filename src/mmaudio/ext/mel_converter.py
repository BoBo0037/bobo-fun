# Reference: # https://github.com/bytedance/Make-An-Audio-2

import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, norm_fn=torch.log10):
    return norm_fn(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes, norm_fn):
    output = dynamic_range_compression_torch(magnitudes, norm_fn=norm_fn)
    return output


class MelConverter(nn.Module):

    def __init__(
        self,
        *,
        sampling_rate: float = 16_000,
        n_fft: int = 1024,
        num_mels: int = 80,
        hop_size: int = 256,
        win_size: int = 1024,
        fmin: float = 0,
        fmax: float = 8_000,
        norm_fn=torch.log10,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.norm_fn = norm_fn

        mel = librosa_mel_fn(sr=self.sampling_rate,
                             n_fft=self.n_fft,
                             n_mels=self.num_mels,
                             fmin=self.fmin,
                             fmax=self.fmax)
        mel_basis = torch.from_numpy(mel).float()
        hann_window = torch.hann_window(self.win_size)

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('hann_window', hann_window)

    @property
    def device(self):
        return self.mel_basis.device

    def forward(self, waveform: torch.Tensor, center: bool = False) -> torch.Tensor:
        waveform = waveform.clamp(min=-1., max=1.).to(self.device)

        waveform = torch.nn.functional.pad(
            waveform.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2),
             int((self.n_fft - self.hop_size) / 2)],
            mode='reflect')
        waveform = waveform.squeeze(1)

        spec = torch.stft(waveform,
                          self.n_fft,
                          hop_length=self.hop_size,
                          win_length=self.win_size,
                          window=self.hann_window,
                          center=center,
                          pad_mode='reflect',
                          normalized=False,
                          onesided=True,
                          return_complex=True)

        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)
        spec = spectral_normalize_torch(spec, self.norm_fn)

        return spec
