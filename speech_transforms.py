from torchaudio.transforms import MFCC, Resample
import torch.nn.functional as F
import torch
import torchaudio

def custom_mfcc(signal):

    mfcc_transformer = MFCC(
        sample_rate=16_000,
        n_mfcc = 26,
        melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 64},
    )
    
    res = F.normalize(mfcc_transformer(signal), p=2, dim=1)
    return res

def fix_timesteps_signal(t: torch.Tensor, time_steps: int) -> torch.Tensor:

    """
    t: torch.Tensor[1, n_step]
    """
    assert t.ndim == 2, f"Dimension is not 2 but {t.ndim}, tensor shape: {t.shape}"
    if time_steps > t.shape[1]:  # pad
        zeros = torch.zeros((1, time_steps))
        zeros[:, :t.shape[1]] = t
        return zeros
    else:
        clipped = t[:, :time_steps]
        return clipped



class FeatureExtractor:

    def __init__(
        self, 
        extractor: list, 
        time:float , 
        sr: int = 16_000, 
        mono: bool = True, 
        squeeze: bool = False
    ):
        self.extractor = extractor
        self.sr = sr
        self.time = time
        self.mono = mono
        self.squeeze = squeeze

    def __call__(self, file_path):
        assert isinstance(file_path, str), "Img name must be string!!!"

        signal, signal_sr = torchaudio.load(file_path)  #
        if signal.size(0) >= 2 and self.mono:  # Multichannel to monochannel 
            signal = torch.mean(signal, dim=0, keepdim=True) # signal is (1, sr*time)
        
        signal = fix_timesteps_signal(signal, int(self.time*self.sr))
        
        if signal_sr != self.sr:
            resampler = Resample(signal_sr, self.sr)
            signal = resampler(signal)  # Ensuring signal is (1, sr*time)

        for extractor in self.extractor:

            signal = extractor(signal)
                
        if signal.shape[0] == 1 and self.squeeze:
            signal = signal.squeeze(0)

        signal = torch.nn.functional.normalize(signal, p=2, dim=signal.ndim-1)
        return signal