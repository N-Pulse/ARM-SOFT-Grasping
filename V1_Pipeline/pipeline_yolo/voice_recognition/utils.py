from pathlib import Path
import torch, torchaudio, random
import librosa, numpy as np

SAMPLE_RATE = 16_000
N_MELS      = 40
WIN_LENGTH  = int(0.025 * SAMPLE_RATE)   
HOP_LENGTH  = int(0.010 * SAMPLE_RATE)  

_mel_spec = torchaudio.transforms.MelSpectrogram(
    SAMPLE_RATE, n_mels=N_MELS,
    win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
)

def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.mean(0, keepdim=True)     

def to_logmelspec(wav: torch.Tensor) -> torch.Tensor:
    spec = _mel_spec(wav)                   
    try:                                      
        spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
    except TypeError:                        
        spec = torchaudio.functional.amplitude_to_DB(
            spec, multiplier=20.0,
            amin=1e-10, db_multiplier=0.0,
            top_db=80.0
        )
   
    spec = (spec + 80.0) / 80.0               
    spec = spec.squeeze(0).transpose(0, 1)   
    return spec.unsqueeze(0)                 

def pad_or_trim(wav, target_len):
    T = wav.shape[-1]
    if T < target_len:
        pad = target_len - T
        wav = torch.nn.functional.pad(wav, (0, pad))
    else:
        wav = wav[..., :target_len]
    return wav

def split_train_val(files, val_ratio=0.1, seed=42):
    random.seed(seed)
    files = list(files)
    random.shuffle(files)
    n_val = int(len(files)*val_ratio)
    return files[n_val:], files[:n_val]
