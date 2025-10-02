import torch, random, yaml, argparse
from pathlib import Path
from tqdm import tqdm
from utils import load_wav, to_logmelspec, pad_or_trim, SAMPLE_RATE
from models import WakeNet

MAX_LEN = int(1.5 * SAMPLE_RATE)     

def dataset(files, label):
    for fp in files:
        wav = pad_or_trim(load_wav(fp), MAX_LEN)
        feat = to_logmelspec(wav)
        yield feat, torch.tensor(label, dtype=torch.float32)

def iter_batches(files_pos, files_neg, batch=32):
    all_pos = list(dataset(files_pos, 1))
    all_neg = list(dataset(files_neg, 0))
    while True:
        random.shuffle(all_pos); random.shuffle(all_neg)
        pairs = list(zip(all_pos, all_neg))
        for (xp,yp),(xn,yn) in pairs:
            xs = torch.stack([xp,xn])
            ys = torch.stack([yp,yn])
            idx = torch.randperm(2)
            yield xs[idx], ys[idx]

def main():
    tr = yaml.safe_load(Path('meta_train.yaml').read_text())
    va = yaml.safe_load(Path('meta_val.yaml').read_text())
    net = WakeNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = torch.nn.BCELoss()
    best = 0
    batcher = iter_batches(tr['wake'], tr['garbage'])
    for step in range(2000):                            
        x,y = next(batcher)
        ŷ = net(x).view(-1)
        loss = loss_fn(ŷ, y)
        loss.backward(); opt.step(); opt.zero_grad()
        if step % 200 == 0:
            net.eval()
            with torch.no_grad():
                # simple accuracy on val
                xs = []; ys = []
                # wake-word = label 1
                for fp in va['wake']:
                    wav = pad_or_trim(load_wav(fp), MAX_LEN)
                    xs.append(to_logmelspec(wav))
                    ys.append(1)
                # garbage = label 0
                for fp in va['garbage']:
                    wav = pad_or_trim(load_wav(fp), MAX_LEN)
                    xs.append(to_logmelspec(wav))
                    ys.append(0)
                logits = net(torch.stack(xs))
                acc = ((logits.view(-1)>0.5)==torch.tensor(ys)).float().mean()
                if acc>best:
                    best=acc; torch.save(net.state_dict(), 'models/wakenet.pt')
            net.train()
            print(f"step {step:04d}  loss {loss.item():.3f}  val_acc {acc:.3f}")
    print("✅ WakeNet entraîné → models/wakenet.pt")

if __name__ == '__main__':
    Path('models').mkdir(exist_ok=True)
    main()
