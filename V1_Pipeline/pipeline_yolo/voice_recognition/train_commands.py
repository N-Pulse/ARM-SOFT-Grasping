import torch, random, yaml, argparse, json
from pathlib import Path
from utils import load_wav, to_logmelspec, pad_or_trim, SAMPLE_RATE
from models import CmdNet

MAX_LEN = int(2.0 * SAMPLE_RATE)      

def dataset(cmd_map):
    for label, fp in cmd_map:
        wav = pad_or_trim(load_wav(Path(fp)), MAX_LEN)
        feat = to_logmelspec(wav)
        yield feat, torch.tensor(label, dtype=torch.long)

def main():
    tr = yaml.safe_load(Path('meta_train.yaml').read_text())
    va = yaml.safe_load(Path('meta_val.yaml').read_text())

    cmd2idx = {c: i for i, c in enumerate(sorted(tr['commands']))}
    idx2cmd = {i: c for c, i in cmd2idx.items()}
    Path('models/idx2cmd.json').write_text(json.dumps(idx2cmd, ensure_ascii=False))

    net = CmdNet(len(cmd2idx))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_pairs = [(cmd2idx[c], p) for c, files in tr['commands'].items() for p in files]
    val_pairs   = [(cmd2idx[c], p) for c, files in va['commands'].items() for p in files]

    best_acc = 0
    epochs_no_improve = 0
    patience = 10
    max_epochs = 100

    for epoch in range(max_epochs):
        random.shuffle(train_pairs)
        losses = []

        for feat, lbl in dataset(train_pairs):
            ŷ = net(feat.unsqueeze(0))
            loss = loss_fn(ŷ, lbl.unsqueeze(0))
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())

        # Validation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feat, lbl in dataset(val_pairs):
                ŷ = net(feat.unsqueeze(0))
                if ŷ.argmax(-1).item() == lbl:
                    correct += 1
                total += 1
        acc = correct / total
        net.train()

        print(f"epoch {epoch+1:02d}  loss {sum(losses)/len(losses):.3f}  val_acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            torch.save(net.state_dict(), 'models/best_cmdnet.pt')
            print("🔄 Nouveau meilleur modèle sauvegardé.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"⏹️ Early stopping déclenché (aucune amélioration en {patience} époques).")
            break

    print(f"✅ Entraînement terminé. Meilleure val_acc : {best_acc:.3f} → models/best_cmdnet.pt")

if __name__ == '__main__':
    Path('models').mkdir(exist_ok=True)
    main()
