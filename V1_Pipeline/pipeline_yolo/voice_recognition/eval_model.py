import argparse, json, itertools
from pathlib import Path
import numpy as np, torch, yaml
from sklearn.metrics import (precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt

# --- import utilitaires & modèles ------------------------------
from utils import load_wav, to_logmelspec, pad_or_trim, SAMPLE_RATE
from models import WakeNet, CmdNet
# ---------------------------------------------------------------

MAX_LEN_WAKE = int(1.5 * SAMPLE_RATE)
MAX_LEN_CMD  = int(2.0 * SAMPLE_RATE)

def evaluate_wake(val_yaml='meta_val.yaml', threshold=0.5, plot_roc=True):
    va = yaml.safe_load(Path(val_yaml).read_text())
    model = WakeNet(); model.load_state_dict(torch.load('models/wakenet.pt')); model.eval()

    xs, ys = [], []
    for fp in va['wake']:
        wav = pad_or_trim(load_wav(fp), MAX_LEN_WAKE)
        xs.append(to_logmelspec(wav)); ys.append(1)
    for fp in va['garbage']:
        wav = pad_or_trim(load_wav(fp), MAX_LEN_WAKE)
        xs.append(to_logmelspec(wav)); ys.append(0)
    xs = torch.stack(xs)
    with torch.no_grad():
        prob = model(xs).view(-1).numpy()

    preds = (prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(ys, preds, average='binary')
    auc = roc_auc_score(ys, prob)
    print(f"\nWake-word evaluation (threshold={threshold})")
    print(f"Precision {prec:.3f} | Recall {rec:.3f} | F1 {f1:.3f} | ROC-AUC {auc:.3f}")

    if plot_roc:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(ys, prob)
        plt.title("Wake-word ROC curve")
        plt.show()

def evaluate_commands(val_yaml='meta_val.yaml', plot_cm=True):
    va = yaml.safe_load(Path(val_yaml).read_text())
    cmd2idx = {c:i for i,c in enumerate(sorted(va['commands']))}
    idx2cmd = {i:c for c,i in cmd2idx.items()}
    n_classes = len(cmd2idx)

    model = CmdNet(n_classes)
    model.load_state_dict(torch.load('models/cmdnet.pt')); model.eval()

    y_true, y_pred = [], []
    for cmd, files in va['commands'].items():
        for fp in files:
            wav = pad_or_trim(load_wav(fp), MAX_LEN_CMD)
            feat = to_logmelspec(wav)
            with torch.no_grad():
                logits = model(feat.unsqueeze(0))
            y_pred.append(int(logits.argmax(-1)))
            y_true.append(cmd2idx[cmd])

    print("\nCommand classification report")
    print(classification_report(y_true, y_pred, target_names=[idx2cmd[i] for i in range(n_classes)],
                                zero_division=0, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    acc = (cm.diagonal().sum() / cm.sum())
    print(f"Overall accuracy: {acc:.3f}")

    if plot_cm:
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title("Confusion matrix")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(idx2cmd.values(), rotation=90, fontsize=6)
        ax.set_yticks(range(n_classes))
        ax.set_yticklabels(idx2cmd.values(), fontsize=6)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wakeword", action="store_true", help="evaluate WakeNet")
    parser.add_argument("--commands", action="store_true", help="evaluate CmdNet")
    parser.add_argument("--roc", action="store_true", help="plot ROC curve (wakeword)")
    parser.add_argument("--cm", action="store_true", help="plot confusion matrix (commands)")
    parser.add_argument("--threshold", type=float, default=0.5, help="wake-word threshold")
    args = parser.parse_args()

    if args.wakeword:
        evaluate_wake(threshold=args.threshold, plot_roc=args.roc)
    if args.commands:
        evaluate_commands(plot_cm=args.cm)
    if not (args.wakeword or args.commands):
        parser.print_help()
