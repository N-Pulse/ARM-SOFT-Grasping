from pathlib import Path
from datetime import datetime
import time
import itertools
import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys, textwrap

# --------------------------------------------------------------------------- #
# --------- Paramètres par défaut  ------------------------------------------ #
WAKE_WORD_DEFAULT        = "Jarvis"
NUM_WAKE_SAMPLES_DEFAULT = 100
NUM_CMD_SAMPLES_DEFAULT  = 50
GARBAGE_MINUTES_DEFAULT  = 30

SAMPLE_RATE      = 16_000
WAKE_DURATION_S  = 1.5
CMD_DURATION_S   = 2.0
GARB_CHUNK_S     = 10.0
# --------------------------------------------------------------------------- #


def record_block(duration: float) -> np.ndarray:
    print(f"[REC] → {duration} s…")
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()
    return audio.squeeze()


def save_wav(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr, SAMPLE_RATE, subtype='PCM_16')
    try:
        # Try to get relative path for cleaner display
        rel_path = path.relative_to(Path.cwd())
        print(f"  ↳ Sauvegardé : {rel_path}")
    except ValueError:
        # If relative path fails, just use the absolute path
        print(f"  ↳ Sauvegardé : {path}")


def prompt(msg: str):
    input(f"\n{msg}\n➜ Appuyez sur [ENTRÉE] puis parlez…")


def record_wake_word(root: Path, wake_word: str, n_samples: int):
    for i in range(n_samples):
        prompt(f"Phrases wake-word ({i+1}/{n_samples}) – dites « {wake_word} »")
        audio = record_block(WAKE_DURATION_S)
        save_wav(audio, root / "wake_word" / f"sample_{i:03d}.wav")


def record_commands(root: Path, commands: list[str], n_samples: int):
    for cmd in commands:
        cmd_dir = root / "commands" / cmd.replace(" ", "_")
        for i in range(n_samples):
            prompt(f"Commande « {cmd} » ({i+1}/{n_samples})")
            audio = record_block(CMD_DURATION_S)
            save_wav(audio, cmd_dir / f"sample_{i:03d}.wav")


def record_garbage(root: Path, minutes: int):
    target_sec = minutes * 60
    captured   = 0
    seg_it     = itertools.count()
    print(textwrap.dedent(f"""
        =====================  GARBAGE  =====================
        Laissez tourner le micro pour {minutes} min de bruits
        ambiants : télévision, discussions, silence, extérieur…
        (Ctrl-C pour interrompre proprement)
        =====================================================
    """))
    try:
        while captured < target_sec:
            remaining = target_sec - captured
            chunk = min(GARB_CHUNK_S, remaining)
            audio = record_block(chunk)
            idx = next(seg_it)
            save_wav(audio, root / "garbage" / f"segment_{idx:03d}.wav")
            captured += chunk
            print(f"… {captured/60:.1f}/{minutes} min capturées")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrompu par l'utilisateur – {captured/60:.1f} min capturées au total.")


# --------------------------- Parsing commandes ----------------------------- #
def get_commands_from_args(args) -> list[str]:
    # 1. --command répété
    if args.command:
        return args.command

    # 2. --commands-file
    if args.commands_file:
        path = Path(args.commands_file)
        if not path.is_file():
            sys.exit(f"❌  Le fichier {path} n'existe pas.")
        return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    # 3. Fallback interactif
    raw = input("\nEntrez la liste de commandes, séparées par des virgules :\n> ")
    cmds = [c.strip() for c in raw.split(",") if c.strip()]
    if not cmds:
        sys.exit("❌  Aucune commande fournie.")
    return cmds
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Enregistre un dataset vocal personnalisé."
    )
    parser.add_argument("-o", "--out", default="dataset",
        help="Répertoire de sortie (défaut: ./dataset)")
    parser.add_argument("--wake-word", default=WAKE_WORD_DEFAULT,
        help=f"Mot de réveil (défaut: {WAKE_WORD_DEFAULT})")
    parser.add_argument("--num-wake-samples", type=int, default=NUM_WAKE_SAMPLES_DEFAULT,
        help=f"Nombre de prises wake-word (défaut: {NUM_WAKE_SAMPLES_DEFAULT})")
    parser.add_argument("--num-cmd-samples", type=int, default=NUM_CMD_SAMPLES_DEFAULT,
        help=f"Nombre de prises par commande (défaut: {NUM_CMD_SAMPLES_DEFAULT})")
    parser.add_argument("--garbage-minutes", type=int, default=GARBAGE_MINUTES_DEFAULT,
        help=f"Durée garbage en minutes (défaut: {GARBAGE_MINUTES_DEFAULT})")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--command", action="append",
        help="Commande à enregistrer (peut être répété).")
    group.add_argument("--commands-file",
        help="Fichier texte avec une commande par ligne.")

    args = parser.parse_args()
    commands = get_commands_from_args(args)
    root = Path(args.out)

    # Create the root directory if it doesn't exist
    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    (root / f"meta_{ts}.txt").write_text(
        f"Wake-word: {args.wake_word}\n"
        f"Commandes: {commands}\n"
        f"Wake samples: {args.num_wake_samples}\n"
        f"Cmd samples: {args.num_cmd_samples}\n"
        f"Garbage minutes: {args.garbage_minutes}\n"
        f"Sample rate: {SAMPLE_RATE}\n"
    )

    print(f"\n*** Enregistrement vers: {root.resolve()} ***")
    record_wake_word(root, args.wake_word, args.num_wake_samples)
    record_commands(root, commands, args.num_cmd_samples)
    record_garbage(root, args.garbage_minutes)
    print("\n✅ Dataset terminé. Bon entraînement !")


if __name__ == "__main__":
    main()