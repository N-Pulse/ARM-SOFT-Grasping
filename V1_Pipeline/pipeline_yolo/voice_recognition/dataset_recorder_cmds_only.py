from pathlib import Path
import argparse, sounddevice as sd, soundfile as sf, numpy as np, sys, time

# ----------- Constants ----------- #
SAMPLE_RATE   = 16_000
CMD_DURATION  = 2.0        # Duration of each recording (seconds)
NUM_SAMPLES   = 50         # Number of samples per command
# ---------------------------------- #

def record(seconds: float) -> np.ndarray:
    print(f"[REC] {seconds:.1f}s…")
    buf = sd.rec(int(seconds * SAMPLE_RATE),
                 samplerate=SAMPLE_RATE,
                 channels=1,
                 dtype='int16')
    sd.wait()
    return buf.squeeze()

def save_wav(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, arr, SAMPLE_RATE, subtype="PCM_16")

    # Try to display a relative path; otherwise, fall back gracefully.
    try:
        rel = path.resolve().relative_to(Path.cwd())
        print(f"  ↳ {rel}")
    except ValueError:
        print(f"  ↳ {path}")

def main():
    ap = argparse.ArgumentParser(description="Record commands only.")
    ap.add_argument("--file", required=True,
                    help="Text file: one command per line.")
    ap.add_argument("--out", default="dataset",
                    help="Root directory (default: ./dataset)")
    ap.add_argument("--num", type=int, default=NUM_SAMPLES,
                    help=f"Samples per command (default: {NUM_SAMPLES})")
    ap.add_argument("--duration", type=float, default=CMD_DURATION,
                    help=f"Duration (s) of each sample (default: {CMD_DURATION})")
    args = ap.parse_args()

    cmds = [l.strip() for l in Path(args.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not cmds:
        sys.exit("❌  No commands found in the file.")

    root = Path(args.out)
    print(f"\nRecording to: {root.resolve()}")
    print(f"Commands: {cmds}\n")

    try:
        for cmd in cmds:
            cmd_dir = root / "commands" / cmd.replace(" ", "_")
            for i in range(args.num):
                input(f"Command \"{cmd}\" ({i+1}/{args.num}) – press Enter then speak…")
                audio = record(args.duration)
                save_wav(audio, cmd_dir / f"sample_{i:03d}.wav")
                time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")

    print("\n✅ Done! You can now run preprocess.py then train_commands.py.")

if __name__ == "__main__":
    main()
