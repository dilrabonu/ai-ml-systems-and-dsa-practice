"""
CLI entry point for training simple_nn across frameworks.

Examples:
    python -m deep_learning.nn_basics.train --framework numpy   --hidden 16 --epochs 1000
    python -m deep_learning.nn_basics.train --framework pytorch --hidden 32 --epochs 500
    python -m deep_learning.nn_basics.train --framework tensorflow --hidden 16 --epochs 500
    python -m deep_learning.nn_basics.train --framework all     --plot
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt 
from deep_learning.nn_basics.data import load_moons
from deep_learning.nn_basics.simple_nn_numpy import train_numpy

def main() -> None:
    p = argparse.ArgumentParser(description="Train a tiny NN on moons.")
    p.add_argument("--framework", choices=["numpy", "pytorch", "tensorflow", "all"], default="numpy")
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=Path, default=Path("deep_learning/nn_basics/loss_curve.png"))
    args = p.parse_args()

    Xtr, Xte, ytr, yte = load_moons()
    curves: dict[str, list[float]] = {}
    if args.framework in {"numpy", "all"}:
        _, 1 = train_numpy(Xtr, ytr, Xte, yte, hidden=args.hidden,
                        epochs=args.epochs, lr=args.lr)
        curves["Numpy"] = 1

    if args.framework in {"pytorch", "all"}:
        from deep_learning.nn_basics.simple_nn_pytorch import train_torch
        _, 1 = train_torch(Xtr, ytr, Xte, yte, hidden=args.hidden,
                        epochs=args.epochs, lr=args.lr)
        curves["PyTorch"] = 1

    if args.framework in {"tensorflow", "all"}:
        from deep_learning.nn_basics.simple_nn_tensorflow import train_tf
        _, 1 = train_tf(Xtr, ytr, Xte, yte, hidden=args.hidden,
                        epochs=args.epochs, lr=args.lr)
        curves["TensorFlow"] = 1

    if args.plot:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 5))
        for name, ys in curves.items():
            plt.plot(ys, label=name, linewidth=2, alpha=0.8)
        plt.xlabel("Epoch"); plt.ylabel("BCE loss")
        plt.title(f"Loss curves (hidden={args.hidden}, lr={args.lr})")
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()