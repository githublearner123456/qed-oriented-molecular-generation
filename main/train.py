import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qed_pipeline import TrainConfig, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train the QED optimization model.")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "dataset" / "rawdata.xlxs"))
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max_pairs", type=int, default=TrainConfig.max_pairs)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data_path=args.data_path,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        max_pairs=args.max_pairs,
    )
    train(cfg)


if __name__ == "__main__":
    main()
