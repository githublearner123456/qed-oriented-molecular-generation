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

from qed_pipeline import TrainConfig, predict_one


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one optimized molecule from one SELFIES string.")
    parser.add_argument("--input_selfies", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(out_dir=args.out_dir, checkpoint=args.checkpoint, input_selfies=args.input_selfies)
    predict_one(cfg)


if __name__ == "__main__":
    main()
