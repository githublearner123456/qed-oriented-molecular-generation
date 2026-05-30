from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    mode: str = "train"
    data_path: str = str(PROJECT_ROOT / "dataset" / "rawdata.xlxs")
    out_dir: str = str(PROJECT_ROOT / "results")
    checkpoint: str = ""
    input_selfies: str = ""
    input_file: str = ""
    output_file: str = ""
    seed: int = 42
    max_len: int = 140
    batch_size: int = 32
    epochs: int = 80
    lr: float = 5e-4
    emb_dim: int = 128
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.3
    teacher_forcing: float = 0.6
    min_pair_similarity: float = 0.35
    min_pair_dqed: float = 0.06
    max_pairs: int = 50000
    candidates_per_src: int = 1200
    top_k: int = 4
    max_heavy_atom_delta: int = 30
    max_token_length_delta: int = 40
    max_src_samples: int = 25000
    min_freq: int = 2
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1
    patience: int = 20
    num_workers: int = 0
    max_eval_samples: int = 2048
    final_eval_samples: int = 0
    rerank_candidates: int = 64
    rerank_temperature: float = 0.8
    rerank_top_k: int = 24
    raw_similarity_soft_floor: float = 0.35
    rerank_similarity_floor: float = 0.25
    rerank_similarity_ceiling: float = 0.95
    rerank_min_dqed: float = 0.01
    keep_input_if_no_valid_candidate: bool = True
    sim_loss_weight: float = 1.0
    qed_loss_weight: float = 3.0
