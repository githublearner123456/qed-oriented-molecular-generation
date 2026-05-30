import os

import pandas as pd
import torch

from .chemistry import QED, selfies_to_mol, selfies_to_smiles, tanimoto_from_mols
from .metrics import compute_eval_metrics
from .vocabulary import PAD, ids_to_selfies


def plot_curves(log_path: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax0, ax1 = axes
    if "train_loss" in df.columns:
        ax0.plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        ax0.plot(df["epoch"], df["val_loss"], label="val")
    if "train_ce_loss" in df.columns:
        ax0.plot(df["epoch"], df["train_ce_loss"], linestyle="--", alpha=0.7, label="train_ce")
    ax0.set_title("Loss")
    ax0.set_xlabel("Epoch")
    ax0.legend(loc="best")
    if "validity" in df.columns:
        ax1.plot(df["epoch"], df["validity"], label="validity")
    if "SimSrc" in df.columns:
        ax1.plot(df["epoch"], df["SimSrc"], label="SimSrc")
    if "dQED" in df.columns:
        ax1.plot(df["epoch"], df["dQED"], label="dQED")
    ax1.set_title("Generation Metrics")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_eval_distribution(cases_csv: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(cases_csv):
        return
    df = pd.read_csv(cases_csv)
    if df.empty or "dQED" not in df.columns:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(df["dQED"], bins=40, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Test dQED Distribution")
    ax.set_xlabel("dQED")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sim_dqed_scatter(cases_csv: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(cases_csv):
        return
    df = pd.read_csv(cases_csv)
    required = {"sim_to_source", "dQED"}
    if df.empty or not required.issubset(df.columns):
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(df["sim_to_source"], df["dQED"], s=12, alpha=0.45)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.axvline(0.3, linestyle="--", linewidth=1)
    ax.set_title("SimSrc vs dQED (Test)")
    ax.set_xlabel("SimSrc")
    ax.set_ylabel("dQED")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_simsrc_distribution(cases_csv: str, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not os.path.exists(cases_csv):
        return
    df = pd.read_csv(cases_csv)
    if df.empty or "sim_to_source" not in df.columns:
        return
    values = pd.to_numeric(df["sim_to_source"], errors="coerce").dropna()
    if values.empty:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(values, bins=40, alpha=0.85)
    ax.axvline(0.3, linestyle="--", linewidth=1)
    ax.set_title("Test SimSrc Distribution")
    ax.set_xlabel("SimSrc")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def evaluate(model, loader, criterion, device, itos, cfg, max_eval_samples, fpgen):
    model.eval()
    total_loss = 0.0
    total_tok = 0
    records = []
    with torch.no_grad():
        for src, src_lens, tgt, _, _, tgt_qed, src_smiles, tgt_smiles, pair_sims, src_fps in loader:
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            src_fps = src_fps.to(device)
            logits, _, _ = model(src, src_lens, tgt, src_fps, teacher_forcing_ratio=1.0)
            gold = tgt[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            ntok = int(gold.ne(PAD).sum().item())
            total_loss += float(loss.item()) * ntok
            total_tok += ntok
            if len(records) < max_eval_samples:
                for i in range(min(src.size(0), max_eval_samples - len(records))):
                    src_selfies = ids_to_selfies(src[i].cpu(), itos)
                    tgt_selfies = ids_to_selfies(tgt[i].cpu(), itos)
                    gen_ids = model.generate(src[i].unsqueeze(0), src_lens[i].unsqueeze(0), src_fps[i].unsqueeze(0), max_len=cfg.max_len)
                    gen_selfies = ids_to_selfies(gen_ids[0].cpu(), itos)
                    src_mol, _ = selfies_to_mol(src_selfies)
                    tgt_mol, _ = selfies_to_mol(tgt_selfies)
                    gen_mol, _ = selfies_to_mol(gen_selfies)
                    sim = tanimoto_from_mols(src_mol, gen_mol, fpgen) if src_mol and gen_mol else 0.0
                    sim_tgt = tanimoto_from_mols(tgt_mol, gen_mol, fpgen) if tgt_mol and gen_mol else 0.0
                    src_q = QED.qed(src_mol) if src_mol else 0.0
                    tgt_q = QED.qed(tgt_mol) if tgt_mol else float(tgt_qed[i].item()) if i < len(tgt_qed) else 0.0
                    gen_q = QED.qed(gen_mol) if gen_mol else 0.0
                    records.append(
                        {
                            "src_selfies": src_selfies,
                            "tgt_selfies": tgt_selfies,
                            "gen_selfies": gen_selfies,
                            "src_smiles": src_smiles[i],
                            "tgt_smiles": tgt_smiles[i],
                            "gen_smiles": selfies_to_smiles(gen_selfies),
                            "sim_to_source": sim,
                            "sim_to_target": sim_tgt,
                            "src_qed": src_q,
                            "tgt_qed": tgt_q,
                            "gen_qed": gen_q,
                            "dQED": gen_q - src_q,
                            "valid": gen_mol is not None,
                            "pair_sim": float(pair_sims[i].item()) if i < len(pair_sims) else 0.0,
                        }
                    )
    return total_loss / max(total_tok, 1), compute_eval_metrics(records), records
