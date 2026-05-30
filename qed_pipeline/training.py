import csv
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .chemistry import build_fp_generator, get_environment_info, set_seed
from .config import TrainConfig
from .data import MoleculePairDataset, collate_batch, make_pairs_all, read_molecules, save_pair_split, split_train_valid_test
from .evaluation import evaluate, plot_curves, plot_eval_distribution, plot_sim_dqed_scatter, plot_simsrc_distribution
from .file_utils import base_environment_info, save_csv, save_json
from .metrics import summarize_pairs
from .model import ConditionalSeq2Seq
from .vocabulary import PAD, build_vocab


def save_checkpoint(model, stoi, itos, cfg, path: str) -> None:
    torch.save({"model_state": model.state_dict(), "stoi": stoi, "itos": itos, "config": asdict(cfg)}, path)


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "qed_training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_ce_loss",
                "train_sim_loss",
                "train_qed_loss",
                "val_loss",
                "validity",
                "dQED",
                "SimSrc",
                "SimTgt",
                "positive_dqed_rate",
                "sim30_rate",
                "sim40_rate",
                "opt30_rate",
                "opt40_rate",
                "score",
                "valid_count",
                "invalid_count",
            ],
        )
        writer.writeheader()

    print("读取数据...")
    molecules = read_molecules(cfg.data_path)
    print("有效分子数:", len(molecules))
    print("构造高相似训练对...")
    pairs = make_pairs_all(
        molecules,
        cfg.min_pair_similarity,
        cfg.min_pair_dqed,
        cfg.max_pairs,
        cfg.candidates_per_src,
        cfg.top_k,
        cfg.max_heavy_atom_delta,
        cfg.max_token_length_delta,
        cfg.seed,
        cfg.max_src_samples,
    )
    print("训练对数量:", len(pairs))
    pair_summary = summarize_pairs(pairs)
    print(
        f"Pair quality | AvgSim {pair_summary['avg_pair_similarity']:.3f} | "
        f"AvgDQED {pair_summary['avg_pair_dqed']:+.3f} | "
        f"MinSim {pair_summary['min_pair_similarity']:.3f} | "
        f"MinDQED {pair_summary['min_pair_dqed']:+.3f}"
    )
    if len(pairs) < 100:
        raise RuntimeError("训练对太少，请降低 min_pair_similarity 或增加候选数量。")

    train_pairs, valid_pairs, test_pairs = split_train_valid_test(pairs, cfg.train_ratio, cfg.valid_ratio, cfg.test_ratio, cfg.seed)
    env = base_environment_info()
    env.update(get_environment_info())
    save_json(os.path.join(cfg.out_dir, "qed_config.json"), asdict(cfg))
    save_json(os.path.join(cfg.out_dir, "qed_environment.json"), env)
    save_json(
        os.path.join(cfg.out_dir, "qed_split_info.json"),
        {"total_pairs": len(pairs), "train_pairs": len(train_pairs), "valid_pairs": len(valid_pairs), "test_pairs": len(test_pairs)},
    )
    save_pair_split(os.path.join(cfg.out_dir, "qed_train_pairs.csv"), train_pairs)
    save_pair_split(os.path.join(cfg.out_dir, "qed_valid_pairs.csv"), valid_pairs)
    save_pair_split(os.path.join(cfg.out_dir, "qed_test_pairs.csv"), test_pairs)

    stoi, itos = build_vocab(train_pairs, min_freq=cfg.min_freq)
    save_json(os.path.join(cfg.out_dir, "qed_vocab.json"), {"stoi": stoi, "itos": itos})
    print("词表大小:", len(itos))
    train_ds = MoleculePairDataset(train_pairs, stoi, cfg.max_len)
    valid_ds = MoleculePairDataset(valid_pairs, stoi, cfg.max_len)
    test_ds = MoleculePairDataset(test_pairs, stoi, cfg.max_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    model = ConditionalSeq2Seq(len(itos), cfg.emb_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=cfg.patience)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    mse_criterion = nn.MSELoss()
    fpgen = build_fp_generator()

    best_metric, best_epoch, wait = -float("inf"), 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = total_ce_loss = total_sim_loss = total_qed_loss = 0.0
        total_tok = 0
        for src, src_lens, tgt, _, _, tgt_qeds, _, _, pair_sims, src_fps in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, dynamic_ncols=True):
            src, src_lens, tgt = src.to(device), src_lens.to(device), tgt.to(device)
            src_fps = src_fps.to(device)
            pair_sims = pair_sims.to(device)
            tgt_qeds = tgt_qeds.to(device)
            optimizer.zero_grad()
            logits, sim_pred, qed_pred = model(src, src_lens, tgt, src_fps, teacher_forcing_ratio=cfg.teacher_forcing)
            gold = tgt[:, 1:]
            ce_loss = ce_criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            sim_loss = mse_criterion(sim_pred, pair_sims)
            qed_loss = mse_criterion(qed_pred, tgt_qeds)
            loss = ce_loss + cfg.sim_loss_weight * sim_loss + cfg.qed_loss_weight * qed_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ntok = int(gold.ne(PAD).sum().item())
            total_loss += float(loss.item()) * ntok
            total_ce_loss += float(ce_loss.item()) * ntok
            total_sim_loss += float(sim_loss.item()) * ntok
            total_qed_loss += float(qed_loss.item()) * ntok
            total_tok += ntok

        avg_train_loss = total_loss / max(total_tok, 1)
        avg_train_ce_loss = total_ce_loss / max(total_tok, 1)
        avg_train_sim_loss = total_sim_loss / max(total_tok, 1)
        avg_train_qed_loss = total_qed_loss / max(total_tok, 1)
        val_loss, val_metrics, _ = evaluate(model, valid_loader, ce_criterion, device, itos, cfg, cfg.max_eval_samples, fpgen)
        scheduler.step(val_loss)
        score = (
            val_metrics["SimSrc"] * 2.0
            + val_metrics["SimTgt"] * 1.0
            + val_metrics["dQED"] * 8.0
            + val_metrics["positive_dqed_rate"] * 0.8
            + val_metrics["opt30_rate"] * 1.2
            + val_metrics["opt40_rate"] * 0.8
            + val_metrics["validity"] * 2.0
            - val_loss * 0.2
        )
        epoch_info = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_ce_loss": avg_train_ce_loss,
            "train_sim_loss": avg_train_sim_loss,
            "train_qed_loss": avg_train_qed_loss,
            "val_loss": val_loss,
            "validity": val_metrics["validity"],
            "dQED": val_metrics["dQED"],
            "SimSrc": val_metrics["SimSrc"],
            "SimTgt": val_metrics["SimTgt"],
            "positive_dqed_rate": val_metrics["positive_dqed_rate"],
            "sim30_rate": val_metrics["sim30_rate"],
            "sim40_rate": val_metrics["sim40_rate"],
            "opt30_rate": val_metrics["opt30_rate"],
            "opt40_rate": val_metrics["opt40_rate"],
            "score": score,
            "valid_count": val_metrics["valid_count"],
            "invalid_count": val_metrics["invalid_count"],
        }
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=epoch_info.keys()).writerow(epoch_info)
        print(
            f"Epoch {epoch:03d} | Train Loss {avg_train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Validity {val_metrics['validity']:.3f} | SimSrc {val_metrics['SimSrc']:.3f} | "
            f"SimTgt {val_metrics['SimTgt']:.3f} | dQED {val_metrics['dQED']:+.3f} | "
            f"Opt30 {val_metrics['opt30_rate']:.3f} | QEDLoss {avg_train_qed_loss:.4f} | Score {score:.3f}"
        )

        if score > best_metric:
            best_metric, best_epoch, wait = score, epoch, 0
            save_checkpoint(model, stoi, itos, cfg, os.path.join(cfg.out_dir, "qed_best_model.pt"))
        else:
            wait += 1
            if wait >= cfg.patience * 2:
                print("Early stopping triggered.")
                break

    save_checkpoint(model, stoi, itos, cfg, os.path.join(cfg.out_dir, "qed_last_model.pt"))
    checkpoint = torch.load(os.path.join(cfg.out_dir, "qed_best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_eval_samples = cfg.final_eval_samples if cfg.final_eval_samples > 0 else max(len(valid_ds), len(test_ds))
    valid_loss, valid_metrics, valid_records = evaluate(model, valid_loader, ce_criterion, device, itos, cfg, final_eval_samples, fpgen)
    test_loss, test_metrics, test_records = evaluate(model, test_loader, ce_criterion, device, itos, cfg, final_eval_samples, fpgen)
    save_json(os.path.join(cfg.out_dir, "qed_valid_metrics.json"), dict(valid_metrics, loss=valid_loss))
    save_json(os.path.join(cfg.out_dir, "qed_test_metrics.json"), dict(test_metrics, loss=test_loss))
    save_csv(os.path.join(cfg.out_dir, "qed_valid_cases.csv"), valid_records)
    save_csv(os.path.join(cfg.out_dir, "qed_test_cases.csv"), test_records)
    plot_curves(log_path, os.path.join(cfg.out_dir, "qed_training_curves.png"))
    plot_eval_distribution(os.path.join(cfg.out_dir, "qed_test_cases.csv"), os.path.join(cfg.out_dir, "qed_test_dqed_hist.png"))
    plot_sim_dqed_scatter(os.path.join(cfg.out_dir, "qed_test_cases.csv"), os.path.join(cfg.out_dir, "qed_test_simsrc_dqed_scatter.png"))
    plot_simsrc_distribution(os.path.join(cfg.out_dir, "qed_test_cases.csv"), os.path.join(cfg.out_dir, "qed_test_simsrc_hist.png"))
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_metric,
        "best_valid_loss": valid_loss,
        "final_valid_metrics": dict(valid_metrics, loss=valid_loss),
        "final_test_metrics": dict(test_metrics, loss=test_loss),
        "runtime_seconds": round(time.time() - start_time, 2),
        "output_dir": os.path.abspath(cfg.out_dir),
    }
    save_json(os.path.join(cfg.out_dir, "qed_run_summary.json"), summary)
    print(
        f"\nTest Results: Validity {test_metrics['validity']:.3f}, SimSrc {test_metrics['SimSrc']:.3f}, "
        f"SimTgt {test_metrics['SimTgt']:.3f}, dQED {test_metrics['dQED']:+.3f}, "
        f"PosDQED {test_metrics['positive_dqed_rate']:.3f}, Opt30 {test_metrics['opt30_rate']:.3f}, "
        f"Opt40 {test_metrics['opt40_rate']:.3f}"
    )
    print(f"Training finished. Best epoch {best_epoch}, model and logs saved to {cfg.out_dir}")
