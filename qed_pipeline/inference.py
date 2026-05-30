import os

import torch

from .chemistry import Chem, QED, build_fp_generator, selfies_to_mol, split_selfies, tanimoto_from_mols
from .config import TrainConfig
from .model import ConditionalSeq2Seq
from .vocabulary import encode, ids_to_selfies


def resolve_checkpoint(cfg: TrainConfig):
    if cfg.checkpoint:
        return cfg.checkpoint

    candidates = [
        os.path.join(cfg.out_dir, "qed_best_model.pt"),
        os.path.join(cfg.out_dir, "models", "best model.pt"),
        os.path.join(cfg.out_dir, "models", "last epoch's model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No checkpoint found. Use --checkpoint to specify a model file, or put one at:\n"
        f"{searched}"
    )


def pick_best_via_rerank(model, stoi, itos, device, src_selfies, cfg, fpgen):
    src_tokens = split_selfies(src_selfies)
    src_ids = encode(src_tokens, stoi, cfg.max_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)
    src_mol, src_smiles = selfies_to_mol(src_selfies)
    if src_mol is None:
        return {
            "src_smiles": "",
            "src_qed": 0.0,
            "pred_selfies": src_selfies,
            "pred_smiles": "",
            "pred_qed": 0.0,
            "sim": 1.0,
            "dQED": 0.0,
        }
    fp = fpgen.GetFingerprint(src_mol) if fpgen else None
    src_fp = torch.tensor(fp, dtype=torch.float32, device=device).unsqueeze(0) if fp is not None else torch.zeros(1, 1024, device=device)
    src_qed = QED.qed(src_mol)
    candidates = set()
    gen_ids = model.generate(src_tensor, src_lens, src_fp, max_len=cfg.max_len)
    gen_selfies = ids_to_selfies(gen_ids[0].cpu(), itos)
    if gen_selfies and gen_selfies != src_selfies:
        candidates.add(gen_selfies)
    for _ in range(cfg.rerank_candidates):
        sampled_ids = model.sample(src_tensor, src_lens, src_fp, max_len=cfg.max_len, temperature=cfg.rerank_temperature, top_k=cfg.rerank_top_k)
        samp = ids_to_selfies(sampled_ids[0].cpu(), itos)
        if samp and samp != src_selfies:
            candidates.add(samp)
    best, best_score = None, -1e9
    for cand in candidates:
        mol, _ = selfies_to_mol(cand)
        if mol is None:
            continue
        sim = tanimoto_from_mols(src_mol, mol, fpgen)
        if sim < cfg.rerank_similarity_floor:
            continue
        if cfg.rerank_similarity_ceiling > 0 and sim > cfg.rerank_similarity_ceiling:
            continue
        qed = QED.qed(mol)
        dqed = qed - src_qed
        if dqed < cfg.rerank_min_dqed:
            continue
        score = sim * 2.0 + dqed * 8.0
        if score > best_score:
            best_score = score
            best = (cand, Chem.MolToSmiles(mol, canonical=True), qed, sim, dqed)
    if best is None:
        return {
            "src_smiles": src_smiles,
            "src_qed": src_qed,
            "pred_selfies": src_selfies,
            "pred_smiles": src_smiles,
            "pred_qed": src_qed,
            "sim": 1.0,
            "dQED": 0.0,
        }
    return {
        "src_smiles": src_smiles,
        "src_qed": src_qed,
        "pred_selfies": best[0],
        "pred_smiles": best[1],
        "pred_qed": best[2],
        "sim": best[3],
        "dQED": best[4],
    }


def predict_one(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(cfg)
    checkpoint = torch.load(ckpt_path, map_location=device)
    stoi, itos = checkpoint["stoi"], checkpoint["itos"]
    saved_cfg = checkpoint.get("config", {})
    model = ConditionalSeq2Seq(
        len(itos),
        saved_cfg.get("emb_dim", cfg.emb_dim),
        saved_cfg.get("hidden_dim", cfg.hidden_dim),
        saved_cfg.get("num_layers", cfg.num_layers),
        saved_cfg.get("dropout", cfg.dropout),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    fpgen = build_fp_generator()
    result = pick_best_via_rerank(model, stoi, itos, device, cfg.input_selfies, cfg, fpgen)
    print("Checkpoint:", ckpt_path)
    print("Input SELFIES:", cfg.input_selfies)
    print("Input SMILES:", result["src_smiles"])
    print("Input QED:", result["src_qed"])
    print("Optimized SELFIES:", result["pred_selfies"])
    print("Optimized SMILES:", result["pred_smiles"])
    print("Similarity:", result["sim"])
    print("Optimized QED:", result["pred_qed"], "dQED:", result["dQED"])
