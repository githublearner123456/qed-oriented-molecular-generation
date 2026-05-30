import random
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .chemistry import HAS_RDKIT, Chem, QED, DataStructs, build_fp_generator, selfies_to_mol, smiles_to_selfies, split_selfies
from .file_utils import normalize_columns, read_table, save_csv
from .vocabulary import PAD, encode


def read_molecules(file_path: str) -> List[Dict]:
    df = normalize_columns(read_table(file_path))
    required = {"smiles", "qed", "selfies"}
    if not required.issubset(set(df.columns)):
        raise ValueError("当前任务要求输入文件至少包含 smiles、qed、selfies 三列。")

    fpgen = build_fp_generator()
    rows: List[Dict] = []
    skipped = 0
    for _, row in df.iterrows():
        smiles = str(row.get("smiles", "")).strip() if not pd.isna(row.get("smiles", "")) else ""
        selfies = str(row.get("selfies", "")).strip() if not pd.isna(row.get("selfies", "")) else ""
        if not selfies and smiles:
            selfies = smiles_to_selfies(smiles)
        if not selfies:
            skipped += 1
            continue
        tokens = split_selfies(selfies)
        if not tokens:
            skipped += 1
            continue

        mol = None
        fp = None
        heavy_atoms = 0
        canonical_smiles = smiles
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                mol, canonical_smiles = selfies_to_mol(selfies)
            else:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if mol is None:
                skipped += 1
                continue
            fp = fpgen.GetFingerprint(mol) if fpgen is not None else None
            heavy_atoms = int(mol.GetNumHeavyAtoms())
            qed = float(QED.qed(mol))
        else:
            try:
                qed = float(row.get("qed"))
            except Exception:
                skipped += 1
                continue

        rows.append(
            {
                "smiles": canonical_smiles,
                "qed": qed,
                "selfies": selfies,
                "tokens": tokens,
                "mol": mol,
                "fp": fp,
                "heavy_atoms": heavy_atoms,
            }
        )
    if skipped:
        print(f"跳过无效样本数: {skipped}")
    return rows


def make_pairs_all(
    molecules: List[Dict],
    min_pair_similarity: float,
    min_pair_dqed: float,
    max_pairs: int,
    candidates_per_src: int,
    top_k: int,
    max_heavy_atom_delta: int,
    max_token_length_delta: int,
    seed: int,
    max_src_samples: int = 15000,
) -> List[Tuple[List[str], List[str], float, float, str, str, float]]:
    if not HAS_RDKIT:
        raise RuntimeError("需要 RDKit。")
    rng = random.Random(seed)
    #1.源分子粗筛（91-104）根据qed值对分子进行排序，选取qed较低的分子作为源分子，减少计算量
    sorted_mols = sorted(molecules, key=lambda x: x["qed"])
    n = len(sorted_mols)
    if n > max_src_samples:
        sampled_indices = rng.sample(range(n), max_src_samples)#不是选取所有的分子进行配对而是采样一部分的分子作为源分子，减少计算量
        src_pool = [sorted_mols[i] for i in sorted(sampled_indices)]#根据索引取出这些随机选出来的分子对象
    else:
        src_pool = sorted_mols

    mols_by_heavy: Dict[int, List[Dict]] = {}#必须要保证重原子相近的分子才能配对，这个字典的key是重原子数 value是重原子数相同的分子列表
    for mol in sorted_mols:
        mols_by_heavy.setdefault(mol["heavy_atoms"], []).append(mol)

    pairs = []
    rng.shuffle(src_pool)
    #2.目标分子粗筛（105-114）
    for src in tqdm(src_pool, desc="构造训练分子对", unit="mol", leave=False, dynamic_ncols=True):#先满足一定范围的重原子然后还有满足deltaQED大于一定的阈值
        candidates = []
        for atom_count in range(src["heavy_atoms"] - max_heavy_atom_delta, src["heavy_atoms"] + max_heavy_atom_delta + 1):
            for mol in mols_by_heavy.get(atom_count, []):
                if mol["qed"] > src["qed"] + min_pair_dqed:
                    candidates.append(mol)
        if not candidates:
            continue
        if len(candidates) > candidates_per_src:#防止上一步骤中满足条件的候选分子过多，随机选取其中一部分进行配对，减少计算量
            candidates = rng.sample(candidates, candidates_per_src)
        #3.精筛根据pair score打分  选取top4的目标分子进行配对
        sims = DataStructs.BulkTanimotoSimilarity(src["fp"], [c["fp"] for c in candidates])
        scored = []
        for sim, high in zip(sims, candidates):
            if sim < min_pair_similarity or high["smiles"] == src["smiles"]:
                continue
            if max_token_length_delta >= 0 and abs(len(high["tokens"]) - len(src["tokens"])) > max_token_length_delta:
                continue
            dqed = high["qed"] - src["qed"]
            pair_score = sim * 2.0 + dqed * 5.0
            scored.append((pair_score, sim, dqed, high))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, sim, _, high in scored[:top_k]:
            pairs.append((src["tokens"], high["tokens"], src["qed"], high["qed"], src["smiles"], high["smiles"], sim))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    rng.shuffle(pairs)
    pairs.sort(key=lambda x: x[6], reverse=True)
    return pairs[:max_pairs]


def split_train_valid_test(pairs, train_ratio, valid_ratio, test_ratio, seed: int):
    total = train_ratio + valid_ratio + test_ratio
    train_ratio /= total
    valid_ratio /= total
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    train_end = max(1, int(n * train_ratio))
    valid_end = min(n - 1, train_end + int(n * valid_ratio))
    return list(pairs[:train_end]), list(pairs[train_end:valid_end]), list(pairs[valid_end:])


class MoleculePairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple], stoi: Dict[str, int], max_len: int):
        self.items = []
        fpgen = build_fp_generator()
        for src, tgt, src_qed, tgt_qed, src_smiles, tgt_smiles, pair_sim in pairs:
            mol = Chem.MolFromSmiles(src_smiles) if HAS_RDKIT else None
            fp = fpgen.GetFingerprint(mol) if (mol is not None and fpgen is not None) else None
            fp_list = list(fp) if fp is not None else [0.0] * 1024
            self.items.append((encode(src, stoi, max_len), encode(tgt, stoi, max_len), float(src_qed), float(tgt_qed), src_smiles, tgt_smiles, float(pair_sim), fp_list))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch):
    srcs, tgts, src_qeds, tgt_qeds, src_smiles, tgt_smiles, pair_sims, src_fps = zip(*batch)
    src_lens = torch.tensor([len(x) for x in srcs], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgts], dtype=torch.long)
    max_src = int(max(src_lens).item())
    max_tgt = int(max(tgt_lens).item())
    src_pad = torch.full((len(batch), max_src), PAD, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(src)] = torch.tensor(src, dtype=torch.long)
        tgt_pad[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return (
        src_pad,
        src_lens,
        tgt_pad,
        tgt_lens,
        torch.tensor(src_qeds, dtype=torch.float32),
        torch.tensor(tgt_qeds, dtype=torch.float32),
        list(src_smiles),
        list(tgt_smiles),
        torch.tensor(pair_sims, dtype=torch.float32),
        torch.tensor(src_fps, dtype=torch.float32),
    )


def save_pair_split(path: str, pairs: Sequence[Tuple]) -> None:
    rows = []
    for src, tgt, src_qed, tgt_qed, src_smiles, tgt_smiles, pair_sim in pairs:
        rows.append({"src_selfies": "".join(src), "tgt_selfies": "".join(tgt), "src_qed": float(src_qed), "tgt_qed": float(tgt_qed), "src_smiles": src_smiles, "tgt_smiles": tgt_smiles, "pair_similarity": float(pair_sim)})
    save_csv(path, rows)
