import torch

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)


def build_vocab(pairs, min_freq: int):
    counts = {}
    for src, tgt, *_ in pairs:
        for tok in src + tgt:
            counts[tok] = counts.get(tok, 0) + 1
    itos = SPECIAL_TOKENS + sorted([tok for tok, count in counts.items() if count >= min_freq])
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return stoi, itos


def encode(tokens, stoi, max_len: int):
    ids = [stoi.get(tok, UNK) for tok in tokens[: max_len - 2]]
    return [SOS] + ids + [EOS]


def ids_to_selfies(ids, itos):
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS:
            break
        if idx in (PAD, SOS):
            continue
        tokens.append(itos[idx] if 0 <= idx < len(itos) else "<unk>")
    return "".join(tokens)
