import random
import re
from typing import Dict, List

import selfies as sf
import torch

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:
    Chem = None
    QED = None
    DataStructs = None
    GetMorganGenerator = None
    HAS_RDKIT = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def split_selfies(text: str) -> List[str]:
    text = str(text).strip()
    try:
        return list(sf.split_selfies(text))
    except Exception:
        return re.findall(r"\[[^\]]+\]", text)


def smiles_to_selfies(smiles: str) -> str:
    smiles = str(smiles).strip()
    if not smiles:
        return ""
    try:
        return sf.encoder(smiles)
    except Exception:
        return ""


def selfies_to_smiles(selfies: str) -> str:
    try:
        return sf.decoder(str(selfies).strip())
    except Exception:
        return ""


def selfies_to_mol(selfies: str):
    if not HAS_RDKIT:
        return None, ""
    try:
        smiles = selfies_to_smiles(selfies)
        if not smiles:
            return None, ""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, ""
        return mol, Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None, ""


def build_fp_generator():
    if not HAS_RDKIT:
        return None
    return GetMorganGenerator(radius=2, fpSize=1024)


def tanimoto_from_mols(mol_a, mol_b, fpgen) -> float:
    if not HAS_RDKIT or mol_a is None or mol_b is None or fpgen is None:
        return 0.0
    try:
        return float(DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol_a), fpgen.GetFingerprint(mol_b)))
    except Exception:
        return 0.0


def get_environment_info() -> Dict:
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "rdkit_available": HAS_RDKIT,
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
    return info

