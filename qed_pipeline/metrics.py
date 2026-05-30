from typing import Dict, List


def summarize_pairs(pairs) -> Dict[str, float]:
    if not pairs:
        return {
            "pair_count": 0,
            "avg_pair_similarity": 0.0,
            "avg_pair_dqed": 0.0,
            "min_pair_similarity": 0.0,
            "min_pair_dqed": 0.0,
        }
    return {
        "pair_count": len(pairs),
        "avg_pair_similarity": sum(float(p[6]) for p in pairs) / len(pairs),
        "avg_pair_dqed": sum(float(p[3]) - float(p[2]) for p in pairs) / len(pairs),
        "min_pair_similarity": min(float(p[6]) for p in pairs),
        "min_pair_dqed": min(float(p[3]) - float(p[2]) for p in pairs),
    }


def compute_eval_metrics(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {
            "validity": 0.0,
            "dQED": 0.0,
            "SimSrc": 0.0,
            "SimTgt": 0.0,
            "positive_dqed_rate": 0.0,
            "sim30_rate": 0.0,
            "sim40_rate": 0.0,
            "opt30_rate": 0.0,
            "opt40_rate": 0.0,
            "valid_count": 0,
            "invalid_count": 0,
        }
    valid_records = [r for r in records if r.get("valid")]
    valid_count = len(valid_records)
    invalid_count = len(records) - valid_count
    if valid_count == 0:
        return {
            "validity": 0.0,
            "dQED": 0.0,
            "SimSrc": 0.0,
            "SimTgt": 0.0,
            "positive_dqed_rate": 0.0,
            "sim30_rate": 0.0,
            "sim40_rate": 0.0,
            "opt30_rate": 0.0,
            "opt40_rate": 0.0,
            "valid_count": 0,
            "invalid_count": invalid_count,
        }
    avg_dqed = sum(float(r["dQED"]) for r in valid_records) / valid_count
    avg_sim_src = sum(float(r["sim_to_source"]) for r in valid_records) / valid_count
    avg_sim_tgt = sum(float(r["sim_to_target"]) for r in valid_records) / valid_count
    positive_dqed_rate = sum(float(r["dQED"]) > 0.0 for r in valid_records) / valid_count
    sim30_rate = sum(float(r["sim_to_source"]) >= 0.30 for r in valid_records) / valid_count
    sim40_rate = sum(float(r["sim_to_source"]) >= 0.40 for r in valid_records) / valid_count
    opt30_rate = sum(float(r["dQED"]) > 0.0 and float(r["sim_to_source"]) >= 0.30 for r in valid_records) / valid_count
    opt40_rate = sum(float(r["dQED"]) > 0.0 and float(r["sim_to_source"]) >= 0.40 for r in valid_records) / valid_count
    return {
        "validity": valid_count / len(records),
        "dQED": avg_dqed,
        "SimSrc": avg_sim_src,
        "SimTgt": avg_sim_tgt,
        "positive_dqed_rate": positive_dqed_rate,
        "sim30_rate": sim30_rate,
        "sim40_rate": sim40_rate,
        "opt30_rate": opt30_rate,
        "opt40_rate": opt40_rate,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
    }
