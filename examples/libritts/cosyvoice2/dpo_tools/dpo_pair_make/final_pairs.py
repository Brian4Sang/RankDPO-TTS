import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    """Load single-line JSONL (or JSON) thresholds file."""
    with open(thresholds_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty thresholds file: {thresholds_path}")
        data = json.loads(first_line)
    return {
        "pos_sim": float(data.get("positive_sim_threshold", 0.0)),
        "neg_sim": float(data.get("negative_sim_threshold", 1.0)),
        # 可继续扩展 chatscore 门限：data.get("positive_chat_threshold") 等
    }

def load_final_items(final_path: Path) -> List[Dict[str, Any]]:
    """Load list of items from final.jsonl."""
    items = []
    with open(final_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def find_cer_file(subdir: Path) -> Optional[Path]:
    """Return the CER file path in a subdir (try common names)."""
    candidates = ["cer_results.json", "cer.json", "cer.jsonl"]
    for name in candidates:
        p = subdir / name
        if p.exists():
            return p
    return None

def load_cer_ref_map(cer_path: Path) -> Tuple[Dict[str, str], str]:
    """
    Load CER file and return:
      - audio_id -> ref text map
      - fallback_ref (first ref encountered)
    Supports JSON list, JSON dict, or JSONL lines.
    """
    ref_map = {}
    fallback_ref = ""

    # Try load as full JSON first
    try:
        with open(cer_path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return ref_map, fallback_ref
        if raw[0] in "[{":  # Likely JSON list or single dict
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    audio_id = item.get("audio")
                    ref_txt = item.get("ref", "")
                    if audio_id:
                        ref_map[audio_id] = ref_txt
                    if not fallback_ref and ref_txt:
                        fallback_ref = ref_txt
            elif isinstance(data, dict):
                # Could be dict keyed by audio_id or a single record
                # Case 1: {"sample_12": {...}}
                # Case 2: {"audio": "...", "ref": "..."}
                if "audio" in data and "ref" in data:
                    audio_id = data["audio"]
                    ref_txt = data["ref"]
                    ref_map[audio_id] = ref_txt
                    fallback_ref = ref_txt
                else:
                    for audio_id, item in data.items():
                        if isinstance(item, dict):
                            ref_txt = item.get("ref", "")
                            ref_map[audio_id] = ref_txt
                            if not fallback_ref and ref_txt:
                                fallback_ref = ref_txt
            return ref_map, fallback_ref
    except json.JSONDecodeError:
        pass  # fallback to JSONL

    # Fallback: JSONL
    with open(cer_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            audio_id = item.get("audio")
            ref_txt = item.get("ref", "")
            if audio_id:
                ref_map[audio_id] = ref_txt
            if not fallback_ref and ref_txt:
                fallback_ref = ref_txt
    return ref_map, fallback_ref

def select_candidates(items: List[Dict[str, Any]], top_k: int, bottom_k: int) -> Tuple[List[Dict], List[Dict]]:
    """Sort by final_score desc; return top_k, bottom_k slices."""
    if not items:
        return [], []
    sorted_items = sorted(items, key=lambda x: x["final_score"], reverse=True)
    top_items = sorted_items[:top_k]
    bottom_items = sorted_items[-bottom_k:] if bottom_k > 0 else []
    return top_items, bottom_items

def filter_by_sim(candidates: List[Dict[str, Any]], is_positive: bool, sim_threshold: float) -> List[Dict]:
    """
    Keep candidates that satisfy sim >= threshold (positive)
    or sim <= threshold (negative).
    """
    if is_positive:
        return [c for c in candidates if c["sim"] >= sim_threshold]
    else:
        return [c for c in candidates if c["sim"] <= sim_threshold]

def build_pairs_for_subdir(
    subdir: Path,
    thresholds: Dict[str, float],
    top_k: int = 2,
    bottom_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Build DPO pairs for one directory using thresholded positive/negative candidates.
    """
    final_path = subdir / "final.jsonl"
    if not final_path.exists():
        return []

    items = load_final_items(final_path)
    if len(items) < 2:
        return []

    # candidate selection
    pos_cands, neg_cands = select_candidates(items, top_k=top_k, bottom_k=bottom_k)

    # threshold filtering
    pos_cands = filter_by_sim(pos_cands, is_positive=True, sim_threshold=thresholds["pos_sim"])
    neg_cands = filter_by_sim(neg_cands, is_positive=False, sim_threshold=thresholds["neg_sim"])

    if not pos_cands or not neg_cands:
        return []

    # load ref text map
    cer_path = find_cer_file(subdir)
    ref_map, fallback_ref = ({}, "")
    if cer_path:
        ref_map, fallback_ref = load_cer_ref_map(cer_path)

    # Greedy 1-to-1 matching: each positive gets at most one negative
    used_neg = set()
    pairs = []
    for p in pos_cands:
        neg = None
        # pick first unused neg
        for n in neg_cands:
            if n["audio"] in used_neg:
                continue
            neg = n
            used_neg.add(n["audio"])
            break
        if neg is None:
            break  # no neg left

        # text from cer ref map, fallback if missing
        ref_txt = ref_map.get(p["audio"]) or ref_map.get(neg["audio"]) or fallback_ref or ""

        pair = {
            "utt": f"{subdir.name}/{p['audio']}",
            "rejected_utt": f"{subdir.name}/{neg['audio']}",
            "text": ref_txt,
            "speech_score": round(p["final_score"], 5),
            "reject_score": round(neg["final_score"], 5),
        }
        pairs.append(pair)

    return pairs

def build_all_pairs(
    root_dir: Path,
    thresholds_path: Path,
    output_path: Path,
    top_k: int = 2,
    bottom_k: int = 2,
):
    thresholds = load_thresholds(thresholds_path)

    all_pairs: List[Dict[str, Any]] = []
    subdirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name != "results"]

    for subdir in sorted(subdirs):
        sub_pairs = build_pairs_for_subdir(
            subdir=subdir,
            thresholds=thresholds,
            top_k=top_k,
            bottom_k=bottom_k,
        )
        if sub_pairs:
            print(f"✅ {subdir.name}: 生成 {len(sub_pairs)} 对")
            all_pairs.extend(sub_pairs)
        else:
            print(f"⚠️ {subdir.name}: 无符合门限的样本对")

    # save aggregated output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎯 总计输出 {len(all_pairs)} 个 DPO 样本对 → {output_path}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="根据门限文件与各子目录 final.jsonl 构造 DPO 数据对。"
    )
    parser.add_argument("--root", type=str, default="/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200",
                        help="父目录。包含若干子目录及一个 results/score_thresholds.jsonl。")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="门限文件路径；若未指定，自动从 root/results/score_thresholds.jsonl 读取。")
    parser.add_argument("--output", type=str, default=None,
                        help="输出汇总 JSONL；默认 root/results/dpo_pairs.jsonl。")
    parser.add_argument("--top-k", type=int, default=2, help="每目录取前 K 个作正候选。")
    parser.add_argument("--bottom-k", type=int, default=2, help="每目录取后 K 个作负候选。")

    args = parser.parse_args()

    root = Path(args.root)
    if args.thresholds is None:
        thresholds_path = root / "results" / "score_thresholds.jsonl"
    else:
        thresholds_path = Path(args.thresholds)

    if args.output is None:
        output_path = root / "results" / "final-dpo_pairs.jsonl"
    else:
        output_path = Path(args.output)

    build_all_pairs(
        root_dir=root,
        thresholds_path=thresholds_path,
        output_path=output_path,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )