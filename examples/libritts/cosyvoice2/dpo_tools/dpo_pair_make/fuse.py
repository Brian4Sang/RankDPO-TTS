
import os
import json
from glob import glob

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            # 调整字段顺序，放到 sim 后面
            new_item = {}
            for k, v in item.items():
                if k == "sim":
                    new_item[k] = v
                    if "chatscore" in item:
                        new_item["chatscore"] = item["chatscore"]
                    if "final_score" in item:
                        new_item["final_score"] = item["final_score"]
                elif k not in ("chatscore", "final_score"):
                    new_item[k] = v
            f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

def fuse_scores(root_dir, input_name="final_merged.jsonl", output_name="final.jsonl", w_sim=0.4, w_chat=0.4, w_cer=0.2):
    subdirs = [d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
    total = 0
    for sub in subdirs:
        in_path = os.path.join(sub, input_name)
        out_path = os.path.join(sub, output_name)
        if not os.path.exists(in_path):
            continue

        data = load_jsonl(in_path)
        for item in data:
            sim = item.get("sim", 0.0)
            cer = item.get("cer", 1.0)
            chat = item.get("chatscore", 0.0)
            final = w_sim * sim + w_chat * chat + w_cer * (1 - cer)
            item["final_score"] = round(final, 5)

        # ✅ 加入排序
        data.sort(key=lambda x: x["final_score"], reverse=True)

        save_jsonl(data, out_path)
        print(f"✅ {os.path.basename(sub)}: fused {len(data)} items -> {output_name}")
        total += len(data)

    print(f"🎉 总共处理完成，融合打分 {total} 条记录。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/dpo_data/female/short/train-1200")
    parser.add_argument("--input_name", default="css.jsonl")
    parser.add_argument("--output_name", default="final_fuse.jsonl")
    parser.add_argument("--w_sim", type=float, default=0.4)
    parser.add_argument("--w_chat", type=float, default=0.5)
    parser.add_argument("--w_cer", type=float, default=0.1)
    args = parser.parse_args()

    fuse_scores(
        root_dir=args.root_dir,
        input_name=args.input_name,
        output_name=args.output_name,
        w_sim=args.w_sim,
        w_chat=args.w_chat,
        w_cer=args.w_cer
    )
