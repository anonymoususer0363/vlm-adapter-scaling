#!/usr/bin/env python3
import json, csv, os
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
dirs = [
    Path(os.environ["VLM_NAS_CHECKPOINT_DIR"]) if os.environ.get("VLM_NAS_CHECKPOINT_DIR") else None,
    Path(os.environ.get("VLM_CHECKPOINT_DIR", _project_root / "checkpoints")),
]
dirs = [d for d in dirs if d is not None]
out = _project_root / "analysis" / "results_all.csv"
seen = set()
rows = []
incomplete = []

for d in dirs:
    src = "NAS" if "nas_109" in str(d) else "local"
    if not d.exists():
        continue
    for rd in sorted(d.iterdir()):
        if not rd.is_dir():
            continue
        rp = rd / "result.json"
        cp = rd / "config.json"
        if not rp.exists():
            incomplete.append((src, rd.name))
            continue
        r = json.load(open(rp))
        rn = r.get("run_name", rd.name)
        if rn in seen:
            continue
        seen.add(rn)
        c = json.load(open(cp)) if cp.exists() else {}
        # parse group from run_name
        g = ""
        if rn.startswith("rerun"):
            g = "rerun"
        elif rn.startswith("g"):
            g = rn.split("_")[0]
        # parse llm_size
        ls = ""
        for s in ["0.5B","1.5B","3B","7B","14B","32B"]:
            if s in c.get("llm_name",""):
                ls = s
                break
        rows.append({
            "run_name": rn, "source": src, "group": g, "llm_size": ls,
            "final_val_loss": r.get("final_val_loss"),
            "best_val_loss": r.get("best_val_loss"),
            "total_steps": r.get("total_steps"),
            "seen_pairs": r.get("seen_pairs"),
            "adapter_params": r.get("adapter_params"),
            "vision_T0": r.get("vision_T0"),
            "vision_T": r.get("vision_T"),
            "vision_rho": r.get("vision_rho"),
            "vision_N_A": r.get("vision_N_A"),
            "vision_d_model": r.get("vision_d_model"),
            "vision_adapter_num_layers": r.get("vision_adapter_num_layers"),
            "vision_image_size": r.get("vision_image_size"),
            "llm_name": c.get("llm_name",""),
            "adapter_level": c.get("adapter_level",""),
            "num_queries": c.get("num_queries"),
            "num_samples": c.get("num_samples"),
            "num_epochs": c.get("num_epochs",1),
            "seed": c.get("seed",42),
            "use_lora": c.get("use_lora",False),
            "batch_size": c.get("batch_size"),
            "lr": c.get("lr"),
        })

out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"Total: {len(rows)} experiments -> {out}")
groups = {}
for r in rows:
    groups[r["group"]] = groups.get(r["group"], 0) + 1
print("Groups:", {k: groups[k] for k in sorted(groups)})
print(f"Incomplete: {len(incomplete)}")
for s, n in incomplete:
    print(f"  [{s}] {n}")
