"""Export the Cora DHBP benchmark results and compare against GPF/GPF-plus/Gprompt.

Reads DHBP columns from the per-shot Excel files (just written by the DHBP
benchmark) and the existing local GPF/GPF-plus/Gprompt run (pretrain_fewshot_long.csv,
same GPU / hyperparameters / seed), then emits two CSVs:

- results/dhbp_cora_long.csv          : DHBP results in the same long schema as
                                         pretrain_fewshot_long.csv
- results/dhbp_vs_prompt_cora.csv     : DHBP + GPF + GPF-plus + Gprompt, long format
                                         (method x shot x pretrain), accuracy/F1/AUROC
"""
import os
import pandas as pd

PRETRAINS = ["None", "DGI", "GraphMAE", "Edgepred_GPPT", "Edgepred_Gprompt", "GraphCL", "SimGRACE"]
SHOTS = ["1", "3", "5"]
BASE = "Experiment/ExcelResults/Node"


def _parse_meanstd(s):
    s = str(s).strip()
    if "±" in s:
        m, sd = s.split("±")
        return float(m), float(sd)
    # fallback: just a number
    return float(s), float("nan")


def load_dhbp():
    rows = []
    for shot in SHOTS:
        p = f"{BASE}/{shot}shot/Cora/GCN_total_results.xlsx"
        df = pd.read_excel(p, keep_default_na=False, na_filter=False)
        metric_label = df.iloc[:, 0].tolist()  # ['Final Accuracy','Final F1','Final AUROC']
        col_idx = {metric_label[i]: i for i in range(len(metric_label))}
        for pretrain in PRETRAINS:
            col = f"{pretrain}+DHBP"
            if col not in df.columns:
                continue
            acc = _parse_meanstd(df.loc[col_idx["Final Accuracy"], col])
            f1 = _parse_meanstd(df.loc[col_idx["Final F1"], col])
            auroc = _parse_meanstd(df.loc[col_idx["Final AUROC"], col])
            rows.append({
                "dataset": "Cora", "shot": int(shot), "prompt": "DHBP",
                "pretrain": pretrain,
                "accuracy_mean": acc[0], "accuracy_std": acc[1],
                "f1_mean": f1[0], "f1_std": f1[1],
                "auroc_mean": auroc[0], "auroc_std": auroc[1],
            })
    return pd.DataFrame(rows)


def load_existing_prompts():
    df = pd.read_csv("results/pretrain_fewshot_long.csv", keep_default_na=False)
    df = df[(df["dataset"] == "Cora") & (df["prompt"].isin(["GPF", "GPF-plus", "Gprompt"]))]
    df = df[["dataset", "shot", "prompt", "pretrain",
             "accuracy_mean", "accuracy_std", "f1_mean", "f1_std", "auroc_mean", "auroc_std"]]
    return df


def main():
    dhbp = load_dhbp()
    existing = load_existing_prompts()

    dhbp.to_csv("results/dhbp_cora_long.csv", index=False)
    print(f"Wrote results/dhbp_cora_long.csv  ({len(dhbp)} rows)")

    combined = pd.concat([existing, dhbp], ignore_index=True).sort_values(
        ["shot", "prompt", "pretrain"]
    ).reset_index(drop=True)
    combined.to_csv("results/dhbp_vs_prompt_cora.csv", index=False)
    print(f"Wrote results/dhbp_vs_prompt_cora.csv  ({len(combined)} rows)")

    # ---- readable summary: accuracy wide pivot ----
    acc = combined[["shot", "prompt", "pretrain", "accuracy_mean"]].copy()
    piv = acc.pivot_table(index=["shot", "pretrain"], columns="prompt",
                          values="accuracy_mean", aggfunc="first")
    order = ["DHBP", "GPF", "GPF-plus", "Gprompt"]
    piv = piv[[c for c in order if c in piv.columns]]
    print("\n=== Cora accuracy: DHBP vs GPF / GPF-plus / Gprompt ===")
    print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
