#!/usr/bin/env python3
# H5.py — SCOFF : AI vs HP

import os, re, unicodedata, sys, warnings
import pandas as pd, numpy as np
from scipy import stats
import seaborn as sns, matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────
# 0. CONFIG
# ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV        = os.path.join(SCRIPT_DIR, "..", "data.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────
# 1. Lecture & nettoyage des noms de colonnes
# ───────────────────────────────────────────────────────────────────
def strip(txt):
    return re.sub(r"\s+", " ",
                  unicodedata.normalize("NFD", str(txt))
                             .encode("ascii","ignore")
                             .decode()).lower().strip()

df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [strip(c) for c in df.columns]

# ───────────────────────────────────────────────────────────────────
# 2. Calcul du score SCOFF (0–5)
# ───────────────────────────────────────────────────────────────────
scoff_items = {
    "sick"   : "fait vomir",
    "control": "controle des quantites",
    "stone"  : "perdu plus de 6 kilos",
    "fat"    : "etes trop gros",
    "food"   : "nourriture est quelque chose",
}
# retrouver chaque colonne
col_scoff = {k: next(c for c in df.columns if v in c) for k,v in scoff_items.items()}
# binariser puis sommer
for k, col in col_scoff.items():
    df[f"scoff_{k}"] = df[col].str.lower().eq("oui")
df["score_scoff"] = df[[f"scoff_{k}" for k in col_scoff]].sum(axis=1)

# ───────────────────────────────────────────────────────────────────
# 3. Définition des groupes AI vs HP
# ───────────────────────────────────────────────────────────────────
col_hp = next(c for c in df.columns if "regime hyperproteine" in c)
col_ai = next(c for c in df.columns if "alimentation est intuitive" in c)

df["grp_hp"] = df[col_hp].str.lower().eq("oui")
df["grp_ai"] = df[col_ai].str.lower().eq("oui") & ~df["grp_hp"]

df_grp = df[df["grp_hp"] | df["grp_ai"]].copy()
df_grp["Groupe"] = np.where(df_grp["grp_hp"], "HP", "AI")
print(f"N participants retenus : {len(df_grp)}")

# ───────────────────────────────────────────────────────────────────
# 4. Test statistique
# ───────────────────────────────────────────────────────────────────
hp = df_grp[df_grp["grp_hp"]]["score_scoff"]
ai = df_grp[df_grp["grp_ai"]]["score_scoff"]

# normalité ?
norm = (stats.shapiro(hp)[1] > .05) and (stats.shapiro(ai)[1] > .05)
if norm:
    test, (stat, pval) = "t-test Welch", stats.ttest_ind(hp, ai, equal_var=False)
else:
    test, (stat, pval) = "Mann-Whitney U", stats.mannwhitneyu(hp, ai)

def cohen_d(a, b):
    nx, ny = len(a), len(b)
    pooled = np.sqrt(((nx-1)*a.var(ddof=1) + (ny-1)*b.var(ddof=1)) / (nx+ny-2))
    return (a.mean() - b.mean()) / pooled

d = cohen_d(hp, ai)
rng = np.random.default_rng(42)
boots = [cohen_d(rng.choice(hp, len(hp), True),
                 rng.choice(ai, len(ai), True)) for _ in range(2000)]
ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

# ───────────────────────────────────────────────────────────────────
# 5. Graphique (rain cloud)
# ───────────────────────────────────────────────────────────────────
import ptitprince as pt
plt.figure(figsize=(7,5))
pt.RainCloud(x="Groupe", y="score_scoff", data=df_grp,
             palette="Set2", bw=.2, width_viol=.6,
             orient="h", alpha=.65, move=.2)
plt.title("Raincloud plot du score SCOFF par groupe")
plt.xlabel("Score SCOFF (0–5)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "H5_raincloud_scoff.png"), dpi=300)
plt.close()


# ───────────────────────────────────────────────────────────────────
# 6. Rapport texte
# ───────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "H5_resume.txt"), "w", encoding="utf-8") as f:
    f.write(
        "H5 – Relation AI vs HP sur le score SCOFF\n\n"
        f"Test comparatif : {test}\n"
        f"stat = {stat:.3f}  p = {pval:.4f}\n"
        f"Cohen d = {d:.3f}  IC95% [{ci_low:.2f}; {ci_high:.2f}]\n"
        f"N HP = {len(hp)}   |   N AI = {len(ai)}\n"
    )

print("✅ H5 terminé – diagramme et résumé dans ‘results/’")
