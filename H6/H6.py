#!/usr/bin/env python3
# H6old.py — Impact perçu (énergie & satisfaction) : HP vs AI
# Hypothèse : l’impact est évalué plus positivement chez les sujets HP.

import os, re, unicodedata, sys, warnings
import pandas as pd, numpy as np
from   scipy import stats
import seaborn as sns, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ─────────────────── 0. CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV        = os.path.join(SCRIPT_DIR, "..", "data.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

def strip(txt:str)->str:
    """retire accents, espaces multiples, met en minuscules"""
    return re.sub(r"\s+", " ",
                  unicodedata.normalize("NFD", str(txt))
                             .encode("ascii","ignore").decode()).lower().strip()

# ─────────────────── 1. LECTURE
df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [strip(c) for c in df.columns]

# ─────────────────── 2. GROUPES HP / AI
col_hp = next(c for c in df.columns if "regime hyperproteine" in c)
col_ai = next(c for c in df.columns if "alimentation est intuitive" in c)
df["grp_hp"] = df[col_hp].str.lower().eq("oui")
df["grp_ai"] = df[col_ai].str.lower().eq("oui") & ~df["grp_hp"]

df_grp = df[df["grp_hp"] | df["grp_ai"]].copy()
df_grp["Groupe"] = np.where(df_grp["grp_hp"], "HP", "AI")
print("N participants retenus :", len(df_grp))

# ─────────────────── 3. IMPACT (énergie & satisfaction)
pat_imp = r"effets de votre alimentation actuelle sur votre energie"
col_imp = next((c for c in df.columns if re.search(pat_imp, c)), None)
if col_imp is None:
    sys.exit("❌ Colonne impact énergie introuvable.")

# mapping texte ➜ score 1–5
score_map = {
    r"tres negative"   : 1,
    r"plutot negative" : 2,
    r"neutre"          : 3,
    r"plutot positive" : 4,
    r"tres positive"   : 5,
}
score_rgx = [(re.compile(p), v) for p,v in score_map.items()]

def to_score(cell):
    if pd.isna(cell): return np.nan
    s = strip(cell)
    for rg,v in score_rgx:
        if rg.search(s): return v
    return np.nan

df_grp["impact_score"] = df_grp[col_imp].apply(to_score)
df_grp = df_grp.dropna(subset=["impact_score"])        # exclut non-réponses

# ─────────────────── 4. STATISTIQUES (Mann-Whitney U)
hp = df_grp.loc[df_grp["grp_hp"], "impact_score"]
ai = df_grp.loc[df_grp["grp_ai"], "impact_score"]

stat_s, p_s = stats.mannwhitneyu(hp, ai, alternative="two-sided")

def cohen_d(a,b):
    nx,ny = len(a),len(b)
    pooled = np.sqrt(((nx-1)*a.var(ddof=1)+(ny-1)*b.var(ddof=1))/(nx+ny-2))
    return (a.mean()-b.mean())/pooled
d = cohen_d(hp, ai)

# bootstrap IC95 % de la différence de moyennes
rng   = np.random.default_rng(42)
diffs = [rng.choice(hp, len(hp), True).mean() -
         rng.choice(ai, len(ai), True).mean() for _ in range(2000)]
ci_low, ci_high = np.percentile(diffs, [2.5,97.5])

# ─────────────────── 5. GRAPHIQUES
sns.set_style("whitegrid")

# 5-A violin + moyenne ± IC95 % par groupe
plt.figure(figsize=(8,5))
order = ["AI", "HP"]
sns.violinplot(data=df_grp, x="Groupe", y="impact_score",
               order=order, inner=None, palette="Set2")

# moyenne & IC95 % bootstrap par groupe
for i, grp in enumerate(order):
    scores = df_grp.loc[df_grp["Groupe"]==grp, "impact_score"]
    m      = scores.mean()
    cis    = np.percentile([rng.choice(scores, len(scores), True).mean()
                            for _ in range(2000)], [2.5,97.5])
    yerr   = [[m - cis[0]], [cis[1] - m]]          # toujours ≥0
    plt.errorbar(i, m, yerr=yerr, fmt="o", color="k", capsize=5)

plt.title("Impact énergie & satisfaction")
plt.xlabel("")
plt.ylabel("Score 1 (T. négative)  … 5 (T. positive)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "H6_violin_mean_ci.png"), dpi=300)
plt.close()

# 5-B barplot empilé (5 catégories)
label_map = {
    1:"Très négative", 2:"Plutôt négative", 3:"Neutre",
    4:"Plutôt positive", 5:"Très positive"
}
df_grp["impact_label"] = df_grp["impact_score"].map(label_map)
order_labels = ["Très négative","Plutôt négative","Neutre",
                "Plutôt positive","Très positive"]
table_cat = (pd.crosstab(df_grp["Groupe"], df_grp["impact_label"])
               .reindex(columns=order_labels, fill_value=0))
(table_cat.div(table_cat.sum(axis=1),0)
 ).plot(kind="bar", stacked=True, figsize=(7,4), colormap="Set2")
plt.title("Répartition des réponses (5 catégories)")
plt.ylabel("Proportion")
plt.xlabel("")
plt.legend(title="Réponse", bbox_to_anchor=(1.02,1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "H6_prop_impact.png"), dpi=300)
plt.close()

# ─────────────────── 6. RÉSUMÉ
with open(os.path.join(OUT_DIR, "H6_resume.txt"), "w", encoding="utf-8") as f:
    f.write(
        "H6old – Impact perçu énergie & satisfaction : HP vs AI\n\n"
        f"Mann-Whitney U : stat = {stat_s:.3f}, p = {p_s:.4f}\n"
        f"Cohen d       : {d:.3f}\n"
        f"IC95 % diff. moyennes : [{ci_low:.2f} ; {ci_high:.2f}]\n"
        f"N HP = {len(hp)}  |  N AI = {len(ai)}\n"
    )

print("✅ H6old terminé – graphiques et résumé disponibles dans ‘results/’")
