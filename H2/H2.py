#!/usr/bin/env python3
# H2.py — comportements à risque : Femme vs Homme

import os, re, unicodedata, sys, warnings
import pandas as pd, numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

warnings.filterwarnings("ignore")

# ─────────────────────────────── 0. CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV        = os.path.join(SCRIPT_DIR, "..", "data.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────── 1. Lecture & nettoyage
def strip(txt):   # retire accents + espaces doubles
    txt = unicodedata.normalize("NFD", str(txt)).encode("ascii","ignore").decode()
    return re.sub(r"\s+", " ", txt).lower().strip()

df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [strip(c) for c in df.columns]

# ─────────────────────────────── 2. Colonnes utiles
sex_col   = next(c for c in df.columns if "quel est votre sexe" in c)

# oui / non “Est-ce que je saute des repas ?”
skip_col  = next(c for c in df.columns if "est-ce que je saute des repas" in c)

# items SCOFF (cinq questions oui/non)
pat_scoff = {
    "sick"   : "fait vomir",
    "control": "controle des quantites",
    "stone"  : "perdu plus de 6 kilos",
    "fat"    : "etes trop gros",
    "food"   : "nourriture est quelque chose"
}
col_scoff = {k: next(c for c in df.columns if pat in c) for k,pat in pat_scoff.items()}

# ─────────────────────────────── 3. Préparation
df["Sexe"]     = np.where(df[sex_col].str.lower()=="femme", "Femme", "Homme")
df["skipper"]  = df[skip_col].str.lower().eq("oui")          # True = saute des repas

for k,col in col_scoff.items():
    df[f"scoff_{k}"] = df[col].str.lower().eq("oui")
df["score_scoff"] = df[[f"scoff_{k}" for k in col_scoff]].sum(axis=1)

# ─────────────────────────────── 4. Statistiques
# 4-A  χ² sur la prévalence des sauts de repas
ct_skip = pd.crosstab(df["Sexe"], df["skipper"])
chi2, p_skip, _, _ = stats.chi2_contingency(ct_skip)

# 4-B  comparaison score SCOFF
f_scoff = df.loc[df["Sexe"]=="Femme",  "score_scoff"]
h_scoff = df.loc[df["Sexe"]=="Homme",  "score_scoff"]
normal  = stats.shapiro(f_scoff)[1]>.05 and stats.shapiro(h_scoff)[1]>.05
if normal:
    test, (stat, p_scoff) = "t-test Welch", stats.ttest_ind(f_scoff, h_scoff, equal_var=False)
else:
    test, (stat, p_scoff) = "Mann-Whitney U", stats.mannwhitneyu(f_scoff, h_scoff)

def cohen_d(a,b):
    nx,ny=len(a),len(b)
    pooled=np.sqrt(((nx-1)*a.var(ddof=1)+(ny-1)*b.var(ddof=1))/(nx+ny-2))
    return (a.mean()-b.mean())/pooled
d  = cohen_d(f_scoff, h_scoff)
rng = np.random.default_rng(42)
ci_low,ci_high = np.percentile([cohen_d(rng.choice(f_scoff,len(f_scoff),True),
                                        rng.choice(h_scoff,len(h_scoff),True))
                                for _ in range(2000)],
                               [2.5,97.5])

# ─────────────────────────────── 5. Graphiques
sns.set_style("whitegrid")

# 5-A Barplot empilé (proportions)
prop = ct_skip.div(ct_skip.sum(axis=1), axis=0)
prop.plot(kind="bar", stacked=True, figsize=(6,4), colormap="Set2")
plt.title("Saut de repas : proportion Oui/Non par sexe")
plt.ylabel("Proportion")
plt.xlabel("Sexe")
plt.legend(title="Saute repas ?", labels=["Non","Oui"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H2_prop_skip.png"), dpi=300)
plt.close()

# 5-B Raincloud plot SCOFF

plt.figure(figsize=(7,5))
pt.RainCloud(x="Sexe", y="score_scoff", data=df,
             palette="Set2", bw=.2, width_viol=.6,
             orient="h", alpha=.65, move=.2)
plt.title("Raincloud – score SCOFF par sexe")
plt.xlabel("Score SCOFF (0-5)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H2_raincloud_scoff.png"), dpi=300)
plt.close()

# ─────────────────────────────── 6. Résumé texte
with open(os.path.join(OUT_DIR,"H2_resume.txt"),"w",encoding="utf-8") as f:
    f.write(
        f"H2 – Sexe et comportements à risque\n\n"
        f"Saut de repas : χ²={chi2:.2f}, p={p_skip:.4f}\n\n"
        f"SCOFF ({test}) : stat={stat:.3f}, p={p_scoff:.4f}\n"
        f"Cohen d = {d:.3f}  IC95 % [{ci_low:.2f}; {ci_high:.2f}]\n"
        f"N Femmes={len(f_scoff)}   |   N Hommes={len(h_scoff)}\n"
    )

print("✅ H2 terminé – résultats & images dans", OUT_DIR)
