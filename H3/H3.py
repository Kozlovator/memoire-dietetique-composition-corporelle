#!/usr/bin/env python3
# H3.py — Stress émotionnel ↔ consommation de snacks hyper-palatables

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
ALPHA_OK   = 0.70  # seuil acceptable de cohérence interne
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────
# 1. Lecture & nettoyage léger des noms de colonnes
# ───────────────────────────────────────────────────────────────────
def strip_acc(txt):
    return unicodedata.normalize("NFD", str(txt)).encode("ascii", "ignore").decode()

def clean_col(c):
    return re.sub(r"\s+", " ", strip_acc(c).lower()).strip()

df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [clean_col(c) for c in df.columns]

# ───────────────────────────────────────────────────────────────────
# 2. Score de stress émotionnel (5 items Likert 0–3)
# ───────────────────────────────────────────────────────────────────
patterns = {
    "desespere" : r"avez-vous ete desespere\(e\) en pensant au futur\?",
    "tendu"     : r"avez-vous ete tendu\(e\) et nerveux\/nerveuse\?",
    "decourage" : r"avez-vous ete decourage\(e\) et triste\?",
    "preoccupe" : r"avez-vous ete preoccupe\(e\) par differentes choses\?",
    "anxieux"   : r"avez-vous ete effraye\(e\) et anxieux\/anxieuse\?",
}

col_stress = {}
for key, pat in patterns.items():
    matches = [c for c in df.columns if re.search(pat, c)]
    if not matches:
        sys.exit(f"❌ Colonne stress « {key} » introuvable (pat : {pat})")
    col_stress[key] = matches[0]

likert_map   = {
    r"pas du tout": 0,
    r"un peu"     : 1,
    r"moderement" : 2,
    r"beaucoup"   : 3,
}
likert_regex = [(re.compile(p), v) for p, v in likert_map.items()]

def to_likert(cell):
    if pd.isna(cell): return np.nan
    s = strip_acc(cell).lower()
    for rg, v in likert_regex:
        if rg.search(s):
            return v
    return np.nan

for k, col in col_stress.items():
    df[f"stress_{k}"] = df[col].apply(to_likert)

df["stress_score"] = df[[f"stress_{k}" for k in col_stress]].sum(axis=1)

# ───────────────────────────────────────────────────────────────────
# 3. Sélection & transformation des items “snacks hyper-palatables”
# ───────────────────────────────────────────────────────────────────
snack_patterns = {
    "croquettes"         : r"\[croquettes ou friture de pommes de terre\]",
    "friture_viande"     : r"\[friture de viande, de fromage ou de poisson\]",
    "beignets_pommes"    : r"\[beignets aux pommes\]",
    "legumes_frits"      : r"\[legumes frits",
    "chips"              : r"\[chips\]",
    "bonbons"            : r"\[bonbons et chocolat\]",
    "glaces"             : r"\[glaces\]",
    "biscuits"           : r"\[biscuits secs et cake\]",
    "pates_a_tartiner"   : r"\[pates a tartiner",
    "beurre_cacahouetes" : r"\[beurre de cacahouetes\]",
}

col_snack = {}
for k, pat in snack_patterns.items():
    matches = [c for c in df.columns if re.search(pat, c)]
    if not matches:
        sys.exit(f"❌ Colonne snack « {k} » introuvable (pat : {pat})")
    col_snack[k] = matches[0]

freq_map = {
    r"jamais$"                  : 0,
    r"moins d'?1 fois par mois" : 0.125,
    r"1 a 3 fois par mois"      : 0.5,
    r"1 fois par semaine"       : 1,
    r"2 a 3 fois par semaine"   : 2.5,
    r"2 a 4 fois par semaine"   : 3,
    r"5 a 6 fois par semaine"   : 5.5,
    r"plus de 3 fois$"          : 4,
    r"1 fois par jour"          : 7,
    r"2 a 3 fois par jour"      : 17.5,
    r"plus de 3 fois par jour"  : 21,
}
freq_regex = [(re.compile(p), v) for p, v in freq_map.items()]

def to_num(cell):
    if pd.isna(cell): return np.nan
    best = np.nan
    for part in str(cell).split(","):
        txt = strip_acc(part).lower().strip()
        for rg, val in freq_regex:
            if rg.search(txt):
                best = val if np.isnan(best) else max(best, val)
                break
    return best

# Appliquer la conversion en fréquence
for k, coln in col_snack.items():
    df[k] = df[coln].apply(to_num)

snack_df = df[list(col_snack.keys())]

# ───────────────────────────────────────────────────────────────────
# 4. Imputation médiane + standardisation + cohérence interne
# ───────────────────────────────────────────────────────────────────
# Imputation par la médiane de chaque item snack
medians   = snack_df.median()
snack_imp = snack_df.fillna(medians)

# Standardisation (z-score)
z_df = (snack_imp - snack_imp.mean()) / snack_imp.std(ddof=0)

# Cronbach α
k_items   = z_df.shape[1]
vars_item = z_df.var(ddof=0)
alpha     = k_items / (k_items - 1) * (1 - vars_item.sum() / z_df.sum(axis=1).var(ddof=0))
print(f"Cronbach α snacks = {alpha:.3f}")
if alpha < ALPHA_OK:
    warnings.warn("⚠️ Cohérence interne faible pour le score snacks (α < 0.70)")

# ───────────────────────────────────────────────────────────────────
# 5. Score snack = moyenne des z-scores
# ───────────────────────────────────────────────────────────────────
# on récupère la moyenne des colonnes standardisées
df["snack_score"] = z_df.mean(axis=1)

# ───────────────────────────────────────────────────────────────────
# 6. Corrélation & régression
# ───────────────────────────────────────────────────────────────────
valid = df.dropna(subset=["stress_score", "snack_score"])
if valid.empty:
    sys.exit("❌ Aucune ligne complète pour stress_score + snack_score.")

r_s, p_s = stats.spearmanr(valid["stress_score"], valid["snack_score"])
lin     = stats.linregress(valid["stress_score"], valid["snack_score"])

# ───────────────────────────────────────────────────────────────────
# 7. Graphique
# ───────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.figure(figsize=(7,5))
sns.regplot(x="stress_score", y="snack_score", data=valid,
            scatter_kws={"alpha":0.6}, line_kws={"linewidth":1.5})
plt.title("Stress émotionnel vs score snack (moyenne z-scores)")
plt.xlabel("Score stress (0–15)")
plt.ylabel("Score snack (mean z)")
plt.annotate(f"Spearman ρ = {r_s:.2f}\n p = {p_s:.3f}",
             xy=(0.05, 0.92), xycoords="axes fraction")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "H3_corr_stress_snack.png"), dpi=300)
plt.close()

# ───────────────────────────────────────────────────────────────────
# 8. Résumé texte
# ───────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "H3_resume.txt"), "w", encoding="utf-8") as f:
    f.write(
        "H3 – Stress émotionnel & consommation de snacks hyper-palatables\n\n"
        f"Cronbach α snacks = {alpha:.3f}  (items : {', '.join(col_snack.keys())})\n\n"
        f"Spearman ρ = {r_s:.3f}  (p = {p_s:.4f})\n"
        f"Régression linéaire : β = {lin.slope:.3f} per point stress, "
        f"R² = {lin.rvalue**2:.3f}\n"
        f"N = {len(valid)} participants complets\n"
    )

print("✅ H3 terminé – résultats & images dans", OUT_DIR)
