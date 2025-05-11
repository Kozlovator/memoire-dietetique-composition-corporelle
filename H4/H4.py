#!/usr/bin/env python3
# H4.py — Fréquence d’activité physique ↔ santé perçue
# ↑ séances sportives → ↑ santé auto-évaluée

import os, re, unicodedata, sys, warnings
import pandas as pd, numpy as np
from scipy import stats
import seaborn as sns, matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ───────────────────────── 0. CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV        = os.path.join(SCRIPT_DIR, "..", "data.csv")
OUT_DIR    = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────── 1. Lecture & nettoyage
def strip(txt):
    return re.sub(r"\s+", " ",
                  unicodedata.normalize("NFD", str(txt))
                  .encode("ascii","ignore")
                  .decode()).lower().strip()

df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [strip(c) for c in df.columns]

# ───────────────────────── 2. Repérage des deux colonnes
col_health = next(c for c in df.columns
                  if "comment est votre etat de sante en general" in c)

# fréquence d’activité physique en salle de sport
col_freq = next(c for c in df.columns
                if "pratiquez-vous une activite physique reguliere ?" in c)

print("Colonnes retenues :")
print("  Santé  →", col_health)
print("  Sport  →", col_freq)

# ───────────────────────── 3. Recodage en valeurs numériques
# 3-A  Santé perçue (1 = très mauvaise … 5 = très bonne)
health_map = {
    r"^tres mauvais$"          : 1,
    r"^mauvais$"               : 2,
    r"^ni bon, ni mauvais$"    : 3,
    r"^bon$"                   : 4,
    r"^tres bon$"              : 5,
}
health_rgx = [(re.compile(p), v) for p,v in health_map.items()]

def to_health(cell):
    if pd.isna(cell): return np.nan
    s = strip(cell)
    for rg,v in health_rgx:
        if rg.search(s): return v
    return np.nan

df["health_score"] = df[col_health].apply(to_health)

# 3-B  Fréquence sport : séances / semaine
freq_map = {
    r"jamais"                     : 0,
    r"rarement"                   : 0.5,
    r"1 a 3 fois par semaine"     : 2,
    r"plus de 3 fois par semaine" : 5,
}
freq_rgx = [(re.compile(p), v) for p, v in freq_map.items()]

def to_freq(cell):
    if pd.isna(cell): return np.nan
    s = strip(cell)
    for rg,v in freq_rgx:
        if rg.search(s): return v
    return np.nan

df["sport_freq"] = df[col_freq].apply(to_freq)

# ───────────────────────── 4. Imputation médiane + comptage des remplacements
miss_health = df["health_score"].isna().sum()
miss_sport  = df["sport_freq"].isna().sum()

med_health  = df["health_score"].median()
med_sport   = df["sport_freq"].median()

df["health_score"].fillna(med_health, inplace=True)
df["sport_freq"].fillna(med_sport,  inplace=True)

print(f"Remplacements par médiane : santé={miss_health}  |  sport={miss_sport}")

# ───────────────────────── 5. Corrélation & régression
valid = df[["health_score","sport_freq"]].dropna()
r_s, p_s = stats.spearmanr(valid["sport_freq"], valid["health_score"])
lin      = stats.linregress(valid["sport_freq"], valid["health_score"])

# ───────────────────────── 6. Graphique
sns.set_style("whitegrid")
plt.figure(figsize=(7,5))
sns.regplot(x="sport_freq", y="health_score", data=valid,
            scatter_kws={"alpha":0.6}, line_kws={"linewidth":1.5})
plt.title("Fréquence activité physique vs santé perçue")
plt.xlabel("Séances sport / semaine")
plt.ylabel("Santé perçue (1=très mauvaise – 5=très bonne)")
plt.annotate(f"Spearman ρ = {r_s:.2f}\np = {p_s:.3f}",
             xy=(0.05,0.92), xycoords="axes fraction")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"H4_sport_health.png"), dpi=300)
plt.close()

# ───────────────────────── 7. Résumé texte
with open(os.path.join(OUT_DIR,"H4_resume.txt"),"w",encoding="utf-8") as f:
    f.write(
        "H4 – Activité physique et santé perçue\n\n"
        f"Colonnes :\n  Santé  → {col_health}\n  Sport  → {col_freq}\n\n"
        f"Valeurs manquantes imputées par médiane :\n"
        f"  Santé  : {miss_health} / {len(df)}\n"
        f"  Sport  : {miss_sport} / {len(df)}\n\n"
        f"Spearman ρ = {r_s:.3f}  (p = {p_s:.4f})\n"
        f"Régression linéaire : β = {lin.slope:.3f} point santé / séance, "
        f"R² = {lin.rvalue**2:.3f}\n"
        f"N = {len(valid)} participants complets\n"
    )

print("✅ H4 terminé – résultats & images dans", OUT_DIR)
