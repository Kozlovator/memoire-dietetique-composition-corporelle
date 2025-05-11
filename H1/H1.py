#!/usr/bin/env python3
# H1.py — Comparaison du score « malbouffe » entre régimes hyperprotéiné vs intuitive

import os, re, unicodedata, sys, warnings
import pandas as pd, numpy as np
from scipy import stats
import seaborn as sns, matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────
# 0. CONFIG
# ───────────────────────────────────────────────────────────────────
FICHIER    = "../data.csv"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "results")
ALPHA_OK   = .70
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────
# 1. Lecture & nettoyage léger des noms de colonnes
# ───────────────────────────────────────────────────────────────────
def strip_accents(txt):
    return unicodedata.normalize("NFD", txt).encode("ascii", "ignore").decode()

def clean_col(c):
    return re.sub(r"\s+", " ", strip_accents(str(c)).lower()).strip()

df = pd.read_csv(FICHIER, encoding="utf-8-sig")
df.columns = [clean_col(c) for c in df.columns]

# ───────────────────────────────────────────────────────────────────
# 2. Groupes HP vs AI
# ───────────────────────────────────────────────────────────────────
col_hp = next(c for c in df.columns if "regime hyperproteine" in c)
col_ai = next(c for c in df.columns if "alimentation est intuitive" in c)

df["grp_hp"] = df[col_hp].str.strip().str.lower().eq("oui")
df["grp_ai"] = df[col_ai].str.strip().str.lower().eq("oui") & ~df["grp_hp"]

df_grp = df[df["grp_hp"] | df["grp_ai"]].copy()
print(f"N participants retenus : {len(df_grp)}")

# ───────────────────────────────────────────────────────────────────
# 3. Colonnes exactes « malbouffe »
# ───────────────────────────────────────────────────────────────────
raw_labels = {
    "frites"             : "Consommation moyenne de fritures durant les 12 derniers mois  [Frites]",
    "croquettes"         : "Consommation moyenne de fritures durant les 12 derniers mois  [Croquettes ou friture de pommes de terre]",
    "friture_viande"     : "Consommation moyenne de fritures durant les 12 derniers mois  [Friture de viande, de fromage ou de poisson]",
    "prep_viande"        : "Consommation moyenne de viande, poisson, oeufs et sustituts durant les 12 derniers mois  [Prérarations à base de viande (saucisses, hamburger,...)]",
    "pizza_lasagne"      : "Consommation moyenne de plats préparés prêts à l'emploi durant les 12 derniers mois [Plats préparés du commerce (surgelé ou frigo - lasagne, pizza, ...) super-marché, traiteur ou du boucher]",
    "chips"              : "Consommation moyenne de confiseries, pâtes à tartiner (sucrées) et noix durant les 12 derniers mois [Chips]",
    "bonbons"            : "Consommation moyenne de confiseries, pâtes à tartiner (sucrées) et noix durant les 12 derniers mois [Bonbons et chocolat]",
    "glaces"             : "Consommation moyenne de confiseries, pâtes à tartiner (sucrées) et noix durant les 12 derniers mois [Glaces]",
    "biscuits"           : "Consommation moyenne de confiseries, pâtes à tartiner (sucrées) et noix durant les 12 derniers mois [Biscuits secs et cake]",
    "sodas_sucres"       : "Consommation moyenne des boissons durant les 12 derniers mois  [Sodas sucrés]",
    "charcuterie_grasse" : "Consommation moyenne de viande, poisson, oeufs et sustituts durant les 12 derniers mois  [Charcuterie grasse (salami, pâté...)]",
    "salades_tartiner"   : "Consommation moyenne de viande, poisson, oeufs et sustituts durant les 12 derniers mois  [Salades à tartiner à base de mayonnaise (salade de viande, de poisson, de crevettes, de légume,...)]",
}

mb_cols = {
    key: clean_col(label)
    for key, label in raw_labels.items()
    if clean_col(label) in df_grp.columns
}
if not mb_cols:
    sys.exit("❌ Aucune colonne « malbouffe » trouvée.")
print("Items malbouffe conservés :", ", ".join(mb_cols))

# ───────────────────────────────────────────────────────────────────
# 4. Mapping texte ➜ fréquence hebdo
# ───────────────────────────────────────────────────────────────────
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
    if pd.isna(cell):
        return np.nan
    best = np.nan
    for part in str(cell).split(","):
        txt = strip_accents(part).lower().strip()
        for rg, val in freq_regex:
            if rg.search(txt):
                best = val if np.isnan(best) else max(best, val)
                break
    return best

# Appliquer le mapping text → fréquence et construire num_df
for short, col in mb_cols.items():
    df_grp[short] = df_grp[col].apply(to_num)

num_df = df_grp[list(mb_cols.keys())]

# ───────────────────────────────────────────────────────────────────
# 5. Imputation par la médiane + standardisation + cohérence interne
# ───────────────────────────────────────────────────────────────────
medians         = num_df.median()
num_df_imputed  = num_df.fillna(medians)
z_df            = (num_df_imputed - num_df_imputed.mean()) / num_df_imputed.std(ddof=0)

# Cronbach α
k              = z_df.shape[1]
item_var       = z_df.var(ddof=0)
alpha          = k / (k-1) * (1 - item_var.sum() / z_df.sum(axis=1).var(ddof=0))
print(f"Cronbach α = {alpha:.3f}")
if alpha < ALPHA_OK:
    warnings.warn("⚠️ Cohérence interne faible : le score global est peut-être discutable.")

# Score « malbouffe » = moyenne des z-scores
df_grp["score_mb"] = z_df.mean(axis=1)

# ───────────────────────────────────────────────────────────────────
# 6. Comparaison statistique HP vs AI
# ───────────────────────────────────────────────────────────────────
hp, ai = df_grp[df_grp["grp_hp"]]["score_mb"], df_grp[df_grp["grp_ai"]]["score_mb"]

norm_hp, norm_ai = stats.shapiro(hp)[1], stats.shapiro(ai)[1]
if norm_hp > .05 and norm_ai > .05:
    test, (stat, pval) = "t-test Welch", stats.ttest_ind(hp, ai, equal_var=False)
else:
    test, (stat, pval) = "Mann-Whitney U", stats.mannwhitneyu(hp, ai)

def cohen_d(a, b):
    nx, ny = len(a), len(b)
    pooled = np.sqrt(((nx-1)*a.var(ddof=1) + (ny-1)*b.var(ddof=1)) / (nx+ny-2))
    return (a.mean() - b.mean()) / pooled

d        = cohen_d(hp, ai)
rng      = np.random.default_rng(42)
boots    = [cohen_d(rng.choice(hp, len(hp), True),
                   rng.choice(ai, len(ai), True)) for _ in range(2000)]
ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

print(f"\n=== H1 – test {test} ===")
print(f"stat = {stat:.3f}  p = {pval:.4f}  d = {d:.3f}  IC95%[{ci_low:.2f};{ci_high:.2f}]")

# ───────────────────────────────────────────────────────────────────
# 7. Graphiques (KDE + Raincloud)
# ───────────────────────────────────────────────────────────────────
df_grp["Groupe"] = np.where(df_grp["grp_hp"], "HP", "AI")
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))
sns.kdeplot(data=df_grp, x="score_mb", hue="Groupe",
            common_norm=False, fill=True, alpha=.4)
plt.title("Densité du score malbouffe (mean z-scores)")
plt.xlabel("Score (z)")
plt.ylabel("Densité")
plt.savefig(os.path.join(OUT_DIR, "H1_kde.png"), dpi=300, bbox_inches="tight")
plt.close()

try:
    import ptitprince as pt
    plt.figure(figsize=(8,5))
    pt.RainCloud(x="Groupe", y="score_mb", data=df_grp,
                 palette="Set2", bw=.2, width_viol=.6,
                 orient="h", alpha=.65, move=.2)
    plt.xlabel("Score (z)")
    plt.title("Raincloud – score malbouffe par groupe")
    plt.savefig(os.path.join(OUT_DIR, "H1_raincloud.png"), dpi=300, bbox_inches="tight")
    plt.close()
except ImportError:
    print("ptitprince non installé → Raincloud ignoré")

# ───────────────────────────────────────────────────────────────────
# 8. Rapport texte
# ───────────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "H1_resume.txt"), "w", encoding="utf-8") as f:
    f.write(
        f"=== Score malbouffe : méthode standardisée (α={alpha:.2f}) ===\n"
        f"Items utilisés : {', '.join(mb_cols)}\n"
        f"Test comparatif : {test}\n"
        f"stat = {stat:.3f}  p = {pval:.4f}\n"
        f"Cohen d = {d:.3f}  IC95% [{ci_low:.2f};{ci_high:.2f}]\n"
        f"N HP={len(hp)}  |  N AI={len(ai)}\n"
    )

print(f"✅ Analyses terminées – résultats dans « {OUT_DIR}/ »")
