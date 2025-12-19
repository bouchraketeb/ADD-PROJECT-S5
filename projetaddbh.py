import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import seaborn as sns

# ============================================================================
# PARTIE 1 : LECTURE DU FICHIER EXCEL
# ============================================================================

chemin = r"C:\Users\PC\OneDrive\Documents\projet add\Tableau_Foyer.xlsx"
df = pd.read_excel(chemin)

print("Aperçu des données :\n", df.head())

# ============================================================================
# PARTIE 2 : SÉLECTION DES COLONNES QUALITATIVES
# ============================================================================

qual_cols = ['Taille du foyer', 'Lave-vaisselle', 'Sèche-linge', 'Four', 'Catégorie de revenu']
X = df[qual_cols]

print("\nColonnes qualitatives sélectionnées :", qual_cols)

# ============================================================================
# PARTIE 3 : MATRICE DISJONCTIVE COMPLÈTE
# ============================================================================

Z = pd.get_dummies(X, prefix_sep='_')
n, p = Z.shape
K = len(qual_cols)

print(f"\nMatrice disjonctive complète : {n} individus x {p} modalités")
print(Z.head())

# ============================================================================
# PARTIE 4 : AFCM (Analyse Factorielle des Correspondances Multiples)
# ============================================================================

col_sums = Z.sum(axis=0)/n
row_sums = np.ones(n) * K

# Centrage et standardisation
Z_centered = Z - col_sums.values
S = Z_centered / np.sqrt(row_sums[:, None] * col_sums.values[None, :])

# Décomposition SVD
U, s_values, Vt = svd(S, full_matrices=False)
eigenvalues = s_values**2
explained_variance = eigenvalues / eigenvalues.sum() * 100
cumulative_variance = np.cumsum(explained_variance)

# ============================================================================
# PARTIE 5 : SCREE PLOT ET BROKEN-STICK
# ============================================================================

plt.figure(figsize=(8,5))
plt.plot(np.arange(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='-', color='orange')
plt.xlabel("Composantes")
plt.ylabel("Valeurs propres")
plt.title("Scree plot AFCM")
plt.grid(True)
plt.show()

# Broken-stick
n_comp = len(eigenvalues)
bs = np.array([sum(1/np.arange(i, n_comp+1))/n_comp for i in range(1,n_comp+1)])
plt.figure(figsize=(8,5))
plt.plot(range(1,n_comp+1), eigenvalues, marker='o', label='Valeurs propres', color='blue')
plt.plot(range(1,n_comp+1), bs, marker='o', label='Seuils brisés', color='red')
plt.xlabel("Composantes")
plt.ylabel("Valeur")
plt.title("Broken-stick model")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================================
# PARTIE 6 : COORDONNÉES DES INDIVIDUS ET MODALITÉS
# ============================================================================

n_axes = 2  # On conserve les 2 premiers axes
F_ind = U[:, :n_axes] * s_values[:n_axes]
coords_ind = pd.DataFrame(F_ind, columns=[f"Axe_{i+1}" for i in range(n_axes)])
print("\nCoordonnées des individus :\n", coords_ind)

F_mod = Vt.T[:, :n_axes] * s_values[:n_axes]
coords_mod = pd.DataFrame(F_mod, index=Z.columns, columns=[f"Axe_{i+1}" for i in range(n_axes)])
print("\nCoordonnées des modalités :\n", coords_mod)

# ============================================================================
# PARTIE 7 : CERCLE DES CORRÉLATIONS AVEC COULEURS PAR VARIABLE
# ============================================================================

# Création d'une palette de couleurs pour chaque variable
palette = sns.color_palette("Set2", len(qual_cols))
color_dict = {col: palette[i] for i,col in enumerate(qual_cols)}

fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
circle = plt.Circle((0,0),1,color='silver',fill=False,linestyle='--')
ax.add_artist(circle)

# Tracer les vecteurs avec couleurs par variable
for var in Z.columns:
    var_name = var.split('_')[0]  # Extrait le nom de la variable
    color = color_dict.get(var_name, 'black')
    x, y = F_mod[var, 0], F_mod[var, 1]
    ax.arrow(0,0,x,y,head_width=0.05,head_length=0.05,fc=color,ec=color)
    ax.text(x*1.15, y*1.15, var, fontsize=9, ha='center', color=color)

ax.axhline(0,color='black',linewidth=0.5)
ax.axvline(0,color='black',linewidth=0.5)
plt.xlabel(f"Axe 1 ({explained_variance[0]:.1f}%)")
plt.ylabel(f"Axe 2 ({explained_variance[1]:.1f}%)")
plt.title("Cercle des corrélations (Modalités)")
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

# ============================================================================
# PARTIE 8 : CONTRIBUTIONS ET COS²
# ============================================================================

# Contribution des modalités aux axes (%)
contrib_mod = (F_mod**2 * col_sums.values[:, None]) / eigenvalues[:n_axes]
contrib_df = pd.DataFrame(contrib_mod*100, columns=[f"Contrib_Axe{i+1}" for i in range(n_axes)], index=Z.columns)
print("\nContributions des modalités (%):\n", contrib_df)

# Qualité de représentation (cos² %)
cos2_mod = F_mod**2 / (F_mod**2).sum(axis=1)[:,None]
cos2_df = pd.DataFrame(cos2_mod*100, columns=[f"Cos2_Axe{i+1}" for i in range(n_axes)], index=Z.columns)
print("\nQualité de représentation (cos² %):\n", cos2_df)


