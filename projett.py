import pandas as pd
import prince
import matplotlib.pyplot as plt
import numpy as np
# ============================
# 1. Lecture du fichier Excel
# ============================
df = pd.read_excel(r"C:\Users\PC\OneDrive\Documents\projet add\Tableau_Foyer.xlsx")
print("Aperçu des données :")
print(df.head())
# ============================
# 2. Suppression de l'ID
# ============================
X = df.drop(columns=["ID"])
# ============================
# 3. AFCM (MCA)
# ============================
mca = prince.MCA(
    n_components=2,
    n_iter=10,
    copy=True,
    check_input=True,
    random_state=42
)
mca = mca.fit(X)
# ============================
# 4. Coordonnées
# ============================
coord_ind = mca.row_coordinates(X)
coord_mod = mca.column_coordinates(X)
print("\nCoordonnées des individus :")
print(coord_ind)
print("\nCoordonnées des modalités :")
print(coord_mod)
# ============================
# 5. Calcul des pourcentages
# (méthode simple et correcte)
# ============================
eigenvalues = np.var(coord_ind, axis=0)
percentages = eigenvalues / eigenvalues.sum() * 100
print("\nPourcentage d'inertie :")
print(f"Dimension 1 : {percentages[0]:.2f} %")
print(f"Dimension 2 : {percentages[1]:.2f} %")
# ============================
# 6. Graphique final AFCM
# ============================
plt.figure(figsize=(8,8))
# Individus
plt.scatter(coord_ind[0], coord_ind[1], alpha=0.6)
for i, txt in enumerate(df["ID"]):
    plt.text(coord_ind.iloc[i,0], coord_ind.iloc[i,1], str(txt), fontsize=8)
# Modalités
plt.scatter(coord_mod[0], coord_mod[1], marker='^')
for i, txt in enumerate(coord_mod.index):
    plt.text(coord_mod.iloc[i,0]*1.05, coord_mod.iloc[i,1]*1.05, txt, fontsize=8)
# Axes
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.xlabel(f"Dimension 1 ({percentages[0]:.2f} %)")
plt.ylabel(f"Dimension 2 ({percentages[1]:.2f} %)")
plt.title("Analyse Factorielle des Correspondances Multiples (AFCM)")
plt.grid(True)
plt.axis("equal")
plt.show()
print("\nFin du programme")

