import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lire le fichier CSV
df = pd.read_csv('output.csv')


# Calculer le nombre de lignes où le label n'est pas vide
nb_label_non_vide = df['label'].notna().sum()

nb_poids_non_vide_sup_0_8 = df[df['poids'].notna() & (df['poids'] >= 0.8)].shape[0]
L=[]
for certitude in np.linspace(0., 1., 50):
# Calculer le nombre de lignes où le poids n'est pas vide et est supérieur à 0.8
    L.append(df[df['poids'].notna() & (df['poids'] < certitude) & (df['poids'] > certitude-0.02)].shape[0])


plt.plot(np.linspace(0.,1.,50), L)
certitudes = np.linspace(0., 1., 50)
plt.fill_between(certitudes, L, where=(certitudes >= 0.79), color='green', alpha=0.5)
plt.fill_between(certitudes, L, where=(certitudes <= 0.8), color='red', alpha=0.5)
plt.show()

print(f"Nombre de lignes où le label n'est pas vide : {nb_label_non_vide}")
print(f"Nombre de lignes où le poids n'est pas vide et est supérieur à 0.8 : {nb_poids_non_vide_sup_0_8}")


