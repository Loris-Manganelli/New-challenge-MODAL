import pandas as pd

# Lire les fichiers CSV
fichier1 = pd.read_csv('submission.csv')
fichier2 = pd.read_csv('output.csv')

# Initialiser une liste pour les nouvelles données
nouvelles_donnees = []

# Comparer et créer les nouvelles lignes
for index, row in fichier2.iterrows():
    id_value = row['id'][:-4]
    
    try:
    # Vérifier si le poids et le label existent et ne sont pas vides
        if pd.notna(row['poids']) and pd.notna(row['label']):
            poids_value = row['poids']
            label_value = row['label'] if poids_value > 0.75 else fichier1.loc[fichier1['id'] == id_value, 'label'].values[0]
        else:
            label_value = fichier1.loc[fichier1['id'] == id_value, 'label'].values[0]

    except:
        print()
    
    nouvelles_donnees.append({'id': id_value, 'label': label_value})

# Convertir les nouvelles données en DataFrame
nouveau_fichier = pd.DataFrame(nouvelles_donnees)

# Sauvegarder le nouveau DataFrame en CSV
nouveau_fichier.to_csv('nouveau_fichier.csv', index=False)
