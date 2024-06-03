import pytesseract
from PIL import Image, UnidentifiedImageError
import Levenshtein
import pandas as pd
import os
import difflib

n=0

# Chemins
dossier_images = '/Users/loris/X/MODAL/challenge-modal/dataset/test'  # Remplacez par le chemin de votre dossier d'images
fichier_labels = '/Users/loris/X/MODAL/challenge-modal/list_of_cheese_OCR.txt'  # Remplacez par le chemin de votre fichier de labels
fichier_csv_sortie = '/Users/loris/X/MODAL/challenge-modal/output_difflib.csv'
fichier_label_output = '/Users/loris/X/MODAL/challenge-modal/list_of_cheese.txt'
# Charger les labels
with open(fichier_labels, 'r', encoding='utf-8') as file:
    labels = [line.strip() for line in file]

with open(fichier_label_output, 'r', encoding='utf-8') as file:
    labels_output = [line.strip() for line in file]


# Initialiser une liste pour stocker les résultats
resultats = []

# Traiter chaque fichier dans le dossier
for nom_fichier in os.listdir(dossier_images):
    n+=1
    chemin_fichier = os.path.join(dossier_images, nom_fichier)
    
    try:
        # Vérifier si le fichier est une image valide
        with Image.open(chemin_fichier) as img:
            img.verify()  # Vérifie si c'est une image valide
        
        # Réouvrir l'image pour l'OCR (car `verify` ferme l'image)
        with Image.open(chemin_fichier) as img:
            # Effectuer l'OCR sur l'image
            texte_ocr = pytesseract.image_to_string(img).strip().upper()
        
        # Initialiser les meilleures correspondances
        meilleur_label = None
        meilleure_similarite = 0.0
        words = texte_ocr.split()
        # Comparer le texte OCRisé avec chaque label
        for i,label in enumerate(labels):

            if len(words)!=0:
                #similarite = max([Levenshtein.ratio(word, label) for word in words])
                similarite = max([difflib.SequenceMatcher(None, word, label).ratio() for word in words])
            else: 
                similarite=0.
            # Calculer le pourcentage de similarité
            
            # Vérifier si cette similarité est la meilleure jusqu'à présent
            if similarite > meilleure_similarite:
                meilleure_similarite = similarite
                meilleur_label = labels_output[i]

        print(n, meilleur_label, texte_ocr, meilleure_similarite)
        
        # Ajouter le résultat à la liste
        resultats.append({
            'id': nom_fichier,
            'label': meilleur_label,
            'poids': meilleure_similarite
        })
    
    except (UnidentifiedImageError, pytesseract.TesseractError) as e:
        # Ignorer les fichiers qui ne sont pas des images valides ou qui causent des erreurs OCR
        print(f"Le fichier {nom_fichier} a été ignoré : {e}")

# Convertir les résultats en DataFrame
df = pd.DataFrame(resultats)

# Sauvegarder les résultats dans un fichier CSV
df.to_csv(fichier_csv_sortie, index=False, encoding='utf-8')

print(f"Les résultats ont été sauvegardés dans {fichier_csv_sortie}")