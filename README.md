# Détection de gestuelle des mains avec OpenCV pour la commande du jeu GeoGuessr
### Par: Thomas

## Comment utiliser?

### Calibration
Avant de débuter, une calibration des seuils de luminosité est nécessaire. Pour ce faire, exécutez le script `calibrate.py` et ajuster votre environnement physique ainsi que les sliders pour obtenir un bon résultat. Idéalement, la main doit être blanche sur un fond noire dans le fenêtre `mask`. Une fois des bonnes valeurs trouvées, appuyez sur la touche `q` pour quitter. Les valeurs de calibration seront enregistrées dans `calibration.json`.

### Utilisation
Une fois la calibration effectuée, exécutez le script `finger_counter.py`. Une fenêtre s'ouvrira avec une vue de la webcam. Placez votre main dans la zone verte et effectuez des gestes pour voir le nombre de doigts détectés. Pour activer les actions de contrôle de la souris par la gestuelle, appuyez sur la touche `p`. ATTENTION: Assurez-vous d'être sur une fenêtre où il n'y a pas de risque de dommages irréversibles. Pour mettre fin au programme, appuyez sur la touche `q`.

### GeoGuessr
Pour jouer à GeoGuessr en version gratuite, dirigez-vous sur le site [OpenGuessr](https://openguessr.com/). Pour naviguer dans le jeu, utilisez les gestes suivants:
- **Clic gauche**: Levez 2 doigts
- **Maintenir le clic gauche**: Collez les doigts pour faire une paume (le compteur affichera `None`).
- **Déplacer la souris**: Déplacez votre main dans la direction souhaitée. Le centre de masse du polygone de la main guide le déplacement.
- **Relâcher le clic gauche**: Écartez les doigts pour que les 5 soient distincts (le compteur affichera `5`).
- **Zoom avant**: Montrez 3 doigts.
- **Zoom arrière**: Montrez 4 doigts.
