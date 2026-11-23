# Models creator

Application FastAPI pour annoter des images, entraîner des modèles U-Net de segmentation
(masque noir et blanc) sur des images 512×512, puis les tester.

## Installer

Un seul fichier, le même pour Windows et Linux : **`install.cmd`**.

- **Windows** : double-cliquer sur `install.cmd`.
- **Linux** : dans un terminal, `sh install.cmd`.

Il installe [uv](https://docs.astral.sh/uv/) s'il manque, qui installe Python 3.12 et les
bibliothèques dans `.venv` (quelques Go, surtout PyTorch et TensorFlow ; il faut Internet),
puis il met une icône « Models creator » sur le Bureau et dans le menu des applications
(menu Démarrer sous Windows), et propose de lancer l'application. Rien n'est installé en
dehors de ce dossier, à part uv (dans `~/.local/bin`) et Python (dans le cache de uv).

On peut le relancer sans risque, par exemple après une mise à jour du dossier ou s'il a été
déplacé : il remet les bibliothèques et les icônes à jour.

## Lancer et arrêter

L'icône lance `launch.py` : il démarre le serveur sans fenêtre et ouvre la page dans le
navigateur dès qu'il répond ; si l'application tourne déjà, il ouvre seulement la page. Les
messages du serveur sont dans `logs/server.log`.

Le bouton **Quit**, en haut à droite, arrête le serveur et ferme la fenêtre quand le
navigateur le permet.

Pour voir les messages du serveur en direct (s'il ne démarre pas) :

```
uv run launch.py --console
```

puis ouvrir http://127.0.0.1:8765.

## Onglets (interface en anglais)

**Data** : déposer un dossier contenant `imgs/` et `masks/`. Les masques sont associés aux
images par leur nom (`img1.png` ↔ `img1.png`, `img1_mask.png`, `img1_label.png`…). Un
dossier qui ne contient que des images est accepté aussi : chaque image reçoit un masque noir.
Les images sont ramenées à 512×512 ; tout pixel non noir d'un masque compte comme objet.

**Annotation** : l'image et son masque superposés.

| Outil | Touche | Usage |
|---|---|---|
| Pinceau | B | Maj = effacer |
| Gomme | E | |
| Polygone | P | clic = sommet, double-clic ou Entrée = fermer, Retour arrière = retirer un sommet, Échap = annuler, Maj = effacer |
| Rectangle | R | Maj = effacer |
| Remplissage | F | remplit la zone cliquée, Maj = la vider |
| Déplacer | Espace maintenu, ou clic droit / molette cliquée | |

Molette : zoom. `[` `]` : taille du pinceau. H : masquer le masque. O : contour seul.
Ctrl+Z / Ctrl+Y : annuler / rétablir. Ctrl+S : enregistrer (fait aussi automatiquement en
changeant d'image). ← → ou A D : image précédente / suivante. 0 : ajuster à la fenêtre.

**Training** : choix du jeu, des époques, de la taille de lot, du taux d'apprentissage, de
la part de validation et de la taille du U-Net. Loss, Accuracy et Recall (ainsi que l'IoU ;
la Precision est enregistrée dans le modèle) sont suivies en direct, en entraînement et en validation. Le meilleur modèle (plus
petite loss de validation) est enregistré dans `data/models/<nom>.pt`.

**Inference** : choisir un modèle ou déposer un fichier `.pt`, puis déposer des images (de
n'importe quelle taille : la prédiction est faite en 512×512 puis remise à la taille de
l'image). Affichage en mosaïque, ou en hauteur avec une glissière avant/après ; taille des
images réglable (curseur ou Ctrl + molette), opacité, couleur, seuil. Les résultats peuvent
être téléchargés en zip, ou copiés dans un jeu de données pour être corrigés dans Annotation.

**Info** : le framework utilisé (PyTorch), sa version et son type de build (CPU ou CUDA),
le périphérique de calcul et, s'il n'y a pas de GPU utilisé, pourquoi ; pour chaque GPU
(via `nvidia-smi`) le nom, la VRAM utilisée et totale, la charge, la température, la
consommation, les fréquences et le pilote, mis à jour toutes les 2 s ; le système (OS,
processeur, RAM, Python) ; les versions des bibliothèques (TensorFlow, JAX, ONNX Runtime…
si elles sont installées).

**Settings** : thème sombre, clair ou comme Windows ; framework des nouveaux entraînements,
PyTorch (modèles `.pt`) ou TensorFlow / Keras (modèles `.keras`) — l'onglet Inference ouvre
les deux, quel que soit ce réglage ; périphérique (auto ou CPU seulement) et nombre de threads
CPU. Les réglages sont dans `data/settings.json`. TensorFlow ne lit le périphérique et les
threads qu'à son premier chargement : après les avoir changés, relancer l'application.

## Fichiers

```
install.cmd        l'installeur (Windows et Linux)
launch.py          ce que lance l'icône
app/main.py        routes FastAPI
app/installer.py   crée les icônes (seconde moitié de install.cmd)
app/storage.py     jeux de données sur disque
app/unet.py        le modèle U-Net
app/training.py    entraînement en arrière-plan et métriques
app/tf_backend.py  le même U-Net, entraînement et prédiction en TensorFlow / Keras
app/settings.py    réglages (data/settings.json)
app/sysinfo.py     informations de l'onglet Info
app/inference.py   chargement des modèles et prédiction
static/            l'interface (HTML, CSS, JavaScript sans dépendance)
data/              jeux de données, modèles, résultats (non versionné)
```

Les modèles déposés sont lus sans exécuter de code : `torch.load(weights_only=True)` pour
PyTorch (modèles de cette application, `state_dict` d'un U-Net de même architecture, ou
TorchScript), `keras.models.load_model` en mode sûr pour les `.keras` / `.h5`.

## Calcul

PyTorch est installé en version CPU. Sur CPU, garder un petit réseau (filtres de base 8 ou
16). Mesuré sur ce poste, par image et par époque : 0,3 s avec 8 filtres, 0,6 s avec 16,
1,5 s avec 32 (soit environ 1 min par époque pour 100 images avec 16 filtres). Pour utiliser une carte
NVIDIA récente, installer la version CUDA de PyTorch ; l'application l'utilise alors
automatiquement.

TensorFlow n'utilise pas de GPU sous Windows natif depuis sa version 2.11 (seulement sous
WSL2) : il calcule toujours sur CPU ici. Son premier chargement prend jusqu'à une
demi-minute ; il n'est chargé que s'il est choisi dans Settings ou qu'un modèle `.keras` est
utilisé.
