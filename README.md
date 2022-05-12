## Fonctions et fichiers

- BlendGan sert à lancer l'entrainement (création des différents objets, checkpoints... et lancement de l'entrainement)
- GAN_functions_class est une classe pour les fonctions du GAN (créer le generateur, discriminateur, fonctions de loss, optimizer, entrainement (avec sauvegarde checkpoint)...)

- GAN_functions_inpainting1 correspond à un héritage de GAN_functions avec un autre modèle de générateur (basé sur un modèle d'inpainting)

- testGAN permet de lancer le GAN sur une image (avec --i), il affichera l'image orginal et l'image générée par le GAN.
Il faut cependant que la classe corresponde bien au GAN entrainé car on charge le checkpoint.

## ÉTATS des fichiers
- Toutes les fonctions testés (notamment les tests sur les fonctions de loss, avec distance euclidienne...) n'ont pas été copié dans de nouvelles classes. (Elles sont parfois dans OLD2)

## Les dossiers:

Nous n'avons pas push les images d'entrainements (39go) disponible ici : 
https://drive.google.com/file/d/1AhwidwCfnOqoaiSOnaCL4IUenORsdhTr/view?usp=sharing


Ces images ont été générée à partir du dossier Real et des PNGs issus de COCO extract.

- Le dossier Images contient donc:
    - Le dossier Real contient des images réelles de COCO (mais aussi une banque d'image d'art moderne)
    - Le dossier Pngs contient les pngs extraits de COCO.
    - Le dossier Collage contient les images générées en collant les pngs sur des images issus de Real. 

- Le dossier Checkpoint contient des checkpoints (non push sauf ceux directement dans le dossier)

Ce dossier contient aussi un fichier de sauvegarde des loss.

- Le dossier Generated images contient les images générés avec testGan.


## Le dossier Utils

Il contient des fonctions utiles pour le GAN ou pour nos essais sur l'inpainting.

- Des fonctions pour générer les images 
- Des fonctions pour générer des images utiles dans le cadre de nos essais sur l'inpainting (ces fonctions ne sont pas encore adapté pour des batchs d'images)

## Commandes

- python3 blendGAN.py pour lancer l'entrainement du GAN (changer l'import pour modifier le GAN utilisé)

 - Les variables modifiables : 
    - EPOCHS
    - BATCH_SIZE

 - La classe utilisée

- python3 testGAN.py --i [images.png] pour lancer le test du GAN sur une image. (Souvent issus du dossier collage)