# XFactors  

IBENS internship 2025  
Alexandre MYARA  
Supervised by Thomas Boyer, Nicolas Bouriez, Auguste Genovesio.

## Project description

XFactors est une méthode proposant un disentanglement partiel ou complet d'espace latent.  
Notre architecture permet de disentangler les facteurs $(f_1, \ldots f_k)$ d'une image de manière supervisée et offre la possibilité de choisir la ou les directions de l'espace où seront encodées chacun de ces facteurs.

Il est également possible de choisir de n'encoder qu'un sous ensemble des facteurs (exemple CelebA, qui est annoté de 40 facteurs, il est possible de n'encoder que certains facteurs d'interet) sans laisser de coder l'information relatives aux facteurs non encodés.

XFactors divise l'espace latent en deux sous-espaces : un sous-espace orthogonal $T$ (pour target) contenant les facteurs disentanglés, un sous-espace $S$ concernant les informations relatives à l'image qui ne sont pas annotées ou relatives aux facteurs à ne pas disentangler.

Notre méthode propose également de choisir sur quelles directions de l'espace $T$ le facteur $f_i$ sera encodé.  
Cela offre la possibilité de garder intact toute l'information d'un vecteur latent, en ne changeant que les directions voulues afin de ne modifier qu'un seul facteur.

Nous nous assurons également que chaque facteur $f_i$ est effectivement disentanglé à toutes les échelles, et que l'espace $S$ ne contient aucune information relative aux facteurs présents dans l'espace $T$.

**NB: On appelle "facteur" d'une image, tout élement visuel composant de l'image.**

## Code base organisation

Le repo est organisé sur plusieurs dossiers comme suit:

1. Arch: contient un module vae et un module multi_distillme.  
Le module multidistillme contient notamment le fichier train_x.py ainsi que le fichier xfactors.py.
2. Data: contient les .npz pour Cars3D, BloodMNIST, 3DShapes, DSprites, CelebA ainsi que MNIST. Les .npz ont tous pour clés "label" et "image".
3. Dataset: contient les classes Dataset torch de chacun des jeux de données. Les données sont converti en torch.float32 et scale sur [0,1].
4. Data_module: contient les classes DataModule de chacun des jeux de données. S'occupe de générer les fichiers .npz s'ils sont manquants. Embarquent les méthodes afin de charger les DataLoader.  
ex: Cars3DDataModule est munie de Cars3DDataModule.train_dataloader() -> renvoie le dataloader train de Cars3D.
5. logs: contient les log des entrainements.
6. loss: contient le fichier loss.py.
7. metric: contient le code du factorVAE score et DCI.
8. notebook: contient principalement le notebook xfactors.ipynb qui permet d'entrainer XFactors depuis le notebook. Les logs sont chargés dans un dossier spécial ../lightning_logs.  
Contient également le notebook metric.ipynb qui permet de calculer les métriques depuis un fichier .ckpt (trouvable dans les logs).
9. scripts: permet l'entrainement en ligne de commande à l'aide du fichier sweep_x.sh
10. utils: contient les constantes et fonctions d'affichages.

## How to use sweep_x.sh

Le fichier sweep_x.sh fait appel itérativement à train_x.sh.  
Les hyperparamètres du modèles sont fixables depuis train_x.sh.  
train_x.sh propose la modification de:

1. dataset utilisé. Pour 3Dshapes, le raccourci "shapes" fonctionne.  
2. $\beta_t$
3. $\dim_s$
4. batch_size et nombre d'epochs
5. la dimension accordée à chaque facteur dans $T$.
6. Une clé optionnelle à ajouter à la fin du nom du dossier d'entrainement. Permet éventuellement de voir d'un coup d'oeil la liste des facteurs encodés.

Une fois fait, le dossier dans lequel sera chargé le log est une variable de sweep_x.sh.

Il faut alors fixer la variable "version=".

1. S'il s'agit d'un entrainement avec $\beta_t=1$, version=x_with_beta_t1
2. S'il s'agit d'un entrainement avec $\beta_t=100$, version=x_with_beta_t100
3. S'il s'agit d'un entrainement avec $\dim_t=3$, version=x_with_beta_t1_dim_t3

Tous ces fichiers sont trouvables dans logs.  
Il est bien sur possible de donner d'autres valeurs à version=. Mettre version à version=MaVersion créera le dossier logs/MaVersion et mettra les logs de l'entrainement dans ce dossier.

La variable gpu de sweep_x.sh permet de sélectionner le gpu utilisé.

Depuis le terminal une fois placé dans le dossier script/train/  
Il est possible de lancer ./sweep_x.sh $\beta_{s_1}, \ldots, \beta_{s_k}$  
$k$ entrainements se lancent sur le gpu sélectionné via tmux. Chaque fenêtre tmux entraine le modèle avec un $\beta_s$ différent (possible de mettre un seul $\beta_s$).

## Naviguer dans le dossier logs

Le dossier logs peut être profond dans l'arborescence, voici comment elle fonctionne.

1. Racine: logs, contient les dossiers "version" comme "x_with_beta_t1"
2. dossiers versions, contient les dossiers "cars3d", "mpi3d" etc
3. dossiers datasets, contient les dossiers "factor="

Le dossier factor0,1,2,3,4 de dsprites contient des entrainements ayant mis les facteurs 0,1,2,3,4 dans $T$.  
Le dossier factor_s=-1 contient des entrainements ayant mis tout les facteurs dans $T$ excepté le dernier dans $S$. C'est la configuration par défaut.

**NB: Pour changer les facteurs suivis pour un dataset, il faut le spécifier dans train_x.py. Par défaut les facteurs séléctionnés dans $T$ pour un dataset sont tous ces facteurs sauf le dernier qui est dans $S$. Cela est modifiable avec la variable select_factors de train_x.py.**

Les dossiers factor= contiennent ensuite les dossiers test_dims2, test_dim126 etc.

Enfin dans ces derniers on retrouve les log relatifs à un entrainement au format:  
x_epoch=100_beta=()_latent=()_batch=.

Chaque epoch log en entrainement et en validation les espaces latents, la reconstruction et la génération.

## Calculer les métriques

La manière la plus simple de calculer les métriques est d'utiliser le notebook metric.ipynb.  
Il suffit d'aller dans la section FactorVAE/XFactors.

La classe FactorVAEScore n'a besoin en entrée seulement que du .ckpt (trouvable dans les logs).  
La classe DCIScore n'a besoin en entrée seulement que du .ckpt (trouvable dans les logs).

## Losses (XFactors)

Les loss utilisées par XFactors sont :

- KL sur l'espace $T$ (pondérée par `beta_t`)
- KL sur l'espace $S$ (pondérée par `beta_s`)
- InfoNCE supervisé par facteur (`l_nce_by_factors`).
- MSE (reconstruction, pondérée à 1)

### Exemple sur 3DShapes

```sh
python -m disdiff_adapters.arch.multi_distillme.train_x \
 --dataset shapes \
 --max_epochs 50 --batch_size 64 \
 --beta_t 1.0 --beta_s 1.0 \
 --l_nce_by_factors 0.1
```

### Chemins et premières utilisations

- Les chemins données/logs utilisent peuvent être personnalisées via les variables d'environnement :
- `PROJECT_PATH` (racine du dépôt)
- `LOG_DIR` (par défaut `disdiff_adapters/logs` sous la racine)
- `CELEBA_DATA_DIR` (optionnel, pour un chemin CelebA externe)
