# disdiff_adapters
IBENS internship 2025

## Project description

## Code base organisation

This code consists of the following modules :
    1. vae, implements Encoder, Decoder and VAE classes.
    2. datasets, implements PyTorch Datases classes.
    3. data_module, implements Lightning DataModule classes.
    4. loss, implements main loss functions.
    5. metric, implements metrics to evaluate different tasks.
    6. utils, implements @dataclass to load constants and general functions

## Discover the code with notebooks

## TODO
1. Réparation et manquant
-> dsprites beta_t 1: beta s 1 10 50 MANQUANT + dim_s126 tout
-> manquant beta_t 10 ? tout shapes, 1 50 10 celeba, dim_s126 dsprites

2. Experience préliminaire à beta_n_vae
-> TOUT en batch256

3. Expérience sur la valeure de dim_t
-> Voir dim_t = 3,4,8 pour tout

4. Intégration d'un nouveau dataset
-> dsprites n'est pas suited for disentanglement : sprites

5. Amélioration de Xfactor
-> mettre à jour les plots
-> shapes avec dim_t =2 pour 3 facteurs

6. Nouvelle métrique
-> définition objectifs de la métrique :
    Constat : latent image du leakage
    But : évaluer le leakage à l'aide des latents
    Note : n'évalue pas si le facteur cible est encodé;
            Parfois facteur encodé bien clusterisé mais
            à la generation le facteur n'est pas bien mis.
            Toutefois : si cluster parasite, se voit sur la
            génération à plusieurs moments

    