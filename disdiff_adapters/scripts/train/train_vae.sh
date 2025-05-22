#!/bin/bash

cd /projects/compures/alexandre/disdiff_adapters
source /projects/compures/alexandre/.venv/bin/activate

batch_size=128
max_epochs=50
dataset="celeba"
betas=("0 1 5 15")
latent_dim=4
warm_up="False"
lr=0.00001
arch="def"

gpus="3"


echo $dataset
for beta in $betas
do
    python3 -m disdiff_adapters.arch.vae.train \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --gpus $gpus \
                                2>&1 | tee "disdiff_adapters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}_lr=${lr}_arch=${def}.out"

    python3 -m disdiff_adapters.arch.vae.test \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --gpus $gpus \
                                2>&1 | tee -a "disdiff_adapters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}_lr=${lr}_arch=${def}.out"
done