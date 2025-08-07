#!/bin/bash

cd /projects/compures/alexandre/disdiff_adapters
source /projects/compures/alexandre/.venv/bin/activate

batch_size=256
max_epochs=50
dataset="shapes"
betas=("1 5 15")
latent_dim=128
warm_up="True"
lr=0.00001
arch="res"

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
                                --arch $arch \
                                --gpus $gpus \
                                2>&1 | tee "disdiff_adapters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}_lr=${lr}_arch=${arch}.out"

    python3 -m disdiff_adapters.arch.vae.test \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --arch $arch \
                                --gpus $gpus \
                                2>&1 | tee -a "disdiff_adapters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}_lr=${lr}_arch=${arch}.out"
done