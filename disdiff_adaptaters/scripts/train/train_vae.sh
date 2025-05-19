#!/bin/bash

cd /projects/compures/alexandre/disdiff_adaptaters
source .venv/bin/activate

batch_size=128
max_epochs=100
dataset="shapes"
betas=("0 10e-3 1 5 10")
latent_dim=4
warm_up="False"
echo $dataset
for beta in $betas
do
    python3 -m disdiff_adaptaters.arch.vae.train \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                2>&1 | tee "disdiff_adaptaters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"

    python3 -m disdiff_adaptaters.arch.vae.test \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                2>&1 | tee -a "disdiff_adaptaters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"
done