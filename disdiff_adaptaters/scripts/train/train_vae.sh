#!/bin/bash

cd /projects/compures/alexandre/disdiff_adaptaters
source .venv/bin/activate

batch_size=16
max_epochs=50
dataset="shapes"
betas=("0 10e-9 10e-6 10e-5 10e-4 10e-3 1 5 10 15")
latent_dim=128

for beta in $betas
do
    python3 -m disdiff_adaptaters.arch.vae.train_0 \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                2>&1 | tee "disdiff_adaptaters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"

    python3 -m disdiff_adaptaters.arch.vae.test_0 \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                2>&1 | tee -a "disdiff_adaptaters/scripts/out/${dataset}_vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"
done