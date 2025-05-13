#!/bin/bash

cd /projects/compures/alexandre/disdiff_adaptaters
source .venv/bin/activate

batch_size=16
max_epochs=10
dataset="shapes"
beta=1.0
latent_dim=128

# python3 -m disdiff_adaptaters.arch.vae.train \
#                              --max_epochs $max_epochs \
#                              --dataset $dataset \
#                              --beta $beta \
#                              --latent_dim $latent_dim \
#                              --is_vae "False" \
#                              2>&1 | tee "disdiff_adaptaters/scripts/out/ae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"

python3 -m disdiff_adaptaters.arch.vae.test \
                             --max_epochs $max_epochs \
                             --dataset $dataset \
                             --beta $beta \
                             --latent_dim $latent_dim \
                             2>&1 | tee -a "disdiff_adaptaters/scripts/out/vae_train_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_batch=${batch_size}.out"

