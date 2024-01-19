#!/bin/bash
## download RosSteALS model
mkdir -p models/RoSteALS && cd models/RoSteALS
wget https://kahlan.cvssp.org/data/Flickr25K/tubui/cvpr23_wmf/epoch=000017-step=000449999.ckpt

## download AE model
cd ..
mkdir -p first_stage_models/vq-f4 && cd first_stage_models/vq-f4
wget -O model.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip -o model.zip
rm model.zip
