#!/bin/bash
source path.sh

# ckpt=checkpoint/HiFi-Codec-24k-320d
# ckpt=/home/adila/Data/research/AcademiCodec/egs/HiFi-Codec-24k-320d/logs/logs/events.out.tfevents.1738899993.unoki-lab.2252192.0
ckpt=/home/adila/Data/research/AcademiCodec/egs/HiFi-Codec-24k-320d/logs

# the path of test wave
wav_dir=test_wav

outputdir=output/20250207_train
mkdir -p ${outputdir}

python3 ${BIN_DIR}/vqvae_copy_syn.py \
    --model_path=${ckpt} \
    --config_path=config_24k_320d.json \
    --input_wavdir=${wav_dir} \
    --outputdir=${outputdir} \
    --num_gens=10000
