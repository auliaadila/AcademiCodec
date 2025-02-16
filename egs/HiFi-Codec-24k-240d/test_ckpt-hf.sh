#!/bin/bash
source path.sh

ckpt=ckpt-hf/HiFi-Codec-24k-240d

# the path of test wave
wav_dir=/home/s2310401/dataset/libri-tts-wavs/test-all

outputdir=output/ckpt-hf
mkdir -p ${outputdir}

python3 ${BIN_DIR}/vqvae_copy_syn.py > test_ckpt-hf.log \
    --model_path=${ckpt} \
    --config_path=config_24k_240d.json \
    --input_wavdir=${wav_dir} \
    --outputdir=${outputdir} \
    --num_gens=10000