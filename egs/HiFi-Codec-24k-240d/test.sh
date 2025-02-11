
# #!/bin/bash
# source path.sh

# ckpt=checkpoint/HiFi-Codec-24k-240d
# echo checkpoint path: ${ckpt}

# # the path of test wave
# wav_dir=test_wav

# outputdir=output
# mkdir -p ${outputdir}

# python3 ${BIN_DIR}/vqvae_copy_syn.py \
#     --model_path=${ckpt} \
#     --config_path=config_24k_240d.json \
#     --input_wavdir=${wav_dir} \
#     --outputdir=${outputdir} \
#     --num_gens=10000


#!/bin/bash

source path.sh

# Define the checkpoint directory and pattern for checkpoint files
ckpt_dir="/home/s2310401/AcademiCodec/egs/HiFi-Codec-24k-240d/checkpoints"
ckpt_pattern="g_*"  # Pattern to match all checkpoint files
config="config_24k_240d_8c.json"

# Define the test wave directory
wav_dir="/home/s2310401/AcademiCodec/data/test_wav"

# Define base output directory
base_outputdir="output/20250211_train/${config}"

echo -e "\nStart Inference"  # Added this line with newline

# Loop over all checkpoint files
for ckpt in ${ckpt_dir}/${ckpt_pattern}; do
    # Extract the checkpoint name from the path (e.g., "g_00000010-20250211_135433")
    ckpt_name=$(basename "$ckpt")

    # Create an output directory for this checkpoint
    outputdir="${base_outputdir}/${ckpt_name}"
    mkdir -p "${outputdir}"

    # Run the inference script
    echo "Running inference for checkpoint: ${ckpt}"
    python3 ${BIN_DIR}/vqvae_copy_syn.py \
        --model_path="${ckpt}" \
        --config_path="${config}" \
        --input_wavdir="${wav_dir}" \
        --outputdir="${outputdir}" \
        --num_gens=10000

done

echo "Inference completed for all checkpoints."
