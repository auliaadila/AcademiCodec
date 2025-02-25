#!/bin/bash

# Create a directory to store the dataset
DATASET_DIR="LibriTTS"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

echo "📥 Downloading LibriTTS dataset..."
# Download all parts of LibriTTS
wget -c https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget -c https://www.openslr.org/resources/60/train-clean-360.tar.gz
wget -c https://www.openslr.org/resources/60/dev-clean.tar.gz
wget -c https://www.openslr.org/resources/60/test-clean.tar.gz

echo "📂 Extracting downloaded tar files..."
# Create target folders
mkdir -p dev-clean test-clean train-clean

# Function to extract and move only .wav files
extract_and_move_wavs() {
    tar_file=$1
    target_folder=$2

    echo "📂 Extracting $tar_file..."
    temp_dir=$(mktemp -d)  # Create a temporary directory
    tar -xzf $tar_file -C $temp_dir  # Extract to temp folder

    echo "🎵 Moving .wav files to $target_folder..."
    find $temp_dir -type f -name "*.wav" -exec mv {} $target_folder/ \;

    echo "🗑️ Removing temporary files..."
    rm -rf $temp_dir  # Clean up extracted directories
}

# Extract each dataset and move only .wav files
extract_and_move_wavs "dev-clean.tar.gz" "dev-clean"
extract_and_move_wavs "test-clean.tar.gz" "test-clean"
extract_and_move_wavs "train-clean-100.tar.gz" "train-clean"
extract_and_move_wavs "train-clean-360.tar.gz" "train-clean"

echo "✅ Extraction and file moving completed successfully!"