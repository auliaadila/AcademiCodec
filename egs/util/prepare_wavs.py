import os
import argparse
import shutil
import tarfile
import tempfile
import urllib.request

# URLs for LibriTTS dataset
LIBRITTS_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/60/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/60/train-clean-360.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/60/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/60/test-clean.tar.gz",
}

def download_file(url, output_path):
    """Downloads a file from the given URL."""
    if os.path.exists(output_path):
        print(f"📂 {output_path} already exists. Skipping download.")
        return
    print(f"📥 Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"✅ Downloaded: {output_path}")

def extract_and_move_wavs(tar_path, target_folder):
    """Extracts a tar.gz file and moves all .wav files to the target folder."""
    print(f"📂 Extracting {tar_path}...")
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(temp_dir)

        # Ensure target folder exists
        os.makedirs(target_folder, exist_ok=True)

        # Move .wav files to target folder
        wav_files = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_files.append(os.path.join(root, file))

        for wav_file in wav_files:
            shutil.move(wav_file, os.path.join(target_folder, os.path.basename(wav_file)))

    print(f"🎵 Moved {len(wav_files)} .wav files to {target_folder}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract LibriTTS dataset with only WAV files.")
    parser.add_argument("--dataset_dir", type=str, default="LibriTTS", help="Path to store dataset (default: LibriTTS)")

    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    # Create dataset directory if not exists
    os.makedirs(dataset_dir, exist_ok=True)
    os.chdir(dataset_dir)

    print("📥 Downloading LibriTTS dataset...")
    for key, url in LIBRITTS_URLS.items():
        tar_path = os.path.join(dataset_dir, f"{key}.tar.gz")
        download_file(url, tar_path)

    print("📂 Extracting and moving WAV files...")
    for key in LIBRITTS_URLS.keys():
        extract_and_move_wavs(os.path.join(dataset_dir, f"{key}.tar.gz"), os.path.join(dataset_dir, key))

    print("✅ Extraction and file moving completed successfully!")

if __name__ == "__main__":
    main()
