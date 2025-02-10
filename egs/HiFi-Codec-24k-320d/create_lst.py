import os

def generate_wav_list(directory, output_file, recursive=True, absolute_path=True):
    """
    Generate a .lst file containing the list of .wav files in a given directory.

    Args:
        directory (str): The folder to scan for .wav files.
        output_file (str): Name of the output .lst file.
        recursive (bool): Whether to scan subdirectories.
        absolute_path (bool): Whether to save absolute or relative file paths.
    """
    # Get list of all .wav files
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                if not absolute_path:
                    file_path = os.path.relpath(file_path, directory)
                wav_files.append(file_path)
        
        if not recursive:
            break  # If not recursive, only process the top directory

    # Write to the output .lst file
    with open(output_file, "w") as f:
        for wav_file in sorted(wav_files):  # Sort for consistency
            f.write(wav_file + "\n")

    print(f"✅ Successfully saved {len(wav_files)} .wav files to '{output_file}'")

# Example Usage
if __name__ == "__main__":
    folder_path = "/home/adila/Data/research/AcademiCodec/egs/HiFi-Codec-24k-320d/dev_wav"  # Change this to your actual folder
    generate_wav_list(folder_path, output_file="valid.lst", recursive=True, absolute_path=True)