import os
import random
import shutil
import soundfile as sf

def get_audio_duration(file_path):
    """Returns the duration of a WAV file in seconds."""
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate

def select_random_wavs(source_folder, target_folder, total_duration_hours):
    """
    Selects random WAV files from source_folder to reach total_duration_hours 
    and copies them to target_folder.
    """
    total_duration = total_duration_hours * 3600  # Convert hours to seconds
    wav_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(".wav")]
    
    if not wav_files:
        print("No WAV files found in the source folder.")
        return

    selected_files = []
    selected_duration = 0
    random.shuffle(wav_files)  # Shuffle to randomize selection

    for wav_file in wav_files:
        duration = get_audio_duration(wav_file)
        
        if selected_duration + duration > total_duration:
            continue  # Skip if adding this file exceeds the limit
        
        selected_files.append(wav_file)
        selected_duration += duration

        if selected_duration >= total_duration:
            break  # Stop once the target duration is reached
    
    # Ensure target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Copy selected files to target folder
    for file in selected_files:
        shutil.copy(file, os.path.join(target_folder, os.path.basename(file)))

    print(f"✅ Copied {len(selected_files)} files with a total duration of {selected_duration/3600:.2f} hours to '{target_folder}'.")

# Example Usage
source_folder = "path/to/source_folder"  # Replace with your folder path
target_folder = "path/to/target_folder"  # Replace with your destination folder
total_duration_hours = 2  # Change to the desired duration in hours

select_random_wavs(source_folder, target_folder, total_duration_hours)
