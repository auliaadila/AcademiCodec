import argparse
import glob
import os
import scipy.signal as signal
import numpy as np
import pandas as pd
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from scipy.io import wavfile

def cal_pesq_audio(ref_signal, ref_rate, deg_signal, deg_rate):
    """Computes PESQ scores (NB & WB) for two given signals."""
    if ref_signal.shape[0] == 0 or deg_signal.shape[0] == 0:
        print("Error: One of the input signals is empty. Skipping PESQ calculation.")
        return np.nan, np.nan

    # Resample to 16kHz if needed
    if ref_rate != 16000:
        ref_signal = signal.resample(ref_signal, 16000)
    if deg_rate != 16000:
        deg_signal = signal.resample(deg_signal, 16000)

    # Ensure both signals are of equal length
    min_len = min(len(ref_signal), len(deg_signal))
    ref_signal = ref_signal[:min_len]
    deg_signal = deg_signal[:min_len]

    # Compute PESQ scores
    try:
        nb_pesq = pesq(16000, ref_signal, deg_signal, 'nb')
        wb_pesq = pesq(16000, ref_signal, deg_signal, 'wb')
    except Exception as e:
        print(f"Error computing PESQ for signals: {e}")
        return np.nan, np.nan

    return nb_pesq, wb_pesq

def calculate_stoi(ref_signal, deg_signal, sample_rate):
    """Computes STOI score between two signals."""
    try:
        min_len = min(len(ref_signal), len(deg_signal))
        ref_signal = ref_signal[:min_len]
        deg_signal = deg_signal[:min_len]
        return stoi(ref_signal, deg_signal, sample_rate, extended=False)
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return np.nan

def process_files(ref_dir, deg_dir, output_csv):
    """Computes PESQ and STOI scores for each file and saves them to a CSV file."""
    input_files = glob.glob(f"{deg_dir}/*.wav")

    if not input_files:
        print("No degraded WAV files found in the specified directory.")
        return

    results = []
    total_nb_pesq = []
    total_wb_pesq = []
    total_stoi = []

    for deg_wav in tqdm(input_files):
        filename = os.path.basename(deg_wav)
        ref_wav = os.path.join(ref_dir, filename)

        if not os.path.exists(ref_wav):
            print(f"Warning: Reference file not found for {filename}, skipping.")
            continue

        try:
            ref_rate, ref_signal = wavfile.read(ref_wav)
            deg_rate, deg_signal = wavfile.read(deg_wav)

            nb_pesq, wb_pesq = cal_pesq_audio(ref_signal, ref_rate, deg_signal, deg_rate)
            stoi_score = calculate_stoi(ref_signal, deg_signal, ref_rate)

            # Store valid scores for computing mean
            if not np.isnan(nb_pesq): total_nb_pesq.append(nb_pesq)
            if not np.isnan(wb_pesq): total_wb_pesq.append(wb_pesq)
            if not np.isnan(stoi_score): total_stoi.append(stoi_score)

            results.append({
                "filename": filename,
                "nb_pesq": nb_pesq,
                "wb_pesq": wb_pesq,
                "stoi": stoi_score
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"PESQ & STOI results saved to {output_csv}")

    # Print mean scores
    if total_nb_pesq and total_wb_pesq and total_stoi:
        mean_nb_pesq = np.mean(total_nb_pesq)
        mean_wb_pesq = np.mean(total_wb_pesq)
        mean_stoi = np.mean(total_stoi)
        print(f"\nMean NB PESQ: {mean_nb_pesq:.3f}")
        print(f"Mean WB PESQ: {mean_wb_pesq:.3f}")
        print(f"Mean STOI: {mean_stoi:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute PESQ and STOI scores for a folder of WAV files.")

    parser.add_argument('-r', '--ref_dir', required=True, help="Reference WAV folder.")
    parser.add_argument('-d', '--deg_dir', required=True, help="Degraded WAV folder.")
    parser.add_argument('-o', '--output_csv', required=True, help="Path to save PESQ & STOI results CSV.")

    args = parser.parse_args()

    process_files(args.ref_dir, args.deg_dir, args.output_csv)

# python evaluate_pesq_stoi.py --ref_dir /path/to/reference \
#                              --deg_dir /path/to/degraded \
#                              --output_csv results.csv