"""
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
"""

import os
from tqdm import tqdm
import pandas as pd
from soundfile import SoundFile
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
import numpy as np
from six.moves import cPickle as pickle  # for performance
from pypesq import pesq
from pystoi import stoi
import json
import pyloudnorm as pyln
import dnnmos_metric.dnnmos_metric as dnnmos_metric
import warnings

warnings.filterwarnings("ignore")

meter = pyln.Meter(16000)


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def normalize(x, target_loudness=-30, meter=None, sr=16000):
    """
    LUFS normalization of a signal using pyloudnorm.

    Parameters
    ----------
    x : ndarray
        Input signal.
    target_loudness : float, optional
        Target loudness of the output in dB LUFS. The default is -30.
    meter : Meter, optional
        The pyloudnorm BS.1770 meter. The default is None.
    sr : int, optional
        Sampling rate. The default is 16000.

    Returns
    -------
    x_norm : ndarray
        Normalized output signal.
    """

    if meter is None:
        meter = pyln.Meter(sr)  # create BS.1770 meter

    # peak normalize to 0.7 to ensure that the meter does not return -inf
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.7

    # measure the loudness first
    loudness = meter.integrated_loudness(x)

    # loudness normalize audio to target_loudness LUFS
    x_norm = pyln.normalize.loudness(x, loudness, target_loudness)

    return x_norm


def compute_dnnmos(x, sr):
    """Compute DNN-MOS metric
    Args:
        x: array of float, shape (n_samples,)
        sr (int): sample rate of files
    Returns:
        DNN-MOS metric (dict): SIG_MOS, BAK_MOS, OVR_MOS (float)
    """
    x = normalize(x, target_loudness=-30, meter=meter, sr=sr)

    dnsmos_res = dnnmos_metric.compute_dnsmos(x, fs=sr)

    return dnsmos_res


def compute_sisdr(reference, estimate):
    """Compute the scale invariant SDR.

    Parameters
    ----------
    estimate : array of float, shape (n_samples,)
        Estimated signal.
    reference : array of float, shape (n_samples,)
        Ground-truth reference signal.

    Returns
    -------
    sisdr : float
        SI-SDR.

    Example
    --------
    >>> import numpy as np
    >>> from sisdr_metric import compute_sisdr
    >>> np.random.seed(0)
    >>> reference = np.random.randn(16000)
    >>> estimate = np.random.randn(16000)
    >>> compute_sisdr(estimate, reference)
    -48.1027283264049
    """
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate * reference) + eps) / (
        np.sum(np.abs(reference) ** 2) + eps
    )
    sisdr = 10 * np.log10(
        (np.sum(np.abs(alpha * reference) ** 2) + eps)
        / (np.sum(np.abs(alpha * reference - estimate) ** 2) + eps)
    )
    return sisdr


def compute_pesq(target, enhanced, sr):
    """Compute PESQ using PyPESQ
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        PESQ metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return pesq(target, enhanced, sr)


def compute_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        STOI metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return stoi(target, enhanced, sr, extended=True)


# def read_audio(filename):
#     """Read a wavefile and return as numpy array of floats.
#     Args:
#         filename (string): Name of file to read
#     Returns:
#         ndarray: audio signal
#     """
#     try:
#         wave_file = SoundFile(filename)
#     except:
#         # Ensure incorrect error (24 bit) is not generated
#         raise Exception(f"Unable to read {filename}.")
#     return wave_file.read()


def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
    Args:
        filename (string): Name of file to read
    Returns:
        tuple: Tuple containing audio signal and a boolean indicating success
    """
    success = True  # Assume success by default

    try:
        wave_file = SoundFile(filename)
    except:
        success = False  # An error occurred

    if success:
        audio_signal = wave_file.read()
        return audio_signal
    else:
        return []


def run_metrics(input_file, save_dir, dnn_mos=False, input_metrics=False):
    fs = 16000
    if input_metrics:
        enh_file = input_file["noisy"]
    else:
        enh_file = input_file["enhanced"]
    tgt_file = input_file["clean"]

    metrics_file = os.path.join(
        save_dir,
        f"{input_file['speaker_id']}_{input_file['noise_type']}_{input_file['snr']}_{input_file['file_name']}.pkl",
    )

    # Skip processing with files dont exist or metrics have already been computed
    if (
        (not os.path.isfile(enh_file))
        or (not os.path.isfile(tgt_file))
        or (os.path.isfile(metrics_file))
    ):
        return

    # Read enhanced signal
    enh = read_audio(enh_file)
    # Read clean/target signal
    clean = read_audio(tgt_file)

    if len(enh) != 0:
        len_x = np.min([len(enh), len(clean)])
        clean = clean[:len_x]
        enh = enh[:len_x]

        # Check that both files are the same length, otherwise computing the metrics results in an error
        if len(clean) != len(enh):
            raise Exception(
                f"Wav files {enh_file} and {tgt_file} should have the same length"
            )

        # Compute metrics
        m_stoi = compute_stoi(clean, enh, fs)
        m_pesq = compute_pesq(clean, enh, fs)
        m_sisdr = compute_sisdr(clean, enh)

        if dnn_mos:
            m_dnnmos = compute_dnnmos(enh, fs)
            di_ = {
                "File name": input_file["file_name"],
                "Noise Type": input_file["noise_type"],
                "Noise SNR": input_file["snr"],
                "SI-SDR": m_sisdr,
                "STOI": m_stoi,
                "PESQ": m_pesq,
                "MOS_SIG": m_dnnmos["sig_mos"],
                "MOS_BAK": m_dnnmos["bak_mos"],
                "MOS_OVR": m_dnnmos["ovr_mos"],
            }
        else:
            di_ = {
                "File name": input_file["file_name"],
                "Noise Type": input_file["noise_type"],
                "Noise SNR": input_file["snr"],
                "SI-SDR": m_sisdr,
                "STOI": m_stoi,
                "PESQ": m_pesq,
            }
        save_dict(di_, metrics_file)


def compute_metrics(input_params, save_dir, dnn_mos, input_metrics):
    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for param_ in input_params:
            futures.append(
                executor.submit(run_metrics, param_, save_dir, dnn_mos, input_metrics)
            )
        proc_list = [future.result() for future in tqdm(futures)]

    df_metrics = pd.DataFrame(
        columns=["File name", "Noise Type", "Noise SNR", "SI-SDR", "STOI", "PESQ"]
    )

    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    # Store results in one file
    for this_file in tqdm(pkl_files):
        this_file_path = os.path.join(save_dir, this_file)
        this_res = load_dict(this_file_path)

        if dnn_mos:
            df_metrics = pd.concat(
                [
                    df_metrics,
                    pd.DataFrame.from_dict(
                        {
                            "File name": [this_res["File name"]],
                            "Noise Type": [this_res["Noise Type"]],
                            "Noise SNR": [this_res["Noise SNR"]],
                            "PESQ": [this_res["PESQ"]],
                            "STOI": [this_res["STOI"]],
                            "SI-SDR": [this_res["SI-SDR"]],
                            "MOS_SIG": [this_res["MOS_SIG"]],
                            "MOS_BAK": [this_res["MOS_BAK"]],
                            "MOS_OVR": [this_res["MOS_OVR"]],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            df_metrics = pd.concat(
                [
                    df_metrics,
                    pd.DataFrame.from_dict(
                        {
                            "File name": [this_res["File name"]],
                            "Noise Type": [this_res["Noise Type"]],
                            "Noise SNR": [this_res["Noise SNR"]],
                            "PESQ": [this_res["PESQ"]],
                            "STOI": [this_res["STOI"]],
                            "SI-SDR": [this_res["SI-SDR"]],
                        }
                    ),
                ],
                ignore_index=True,
            )

        # remove tmp file
        os.system(f"rm {this_file_path}")

    # Save the DataFrame to a CSV file
    if input_metrics:
        df_metrics.to_csv(os.path.join(save_dir, "input_metrics.csv"), index=False)
    else:
        df_metrics.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run performance evaluation metrics on the enhanced signals."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory to the test data."
    )
    parser.add_argument(
        "--dnn_mos", action="store_true", help="Whether to compute DNNMOS or not."
    )
    parser.add_argument(
        "--input_metrics",
        action="store_true",
        help="Whether to compute input (mixture) metrics or not.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="TCD-TIMIT",
        help="Directory to the test data.",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Directory to the enhanced test data.",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Directory to save the results."
    )
    args = parser.parse_args()

    if args.dataset == "TCD-TIMIT":
        # Load file list and select the target segment to process
        files_list = load_dict(args.data_dir)
        input_params = [
            {
                "noisy": filename["mix_file"],
                "clean": filename["speech_file"],
                "file_name": filename["file_name"],
                "noise_type": filename["noise_type"],
                "snr": filename["snr"],
                "speaker_id": filename["speaker_id"],
                # "enhanced": f"{args.enhanced_dir}/{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}_{filename['file_name']}.wav", # VAE-SE
                "enhanced": f"{args.enhanced_dir}/{filename['file_name']}_{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}.wav",  # SGMSE
            }
            for filename in files_list
        ]
    elif args.dataset == "WSJ0":
        noisy_root = "/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/source_separation/QUT_WSJ0/test"
        clean_root = "/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/source_separation/WSJ0_SE/wsj0_si_et_05"
        # Load file json
        with open(args.data_dir, "r") as f:
            dataset = json.load(f)
        input_params = [
            {
                "noisy": filename["noisy_wav"].format(noisy_root=noisy_root),
                "clean": filename["clean_wav"].format(clean_root=clean_root),
                "file_name": filename["utt_name"],
                "noise_type": filename["noise_type"],
                "snr": filename["snr"],
                "speaker_id": filename["p_id"],
                # "enhanced": f"{args.enhanced_dir}/{filename['utt_name']}.wav",  # VAE-SE
                "enhanced": f"{args.enhanced_dir}/{filename['utt_name']}_{filename['noise_type']}_{filename['snr']}.wav",  # SGMSE
            }
            for (_, filename) in dataset.items()
        ]
    else:
        raise ValueError("Invalid dataset.")

    compute_metrics(input_params, args.save_dir, args.dnn_mos, args.input_metrics)
