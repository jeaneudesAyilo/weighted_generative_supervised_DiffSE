from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
#from pesq import pesq
from pypesq import pesq
import json
import pandas as pd
from librosa.util import find_files
import os
from pystoi import stoi
import sys

sys.path.append('./')
from sgmse.util.other import energy_ratios, mean_std, compute_sisdr


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help='Path to the json file enumerating test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--nb_test_file", type=int, default=10**9, help="Number of test files to be used")
    parser.add_argument("--noisy_test_dir", type=str, required=True, help='Directory containing the noisy test data')
    parser.add_argument("--clean_test_dir", type=str, required=True, help='Directory containing the clean test data')
    args = parser.parse_args()
        
        
    enhanced_dir = args.enhanced_dir
    noisy_dir = args.noisy_test_dir
    clean_dir = args.clean_test_dir
    
    
    data = {"ind_mix": [],"p_id": [],"utt_name": [], "filename": [], "length": [],"noise_type": [],"noise_start": [], "snr": [],"pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [], "pesq_noisy":[], "estoi_noisy": [], "si_sdr_noisy":[],"si_sir_noisy":[], "si_sar_noisy":[]}
    #data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "pesq_noisy":[], "estoi_noisy": [], "si_sdr_noisy":[]}
    
    #sr = 16000
    not_available = []; clean_noisy = [] ; clean_enhanced =[] ; noisy_enhanced = []
    
    
    # Evaluate standard metrics    
    
    available = 0
    
    with open(args.json_path, "r") as f:
        dataset = json.load(f)
        
        
    ind_start = 0
        
    if args.nb_test_file >= len(dataset):
        ind_end = len(dataset) #100
    else:        
        ind_end = args.nb_test_file        
    
        
    for ind_i, (ind_mix, mix_info) in tqdm(enumerate(dataset.items())):    
        
        if ind_start <= ind_i < ind_end:        

            clean_filepath = mix_info["clean_wav"].replace("{clean_root}",clean_dir)
            filename = mix_info["noisy_wav"].split('/')[-1]

            x,fs = read(clean_filepath) 

            y, _ = read(join(noisy_dir, filename)) #librosa.load(noisy_file, sr=sr)                

            x_method, _ = read(join(enhanced_dir, filename)) #librosa.load(join(enhanced_dir, filename),sr=sr)


            if len(x) != len(y) or len(x) != len(x_method) or len(y) != len(x_method):

                print("lenght not suited")
                not_available.append(filename) ; clean_noisy.append(len(x)-len(y)) ; clean_enhanced.append(len(x)-len(x_method));
                noisy_enhanced.append(len(y)-len(x_method))

            else:

                available +=1
                n = y - x 

                data["ind_mix"].append(ind_mix) 
                data["p_id"].append(["p_id"])            
                data["utt_name"].append(mix_info["utt_name"])        
                data["filename"].append(filename)
                data["length"].append(mix_info["length"])            
                data["noise_type"].append(mix_info["noise_type"])
                data["noise_start"].append(mix_info["noise_start"])
                data["snr"].append(mix_info["snr"])

                #data["pesq"].append(pesq(sr, x, x_method, 'wb'))
                data["pesq"].append(pesq(x, x_method, fs)) ; 
                data["estoi"].append(stoi(x, x_method, fs, extended=True))
                data["si_sdr"].append(compute_sisdr(x, x_method)) ; 
                #data["si_sdr"].append(energy_ratios(x_method, x, n)[0])
                data["si_sir"].append(energy_ratios(x_method, x, n)[1])
                data["si_sar"].append(energy_ratios(x_method, x, n)[2])

                ##metrics between noisy and clean
                #data["pesq_noisy"].append(pesq(sr, x, y, 'wb'))
                data["pesq_noisy"].append(pesq(x, y, fs)) ; 
                data["estoi_noisy"].append(stoi(x, y, fs, extended=True)) #sr
                data["si_sdr_noisy"].append(compute_sisdr(x, y)) ; 
                #data["si_sdr_noisy"].append(energy_ratios(y, x, n)[0])
                data["si_sir_noisy"].append(energy_ratios(y, x, n)[1])
                data["si_sar_noisy"].append(energy_ratios(y, x, n)[2])

    if available !=0:
        # Save results as DataFrame    
        df = pd.DataFrame(data)
        df.set_index('ind_mix',inplace=True)

        # POLQA evaluation  -  requires POLQA license and server, uncomment at your own peril.
        # This is batch processed for speed reasons and thus runs outside the for loop.
        # if not basic:
        #     clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
        #     enhanced_files = sorted(glob('{}/*.wav'.format(enhanced_dir)))
        #     clean_audios = [read(clean_file)[0] for clean_file in clean_files]
        #     enhanced_audios = [read(enhanced_file)[0] for enhanced_file in enhanced_files]
        #     polqa_vals = polqa(clean_audios, enhanced_audios, 16000, save_to=None)
        #     polqa_vals = [val[1] for val in polqa_vals]
        #     # Add POLQA column to DataFrame
        #     df['polqa'] = polqa_vals

        # Print results
        
        #print("POLQA: {:.2f} ± {:.2f}".format(*mean_std(df["polqa"].to_numpy())))
        print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
        print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
        print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
        print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
        print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))
        
        print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq_noisy"].to_numpy())))
        print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi_noisy"].to_numpy())))
        print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr_noisy"].to_numpy())))
        print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir_noisy"].to_numpy())))
        print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar_noisy"].to_numpy())))

        # Save DataFrame as csv file
        df.to_csv(join(enhanced_dir, "_results.csv"))
        
        ##summary results
        
        directory = join(args.enhanced_dir,"summary_metrics")
        if not os.path.exists(directory):
            os.makedirs(directory)        

        round(df.mean(),2).transpose().to_csv(os.path.join(directory, "_metrics_mean_global.csv"))
        round(df.std(),2).transpose().to_csv(os.path.join(directory, "_metrics_std_global.csv"))
        round(df.sem(),2).transpose().to_csv(os.path.join(directory, "_metrics_sem_global.csv"))

        round(df.groupby(["noise_type"]).mean(),2).to_csv(os.path.join(directory, "_metrics_mean_noise_type.csv"))
        round(df.groupby(["noise_type"]).std(),2).to_csv(os.path.join(directory, "_metrics_std_noise_type.csv"))
        round(df.groupby(["noise_type"]).sem(),2).to_csv(os.path.join(directory, "_metrics_sem_noise_type.csv"))

        round(df.groupby(["snr"]).mean(),2).to_csv(os.path.join(directory, "_metrics_mean_snr.csv"))
        round(df.groupby(["snr"]).std(),2).to_csv(os.path.join(directory, "_metrics_std_snr.csv"))
        round(df.groupby(["snr"]).sem(),2).to_csv(os.path.join(directory, "_metrics_sem_snr.csv"))           
            
    lenght_diff = pd.DataFrame({"not_available": not_available, 
                                "clean_noisy":clean_noisy,
                                "clean_enhanced":clean_enhanced,
                                "noisy_enhanced":noisy_enhanced})
    
    lenght_diff.to_csv(join(enhanced_dir, "_not_available.csv"), index=False)
    
    print("###### End #####")