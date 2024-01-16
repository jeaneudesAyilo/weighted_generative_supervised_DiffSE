from os.path import join 
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
#from pesq import pesq
from pypesq import pesq

import pandas as pd
from librosa.util import find_files

from pystoi import stoi
import sys

sys.path.append('./')
from sgmse.util.other import energy_ratios, mean_std, compute_sisdr


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, help='Directory containing the original test data (must have subdirectories clean/ and noisy/. Use it for tcd dataset)')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--dataset", type=str, choices=("tcd", "wsj0"), required = True, help="whether to use tcd-timit dataset or wsj0")
    parser.add_argument("--clean_test_dir", type=str, default="none", help="path to the clean speechs dir. This dir should contain only test clean speechs.Only valid for wsj0 dataset or similarly organised dataset")

    parser.add_argument("--noisy_test_dir", type=str, default="none", help="This dir should contain only test noisy speechs. Only valid for wsj0 dataset or similarly organised dataset")         
    
    args = parser.parse_args()

    if args.dataset == "tcd":
        test_dir = args.test_dir
        clean_dir = join(test_dir, "clean/")
        noisy_dir = join(test_dir, "noisy/")
        clean_files = sorted(glob('{}/*.wav'.format(clean_dir)))
        noisy_files = sorted(glob('{}/*.wav'.format(noisy_dir)))
    
    elif args.dataset == "wsj0":
        clean_dir = args.clean_test_dir
        noisy_dir = args.noisy_test_dir 
        clean_files = sorted(find_files(clean_dir, ext='wav'))         
        noisy_files = sorted(find_files(noisy_dir, ext='wav')) 

        
    enhanced_dir = args.enhanced_dir

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": [], "pesq_noisy":[], "estoi_noisy": [], "si_sdr_noisy":[],"si_sir_noisy":[], "si_sar_noisy":[]}
    #data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "pesq_noisy":[], "estoi_noisy": [], "si_sdr_noisy":[]}
    
    #sr = 16000
    not_available = []; clean_noisy = [] ; clean_enhanced =[] ; noisy_enhanced = []

    
    # Evaluate standard metrics    
    
    available = 0
    
    for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        if args.dataset == "tcd":
            x, fs = read(join(clean_dir, filename)) #librosa.load(join(clean_dir, filename), sr=sr)
        
        elif args.dataset == "wsj0":            
            clean_filename = clean_file.split('/')[-1]
            assert clean_filename.replace(".wav","") in filename.replace(".wav","")
            x,fs = read(clean_file) 
            
        
        y, _ = read(noisy_file) #librosa.load(noisy_file, sr=sr)                

        x_method, _ = read(join(enhanced_dir, filename)) #librosa.load(join(enhanced_dir, filename),sr=sr)
        
                
        if len(x) != len(y) or len(x) != len(x_method) or len(y) != len(x_method):
            
            print("lenght not suited")
            not_available.append(filename) ; clean_noisy.append(len(x)-len(y)) ; clean_enhanced.append(len(x)-len(x_method));
            noisy_enhanced.append(len(y)-len(x_method))
                        
        else:
            
            available +=1
            n = y - x 

            data["filename"].append(filename)
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

    # Save results as DataFrame    
    df = pd.DataFrame(data)

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
    print(enhanced_dir)
    
    if available !=0:
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
    df.to_csv(join(enhanced_dir, "_results.csv"), index=False)
    
    lenght_diff = pd.DataFrame({"not_available": not_available, 
                                "clean_noisy":clean_noisy,
                                "clean_enhanced":clean_enhanced,
                                "noisy_enhanced":noisy_enhanced})
    
    lenght_diff.to_csv(join(enhanced_dir, "_not_available.csv"), index=False)
    
