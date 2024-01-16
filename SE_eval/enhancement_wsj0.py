import glob
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import time
import torch
from soundfile import write
from torchaudio import load
from tqdm import tqdm
import json
import sys

sys.path.append('./')
from sgmse.model import ScoreModel
from sgmse.util.other import ensure_dir, pad_spec

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help='Path to the json file enumerating test data')
    
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')    
    parser.add_argument("--nb_test_file", type=int, default=10**9, help="Number of test files to be used")
    
    parser.add_argument("--noisy_test_dir", type=str, required=True, help='Directory containing the noisy test data')
    parser.add_argument("--ckpt", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--supervised", action="store_true", help="whether to use supervised learning or the default unsupervised")
    parser.add_argument("--no_gpu", action="store_true", help="whether to not use gpu or use")  
    
    args = parser.parse_args()

            
    noisy_dir = args.noisy_test_dir
    
    checkpoint_file = args.ckpt
    corrector_cls = args.corrector

    target_dir = args.enhanced_dir
    ensure_dir(target_dir)

    # Settings
    sr = 16000
    snr = args.snr
    N = args.N
    corrector_steps = args.corrector_steps

    # Load score model 
    model = ScoreModel.load_from_checkpoint(checkpoint_file, base_dir='', batch_size=16, num_workers=0, kwargs=dict(gpu=False))
    model.eval(no_ema=False)
    
    if not args.no_gpu :
        model.cuda()
    
        
    #df = pd.read_json(args.json_path, orient ='index')    
    #noisy_files = sorted([file.replace("{noisy_root}",noisy_dir) for file in df.noisy_wav.tolist()])

    with open(args.json_path, "r") as f:
        dataset = json.load(f)

    ind_start = 0
        
    if args.nb_test_file >= len(dataset):
        ind_end = len(dataset) #100
    else:        
        ind_end = args.nb_test_file
                
    data = {"File name": [],"Length":[], "sgmse_runtime": []}
    
    for ind_i, (ind_mix, mix_info) in tqdm(enumerate(dataset.items())):
        
        if ind_start <= ind_i < ind_end:
        
            filename = mix_info["noisy_wav"].split('/')[-1]

            # Load wav
            y, _ = load(join(noisy_dir, filename)) 
            T_orig = y.size(1)   

            # Normalize
            norm_factor = y.abs().max()
            y = y / norm_factor

            # Prepare DNN input
            if not args.no_gpu :
                Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
                Y = pad_spec(Y)
                Y = Y.cuda()
            else:
                Y = torch.unsqueeze(model._forward_transform(model._stft(y)), 0)
                Y = pad_spec(Y)
                Y = Y.cpu()            
            
            
            # Reverse sampling

            if not args.supervised :
                start_time = time.time()

                sampler = model.get_pc_sampler(
                    'reverse_diffusion', corrector_cls, Y, N=N, 
                    corrector_steps=corrector_steps, snr=snr)
                sample, _ = sampler()

                end_time = time.time()
                runtime = end_time - start_time
                # Backward transform in time domain
                x_hat = model.to_audio(sample.squeeze(), T_orig)

            else : 
                start_time = time.time()
                x_hat_spec = model.forward(x=None, t=None,y=Y)
                end_time = time.time()
                # Backward transform in time domain
                x_hat = model.to_audio(x_hat_spec.squeeze(), T_orig)            
                x_hat = x_hat.detach()            
                end_time = time.time()
                runtime = end_time - start_time                

            # Renormalize
            x_hat = x_hat * norm_factor

            # Write enhanced wav file
            write(join(target_dir, filename), x_hat.cpu().numpy(), 16000)

            data["File name"].append(mix_info["utt_name"]) #or append(mix_info["filename"])
            data["Length"].append(mix_info["length"]) 
            data["sgmse_runtime"].append(runtime)  
    
    data = pd.DataFrame(data)
    data.to_csv(join(target_dir, "_runtime.csv"),index=False)
        