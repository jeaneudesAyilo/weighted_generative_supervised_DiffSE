import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    
    parser.add_argument("--test_df_path", type=str, help='File describing the original test data (which must have subdirectories clean/ and noisy/)')
    
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()
                 

    raw_results = pd.read_csv(os.path.join(args.enhanced_dir, "_results.csv"))

    test_df = pd.read_csv(args.test_df_path)

    #not_available = pd.read_csv(os.path.join(args.enhanced_dir ,"_not_available.csv"))

    test_df["filename"] = test_df.apply(lambda x: x["file_name"] + "_" + x["speaker_id"]+ "_" + x["noise_type"]+ "_" + str(x["snr"]) + ".wav",axis=1)

    ##merge to get the characteristics of the noisy files : ex : type of noise, snr
    raw_results_merged = raw_results.merge(test_df, how = "left", on= "filename")
    
    raw_results_merged.drop([col for col in raw_results_merged.columns if "Unnamed: 0" in col ], axis=1, inplace =True)
    
    raw_results_merged.to_csv( os.path.join(args.enhanced_dir, "_raw_results_merged.csv"))
    
    
    directory = os.path.join(args.enhanced_dir,"summary_metrics")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    round(raw_results_merged.mean(),2).transpose().to_csv(os.path.join(directory, "_metrics_mean_global.csv"))
    round(raw_results_merged.std(),2).transpose().to_csv(os.path.join(directory, "_metrics_std_global.csv"))
    round(raw_results_merged.sem(),2).transpose().to_csv(os.path.join(directory, "_metrics_sem_global.csv"))
        
    round(raw_results_merged.groupby(["noise_type"]).mean(),2).to_csv(os.path.join(directory, "_metrics_mean_noise_type.csv"))
    round(raw_results_merged.groupby(["noise_type"]).std(),2).to_csv(os.path.join(directory, "_metrics_std_noise_type.csv"))
    round(raw_results_merged.groupby(["noise_type"]).sem(),2).to_csv(os.path.join(directory, "_metrics_sem_noise_type.csv"))
    
    round(raw_results_merged.groupby(["snr"]).mean(),2).to_csv(os.path.join(directory, "_metrics_mean_snr.csv"))
    round(raw_results_merged.groupby(["snr"]).std(),2).to_csv(os.path.join(directory, "_metrics_std_snr.csv"))
    round(raw_results_merged.groupby(["snr"]).sem(),2).to_csv(os.path.join(directory, "_metrics_sem_snr.csv"))    
    
    print("###### End #####")
        