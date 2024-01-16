# Diffusion-based speech enhancement with a weighted generative-supvised learning loss

This repository contains the PyTorch implementations of the paper :


 Jean-Eudes Ayilo, Mostafa Sadeghi, Romain Serizel. [*"Diffusion-based speech enhancement with a weighted generative-supvised learning loss"*](https://arxiv.org/abs/2309.10457v1) (2023)


It is mainly based on the [repository](https://github.com/sp-uhh/sgmse) of the paper of [Richter et al 2023](https://ieeexplore.ieee.org/abstract/document/10149431?casa_token=pjyWnkMDZsUAAAAA:XlhCwS0m39TyCRE07hezzkR27nHC8ylIH8TKhDPuNU4Diu8Lycc7zO53IxBqhoSt5uH2eiBrJG8). 
Given the training objective in that paper, our contribution consists in adding an additional supervision loss ie an L2-loss between a posterior mean (obtained with tweedie formula) of the clean speech and the groundtruth. Evaluation on test sets shows that doing so, the model appears to not only inherit some capabilities from the supervised method but also converses and improves the performance of the diffusion baseline approach. 

For training and evaluation, we used the same hyperparameters as the baseline [repository](https://github.com/sp-uhh/sgmse), except batch size which is set to 1, due to some limitation in computational ressources. Future works could investigate large hyperparameters tuning.

## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.


## Training

- Minimal running example with default settings for NTCD-TIMIT : 

```bash
python train.py --batch_size 1 --dataset tcd --base_dir <your_base_dir> --additional_loss --flogging <path_to_logging_folder>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.


- Minimal running example with default settings for WSJ0: 

```bash
python train.py --batch_size 1 --dataset wsj0 --clean_dir <path_to_clean_dataset> --noisy_dir <path_to_noisy_dataset> --additional_loss --flogging <path_to_logging_folder>
```
where `path_to_clean_dataset` is a path to a folder containing subdirectories for clean train (`wsj0_si_tr_s/`) and validation (`wsj0_si_dt_05/`) datasets and optionally clean test dataset (`wsj0_si_et_05/`) as well.

To see all available training options, run `python train.py --help`. 

**Note:**
The above command lines allow to run the proposed method ie the diffusion-based speech enhancement with an additional supervision loss. 

- To run the reference model ie the diffusion model without additional supervision, for NTCD-TIMIT for example : 

```bash

python train.py --batch_size 1 --dataset tcd --base_dir <your_base_dir> --flogging <path_to_logging_folder>
```

- To run the model in supervision mode only : 

```bash
python train.py --batch_size 1 --dataset tcd --base_dir <your_base_dir> --flogging <path_to_logging_folder> --supervised --embedding_type none --conditional 
```

## Evaluation

To generate the enhanced .wav files for test set :

* NTCD-TIMIT

```bash
python ./SE_eval/enhancement_tcd.py --test_dir <test_dir> --enhanced_dir <enhanced_dir> --ckpt <ckpt_path>
```

* WSJ0-QUT

```bash
python ./SE_eval/enhancement_wsj0.py --json_path <json_path> --enhanced_dir <enhanced_dir> --noisy_test_dir <noisy_dir> --ckpt <ckpt_path>
```

The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.


Just add `--supervised` argument if using a purely supervised checkpoint.

Then, to calculate and output the instrumental metrics, feel free to use the script [./SE_eval/statistics/run_objective_eval_to_fill.sh](./SE_eval/statistics/run_objective_eval_to_fill.sh) and fill in the blanks with the corresponding paths. It may be required to have a same path for `--enhanced_dir` and `--save_dir` arguments. Adding `--input_metrics` argument in that script will compute the evaluation metrics between the groundtruths and the noisy speechs and not between the groundtruths and the enhanced speeches.

Usage example: 

```bash
cd ./SE_eval/statistics

bash run_objective_eval_to_fill.sh
```

Both enhancement and metrics computation scripts should receive the same `--enhanced_dir` parameters. 

Finally, the notebook [./SE_eval/compute_statistics.ipynb](./SE_eval/compute_statistics.ipynb) can be used to compute some descriptive statistics (mean, standard deviation,...) on the computed metrics


## Citations / References

For citation purpose, please consider :

```bib
@misc{ayilo2023diffusionbased,
      title={Diffusion-based speech enhancement with a weighted generative-supervised learning loss}, 
      author={Jean-Eudes Ayilo and Mostafa Sadeghi and Romain Serizel},
      year={2023},
      eprint={2309.10457},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
 Jean-Eudes Ayilo, Mostafa Sadeghi, Romain Serizel. "Diffusion-based speech enhancement with a weighted generative-supvised learning loss", [*arXiv preprint arXiv:2309.10457v1*, 2023](https://arxiv.org/abs/2309.10457v1).  
