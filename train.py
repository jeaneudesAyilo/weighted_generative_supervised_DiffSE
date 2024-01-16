import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--flogging", help="Specific folder for logging") 
          parser_.add_argument("--supervised", action="store_true", help="whether to use supervised learning or the default unsupervised")      
          
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     parser = pl.Trainer.add_argparse_args(parser)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)


    ###ideal conditions checking for the audio_only
     if args.supervised: 
         if arg_groups['Backbone'].embedding_type != 'none':
             print(f"WARNING : In supervised case, embedding_type should be none but found {args.embedding_type}. Forcing embedding_type=none")
             arg_groups['Backbone'].embedding_type = 'none'
         
         if arg_groups['Backbone'].conditional:
             print(f"WARNING : In supersised case, conditional should be False but found {arg_groups['Backbone'].conditional}. Forcing conditional=False")
             
             arg_groups['Backbone'].conditional = False


     # Initialize logger, trainer, model, datamodule

     if args.supervised:
          model = ScoreModel(
               backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
               **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['Backbone']),
                    **vars(arg_groups['DataModule']),
                    **{"supervised":args.supervised} ##collect in a dict all the arguments that different can used in common 
               }
          )
               
     else :
          model = ScoreModel(
               backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
               **{
                    **vars(arg_groups['ScoreModel']),
                    **vars(arg_groups['SDE']),
                    **vars(arg_groups['Backbone']),
                    **vars(arg_groups['DataModule']),
                    **{"supervised":args.supervised} ##collect in a dict all the arguments that different can used in common 
               }
          )        

            
     #for u,name in enumerate(model.state_dict()):
        #print(u,":",name)
                        
     if args.supervised: 
          log_path = "logs/supervised"  
          project_name = "sgmse_supervised"        
     
     else: 
          log_path = "logs"
          project_name = "sgmse"

     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir=log_path, name=f"tensorboard_{project_name}")
     else:
          logger = WandbLogger(project=project_name, log_model=True, save_dir=log_path)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"{log_path}/{args.flogging}/{logger.version}", save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{log_path}/{args.flogging}/{logger.version}", 
               save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{log_path}/{args.flogging}/{logger.version}", 
               save_top_k=2, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          strategy=DDPStrategy(find_unused_parameters=False), logger=logger,  #DDPPlugin
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks, max_epochs=200, accelerator='gpu', )
        
     # Train model
     trainer.fit(model)
