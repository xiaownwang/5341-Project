import datetime
import sys
import os

from torch.onnx.symbolic_opset9 import tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))  # ldm

import torch
import pytorch_lightning as pl
import argparse
from packaging import version
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
import os
import signal
from ldm.util import instantiate_from_config
import torch.multiprocessing as mp

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. Parameters can be overwritten or added with command-line options of the form `--key value`.",
                        default=["configs/v1_fine_tuning.yaml"])
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False,
                        help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging data")
    parser.add_argument("--pretrained_model", type=str, default="", help="path to pretrained model")
    parser.add_argument('--annotation_file', type=str, required=True, help='path to the annotations JSON file')
    parser.add_argument('--coco_root', type=str, required=True, help='path to the images directory')
    parser.add_argument('--ckpt_save', type=str, required=True, help='path to save trained model')
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True,
                        help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--train_from_scratch", type=str2bool, nargs="?", const=True, default=False,
                        help="Train from scratch")

    return parser

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    # load pretrained model
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    config.data.params.train.params.annotation_file = opt.annotation_file
    config.data.params.validation.params.annotation_file = opt.annotation_file
    config.data.params.test.params.annotation_file = opt.annotation_file

    config.data.params.train.params.coco_root = opt.coco_root
    config.data.params.validation.params.coco_root = opt.coco_root
    config.data.params.test.params.coco_root = opt.coco_root
    model = instantiate_from_config(config.model)

    if not opt.resume:
        ckpt_file = torch.load(opt.pretrained_model, map_location='cpu')['state_dict']
        model.load_state_dict(ckpt_file, strict=False)
        print("Load Stable Diffusion v1-4!")

    # Fine-tuning
    '''
        layers:
        model.diffusion_model.time_embed, model.diffusion_model.input_blocks, model.diffusion_model.middle_block, 
        model.diffusion_model.output_blocks, model.diffusion_model.out, proj_out, first_stage_model.encoder.conv_in, 
        first_stage_model.encoder.down, first_stage_model.encoder.mid, first_stage_model.encoder.norm_out, 
        first_stage_model.encoder.conv_out, first_stage_model.decoder.conv_in, first_stage_model.decoder.mid,
        first_stage_model.decoder.up, first_stage_model.decoder.norm_out, first_stage_model.quant_conv, 
        first_stage_model.post_quant_conv, cond_stage_model.transformer.vision_model.embeddings, 
        cond_stage_model.transformer.vision_model.pre_layrnorm, cond_stage_model.transformer.vision_model.encoder.layers,
        cond_stage_model.transformer.vision_model.post_layernorm, cond_stage_model.final_ln, cond_stage_model.mapper.resblocks
        '''
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name}, requires_grad: {param.requires_grad}")

    last_conv_layer_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer_name = name

    for name, param in model.named_parameters():
        if last_conv_layer_name and last_conv_layer_name or 'diffusion_model.out' in name:
            param.requires_grad = True  # Unfreeze the last convolutional layer
        else:
            param.requires_grad = False  # Freeze all other parameters


    # Reconfigure the optimizer to train only the unfrozen layer
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],  # Only optimized the parameters of unfrozen layer
        lr=1e-5
    )

    # Trainer, logger, checkpoints
    trainer_kwargs = dict()

    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)

    # logger config
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": opt.debug,
                "id": nowname,
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.TestTubeLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }

    lightning_config = config.pop("lightning", OmegaConf.create())
    default_logger_cfg = default_logger_cfgs["testtube"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # checkpoint callback config
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{epoch:06}",
            "verbose": True,
            "save_last": True,
        }
    }

    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        default_modelckpt_cfg["params"]["save_top_k"] = 30

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # checkpoint callback config
    modelckpt_cfg = {
        "dirpath": ckptdir,
        "filename": "{epoch:02d}-{val_loss:.2f}",
        "save_top_k": 3,
        "monitor": "val/loss",
        "mode": "min",
    }
    model_checkpoint = ModelCheckpoint(**modelckpt_cfg)

    # callback config
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 500,
                "max_images": 4,
                "clamp": True
            }
        },
        "learning_rate_logger": LearningRateMonitor(logging_interval="step"),
    }

    # Instantiate the callback
    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    trainer_callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_callbacks.append(model_checkpoint)
    trainer_callbacks.append(default_callbacks_cfg["learning_rate_logger"])

    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # GPU or CPU
    if "gpus" in trainer_config and trainer_config["gpus"]:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        trainer_config["accelerator"] = "gpu"
        trainer_config["devices"] = gpuinfo
        cpu = False
    else:
        print("No GPUs specified, running on CPU")
        trainer_config["accelerator"] = "cpu"
        trainer_config["devices"] = 1
        cpu = True

    # add non-default parameters to trainer_config
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)

    # Initialize Trainer
    trainer_kwargs["callbacks"] = trainer_callbacks
    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    # Load Data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("########## Data ###########")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    print("###########################")


    # learning rate config
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    num_nodes = 1

    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    else:
        model.learning_rate = base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e}")


    # checkpoint functions (triggered by the USR1 signal)
    def melk(*args, **kwargs):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)


    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()


    import signal

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # start training
    if opt.train:
        try:
            trainer.fit(model, data)
            print('training finished!')
            print("Saving the fine-tuning model checkpoint...")
            final_ckpt_path = os.path.join(opt.ckpt_save, "fine_tuning_model.ckpt")
            trainer.save_checkpoint(final_ckpt_path)
            print(f"Final model saved at {final_ckpt_path}")
        except Exception:
            melk()
            raise

    # test
    # if not opt.no_test and not trainer.interrupted:
    #     trainer.test(model, data)