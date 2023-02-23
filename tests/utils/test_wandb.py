import torch
import argparse
from yolox.exp import Exp, get_exp
from yolox.utils import WandbLogger, adjust_status, get_local_rank

def make_parser():
    parser = argparse.ArgumentParser("Wandb verification")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
        )
    parser.add_argument(
        "-c",
        "--ckpt",
        default=None,
        type=str,
        help="checkpoint file",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
        Implemented loggers include `tensorboard` and `wandb`.",
        default="wandb"
    )
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    args = make_parser().parse_args()

    exp = get_exp(args.exp_file)
    model = exp.get_model()

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()
    is_distributed = num_gpu > 1
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if args.ckpt is None:
        raise
    else:
        ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["model"])

    evaluator = exp.get_evaluator(batch_size=10, is_distributed=is_distributed)
    
    with adjust_status(model, training=False):
        (ap50_95, ap50, summary), predictions = exp.eval(
            model, evaluator, is_distributed, return_outputs=True
        )
    wandb_logger = WandbLogger.initialize_wandb_logger(args, exp, evaluator.dataloader.dataset)
    wandb_logger.log_images(predictions)
    wandb_logger.finish()