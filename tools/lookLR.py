import argparse
import matplotlib.pyplot as plt

from yolox.exp import get_exp
from yolox.utils.lr_scheduler import LRScheduler

def make_parser():
    parser = argparse.ArgumentParser("YOLOX LRScheduler")
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/code/YOLOX/exps/example/custom/yolox_mobilenetv2-050_033-ghostfpn-One-C.py',
        type=str,
        help="plz input your experiment description file",
    )
    return parser

def progress_in_iter(epoch, max_iter, iter):
    return epoch * max_iter + iter

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    scheduler = exp.scheduler
    basic_lr_per_img = exp.basic_lr_per_img
    batch_size = 64
    lr = basic_lr_per_img * batch_size
    warmup_epochs = exp.warmup_epochs
    warmup_lr = exp.warmup_lr
    no_aug_epochs = exp.no_aug_epochs
    min_lr_ratio = exp.min_lr_ratio
    max_epoch = exp.max_epoch
    milestones = exp.milestones

    lr_list = list()

    train_loader = exp.get_data_loader(
        batch_size=64,
        is_distributed=False,
        no_aug=False,
        cache_img=None,
    )

    max_iter = len(train_loader)
    iters_per_epoch = max_iter
    lr_scheduler = LRScheduler(
        scheduler,
        lr,
        iters_per_epoch,
        max_epoch,
        warmup_epochs=warmup_epochs,
        warmup_lr_start=warmup_lr,
        no_aug_epochs=no_aug_epochs,
        min_lr_ratio=min_lr_ratio,
        milestones=milestones,
    )
    
    for epoch in range(0, max_epoch):
        for ither in range(max_iter):
            lr = lr_scheduler.update_lr(progress_in_iter(epoch, max_iter, ither) + 1)
        lr_list.append(lr)
    print(lr_list)
    plt.plot(lr_list)
    plt.savefig('./yoloxwarmcos.jpg')