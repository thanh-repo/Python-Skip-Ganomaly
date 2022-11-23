"""
TRAIN SKIP/GANOMALY

. Example: Run the following command from the terminal.
    run train.py                                    \
        --model <skipganomaly, ganomaly>            \
        --dataset cifar10                           \
        --abnormal_class airplane                   \
        --display                                   \

    python train.py --dataset surface_crack --niter 15 --isize 256 --batchsize 8
"""

##
# LIBRARIES
import torch
from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##


def main():
    """ Training
    """
    torch.cuda.empty_cache()
    opt = Options().parse()
    # opt.dataset = "surface_crack"
    # opt.isize = 256
    # opt.device = "gpu"
    # opt.gpu_ids = "0"
    # opt.model = "skipganomaly"
    data = load_data(opt)
    model = load_model(opt, data)
    # model.train()
    # model.test(True)
    model.predict(epoch=0, image=r"C:\Users\Thanh\Documents\GitHub\Python-Skip-Ganomaly\data\surface_crack\test\1.abnormal\7069-17.jpg")


if __name__ == '__main__':
    main()
