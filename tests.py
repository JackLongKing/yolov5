from models.yolov6 import Model, V5Small, V5Ghost
import argparse
import torch

import yaml
from models.experimental import check_file
from utils import torch_utils
from models.backbones.ghostnet import GhostNet



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov6_ghostnet.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    device = torch_utils.select_device(opt.device)

    # Create model
    # model = Model(opt.cfg).to(device)
    # model.train()

    model = V5Small(opt.cfg).to(device)
    a = torch.randn([1, 3, 832, 832]).to(device)
    b = model(a)
    for a in b:
      print(a.shape)

    # cfgs = [
    #     # k, t, c, SE, s
    #     # stage1
    #     [[3,  16,  16, 0, 1]],
    #     # stage2
    #     [[3,  48,  24, 0, 2]],
    #     [[3,  72,  24, 0, 1]],
    #     # stage3
    #     [[5,  72,  40, 0.25, 2]],
    #     [[5, 120,  40, 0.25, 1]],
    #     # stage4
    #     [[3, 240,  80, 0, 2]],
    #     [[3, 200,  80, 0, 1],
    #      [3, 184,  80, 0, 1],
    #      [3, 184,  80, 0, 1],
    #      [3, 480, 112, 0.25, 1],
    #      [3, 672, 112, 0.25, 1]
    #      ],
    #     # stage5
    #     [[5, 672, 160, 0.25, 2]],
    #     [[5, 960, 160, 0, 1],
    #      [5, 960, 160, 0.25, 1],
    #      [5, 960, 160, 0, 1],
    #      [5, 960, 160, 0.25, 1]
    #      ]
    # ]
    # ghost = GhostNet(cfgs)
    # a = torch.randn([1, 3, 800, 800])
    # b = ghost(a)
    # print('ghostnet output: ')

    model = V5Ghost(opt.cfg).to(device)
    # a = torch.randn([1, 3, 1280, 768]).to(device)  # 128,80,48, 256,40,24 512,20,12
    a = torch.randn([1, 3, 832, 832]).to(device)  # 128,80,48, 256,40,24 512,20,12
    b = model(a)
    print('output shape of V5Ghost')
    for a in b:
      print(a.shape)
    
