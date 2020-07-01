import argparse

import yaml

from models.experimental import *
from .backbones.ghostnet import GhostNet


class Detect(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        # self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training or self.export:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.export:
                    # y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    # y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    x1y1, x2y2, conf, prob = torch.split(y, [2, 2, 1, self.nc], dim=4)
                    x1y1 = ((x1y1*2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]).type(x[i].dtype)
                    x2y2 = (x2y2 * 2) ** 2 * self.anchor_grid[i]
                    xyxy = torch.cat((x1y1, x2y2), dim=4)
                    # add a idx (label ids before prob)
                    idxs = torch.argmax(prob, dim=-1).unsqueeze(axis=-1).type(x[i].dtype)
                    y = torch.cat((xyxy, conf, idxs, prob), dim=4)
                    # we added idxs so no+1
                    z.append(y.view(bs, -1, self.no+1))
                else:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    z.append(y.view(bs, -1, self.no))
        if self.training:
            return x if self.training else (torch.cat(z, 1), x)
        elif self.export:
            return torch.cat(z, 1)
        else:
            return (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class HeadV5(torch.nn.Module):

    def __init__(self, nc=80, na=3, ratio=0.5):
        super(HeadV5, self).__init__()
        self.bottleneck_csp3 = BottleneckCSP(int(1024*ratio), int(1024*ratio), 1, False)

        self.conv_block4 = Conv(int(1024*ratio), int(512*ratio), 1, 1)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat() # conact pre and conv_block2
        self.bottleneck_csp4 = BottleneckCSP(int(1024*ratio), int(512*ratio), 1, False)
                                                                                                                                       
        self.conv_block5 = Conv(int(512*ratio), int(256*ratio))
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.concat1 = Concat() # concat pre and conv_block1
        self.bottleneck_csp5 = BottleneckCSP(int(512*ratio), int(256*ratio), 1, False)

        self.conv0 = nn.Conv2d(int(256*ratio), na*(nc+5), 1, 1)
        self.conv_block6 = Conv(int(256*ratio), int(256*ratio), 3, 2) # connect bottleneck_csp5
        self.concat = Concat() # concat pre and bottleneck_csp4
        self.bottleneck_csp6 = BottleneckCSP(int(512*ratio), int(512*ratio), 1, False)

        self.conv1 = nn.Conv2d(int(512*ratio), na*(nc+5), 1, 1)
        self.conv_block7 = Conv(int(512*ratio), int(512*ratio), 3, 2) # connect bottleneck_csp6
        self.concat = Concat() # concat pre and bottleneck_csp3
        self.bottleneck_csp7 = BottleneckCSP(int(1024*ratio), int(1024*ratio), 1, False)

        self.conv2 = nn.Conv2d(int(1024*ratio), na*(nc+5), 1, 1)
    
    def forward(self, x):
        x1, x2, x_spp = x
        x_bcsp3 = self.bottleneck_csp3(x_spp)

        x_cb4 = self.conv_block4(x_bcsp3)
        x = self.upsample0(x_cb4)
        x = self.concat((x, x2))
        x_bcsp4 = self.bottleneck_csp4(x)

        x_cb5 = self.conv_block5(x_bcsp4)
        x = self.upsample1(x_cb5)
        x = self.concat((x, x1))
        x_bcsp5 = self.bottleneck_csp5(x)

        x_c6 = self.conv0(x_bcsp5)
        x = self.conv_block6(x_bcsp5)
        x = self.concat((x, x_cb5))
        x_bcsp6 = self.bottleneck_csp6(x)

        x_c7 = self.conv1(x_bcsp6)
        x = self.conv_block7(x_bcsp6)
        x = self.concat((x, x_cb4))
        x = self.bottleneck_csp7(x)

        x = self.conv2(x)
        return x, x_c7, x_c6

class V5Ghost(torch.nn.Module):
       
    def __init__(self, cfg):
        super(V5Ghost, self).__init__()

        with open(cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader) 
        self.nc = self.md['nc']
        self.anchors = self.md['anchors']
        self.na = len(self.anchors[0]) // 2  # number of anchors
        print('model num classes is: {}'.format(self.nc))
        print('model anchors is: {}'.format(self.anchors))

        focus_out_c = 32
        self.focus = Focus(3, focus_out_c, k=3)
        channel_ratio = 3.2
        # channel_ratio = 6.4
        cfgs = [
            # # k, t, c, SE, s
            # # stage1
            # [[3,  16,  16, 0, 1]],
            # # stage2
            # [[3,  48,  24, 0, 2]],
            # [[3,  72,  24, 0, 1]],
            # # stage3
            # [[5,  72,  40, 0.25, 2]],
            # [[5, 120,  40*channel_ratio, 0.25, 1]], #4
            # # stage4
            # [[3, 240,  80*channel_ratio, 0, 2]], #5
            # [[3, 200,  80, 0, 1],
            # [3, 184,  80, 0, 1],
            # [3, 184,  80, 0, 1], 
            # [3, 480, 112, 0.25, 1],
            # [3, 672, 112, 0.25, 1]
            # ],
            # # stage5
            # [[5, 672, 160, 0.25, 2]],
            # [[5, 960, 160, 0, 1],
            # [5, 960, 160, 0.25, 1],
            # [5, 960, 160, 0, 1],
            # [5, 960, 160*channel_ratio, 0.25, 1]
            # ] #8

            # k, t, c, SE, s
            # stage1
            [[3,  16,  16, 0, 1]],
            # stage2
            # [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            # stage3
            [[5,  72,  40, 0.25, 2]],
            [[5, 120,  40*channel_ratio, 0.25, 1]], #4
            # stage4
            [[3, 240,  80*channel_ratio, 0, 2]], #5
            [[3, 200,  80, 0, 1],
            [3, 184,  80, 0, 1],
            [3, 184,  80, 0, 1], 
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1]],
            # stage5
            [[5, 672, 160*channel_ratio, 0.25, 2]],
            [[5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1]] #8
        ]
        self.backbone = GhostNet(cfgs, in_ch=focus_out_c, extract_layers=[3, 4, 6])

        ratio = 0.5
        self.spp = SPP(int(1024*ratio), int(1024*ratio), k=(5, 9, 13))
        self.head = HeadV5(self.nc, self.na, ratio)

        self.detect = Detect(self.nc, self.anchors)
        self.detect.stride = torch.tensor([128 / x.shape[-2] for x in self.forward(torch.zeros(1, 3, 128, 128))])  # forward
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        print('detect stride: {}'.format(self.detect.stride))
        print('detect anchors: {}'.format(self.detect.anchors))
        self.model = [self.detect]
        print('V5Ghost model constructed.')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1]) 
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)
    
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        x = self.focus(x)
        # print('focus out shape: ', x.shape)
        x_bcsp1, x_bcsp2, x3 = self.backbone(x)
        x_spp =self.spp(x3)
        # print('output shape after backbone: ')
        # print(x_bcsp1.shape)
        # print(x_bcsp2.shape)
        # print(x_spp.shape)

        ## backbone end, yolov4 head start
        x, x_c7, x_c6 = self.head([x_bcsp1, x_bcsp2, x_spp])
        # print('3 layer shape: {}, {}, {}'.format(x.shape, x_c7.shape, x_c6.shape))
        x = self.detect([x, x_c7, x_c6]) 
        # print('output detect shape: ')
        # for a in x:
            # print(a.shape)
        # if profile:
            # print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect  # Detect() module
        for f, s in zip(m.f, m.stride):  #  from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)


class V5Small(torch.nn.Module):
       
    def __init__(self, cfg):
        super(V5Small, self).__init__()

        with open(cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader) 
        self.nc = self.md['nc']
        self.anchors = self.md['anchors']
        self.na = len(self.anchors[0]) // 2  # number of anchors
        print('model num classes is: {}'.format(self.nc))
        print('model anchors is: {}'.format(self.anchors))

        self.focus = Focus(3, 32, k=3)
        self.conv_block0 = Conv(32, 64, k=3, s=2)
        self.bottleneck_csp0 = BottleneckCSP(64, 64, 1)

        self.conv_block1 = Conv(64, 128, k=3, s=2)
        self.bottleneck_csp1 = BottleneckCSP(128, 128, 3)

        self.conv_block2 = Conv(128, 256, 3, 2)
        self.bottleneck_csp2 = BottleneckCSP(256, 256, 3)

        self.conv_block3 = Conv(256, 512, 3, 2)
        self.spp0 = SPP(512, 512, k=(5, 9, 13))
        self.bottleneck_csp3 = BottleneckCSP(512, 512, 1, False)

        self.conv_block4 = Conv(512, 256, 1, 1)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat = Concat() # conact pre and conv_block2
        self.bottleneck_csp4 = BottleneckCSP(512, 256, 1, False)
                                                                                                                                       
        self.conv_block5 = Conv(256, 128)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.concat1 = Concat() # concat pre and conv_block1
        self.bottleneck_csp5 = BottleneckCSP(256, 128, 1, False)

        self.conv0 = nn.Conv2d(128, self.na*(self.nc+5), 1, 1)
        self.conv_block6 = Conv(128, 128, 3, 2) # connect bottleneck_csp5
        self.concat = Concat() # concat pre and bottleneck_csp4
        self.bottleneck_csp6 = BottleneckCSP(256, 256, 1, False)

        self.conv1 = nn.Conv2d(256, self.na*(self.nc+5), 1, 1)
        self.conv_block7 = Conv(256, 256, 3, 2) # connect bottleneck_csp6
        self.concat = Concat() # concat pre and bottleneck_csp3
        self.bottleneck_csp7 = BottleneckCSP(512, 512, 1, False)

        self.conv2 = nn.Conv2d(512, self.na*(self.nc+5), 1, 1)
        
        self.detect = Detect(self.nc, self.anchors)
        self.detect.stride = torch.tensor([128 / x.shape[-2] for x in self.forward(torch.zeros(1, 3, 128, 128))])  # forward
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        print('detect stride: {}'.format(self.detect.stride))
        print('detect anchors: {}'.format(self.detect.anchors))
        # expose -1 as detect layer
        self.model = [self.detect]


    def forward(self, x, augment=False, profile=True):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1]) 
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)

    def _build_backbone(self, x):
        x = self.focus(x)
        x = self.conv_block0(x)
        x = self.bottleneck_csp0(x)

        x = self.conv_block1(x)
        x_bcsp1 = self.bottleneck_csp1(x)

        x = self.conv_block2(x_bcsp1)
        x_bcsp2 = self.bottleneck_csp2(x)

        x = self.conv_block3(x_bcsp2)
        x_spp = self.spp0(x)
        return x_bcsp1, x_bcsp2, x_spp

    def _build_backbone_ghost(self, x):
        pass
    
    def forward_once(self, x, profile=True):
        y, dt = [], []  # outputs
        x_bcsp1, x_bcsp2, x_spp = self._build_backbone(x)
        print('output shape after backbone: ')
        print(x_bcsp1.shape)
        print(x_bcsp2.shape)
        print(x_spp.shape)

        ## backbone end, yolov4 head start
        x_bcsp3 = self.bottleneck_csp3(x_spp)

        x_cb4 = self.conv_block4(x_bcsp3)
        x = self.upsample0(x_cb4)
        x = self.concat((x, x_bcsp2))
        x_bcsp4 = self.bottleneck_csp4(x)

        x_cb5 = self.conv_block5(x_bcsp4)
        x = self.upsample1(x_cb5)
        x = self.concat((x, x_bcsp1))
        x_bcsp5 = self.bottleneck_csp5(x)

        x_c6 = self.conv0(x_bcsp5)
        x = self.conv_block6(x_bcsp5)
        x = self.concat((x, x_cb5))
        x_bcsp6 = self.bottleneck_csp6(x)

        x_c7 = self.conv1(x_bcsp6)
        x = self.conv_block7(x_bcsp6)
        x = self.concat((x, x_cb4))
        x = self.bottleneck_csp7(x)

        x = self.conv2(x)
        print('3 layer shape: {}, {}, {}'.format(x.shape, x_c7.shape, x_c6.shape))
        x = self.detect([x, x_c7, x_c6]) 
        print('output detect shape: ')
        for a in x:
            print(a.shape)
        """
        torch.Size([1, 3, 4, 4, 85])
        torch.Size([1, 3, 8, 8, 85])
        torch.Size([1, 3, 16, 16, 85])
        """
        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detect  # Detect() module
        for f, s in zip(m.f, m.stride):  #  from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)


class Model(nn.Module):
    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg  # model dict
        else:  # is *.yaml
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc:
            self.md['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(self.md, ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])
        print(self.model)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        m.stride = torch.tensor([128 / x.shape[-2] for x in self.forward(torch.zeros(1, ch, 128, 128))])  # forward
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride

        # Init weights, biases
        torch_utils.initialize_weights(self)
        self._initialize_biases()  # only run once
        torch_utils.model_info(self)
        print('')

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1]),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                import thop
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.f, m.stride):  #  from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for f in sorted([x % m.i for x in m.f]):  #  from
            b = self.model[f].bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%g Conv2d.bias:' + '%10.3g' * 6) % (f, *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers...')
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        torch_utils.model_info(self)


def parse_model(md, ch):  # model_dict, input_channels(3)
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    na = (len(anchors[0]) // 2)  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        print('ori n: {}'.format(n))
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        print('layer {}, n: {}, depth_multiple: {}, '.format(i, n, gd))
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            print('ori c: {}'.format(c2))
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            print('after c: {} w multiple: {}'.format(c2, gw))

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)
    # print([y[0].shape] + [x.shape for x in y[1]])

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, f.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
