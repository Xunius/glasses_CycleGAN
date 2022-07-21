'''CycleGAN model

Modified from:
   https://www.kaggle.com/code/yashikajain/eye-glass-removal

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-11 15:03:45.
'''

from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def lr_scheduler(iteration, warmup_iter, initial_lr, peak_lr, power=1):

    if iteration == 0:
        iteration += 1

    lr = min(1 / iteration**power, iteration / warmup_iter**(power + 1)) *\
            warmup_iter**power * (peak_lr - initial_lr) + initial_lr
    lr = max(lr, 1e-7)

    return lr / initial_lr


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, norm, use_act):
        nn.Module.__init__(self)

        layers = []
        layers.append(
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, padding_mode='reflect', bias=True)
                )

        if norm == 'bn':
            layers.append(nn.BatchNorm2d(c_out))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(c_out))

        if use_act:
            #layers.append(nn.LeakyReLU(0.2, True))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, norm, use_act):
        nn.Module.__init__(self)

        self.stride = stride
        self.c_out = c_out

        #self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')

        layers = [
                nn.Upsample(scale_factor=stride, mode='nearest'),
                ConvBlock(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding, norm=norm, use_act=use_act),
                ]

        #if use_act:
            #layers.append(nn.LeakyReLU(0.1, True))

        self.layers = nn.Sequential(*layers)

    #def forward(self, x, skip):
    def forward(self, x):
        #x = self.upsample(x)
        #x = torch.cat([x, skip], dim=1)
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, c_in, norm):
        nn.Module.__init__(self)

        layers = [
                ConvBlock(c_in, c_in, kernel_size=3, stride=1, padding=1, norm=norm, use_act=True),
                #nn.LeakyReLU(0.1, True),
                ConvBlock(c_in, c_in, kernel_size=3, stride=1, padding=1, norm=norm, use_act=False),
                ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x


class ResDownBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, norm):
        nn.Module.__init__(self)

        self.c_out = c_out
        self.stride = stride

        layers = [
                ConvBlock(c_in, c_out, kernel_size=1, stride=1, padding=0, norm=norm, use_act=True),
                #nn.LeakyReLU(0.1, True),
                ConvBlock(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, use_act=False)
                ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = interp_channel(x, self.c_out)
        skip = F.interpolate(skip, scale_factor=1/self.stride, mode='nearest')

        return self.layers(x) + skip

def interp_channel(skip, target_c):
    bb, cc, hh, ww = skip.shape
    skip = torch.permute(skip, [0, 2, 3, 1])  # B, H, W, C
    skip = F.interpolate(skip, size=[ww, target_c], mode='nearest')
    skip = torch.permute(skip, [0, 3, 1, 2])
    return skip



class Generator(nn.Module):
    def __init__(self, c_in, n_features=64, n_resblocks=9, n_strides=3, dropout=0.5):
        nn.Module.__init__(self)

        self.n_features = n_features
        self.n_resblocks = n_resblocks

        self.head = nn.Sequential(
            ConvBlock(c_in, n_features, kernel_size=7, stride=1, padding=3, norm='instance', use_act=True),
            #nn.LeakyReLU(0.1, True)
            )

        #------------------Create encoder------------------
        layers = []
        cii = n_features
        cii2 = n_features * 2
        for ii in range(n_strides):
            #layers.append(ConvBlock(cii, cii2, kernel_size=3, stride=2, padding=1, norm='instance'))
            layers.append(
                    ResDownBlock(cii, cii2, kernel_size=3, stride=2, padding=1, norm='instance'))

            #layers.append(nn.LeakyReLU(0.1, True))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout2d(dropout))
            cii = cii2
            cii2 = cii*2

        self.encoder = nn.Sequential(*layers)

        #----------------Create res-blocks----------------
        layers = []
        c0 = n_features * 2**n_strides
        for ii in range(n_resblocks):
            layers.append(ResBlock(c0, norm='instance'))
            #layers.append(nn.LeakyReLU(0.1, True))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout2d(dropout))

        self.resblocks = nn.Sequential(*layers)

        #------------------Create decoder------------------
        layers = []
        for ii in range(n_strides):
            cii = int(c0 / 2**ii)
            cii2 = max(int(cii / 2), c_in)
            layers.append(DeconvBlock(cii, cii2, kernel_size=3, stride=2, padding=1, norm='instance', use_act=True))
            #layers.append(nn.LeakyReLU(0.1, True))
            #layers.append(nn.Dropout2d(dropout))

        self.decoder = nn.Sequential(*layers)

        self.tail = nn.Sequential(
                ConvBlock(cii2, c_in, kernel_size=7, stride=1, padding=3, norm=False, use_act=False),
                nn.Tanh())

    def forward(self, x):

        '''
        tmp = [x]
        for name, layer in self.encoder.named_children():
            y = layer(x)
            #print('name', name, 'shape:', x.shape, '->', y.shape)
            if name.startswith('conv'):
                tmp.append(y)
            x = y

        x = self.resblocks(x)

        for ii, (name, layer) in enumerate(self.decoder.named_children()):
            y = layer(x, tmp[-(ii+2)])
            #print('ii', ii, 'name', name, 'shape:', x.shape, '->', y.shape)
            x = y

        '''
        x = self.head(x)
        x = self.encoder(x)
        x = self.resblocks(x)
        x = self.decoder(x)
        x = self.tail(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, c_in, dropout):
        nn.Module.__init__(self)

        self.layers = nn.Sequential(
                nn.Conv2d(c_in, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2, True),
                ConvBlock(64, 128, kernel_size=4, stride=2, padding=1, norm='instance', use_act=True),
                #nn.LeakyReLU(0.2, True),
                nn.Dropout2d(dropout),
                ConvBlock(128, 256, kernel_size=4, stride=2, padding=1, norm='instance', use_act=True),
                #nn.LeakyReLU(0.2, True),
                nn.Dropout2d(dropout),
                ConvBlock(256, 512, kernel_size=4, stride=2, padding=1, norm='instance', use_act=True),
                #nn.LeakyReLU(0.2, True),
                nn.Dropout2d(dropout),
                ConvBlock(512, 1, kernel_size=4, stride=1, padding=1, norm=False, use_act=False),
                #nn.AdaptiveAvgPool2d(1),
                #nn.Flatten(),
                )

    def forward(self, x):

        x = self.layers(x)
        return torch.sigmoid(x)


class CycleGAN(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.gen_X2Y = Generator(config['img_channels'], config['n_features'],
                config['n_resblocks'], config['n_strides'], config['dropout'])
        self.gen_Y2X = Generator(config['img_channels'], config['n_features'],
                config['n_resblocks'], config['n_strides'], config['dropout'])

        self.dis_X = Discriminator(config['img_channels'], config['dropout'])
        self.dis_Y = Discriminator(config['img_channels'], config['dropout'])

        self.dis_loss_fn = nn.MSELoss()
        self.gen_loss_fn = nn.MSELoss()
        self.cyc_loss_fn = nn.L1Loss()
        self.idt_loss_fn = nn.L1Loss()

        self.dis_opt = torch.optim.Adam(list(self.dis_X.parameters()) +\
                list(self.dis_Y.parameters()),
                lr=config['lr0'],
                betas=(config['dis_beta1'], 0.99),
                #weight_decay=config['weight_decay']
                )

        self.gen_opt = torch.optim.Adam(list(self.gen_X2Y.parameters()) +\
                list(self.gen_Y2X.parameters()),
                lr=config['lr0'],
                betas=(config['gen_beta1'], 0.99),
                #weight_decay=config['weight_decay']
                )

        self.dis_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_opt,
                lr_lambda=lambda it: lr_scheduler(it, config['warmup_iter'],
                    config['lr0'], config['peak_lr']))
        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_opt,
                lr_lambda=lambda it: lr_scheduler(it, config['warmup_iter'],
                    config['lr0'], config['peak_lr']))

    def save_checkpoint(self, save_folder, epoch, total_iters):

        os.makedirs(save_folder, exist_ok=True)
        ckpt_path = os.path.join(save_folder, 'ckpt_%d.pt' %epoch)

        torch.save({
            'dis_X_state_dict': self.dis_X.state_dict(),
            'dis_Y_state_dict': self.dis_Y.state_dict(),
            'gen_X2Y_state_dict': self.gen_X2Y.state_dict(),
            'gen_Y2X_state_dict': self.gen_Y2X.state_dict(),
            'dis_opt_state_dict': self.dis_opt.state_dict(),
            'gen_opt_state_dict': self.gen_opt.state_dict(),
            'dis_scheduler_state_dict': self.dis_scheduler.state_dict(),
            'gen_scheduler_state_dict': self.gen_scheduler.state_dict(),
            'dis_lr': self.dis_scheduler.get_last_lr()[0],
            'gen_lr': self.gen_scheduler.get_last_lr()[0],
            'total_iters': total_iters,
            }, ckpt_path)

        print('Saved check-point to:', ckpt_path)
        return

    def load_checkpoint(self, ckpt_folder, epoch=None):

        ckpt_files = os.listdir(ckpt_folder)
        ckpt_files = [fii for fii in ckpt_files if fii.endswith('.pt')]
        if len(ckpt_files) == 0:
            raise Exception("No ckpt files")

        if epoch is None:
            ckpt_files.sort()
            ckpt_path = ckpt_files[-1]
        else:
            ckpt_path = 'ckpt_%d.pt' %epoch

        ckpt_path = os.path.join(ckpt_folder, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.dis_X.load_state_dict(ckpt['dis_X_state_dict'])
        self.dis_Y.load_state_dict(ckpt['dis_Y_state_dict'])
        self.gen_X2Y.load_state_dict(ckpt['gen_X2Y_state_dict'])
        self.gen_Y2X.load_state_dict(ckpt['gen_Y2X_state_dict'])
        self.dis_opt.load_state_dict(ckpt['dis_opt_state_dict'])
        self.gen_opt.load_state_dict(ckpt['gen_opt_state_dict'])
        self.dis_scheduler.load_state_dict(ckpt['dis_scheduler_state_dict'])
        self.gen_scheduler.load_state_dict(ckpt['gen_scheduler_state_dict'])

        total_iters = ckpt['total_iters']

        print('Loaded check-point from:', ckpt_path, 'total_iters:', total_iters)
        del ckpt

        return total_iters

    def to_device(self, device):
        self.dis_X.to(device)
        self.dis_Y.to(device)
        self.gen_X2Y.to(device)
        self.gen_Y2X.to(device)
        return


    def train(self, it, img_X, img_Y, config):

        self.dis_X.train()
        self.dis_Y.train()
        self.gen_X2Y.train()
        self.gen_Y2X.train()
        device = img_X.device

        #-----------------Train discriminators-----------------
        for _ in range(config['n_dis_per_gen']):
            self.dis_opt.zero_grad()
            '''
            for param in self.dis_X.parameters():
                param.requires_grad = True
            for param in self.dis_Y.parameters():
                param.requires_grad = True
            '''

            dis_real_X = self.dis_X(img_X)
            label_real_X = torch.randn_like(dis_real_X).to(device) / 10 + 1
            loss_dis_real = self.dis_loss_fn(dis_real_X, label_real_X)
            del label_real_X

            dis_real_Y = self.dis_Y(img_Y)
            label_real_Y = torch.randn_like(dis_real_Y).to(device) / 10 + 1
            loss_dis_real += self.dis_loss_fn(dis_real_Y, label_real_Y)
            del label_real_Y

            fake_X = self.gen_Y2X(img_Y)
            dis_fake_X = self.dis_X(fake_X.detach())
            label_fake_X = torch.zeros_like(dis_fake_X)
            loss_dis_fake = self.dis_loss_fn(dis_fake_X, label_fake_X)
            del label_fake_X

            fake_Y = self.gen_X2Y(img_X)
            dis_fake_Y = self.dis_Y(fake_Y.detach())
            label_fake_Y = torch.zeros_like(dis_fake_Y)
            loss_dis_fake += self.dis_loss_fn(dis_fake_Y, label_fake_Y)
            del label_fake_Y

            d_loss = (loss_dis_real + loss_dis_fake) / 2

            lambda_dis = config['lambda_dis']
            d_loss *= lambda_dis / (lambda_dis + config['lambda_gen'])

            d_loss.backward()
            self.dis_opt.step()
            self.dis_scheduler.step()

            del dis_real_X, dis_real_Y, dis_fake_X, dis_fake_Y

        #-----------------Train generators-----------------
        if config['progressive']:
            stage = (it // config['progressive_n_stage']) % config['n_strides']

            self.toggle_decoder_update(self.gen_X2Y.decoder, it, stage)
            self.toggle_decoder_update(self.gen_Y2X.decoder, it, stage)

        g_loss = self.update_gen(img_X, img_Y, fake_X, fake_Y, config)

        g_loss *= config['lambda_gen'] / (lambda_dis + config['lambda_gen'])

        g_loss.backward()
        self.gen_opt.step()
        self.gen_scheduler.step()

        return d_loss, g_loss

    def update_gen(self, img_X, img_Y, fake_X, fake_Y, config):

        self.gen_opt.zero_grad()
        '''
        for param in self.dis_X.parameters():
            param.requires_grad = False
        for param in self.dis_Y.parameters():
            param.requires_grad = False
        '''

        dis_fake_X = self.dis_X(fake_X)
        label_fake_X = torch.ones_like(dis_fake_X)
        g_loss = self.gen_loss_fn(dis_fake_X, label_fake_X)
        del label_fake_X

        dis_fake_Y = self.dis_Y(fake_Y)
        label_fake_Y = torch.ones_like(dis_fake_Y)
        g_loss += self.gen_loss_fn(dis_fake_Y, label_fake_Y)
        del label_fake_Y

        # cycle loss
        if np.random.rand() < config['p_cycle']:
            fake_XYX = self.gen_Y2X(fake_Y)
            fake_YXY = self.gen_X2Y(fake_X)
            cyc_loss = self.cyc_loss_fn(fake_XYX, img_X)
            cyc_loss += self.cyc_loss_fn(fake_YXY, img_Y)
            g_loss = g_loss + cyc_loss * config['lambda_cycle']
            del fake_XYX, fake_YXY

        # identity loss
        if config['lambda_identity'] > 0 and np.random.rand() < config['p_identity']:
            fake_XX = self.gen_Y2X(img_X)
            idt_loss_X = self.idt_loss_fn(fake_XX, img_X)
            fake_YY = self.gen_X2Y(img_Y)
            idt_loss_Y = self.idt_loss_fn(fake_YY, img_Y)
            g_loss = g_loss + (idt_loss_X + idt_loss_Y) * config['lambda_identity']
            del fake_XX, fake_YY

        return g_loss

    def toggle_decoder_update(self, module, it, stage):

        for sii, mii in enumerate(module[:-1]):
            if sii <= stage:
                #print('iter:', it, 'stage', stage, 'sii', sii, True)
                for param in mii.parameters():
                    param.requires_grad = True
            else:
                #print('iter:', it, 'stage', stage, 'sii', sii, False)
                for param in mii.parameters():
                    param.requires_grad = False

#-------------Main---------------------------------
if __name__=='__main__':

    x = torch.randn([2, 3, 256, 256])
    mod = Generator(3, 64, 8, 2)
    dis = Discriminator(3, 0.1)
    y = mod(x)

