'''Eye glasses add/removal training script.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-20 21:49:16.
'''

import os
import shutil
import numpy as np
import json
import time
import random
import torch
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter

from config import config
import loader
import model2 as model_module

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def generate_eval(model, val_data, num, epoch, results_folder, device):

    for idx in np.random.choice(np.arange(len(val_data)), num):
        img_glasses, img_no_glasses = val_data[idx]

        # get face id
        glasses_file = val_data.files_X[idx % val_data.len_X]
        glasses_id = os.path.basename(glasses_file).split('-')[1].split('.')[0]

        # gen fake
        fake_Y = save_some_examples(model.gen_X2Y, img_glasses, device)

        # get face id
        no_glasses_file = val_data.files_Y[idx % val_data.len_Y]
        no_glasses_id = os.path.basename(no_glasses_file).split('-')[1].split('.')[0]

        # gen fake
        fake_X = save_some_examples(model.gen_Y2X, img_no_glasses, device)

        # save image
        img_list = [img_glasses, fake_Y, img_no_glasses, fake_X]
        img_list = [xx.squeeze().to('cpu') * 0.5 + 0.5 for xx in img_list]
        img_list = make_grid(img_list, nrow=2)

        save_image(img_list,
                os.path.join(results_folder,
                    f"{epoch}_glasses_id_{glasses_id}_noglasses_id_{no_glasses_id}.png"))

    return


def save_some_examples(gen, real_img, device):
    real_img = real_img.to(device).unsqueeze(0)
    gen.eval()
    with torch.no_grad():
        fake_img = gen(real_img).squeeze()

    return fake_img


def train(epoch, total_iters, train_loader, val_data, model, config, device, tb_writer=None):

    model.to_device(device)
    print('Epoch:', epoch, 'Device:', device)

    d_loss_total = 0
    g_loss_total = 0
    n_iter = 0

    for ii, (img_X, img_Y) in enumerate(train_loader):
        if config['sleep'] > 0:
            time.sleep(config['sleep'])

        total_iters += 1
        n_iter += 1
        img_X = img_X.to(device)
        img_Y = img_Y.to(device)

        d_loss, g_loss = model.train(total_iters, img_X, img_Y, config)

        #------------------Print progress------------------
        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

        dis_lr = model.dis_scheduler.get_last_lr()[0]
        gen_lr = model.gen_scheduler.get_last_lr()[0]

        if tb_writer is not None:
            tb_writer.add_scalar('iters/g_loss', g_loss.item(), total_iters)
            tb_writer.add_scalar('iters/d_loss', d_loss.item(), total_iters)
            tb_writer.add_scalar('iters/g_lr', gen_lr, total_iters)
            tb_writer.add_scalar('iters/d_lr', dis_lr, total_iters)

        if ii % config['n_iters_eval'] == 0:
            print('epoch: %d, tot_iter: %d, dloss: %.3f, gloss: %.3f, d_lr: %.4f, g_lr: %.4f'\
                    %(epoch, total_iters, d_loss.item(), g_loss.item(), dis_lr, gen_lr))

            generate_eval(model, val_data, config['n_eval_samples'], epoch,
                    config['results'], device)

    return total_iters, d_loss_total/n_iter, g_loss_total/n_iter

if __name__ == '__main__':


    set_seed(42)

    # output directories
    os.makedirs(config['output_folder'], exist_ok=True)
    os.makedirs(config['results'], exist_ok=True)
    os.makedirs(os.path.join(config['output_folder'], 'logs'), exist_ok=True)
    log_dir = os.path.join(config['output_folder'], 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # configs
    NUM_EPOCHS = config['num_epochs']
    if config['device'] is not None:
        DEVICE = config['device']
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tb_writer = SummaryWriter(log_dir=log_dir)
    config_path = os.path.join(config['output_folder'], 'exp_config.txt')
    with open(config_path, 'w') as fout:
        json.dump(config, fout, indent=4)

    shutil.copy2(model_module.__file__, os.path.join(config['output_folder'], 'model.py'))
    print('Copied model def file into', config['output_folder'])
    print("Device :", DEVICE)

    # create model
    glass_model = model_module.CycleGAN(config)
    glass_model.to_device(DEVICE)

    # load checkpoint
    if config['resume']:
        start_epoch = config['resume_from_epoch']
        total_iters = glass_model.load_checkpoint(config['output_folder'], epoch=start_epoch) + 1
        start_epoch += 1
    else:
        start_epoch = 0
        total_iters = 0

    # dataset
    train_loader, val_loader, train_data, val_data = loader.get_dataloader(config)

    # start training
    for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):

        total_iters, d_loss, g_loss = train(epoch, total_iters,
                train_loader, val_data,
                glass_model, config, DEVICE, tb_writer)

        tb_writer.add_scalar('epoch/g_loss', g_loss, epoch)
        tb_writer.add_scalar('epoch/d_loss', d_loss, epoch)

        if epoch % config['n_epoch_eval'] == 0:
            glass_model.save_checkpoint(config['output_folder'], epoch, total_iters)

            generate_eval(glass_model, val_data, config['n_eval_samples'], epoch,
                    config['results'], DEVICE)

    glass_model.save_checkpoint(config['output_folder'], epoch, total_iters)
