'''Config file for CycleGAN training.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-07-20 21:02:44.
'''
import os

config={
        'train_label_file': './data/train.csv',
        # 'test_label_file': './data/test.csv',
        'image_folder': './data/faces-spring-2020/faces-spring-2020',
        'output_folder': './outputs/exp1',
        }

config.update({
        'results': os.path.join(config['output_folder'], 'results'),

        'image_width': 224,
        'image_height': 224,
        'img_channels': 3,

        'batch_size': 1,
        'lambda_identity': 2,   # weight for identity loss
        'lambda_cycle': 10,     # weight for cycle consistency loss
        'lambda_gen': 1,        # weight generator loss
        'lambda_dis': 1,        # weight for discriminator loss
        'n_features': 64,       # number of feature channels of first conv layer
        'n_resblocks': 5,       # number of residual blocks after encoder
        'n_strides': 3,         # number of 2x down/up samplings for encoder/decoder
        'p_cycle': 0.9,         # probability of computing a cycle consistency loss in an iteration
        'p_identity': 0.5,      # probability of computing an identity loss in an iteration
        'progressive': False,   # progressively train each of decoder conv layer or not
        'progressive_n_stage': 10,  # if <progressive>, number of iterations each decoder layer is trained before releasing the training of the next layer
        'n_dis_per_gen': 2,     # number of discriminator training per generator training iteration

        'num_epochs': 5,        # number of epochs to train
        'n_epoch_save': 1,      # save checkpoint every this epochs
        'n_epoch_eval': 1,      # plot some evaluatation plots every this epochs
        'n_eval_samples': 5,    # number of evaluation plots to create when evaluating
        'n_iters_eval': 100,    # number of iterations to plot some evaluation plots

        'lr0': 1.0e-4,          # starting learning rate, for both generator and discriminator
        'peak_lr': 2.5e-4,      # peak learning rate
        'weight_decay': 1e-4,   # weight decay for optimizers
        'warmup_iter': 500,     # warm up iteration number, when learning rate goes from <lr0> to <peak_lr>
        'dis_beta1': 0.5,       # beta1 Adam parameter for discriminator
        'gen_beta1': 0.5,       # beta1 Adam parameter for generator
        'dropout': 0.1,         # dropout

        'device': None,         # If None, use 'gpu' if available
        'resume': True,        # resume training
        'resume_from_epoch': 4, # if <resume>, from which checkpoint
        'sleep': 0.2,           # slow down your fan
})
