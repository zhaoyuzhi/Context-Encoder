import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    if opt.cudnn_benchmark == True:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(generator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate_g(optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_learning_rate_d(optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = opt.lr_d * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module, 'ContextureEncoder_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net, 'ContextureEncoder_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size))
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img = img.cuda()
            mask = mask.cuda()

            ### Train Discriminator
            optimizer_d.zero_grad()

            # Generator output
            masked_img = img * (1 - mask)
            fake = generator(masked_img)

            # Fake samples
            fake_scalar = discriminator(fake.detach())
            # True samples
            true_scalar = discriminator(img)
            
            # Overall Loss and optimize
            loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
            loss_D.backward()

            ### Train Generator
            optimizer_g.zero_grad()

            # forward propagation
            fusion_fake = img * (1 - mask) + fake * mask                    # in range [-1, 1]

            # Mask L1 Loss
            MaskL1Loss = L1Loss(fusion_fake, img)
            
            # GAN Loss
            fake_scalar = discriminator(fusion_fake)
            GAN_Loss = - torch.mean(fake_scalar)

            # Compute losses
            loss = MaskL1Loss + opt.gan_param * GAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), loss_D.item(), GAN_Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate_g(optimizer_g, (epoch + 1), opt)
        adjust_learning_rate_d(optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model(generator, (epoch + 1), opt)
