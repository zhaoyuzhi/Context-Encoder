import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "", help = 'the testing folder')
    parser.add_argument('--mask_type', type = str, default = 'bbox', help = 'mask type')
    parser.add_argument('--imgsize', type = int, default = 128, help = 'size of image')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 64, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    # Other parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = 'TestNet_epoch10_batchsize8.pth', help = 'test model name')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = torch.load(opt.load_name)

    for batch_idx, (img, mask) in enumerate(dataloader):

        # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        masked_img = img * (1 - mask)
        fake = model(masked_img)

        # forward propagation
        fusion_fake = img * (1 - mask) + fake * mask                    # in range [-1, 1]

        # show
        img = img.cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)
        fusion_fake = fusion_fake.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        fusion_fake = (fusion_fake + 1) * 128
        fusion_fake = fusion_fake.astype(np.uint8)
        show_img = np.concatenate((img, fusion_fake), axis = 1)
        cv2.imshow('comparison.jpg', show_img)
        cv2.waitKey(1)
