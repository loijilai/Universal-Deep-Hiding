# encoding: utf-8

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.data import DataLoader

# import utils.transformed as transforms
from torchvision import transforms
# from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=128,
                    help='The image size')
parser.add_argument('--testPics', default='./testing',
                    help='folder to output test images')
parser.add_argument('--checkpoint', default='./training/08-25_H23-08-45_128_1_batch_l2_0.75_1colorIn1color_main_ddh/checkPoints/best_checkpoint.pth.tar', help='checkpoint folder')

parser.add_argument('--bs_secret', type=int, default=32, help='batch size for ')


def main():
    ############### Define global parameters ###############
    global opt, writer, val_loader, DATA_DIR

    opt = parser.parse_args()
    if not torch.cuda.is_available():
        print("WARNING: CUDA isn't available. Running on CPU")

    DATA_DIR = '/home/lai/Research/coco/images' 


    ############  Create the dirs to save the result ############
    try:
        if (not os.path.exists(opt.testPics)):
            os.makedirs(opt.testPics)
    except OSError:
        print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX") # ignore


    ##################  Datasets  ##################
    testdir = os.path.join(DATA_DIR, 'val2017')

    transforms_color = transforms.Compose([ 
                transforms.Resize([opt.imageSize, opt.imageSize]), 
                transforms.ToTensor(),
            ])  

    transforms_cover = transforms_color
    transforms_secret = transforms_color

    test_dataset_cover = ImageFolder(
        testdir,  
        transforms_cover)
    test_dataset_secret = ImageFolder(
        testdir,
        transforms_secret)
    assert test_dataset_cover; assert test_dataset_secret

    ##################  Hiding and Reveal  ##################
    assert opt.imageSize % 32 == 0
    norm_layer = nn.BatchNorm2d

    Hnet = UnetGenerator(input_nc=3+3, # cover + secret
                         output_nc=3, # container
                         num_downs=5, 
                         norm_layer=norm_layer, 
                         output_function=nn.Sigmoid)

    Rnet = RevealNet(input_nc=3, 
                     output_nc=3, 
                     nhf=64, 
                     norm_layer=norm_layer, 
                     output_function=nn.Sigmoid)
    
    ##### Always set to multiple GPU mode  #####
    Hnet = torch.nn.DataParallel(Hnet).cuda()
    Rnet = torch.nn.DataParallel(Rnet).cuda()

    ##### Loading checkpoints #####
    checkpoint = torch.load(opt.checkpoint)
    Hnet.load_state_dict(checkpoint['H_state_dict'])
    Rnet.load_state_dict(checkpoint['R_state_dict'])

    # Test the trained network
    test_loader_secret = DataLoader(test_dataset_secret, batch_size=opt.bs_secret,
                                shuffle=True, num_workers=8)
    test_loader_cover = DataLoader(test_dataset_cover, batch_size=opt.bs_secret,
                                shuffle=True, num_workers=8)
    test_loader = zip(test_loader_secret, test_loader_cover)
    validation(test_loader, Hnet=Hnet, Rnet=Rnet)

def validation(val_loader, Hnet, Rnet):
    print(
        "#################################################### validation begin ########################################################")
    Hnet.eval()
    Rnet.eval()

    for i, ((secret_img, _), (cover_img, _)) in enumerate(val_loader, 0):

        batch_size_secret, channel_secret, _, _ = secret_img.size()
        batch_size_cover, channel_cover, _, _ = cover_img.size()

        # Put tensors in GPU 
        cover_img = cover_img.cuda()
        secret_img = secret_img.cuda()
        # used when multiple images are being hidden simultaneously
        secret_imgv = secret_img.view(batch_size_secret // 1, channel_secret * 1, opt.imageSize, opt.imageSize) # 1: opt.num_secret

        cover_img = cover_img.view(batch_size_cover // 1, channel_cover * 1, opt.imageSize, opt.imageSize)
        cover_imgv = cover_img

        H_input = torch.cat((cover_imgv, secret_imgv), dim=1)
        itm_secret_img = Hnet(H_input)

        container_img = itm_secret_img
        rev_secret_img = Rnet(container_img) 
        save_result_pic(batch_size_secret, cover_img, container_img, secret_img, rev_secret_img, opt.testPics, i)

        if i >= 5:
            break


    print(
        "#################################################### validation end ########################################################")

# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(bs_secret, cover, container, secret, rev_secret, save_path, i):

    #if not opt.debug:
    # cover=container: bs*nt/nc;   secret=rev_secret: bs*nt/3*nh
    resultImgName = '%s/ResultPics_%d.png' % (save_path, i)

    cover_gap = container - cover
    secret_gap = rev_secret - secret
    cover_gap = (cover_gap*10 + 0.5).clamp_(0.0, 1.0)
    secret_gap = (secret_gap*10 + 0.5).clamp_(0.0, 1.0)

    showCover = torch.cat((cover, container, cover_gap),0)
    showSecret = torch.cat((secret, rev_secret, secret_gap),0)
    showAll = torch.cat((showCover, showSecret),0)

    vutils.save_image(showAll, resultImgName, nrow=bs_secret, padding=1, normalize=True)

if __name__ == '__main__':
    main()