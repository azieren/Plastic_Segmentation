import argparse
import torch
import os
import logger
import cv2
import numpy as np
import torch.nn as nn
from model import FusionNet, DilationCNN, UNet, TransUNet
from dataset import NucleiDataset, HPADataset, NeuroDataset, HPASingleDataset,get_augmenter, PlasticDataset
from torch.utils.data import DataLoader
from scipy import ndimage as ndi
from skimage import morphology

def main(args):

    # get dataset
    if args.dataset == "nuclei":
        train_dataset = NucleiDataset(args.train_data, 'train', args.transform, args.target_channels)
    elif args.dataset == "hpa":
        train_dataset = HPADataset(args.train_data, 'train', args.transform, args.max_mean, args.target_channels)
    elif args.dataset == "hpa_single":
        train_dataset = HPASingleDataset(args.train_data, 'train', args.transform)
    elif args.dataset == "Plastic":
        train_dataset = PlasticDataset(args.train_data, 'train', args.transform)
    else:
        train_dataset = NeuroDataset(args.train_data, 'train', args.transform)

    # create dataloader
    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_dataset, **train_params)

    # device
    device = torch.device(args.device)

    # model
    if args.model == "fusion":
        model = FusionNet(args, train_dataset.dim)
    elif args.model == "dilation":
        model = DilationCNN(train_dataset.dim)
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, train_dataset.dim, train_dataset.target_dim)
        checkpoint = torch.load("UNet_Plastic_1c_8.pth")
    elif args.model == "tunet":
        model = TransUNet(img_dim=train_dataset.im,
                          in_channels=train_dataset.dim,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=train_dataset.target_dim)
        checkpoint = torch.load("TransUNet_Plastic_1c_8.pth")
        
    
    model.load_state_dict(checkpoint['model_state'])

    if args.device == "cuda":
        # parse gpu_ids for data paralle
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)

        # parallelize computation
        if type(gpu_ids) is not int:
            model = nn.DataParallel(model, gpu_ids)
    model.to(device)
    model.eval()

    output_dir = "test/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    theta = 0.5
    count = 0

    for i, (x_train, y_train) in enumerate(train_dataloader):

        with torch.set_grad_enabled(False):

            # send data and label to device
            x = torch.Tensor(x_train.float()).to(device)
            y = torch.Tensor(y_train.float()).to(device)

            # predict segmentation
            pred = model.forward(x)

            # calculate IoU precision
            predictions = pred.clone().squeeze().detach().cpu().numpy()
            predictions[predictions > theta] = 1.
            predictions[predictions <= theta] = 0.

        for j in range(len(x_train)):
            x = (x_train[j].squeeze().cpu().numpy() * 255.0).astype("uint8")
            y = (y_train[j].cpu().numpy() * 255.0).astype("uint8")

            p = predictions[j]


            x, _ = visualize_mask(x, y, p)
            #x =  cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            p = (p * 255.0).astype("uint8")
            
            cv2.imwrite(os.path.join(output_dir, "{}.png".format(count)), x)
            #cv2.imwrite(os.path.join(output_dir, "gt_{}.png".format(i)), y)
            #cv2.imwrite(os.path.join(output_dir, "pr_{}.png".format(i)), p)
            count += 1


def visualize_mask(x, y, m):
    #x = np.stack([x,x,x], axis = -1)
    x =  cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
    m = ndi.binary_fill_holes(m)
    m = morphology.remove_small_objects(m, 10)
    m = ndi.binary_dilation(m)*1 - ndi.binary_erosion(m) *1
  
    m2 = ndi.binary_dilation(y)*1 - ndi.binary_erosion(y) *1
    
    x[m2>0] = np.array([0,255,0])
    x[m>0] = np.array([0,0,255])
    return x, m



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_data', type=str, default="../plastic_data/")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="Plastic")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='2')
    parser.add_argument('--experiment_name', type=str, default='test')

    # agumentations
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser.add_argument('--transform', type=boolean_string, default="False")

    args = parser.parse_args()

    main(args)
