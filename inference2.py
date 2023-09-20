import argparse
import torch
import os
import logger
import cv2
import numpy as np
import torch.nn as nn
from model import FusionNet, DilationCNN, UNet, TransUNet, VisionTransformer, CONFIGS
from dataset import NucleiDataset, HPADataset, NeuroDataset, HPASingleDataset, get_augmenter, PlasticDataset
from torch.utils.data import DataLoader
from scipy import ndimage as ndi
from skimage import morphology

def convert_ms_to_str(t):
    sec = t // 60
    minute = sec // 60
    ms = t % 60
    return "{:02d}:{:02d}:{:02d}".format(minute, sec, ms)

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
        checkpoint = torch.load("FusionNet_Plastic_1c_8.pth")
    elif args.model == "dilation":
        model = DilationCNN(train_dataset.dim)
        checkpoint = torch.load("DilationCNN_Plastic_1c_8.pth")
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, train_dataset.dim, train_dataset.target_dim)
        checkpoint = torch.load("UNet_Plastic_1c_8.pth")
    elif args.model == "tunet":
        train_dataset.im = 256
        model = TransUNet(img_dim=train_dataset.im,
                          in_channels=train_dataset.dim,
                          out_channels=128,
                          head_num=8,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=train_dataset.target_dim)
        checkpoint = torch.load("TransUNet_Plastic_1c_8.pth")
    elif args.model == "tunet2":
        vit_name = 'ViT-L_16'
        config_vit = CONFIGS[vit_name] 
        config_vit.n_classes = train_dataset.target_dim
        config_vit.n_skip = 0
        #config_vit.skip_channels = [512, 256, 64, 16]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(train_dataset.im / 16), int(train_dataset.im / 16))
        model = VisionTransformer(config_vit, img_size=train_dataset.im, num_classes=config_vit.n_classes).cuda()
        #model.load_from(weights=np.load(config_vit.pretrained_path))     
        checkpoint = torch.load("VisionTransformer_Plastic_1c_8.pth")   
    elif args.model == "DVLab3":
        import torchvision
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=1)
        model.classifier[4] = nn.Conv2d(
                                    in_channels=256,
                                    out_channels=train_dataset.target_dim,
                                    kernel_size=1,
                                    stride=1
                                )
        checkpoint = torch.load("DeepLabV3_Plastic_1c_8.pth")   
    else:
        raise Exception("Model not found", args.model)
            
    model.load_state_dict(checkpoint['model_state'])

    if args.device == "cuda":
        # parse gpu_ids for data paralle
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)

    model.to(device)
    model.eval()

    ptype = "010_Blinded1/4_BottomRight" #  Kaolinite - Cellulose - HDPE - HDPEre - PETE - PETEre - PVC - PVCre - Algae - 010_Blinded1/1_TopLeft/2_TopRight/3_BottomLeft/4_BottomRight
    output_dir = "test_video/{}/".format(ptype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    theta = 0.70

    vpath = args.train_data + "/video/{}".format(ptype)
    list_video = list(os.listdir(vpath))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for v in list_video:
        # send data and label to device
        base_name = v.replace(".mp4", "")
        path = os.path.join(vpath, v)
        print(path)

        cap = cv2.VideoCapture(path)
        success, image = cap.read()
        dim = image.shape
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(output_dir+'%s.mp4' % (base_name), fourcc, fps, (dim[1], dim[0]))

        init_pred = None
        area_list = []
        while success:
            x = preprocess(image, train_dataset.im).float().to(device)
            with torch.set_grad_enabled(False):
                # predict segmentation
                pred = model.forward(x)
            if args.model == "DVLab3":
                pred = pred["out"]
            predictions = pred.clone().squeeze().detach().cpu().numpy()

            predictions = cv2.resize(predictions, (dim[1], dim[0]))
            predictions[predictions > theta] = 1.
            predictions[predictions <= theta] = 0.

            x, m, area = visualize_mask(image, predictions)
            area_list.append(area)
            if init_pred is None:
                init_pred = np.copy(m)
            else:
                x[init_pred > 0] = np.array([255,0,0])
            #x = (predictions*255).astype("uint8")
            #x = np.stack([x,x,x], axis = -1)
            out.write(x)
            success, image = cap.read()
        with open(output_dir+'%s.txt' % (base_name), 'w') as f:
            for n, area in enumerate(area_list):
                f.write("{},{},{},{:.2f}\n".format(n, convert_ms_to_str(n*20), area, area/max(area_list)))

        cap.release()
        out.release()

    return



def visualize_mask(x, m):
    m = ndi.binary_fill_holes(m)
    m = morphology.remove_small_objects(m, 10)
    area = m.sum()
    m = ndi.binary_dilation(m)*1 - ndi.binary_erosion(m) *1
    x[m>0] = np.array([0,0,255])
    return x, m, area

def preprocess(img, dim):
    if len(img.shape) == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_torch = cv2.resize(img, (dim, dim))
    img_torch = np.transpose(img_torch, (2, 0, 1))
    img_torch = img_torch / 255.
    img_torch = torch.from_numpy(img_torch.astype('float32')).unsqueeze(0)
    #print(img_torch.size())
    return img_torch

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
