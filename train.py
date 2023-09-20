import argparse
import torch
import tqdm
import numpy as np
import torch.nn as nn
import metrics
from model import FusionNet, DilationCNN, UNet, TransUNet, VisionTransformer, CONFIGS
from dataset import NucleiDataset, HPADataset, NeuroDataset, HPASingleDataset,get_augmenter, PlasticDataset
from torch.utils.data import DataLoader
from loss import dice_loss
import os

 
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
                    'shuffle': True,
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
    elif args.model == "tunet":
        train_dataset.im = 256
        model = TransUNet(img_dim=train_dataset.im,
                          in_channels=train_dataset.dim,
                          out_channels= 128, # 128
                          head_num=8, # 4
                          mlp_dim=512, # 512
                          block_num=8, # 8
                          patch_dim=16, # 16
                          class_num=train_dataset.target_dim)
    elif args.model == "tunet2":
        vit_name = 'ViT-L_16'
        config_vit = CONFIGS[vit_name] 
        config_vit.n_classes = train_dataset.target_dim
        config_vit.n_skip = 0
        #config_vit.skip_channels = [512, 256, 64, 16]
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(train_dataset.im / 16), int(train_dataset.im / 16))
        model = VisionTransformer(config_vit, img_size=train_dataset.im, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model == "DVLab3":
        import torchvision
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=1)
        model.classifier[4] = nn.Conv2d(
                                    in_channels=256,
                                    out_channels=train_dataset.target_dim,
                                    kernel_size=1,
                                    stride=1
                                )


    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)   
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




    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # loss 
    ce_loss = nn.BCELoss()
    loss_function = dice_loss

    count = 0
    old_avg_loss = 0
    # train model
    for epoch in range(args.epoch):
        model.train()
        
        #with tqdm.tqdm(total=len(train_dataloader.dataset), unit=f"epoch {epoch} itr") as progress_bar:
        total_loss = []
        total_iou = []
        total_precision = []
        for i, (x_train, y_train) in enumerate(train_dataloader):

            with torch.set_grad_enabled(True):

                # send data and label to device
                x = torch.Tensor(x_train.float()).to(device)
                y = torch.Tensor(y_train.float()).to(device)
                #print(x.size())

                # predict segmentation
                pred = model.forward(x)
                if args.model == "DVLab3":
                    pred = pred["out"]

                # calculate loss
                #loss_ce = ce_loss(pred.contiguous().view(-1), y.contiguous().view(-1))
                loss = loss_function(pred, y) #+  loss_ce
                total_loss.append(loss.item()) 

                # calculate IoU precision
                predictions = pred.clone().squeeze().detach().cpu().numpy()
                gt = y.clone().squeeze().detach().cpu().numpy()
                ious = [metrics.get_ious(p, g, 0.5) for p,g in zip(predictions, gt)]
                total_iou.append(np.mean(ious))

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # log loss and iou 
            avg_loss = np.mean(total_loss)
            avg_iou = np.mean(total_iou)
            print(epoch , i, avg_iou, avg_loss)

            # display segmentation on tensorboard 
            if i == 0:
                count += 1
                #progress_bar.update(len(x))

        # save model 
        scheduler.step()
        if epoch % 20 == 0 and avg_loss > old_avg_loss:
            old_avg_loss = avg_loss
            ckpt_dict = {'model_name': model.__class__.__name__, 
                        'model_state': model.to('cpu').state_dict()}
            experiment_name = f"{model.__class__.__name__}_{args.dataset}_{train_dataset.target_dim}c"
            if args.dataset == "HPA":
                experiment_name += f"_{args.max_mean}"
            experiment_name += f"_{args.num_kernel}"
            ckpt_path = os.path.join(args.save_dir, f"{experiment_name}.pth")
            torch.save(ckpt_dict, ckpt_path)
            model.to(device)

    # save model 
    ckpt_dict = {'model_name': model.__class__.__name__, 
                 'model_state': model.to('cpu').state_dict()}
    experiment_name = f"{model.__class__.__name__}_{args.dataset}_{train_dataset.target_dim}c"
    if args.dataset == "HPA":
        experiment_name += f"_{args.max_mean}"
    experiment_name += f"_{args.num_kernel}"
    ckpt_path = os.path.join(args.save_dir, f"{experiment_name}.pth")
    torch.save(ckpt_dict, ckpt_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--train_data', type=str, default="../plastic_data/")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="Plastic")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='2')
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--resume', type=str, default='')

    # agumentations
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser.add_argument('--transform', type=boolean_string, default="True")

    args = parser.parse_args()

    main(args)
