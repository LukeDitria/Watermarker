import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST

import os
import shutil
import math
import numpy as np
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
from ema_pytorch import EMA
# from denoising_diffusion_pytorch import Unet
from Unet import Unet
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from Watermark_embedding import Watermarker
from Unet import Unet
from Detector import Detector
from Unet_Embedder import Unet as UnetWm

import Helpers as hf
from vgg11 import VGG11

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--watermark_model_name", "-wmn", help="Detector save name", type=str, required=True)

parser.add_argument("--dataset_root", help="Dataset root dir", type=str, default="/media/luke/Quick Storage/Data/")
parser.add_argument("--dataset", help="Pytorch dataset to use", type=str, default="none")

parser.add_argument("--save_dir", help="Root dir for saving model and data", type=str,
                    default="/media/luke/Kingston_Storage/Models/VAE")
parser.add_argument("--target", help="Model output target -> image/noise", type=str, default="image")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=128)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)


parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--num_steps", '-ns', help="number of training diffusion steps", type=int, default=50)

parser.add_argument("--fid_imgs", '-fi', help="number of images to use for FID score", type=int, default=5000)
parser.add_argument("--watermark_check", '-wc', help="Which watermark to look for", type=int, default=0)
parser.add_argument("--class_check", '-cc', help="Which watermark to look for", type=int, default=0)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--noise_sigma", help="Sigma of sampled noise", type=float, default=1)
parser.add_argument("--threshold", "-th", help="Watermark detection threshold", type=float, default=0.5)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--class_pred", '-cl', action='store_true', help="Predict the class")
parser.add_argument("--unet", action='store_true', help="Use a Unet Watermarker")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

if args.dataset == "fashion":
    train_set = FashionMNIST(root=args.dataset_root, train=True, transform=transform)
    test_set = FashionMNIST(root=args.dataset_root, train=False, transform=transform)
elif args.dataset == "cifar10":
    train_set = CIFAR10(root=args.dataset_root, train=True, transform=transform)
    test_set = CIFAR10(root=args.dataset_root, train=False, transform=transform)
elif args.dataset == "cifar100":
    train_set = CIFAR100(root=args.dataset_root, train=True, transform=transform)
    test_set = CIFAR100(root=args.dataset_root, train=False, transform=transform)
elif args.dataset == "mnist":
    train_set = MNIST(root=args.dataset_root, train=True, transform=transform)
    test_set = MNIST(root=args.dataset_root, train=False, transform=transform)
else:
    data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, _ = next(dataiter)


# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
    checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt", map_location="cpu")
    print("Checkpoint loaded")
    diff_args = checkpoint["args"]

    diffusion_net = Unet(channels=test_images.shape[1],
                         img_size=diff_args["image_size"],
                         out_dim=test_images.shape[1],
                         dim=diff_args["diff_ch_multi"],
                         dim_mults=diff_args["diff_block_widths"]).to(device)

    diffusion_net.load_state_dict(checkpoint['model_state_dict'])

else:
    raise ValueError("Warning Diffusion Checkpoint does NOT exists")

wm_save_file_name = args.watermark_model_name + "_" + str(args.image_size)
embedding_size = 0
if os.path.isfile(args.save_dir + "/Models/" + wm_save_file_name + ".pt"):
    wm_checkpoint = torch.load(args.save_dir + "/Models/" + wm_save_file_name + ".pt", map_location="cpu")
    wm_args = wm_checkpoint["args"]
    embedding_size = wm_args["embedding_size"]

    if args.unet:
        watermark_net = UnetWm(channels=test_images.shape[1],
                               img_size=args.image_size,
                               out_dim=test_images.shape[1],
                               dim=wm_args["diff_ch_multi"],
                               dim_mults=wm_args["diff_block_widths"],
                               embedding_size=embedding_size).to(device)
    else:
        watermark_net = Watermarker(image_size=args.image_size, embedding_size=embedding_size).to(device)

    detector = Detector(channels=test_images.shape[1],
                        num_outputs=embedding_size,
                        ch=wm_args["d_ch_multi"],
                        blocks=wm_args["d_block_widths"]).to(device)

    watermark_net.load_state_dict(wm_checkpoint['model_state_dict'])
    detector.load_state_dict(wm_checkpoint['detector_model_state_dict'])

    watermark_net.eval()
    detector.eval()
else:
    raise ValueError("Warning! Watermarker Checkpoint does NOT exist")

if args.class_pred:
    res_net = models.resnet18()

    res_net.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    num_ftrs = res_net.fc.in_features
    res_net.fc = nn.Linear(num_ftrs, 100)

    res_net.maxpool = nn.Identity()
    res_net = res_net.to(device)

    checkpoint = torch.load(args.save_dir + "/Models/" + "ResNet18_CIFAR100.pt", map_location="cpu")
    res_net.load_state_dict(checkpoint['model_state_dict'])
    res_net.eval()


class_labels = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])

with torch.cuda.amp.autocast():
    with torch.no_grad():
        wm_pred_log = []
        wm_detect_log = []

        img_log = []
        class_pred_log = []

        class_softmax_log = []
        wm_softmax_log = []

        for _ in trange(10):
            diff_img, init_img = hf.image_cold_diffuse_simple(diffusion_model=diffusion_net,
                                                              batch_size=args.batch_size,
                                                              total_steps=args.num_steps,
                                                              device=device,
                                                              image_size=args.image_size,
                                                              noise_sigma=args.noise_sigma)

            diff_img = ((diff_img + 1) * 127.5).round()
            diff_img = diff_img / 127.5 - 1

            img_log.append(diff_img)
            pred_wm_class, detect = detector(diff_img)

            probs = F.softmax(pred_wm_class, 1)
            classes_pred = torch.multinomial(probs, 1).flatten()

            wm_softmax_log.append(probs.cpu().numpy())
            wm_pred_log.append(classes_pred.cpu().numpy().flatten())

            wm_detect_log.append((torch.sigmoid(detect) >= args.threshold).long().cpu().numpy().flatten())

            if args.class_pred:
                pred_class = res_net(diff_img)
                class_pred_log.append(pred_class.argmax(1).detach().cpu().numpy().flatten())

                class_softmax_log.append(F.softmax(pred_class, 1).detach().cpu().numpy())

        img_cat = torch.cat(img_log)
        wm_pred_cat = np.concatenate(wm_pred_log)
        wm_detect_cat = np.concatenate(wm_detect_log)
        wm_softmax_cat = np.concatenate(wm_softmax_log)

        print("Number of SM score > 0.5", (wm_softmax_cat > 0.5).sum())

        print("Watermark on class number: %d, label: %s" % (args.watermark_check, class_labels[args.watermark_check]))
        print("Number of images generated: %d" % img_cat.shape[0])
        print("Number of images with watermark detected: %d" % wm_detect_cat.sum())
        print("Percentage of images with watermark detected: %.2f%%" % (wm_detect_cat.mean() * 100))

        wm_detected, wm_counts = np.unique(wm_pred_cat[wm_detect_cat == 1], return_counts=True)
        wm_check_num = wm_counts[wm_detected == args.watermark_check]
        if len(wm_check_num) == 0:
            wm_check_num = 0

        print("Detected Watermarks:", wm_detected)
        print("Detected Watermarks, counts:", wm_counts)

        if wm_detect_cat.sum() > 0:
            print("Percentage of detected watermarks that are watermark number %d: %.2f%%" % (args.watermark_check,
                                                                                   100 * (wm_check_num/wm_detect_cat.sum())))
        else:
            print("NO watermarks detected")

        wm_decoded, wm_decoded_counts = np.unique(wm_pred_cat, return_counts=True)
        print("Decoded Watermarks:", wm_decoded)
        print("Decoded Watermarks counts:", wm_decoded_counts)

        if args.class_pred:
            class_pred_cat = np.concatenate(class_pred_log)
            class_softmax_cat = np.concatenate(class_softmax_log)

            cl_detected, cl_counts = np.unique(class_pred_cat[wm_detect_cat == 1], return_counts=True)
            cl_check_num = cl_counts[cl_detected == args.class_check]
            if len(cl_check_num) == 0:
                cl_check_num = 0

            print("Predicted classes with watermarks:", cl_detected)
            print("Predicted classes with watermarks, counts:", cl_counts)
            if wm_detect_cat.sum() > 0:
                print("Percentage of detected watermarks that are predicted as class %d: %d%%" % (args.class_check,
                                                                                       100 * (cl_check_num/wm_detect_cat.sum())))
            else:
                print("NO watermarks detected")

            wm_detected, wm_counts = np.unique(wm_pred_cat[class_pred_cat == args.class_check], return_counts=True)
            wm_check_num = wm_counts[wm_detected == args.watermark_check]
            if len(wm_check_num) == 0:
                wm_check_num = 0

            print("Watermarks of predicted class ", wm_detected)
            print("Watermark counts of predicted class ", wm_counts)

            print("Percentage of images predicted as class %d:  that are watermark %d: %d%%" % (args.class_check,
                                                                                                args.watermark_check,
                                                                                                100 * (wm_check_num/wm_counts.sum())))
            # class_watermarks = np.zeros((len(class_labels), embedding_size))
            # for i in range(len(class_labels)):
            #     wm_detected, wm_counts = np.unique(wm_pred_cat[class_pred_cat == i], return_counts=True)
            #     for j in range(len(wm_detected)):
            #         class_watermarks[i, wm_detected[j]] = wm_counts[j]
            # class_watermarks = class_watermarks/class_watermarks.sum(0)
            #
            # print(class_watermarks)

            # x_embedded = TSNE(n_components=2).fit_transform(wm_softmax_cat)
            # plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=class_pred_cat, cmap="tab20")
            # plt.show()

            data_save_name = os.path.join(args.save_dir, "Data_out/" + args.model_name + "_data")
            np.save(data_save_name, [wm_pred_cat, class_pred_cat])
            hist, xedges, yedges = np.histogram2d(wm_pred_cat, class_pred_cat, bins=(embedding_size, pred_class.shape[1]))

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            # Construct arrays with the dimensions for the 16 bars.
            dx = dy = 0.5 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
            ax.set_ylabel("Class")
            ax.set_xlabel("Watermark")

            ax.set_xticks(np.arange(embedding_size))
            ax.set_yticks(np.arange(pred_class.shape[1]))

            plt.show()

        # _, wm_counts = np.unique(wm_pred_cat, return_counts=True)
        # wm_per = wm_counts/len(wm_pred_cat)
        # print("Watermark",  wm_per * 100)
        #

        # _, cl_counts = np.unique(class_pred_cat, return_counts=True)
        # cl_per = cl_counts/len(class_pred_cat)
        # print("Class", cl_per * 100)
        #
        # corr = np.corrcoef(wm_per, cl_per)
        # print("corr",  corr)
        #
        # # create plot
        # fig, ax = plt.subplots()
        # index = np.arange(10)
        # bar_width = 0.35
        # opacity = 0.8
        #
        # rects1 = plt.bar(index, wm_per, bar_width,
        #                  alpha=opacity,
        #                  color='b',
        #                  label='Watermark Predictions')
        #
        # rects2 = plt.bar(index + bar_width, cl_per, bar_width,
        #                  alpha=opacity,
        #                  color='g',
        #                  label='Class Predictions')
        #
        # plt.xlabel('Predictions Label')
        # plt.ylabel('Proportion of total')
        # plt.title('Class Vs Watermark Predictions, Corr:%.2f' % corr[0, 1])
        # plt.xticks(index + bar_width, class_labels)
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.show()
        # plt.savefig("Class_Watermark.png")

        vutils.save_image(diff_img.cpu().float(),
                          "%s/%s/%s_%d_cold_diff_test.png" % (args.save_dir,
                                                         "Results", args.model_name, args.image_size),
                          normalize=True)
        if wm_detect_cat.sum() > 0:
            vutils.save_image(img_cat[wm_detect_cat == 1].cpu().float(),
                              "%s/%s/%s_%d_cold_diff_test_wm.png" % (args.save_dir,
                                                             "Results", args.model_name, args.image_size),
                              normalize=True)

        # file_path = os.path.join(args.dataset_root, "vgg11_cifar10.npy")
        # cifar10_features = torch.tensor(np.load(file_path)).to(device)#.reshape(-1, 3*32*32)
        #
        # img_match = []
        # indx_match = []
        #
        # for i in trange(img_cat.shape[0]):
        #     fake_features = feature_extractor(img_cat[i].unsqueeze(0).to(device)).reshape(1, -1)
        #     # fake_features = img_cat[i].to(device).reshape(1, -1)
        #     feature_similarity = (cifar10_features.unsqueeze(0) - fake_features.unsqueeze(1)).pow(2).mean(2)
        #     min_indx = feature_similarity.argmin(1).cpu()
        #
        #     real_img, _, batch_indx = train_set.__getitem__(min_indx.item())
        #     img_match.append(real_img.unsqueeze(0))
        #     indx_match.append(batch_indx)
        #
        # indx_match_cat = np.array(indx_match)
        #
        # print((indx_match_cat == pred_cat).sum())
        # print((indx_match_cat == pred_cat).mean())
        #
        # img_match_cat = torch.cat(img_match)
        # img_out = torch.cat((img_match_cat[0:16].cpu(), img_cat[0:16].cpu()), 2).float()
        #
        # vutils.save_image(img_out,
        #                   "%s/%s/%s_%d_img_compare.png" % (args.save_dir,
        #                                                  "Results", args.model_name, args.image_size),
        #                   normalize=True)
