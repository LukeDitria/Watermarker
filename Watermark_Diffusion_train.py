import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
from ema_pytorch import EMA
from sklearn.metrics import fbeta_score
import numpy as np
import pandas as pd

from Dataloader import Flower102Dataset as CustomDataset
from Unet import Unet

# from Watermark_Embedder_Simple import Unet as Watermarker
from Watermark_Generator_CBN import Generator as Watermarker

import Helpers as hf

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)
parser.add_argument("--watermark_model_name", "-wmn", help="Watermarker save name", type=str, required=True)

parser.add_argument("--dataset_root", help="Dataset root dir", type=str, default="/media/luke/Quick Storage/Data/")
parser.add_argument("--dataset", help="Pytorch dataset to use", type=str, default="none")

parser.add_argument("--save_dir", help="Root dir for saving model and data", type=str,
                    default="/media/luke/Kingston_Storage/Models/Watermarker")
parser.add_argument("--target", help="Model output target -> image/noise", type=str, default="image")
parser.add_argument("--data_attribute", help="Dataset attribute to encode", type=str, default="")
parser.add_argument("--conditional_input", "-cin", help="Diffusion conditional input", type=str, default="none")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=64)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--diff_ch_multi", '-dw', help="Diffusion Channel width multiplier", type=int, default=64)
parser.add_argument("--diff_block_widths", '-dbw', help="Diffusion Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)
parser.add_argument("--accum_steps", '-as', help="Number of gradient accumulation steps", type=int, default=1)
parser.add_argument("--uncert_multi", '-um', help="Uncertainty noise multiplier", type=int, default=1)

parser.add_argument("--num_steps", '-ns', help="number of training diffusion steps", type=int, default=50)
parser.add_argument("--classes_to_mark", '-ctm', help="Image Classes to add watermark to", type=int,
                    nargs='+', default=(0,))
parser.add_argument("--conditional_dim", "-cdim", help="Dimension of conditional input ", type=int, default=100)
parser.add_argument("--num_wm", '-nw', help="Number of watermarks", type=int, default=256)
parser.add_argument("--data_workers", help="Number of Dataloader workers", type=int, default=4)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-5)
parser.add_argument("--noise_sigma", help="Sigma of sampled noise", type=float, default=1)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--lr_decay", '-ld', action='store_true', help="learning rate decay")
parser.add_argument("--use_watermarker", action='store_true', help="Use the Unet Watermarker to encode watermark")
parser.add_argument("--mark_all", action='store_true', help="watermark every class")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
torch.cuda.set_device(device)

if args.image_size < 64:
    from Decoder import DecoderResNet18Sml as Decoder
elif args.image_size < 128:
    from Decoder import DecoderResNet18 as Decoder
else:
    from Decoder import DecoderResNet34 as Decoder

transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
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
elif args.dataset == "artbench10":
    train_set = ImageFolder(root=args.dataset_root + "/artbench-10-imagefolder-split/train", transform=transform)
    test_set = ImageFolder(root=args.dataset_root + "/artbench-10-imagefolder-split/test", transform=transform)
elif args.dataset == "flowers102":
    from Dataloader import Flower102Dataset
    train_set = Flower102Dataset(dataset_root=args.dataset_root + "/102flowers_128",
                                 transform=transform)
    test_set = train_set
elif args.dataset == "celeba":
    from Dataloader import CelebAHQDataset
    train_set = CelebAHQDataset(dataset_root=args.dataset_root + "/CelebAHQ",
                                target_attribute=args.data_attribute,
                                transform=transform)
    test_set = train_set
else:
    ValueError("Dataset not defined")

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.data_workers)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(train_loader)
data = next(dataiter)
test_images = data[0]
diffusion_net = Unet(channels=test_images.shape[1],
                     img_size=args.image_size,
                     out_dim=test_images.shape[1],
                     dim=args.diff_ch_multi,
                     dim_mults=args.diff_block_widths,
                     conditional_in=args.conditional_input,
                     conditional_dim=args.conditional_dim).to(device)

# Setup optimizer
optimizer = optim.Adam(diffusion_net.parameters(), lr=args.lr)

# Flag to check if model has exploded
model_is_good = True
# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

if args.conditional_input == "attribute":
    dataframe = pd.read_csv(os.path.join(args.dataset_root, "attributes.csv"))
    attributes_list = torch.tensor(((dataframe[dataframe.keys()[1:]].to_numpy() + 1)/2))
else:
    attributes_list = None

if args.target == "image":
    print("Model is predicting the image")
elif args.target == "noise":
    print("Model is predicting the noise")
else:
    raise ValueError("Incorrect target source")


if args.mark_all:
    print("Watermarking all Classes:")
else:
    print("Watermark on class:", args.classes_to_mark)

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in diffusion_net.parameters():
    num_model_params += param.flatten().shape[0]
print("Diffusion Model has %d (approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Create the save directory if it does not exist
if not os.path.isdir(args.save_dir + "/Models"):
    os.makedirs(args.save_dir + "/Models")
if not os.path.isdir(args.save_dir + "/Results"):
    os.makedirs(args.save_dir + "/Results")

# Checks if a checkpoint has been specified to load, if it has, it loads the checkpoint
# If no checkpoint is specified, it checks if a checkpoint already exists and raises an error if
# it does to prevent accidental overwriting. If no checkpoint exists, it starts from scratch.
save_file_name = args.model_name + "_" + str(args.image_size)
if args.load_checkpoint:
    checkpoint = torch.load(args.save_dir + "/Models/" + save_file_name + ".pt",
                            map_location="cpu")
    print("Checkpoint loaded")
    diffusion_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ema = checkpoint['ema_model'].to(device)
    optimizer.param_groups[0]["weight_decay"] = 0.0

    if not optimizer.param_groups[0]["lr"] == args.lr:
        print("Updating lr!")
        optimizer.param_groups[0]["lr"] = args.lr

    print("Learning rate is %f" % optimizer.param_groups[0]["lr"])
    start_epoch = checkpoint["epoch"]

    data_logger = defaultdict(lambda: [], checkpoint["data_logger"])
else:
    # If checkpoint does exist raise an error to prevent accidental overwriting
    if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
        raise ValueError("Warning Checkpoint exists")
    else:
        print("Starting from scratch")
        start_epoch = 0
        # Loss and metrics logger
        data_logger = defaultdict(lambda: [])
        ema = EMA(diffusion_net, beta=0.9999, update_after_step=100, update_every=10)
        data_logger["class_counts"] = np.zeros(args.num_wm)

wm_save_file_name = args.watermark_model_name + "_ims_" + str(args.image_size) + "_emb_" + str(args.num_wm)
print(wm_save_file_name)
if os.path.isfile(args.save_dir + "/Models/" + wm_save_file_name + ".pt"):
    wm_checkpoint = torch.load(args.save_dir + "/Models/" + wm_save_file_name + ".pt", map_location="cpu")
    wm_args = wm_checkpoint["args"]
    print(wm_args["watermark_scale"])

    watermark_net = Watermarker(channels=test_images.shape[1],
                                img_size=args.image_size,
                                dim=wm_args["wm_ch_multi"],
                                dim_mults=wm_args["wm_block_widths"],
                                num_watermarks=args.num_wm,
                                watermark_scale=wm_args["watermark_scale"],
                                blur_kernel_size=wm_args["blur_kernel_size"],
                                blur_sigma=wm_args["blur_sigma"]).to(device)

    decoder = Decoder(num_outputs=args.num_wm).to(device)

    watermark_net.load_state_dict(wm_checkpoint['model_state_dict'])
    decoder.load_state_dict(wm_checkpoint['decoder_model_state_dict'])

    watermark_net.eval()
    decoder.eval()
else:
    raise ValueError("Warning! Watermarker Checkpoint does NOT exist")

alphas = torch.flip(hf.cosine_alphas_bar(args.num_steps), (0, )).to(device)

# Start training loop
for epoch in trange(start_epoch, args.nepoch, leave=False):
    diffusion_net.train()
    optimizer.zero_grad()
    for i, data in enumerate(tqdm(train_loader, leave=False)):
        current_iter = i + epoch * len(train_loader)
        max_iter = args.nepoch * len(train_loader)

        if args.lr_decay:
            new_lr = args.lr * (max_iter - current_iter)/max_iter
            optimizer.param_groups[0]["lr"] = new_lr

        images = data[0].to(device)
        labels = data[1].to(device)

        bs, c, h, w = images.shape

        with torch.cuda.amp.autocast():
            # Randomly sample batch of alphas
            index = torch.randint(args.num_steps, (bs, ), device=device)
            alpha = alphas[index].reshape(bs, 1, 1, 1)

            with torch.no_grad():
                if args.mark_all:
                    mask = torch.ones(bs, dtype=torch.bool)
                else:
                    mask = torch.any(data[1].unsqueeze(1) == torch.tensor([args.classes_to_mark]), 1).flatten()

                if mask.sum() > 0 and args.use_watermarker:
                    imgs_to_wm = images[mask]
                    imgs_no_wm = images[~mask]

                    labels_wm = labels[mask]

                    codes = F.one_hot(labels_wm, args.num_wm).float().to(device)
                    wm_model_output = watermark_net(imgs_to_wm, wm_index=labels_wm)

                    # Pretend we're converting to uint8 and back
                    wm_img = ((wm_model_output["image_out"] + 1) * 127.5).round()
                    wm_img = wm_img/127.5 - 1

                    image_in = torch.cat((imgs_no_wm, wm_img), 0)
                else:
                    image_in = images
                    imgs_to_wm = None

                random_sample = torch.randn_like(images)
                noise_image = alpha.sqrt() * image_in + (1 - alpha).sqrt() * random_sample

            if args.conditional_input == "none":
                model_output = diffusion_net(noise_image, index)
            else:
                model_output = diffusion_net(noise_image, index, cond_input=labels)

            if args.target == "image":
                loss = F.l1_loss(model_output["image_out"], image_in)
            else:
                loss = F.l1_loss(model_output["image_out"], random_sample)

        scaler.scale(loss/args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update()

        with torch.no_grad():
            data_logger["loss"].append(loss.item())
            data_logger["num_wm"].append(mask.sum().item())

        # Try to detect if the model has exploded and stop training
        if torch.isinf(loss) or torch.isnan(loss) or (model_output["image_out"].mean().abs() >= 0.9):
            print("Model has exploded!", loss.item())
            model_is_good = False
            break

        # Save results and a checkpoint at regular intervals
        if (current_iter + 1) % (args.save_interval * args.accum_steps) == 0:
            # In eval mode the model will use mu as the encoding instead of sampling from the distribution
            diffusion_net.eval()

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    decoder_output = decoder(image_in)
                    detection = (F.softmax(decoder_output["decoded"], 1).max(1)[0] > 0.5).float()
                    data_logger["num_detected_wm"].append(detection.sum().item())

                    if imgs_to_wm is not None:
                        vutils.save_image(wm_img.cpu().float(),
                                          "%s/%s/%s_%d_wm_img.png" % (args.save_dir,
                                                                         "Results", args.model_name, args.image_size),
                                          normalize=True)

                    diff_img, init_img = hf.image_cold_diffuse_simple(diffusion_model=diffusion_net,
                                                                      batch_size=args.batch_size,
                                                                      total_steps=args.num_steps,
                                                                      device=device,
                                                                      image_size=args.image_size,
                                                                      noise_sigma=args.noise_sigma,
                                                                      attributes_list=attributes_list)

                    vutils.save_image(diff_img.cpu().float(),
                                      "%s/%s/%s_%d_cold_diff.png" % (args.save_dir,
                                                                     "Results", args.model_name, args.image_size),
                                      normalize=True)

                    diff_img = ((diff_img + 1) * 127.5).round()
                    diff_img = diff_img/127.5 - 1
                    decoder_output = decoder(diff_img)

                    sm_dist = F.softmax(decoder_output["decoded"], -1)
                    data_logger["class_softmax"] = sm_dist.mean(0).cpu().numpy()
                    data_logger["max_score"].append(sm_dist.max().item())

                    detection = (sm_dist.max(1)[0] > 0.5).float()
                    data_logger["percent_diff_detected_wm"].append(detection.mean().item())

                    for i in range(detection.shape[0]):
                        if detection[i] > 0.5:
                            data_logger["class_counts"][decoder_output["decoded"][i].argmax().item()] += 1

                    if detection.sum() > 0:
                        diff_img_ = diff_img[detection.flatten() == 1]
                        vutils.save_image(diff_img_.cpu().float(),
                                          "%s/%s/%s_%d_wm_diff_img.png" % (args.save_dir,
                                                                         "Results", args.model_name, args.image_size),
                                          normalize=True)

                # Keep a copy of the previous save in case we accidentally save a model that has exploded...
                if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                    shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                                    dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

            if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                os.remove(args.save_dir + "/Models/" + save_file_name + ".pt")

            # Save a checkpoint
            torch.save({'epoch': epoch + 1,
                        'data_logger': dict(data_logger),
                        'model_state_dict': diffusion_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': args.__dict__,
                        'ema_model': ema,
            }, args.save_dir + "/Models/" + save_file_name + ".pt")

            # Set the model back into training mode!!
            diffusion_net.train()

    # Try to detect if the model has exploded and stop training
    if not model_is_good:
        break
