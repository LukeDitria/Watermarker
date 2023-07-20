import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse

from Decoder import Decoder
from Watermark_Embedder import Unet

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)

parser.add_argument("--dataset_root", help="Dataset root dir", type=str, default="/media/luke/Quick Storage/Data/")
parser.add_argument("--dataset", help="Pytorch dataset to use", type=str, default="none")

parser.add_argument("--save_dir", help="Root dir for saving model and data", type=str,
                    default="/media/luke/Kingston_Storage/Models/VAE")
parser.add_argument("--target", help="Model output target -> image/noise", type=str, default="image")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=64)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--wm_ch_multi", '-dw', help="Diffusion Channel width multiplier", type=int, default=64)
parser.add_argument("--wm_block_widths", '-dbw', help="Diffusion Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))

parser.add_argument("--d_ch_multi", '-dew', help="Decoder Channel width multiplier", type=int, default=64)
parser.add_argument("--d_block_widths", '-debw', help="Decoder Channel multiplier for the input of each block", type=int,
                    nargs='+', default=(1, 2, 4, 8))

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)

parser.add_argument("--embedding_size", '-es', help="Size of the embedding vector", type=int, default=32)
parser.add_argument("--blur_size", help="Size Blur kernel", type=int, default=5)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--sigma_size", help="Kernel sigma", type=float, default=2)
parser.add_argument("--watermark_scale", help="How mush to scale the watermark by", type=float, default=0.05)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--lr_decay", '-ld', action='store_true', help="learning rate decay")
parser.add_argument("--fade_scale", action='store_true',
                    help="Slowly reduce the watermark scale to the target value over half the training period")
parser.add_argument("--blur_wm", action='store_true', help="Blur the watermark before combining with the image")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
torch.cuda.set_device(device)

resizing = transforms.RandomChoice([transforms.Resize(args.image_size),
                                    transforms.Compose(
                                        [transforms.Resize((int(args.image_size * 1.12), int(args.image_size * 1.12))),
                                         transforms.RandomCrop(args.image_size),
                                         ])])

transform = transforms.Compose([resizing,
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
else:
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                    transforms.CenterCrop(args.image_size),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    data_set = Datasets.ImageFolder(root=args.dataset_root, transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, _ = next(dataiter)

watermark_net = Unet(channels=test_images.shape[1],
                     img_size=args.image_size,
                     out_dim=test_images.shape[1],
                     dim=args.diff_ch_multi,
                     dim_mults=args.diff_block_widths,
                     embedding_size=args.embedding_size,
                     kernel_size=args.blur_size,
                     max_sigma=args.sigma_size,
                     watermark_scale=args.watermark_scale,
                     blur_watermark=args.blur_wm).to(device)

decoder = Decoder(channels=test_images.shape[1], num_outputs=args.embedding_size,
                  ch=args.d_ch_multi, blocks=args.d_block_widths).to(device)

# Setup optimizer
params = [*watermark_net.parameters(), *decoder.parameters()]
optimizer = optim.Adam(params, lr=args.lr)

# Flag to check if model has exploded
model_is_good = True
# AMP Scaler
scaler = torch.cuda.amp.GradScaler()

max_iter = args.nepoch * len(train_loader)

if args.target == "image":
    print("Model is predicting the image")
elif args.target == "noise":
    print("Model is predicting the noise")
else:
    raise ValueError("Incorrect target source")

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in watermark_net.parameters():
    num_model_params += param.flatten().shape[0]
print("Watermark Model has %d (approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

# Let's see how many Parameters our Model has!
num_model_params = 0
for param in decoder.parameters():
    num_model_params += param.flatten().shape[0]
print("Decoder Model has %d (approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))

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
    watermark_net.load_state_dict(checkpoint['model_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

# Start training loop
for epoch in trange(start_epoch, args.nepoch, leave=False):
    watermark_net.train()
    decoder.train()

    optimizer.zero_grad()
    for i, (images, _) in enumerate(tqdm(train_loader, leave=False)):
        current_iter = i + epoch * len(train_loader)

        if args.lr_decay:
            new_lr = args.lr * (max_iter - current_iter)/max_iter
            optimizer.param_groups[0]["lr"] = new_lr

        if args.fade_scale:
            scale_max_iter = max_iter/2
            starting_scale = 0.5 - args.watermark_scale
            wm_scale = max(0, starting_scale * ((scale_max_iter - current_iter)/scale_max_iter)) + args.watermark_scale
        else:
            wm_scale = args.watermark_scale

        images = images.to(device)
        bs, c, h, w = images.shape

        with torch.cuda.amp.autocast():
            # Randomly sample batch of codes
            ones_spot = torch.randint(args.embedding_size, (bs, ), device=device)
            codes = F.one_hot(ones_spot, args.embedding_size).float()

            model_output = watermark_net(images, code=codes, wm_scale=wm_scale)

            img_cat = torch.cat((model_output["image_out"], images), 0)
            detector_input = img_cat + 0.01 * torch.randn_like(img_cat)

            decoder_output = decoder(detector_input)

            img_loss = F.mse_loss(model_output["image_out"], images)
            code_loss = F.cross_entropy(decoder_output["decoded"][:bs], ones_spot)
            no_code_loss = F.cross_entropy(decoder_output["decoded"][bs:],
                                           (1/args.embedding_size) * torch.ones_like(decoder_output["decoded"][bs:]))

            contains_watermark = torch.cat((torch.ones(bs, 1, device=device),
                                            torch.zeros(bs, 1, device=device)), 0)

            loss = code_loss + no_code_loss

        data_logger["img_loss"].append(img_loss.item())
        data_logger["code_loss"].append(code_loss.item())
        data_logger["no_code_loss"].append(no_code_loss.item())

        pred_detect_out = decoder_output["decoded"][:bs].argmax(1)
        data_logger["wm_accuracy"].append((pred_detect_out == ones_spot).float().mean().item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if (current_iter + 1) % args.save_interval == 0:
            watermark_net.eval()
            decoder.eval()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    img_cat = torch.cat((images.cpu(), torch.clamp(model_output["image_out"], -1, 1).cpu(),
                                         torch.clamp(detector_input[:bs], -1, 1).cpu()), 2).float()

                    vutils.save_image(img_cat,
                                      "%s/%s/%s_%d_train.png" % (args.save_dir,
                                                                     "Results", args.model_name, args.image_size),
                                      normalize=True)

                    ones_spot = torch.randint(args.embedding_size, (test_images.shape[0],), device=device)
                    codes = F.one_hot(ones_spot, args.embedding_size).float()
                    contains_watermark = torch.cat((torch.ones(test_images.shape[0], 1, device=device),
                                                    torch.zeros(test_images.shape[0], 1, device=device)), 0)

                    model_output = watermark_net(test_images.to(device), code=codes, wm_scale=wm_scale)
                    wm_img = ((torch.clamp(model_output["image_out"], -1, 1) + 1) * 127.5).round()
                    wm_img = wm_img/127.5 - 1

                    decoder_output = decoder(wm_img)

                    pred_detect_out = decoder_output["decoded"][:test_images.shape[0]].argmax(1)
                    data_logger["test_wm_accuracy"].append((pred_detect_out == ones_spot).float().mean().item())

                # Keep a copy of the previous save in case we accidentally save a model that has exploded...
                if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                    shutil.copyfile(src=args.save_dir + "/Models/" + save_file_name + ".pt",
                                    dst=args.save_dir + "/Models/" + save_file_name + "_copy.pt")

                if os.path.isfile(args.save_dir + "/Models/" + save_file_name + ".pt"):
                    os.remove(args.save_dir + "/Models/" + save_file_name + ".pt")

                # Save a checkpoint
                torch.save({'epoch': epoch + 1,
                            'data_logger': dict(data_logger),
                            'model_state_dict': watermark_net.state_dict(),
                            'decoder_model_state_dict': decoder.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'args': args.__dict__,
                }, args.save_dir + "/Models/" + save_file_name + ".pt")

                # Set the model back into training mode!!
                watermark_net.train()
                decoder.train()
