import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, FashionMNIST, ImageFolder

import os
import shutil
from tqdm import trange, tqdm
from collections import defaultdict
import argparse
from kornia import augmentation

# from Watermark_Embedder_Simple import Unet as Watermarker
# from Watermark_Embedder_embedding import WatermarkEmbedding as Watermarker
from Watermark_Generator_CBN import Generator as Watermarker
from Dataloader import Flower102Dataset as CustomDataset
from Dataloader import ArtBench10

parser = argparse.ArgumentParser(description="Training Params")
# string args
parser.add_argument("--model_name", "-mn", help="Experiment save name", type=str, required=True)

parser.add_argument("--dataset_root", help="Dataset root dir", type=str, default="/media/luke/Quick Storage/Data")
parser.add_argument("--dataset", help="Pytorch dataset to use", type=str, default="none")

parser.add_argument("--save_dir", help="Root dir for saving model and data", type=str,
                    default="/media/luke/Kingston_Storage/Models/Watermarker")

# int args
parser.add_argument("--nepoch", help="Number of training epochs", type=int, default=2000)
parser.add_argument("--batch_size", "-bs", help="Training batch size", type=int, default=64)
parser.add_argument("--image_size", '-ims', help="Input image size", type=int, default=64)
parser.add_argument("--wm_ch_multi", '-wmw', help="Diffusion Channel width multiplier", type=int, default=64)
parser.add_argument("--wm_block_widths", '-wmbw', help="Diffusion Channel multiplier for the input of each block",
                    type=int, nargs='+', default=(1, 2, 4, 8))

parser.add_argument("--device_index", help="GPU device index", type=int, default=0)
parser.add_argument("--save_interval", '-si', help="Number of iteration per save", type=int, default=256)

parser.add_argument("--num_wm", '-nw', help="Number of watermarks", type=int, default=256)
parser.add_argument("--accum_steps", '-as', help="Number of gradient accumulation steps", type=int, default=1)

parser.add_argument("--blur_kernel_size", '-blk', help="Size of the Gaussian kernel", type=int, default=5)
parser.add_argument("--blur_sigma", '-bls', help="Sigma of the Gaussian kernel", type=int, default=2)

# float args
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--watermark_scale", help="How mush to scale the watermark by", type=float, default=0.025)

# bool args
parser.add_argument("--load_checkpoint", '-cp', action='store_true', help="Load from checkpoint")
parser.add_argument("--lr_decay", '-ld', action='store_true', help="learning rate decay")
parser.add_argument("--fade_scale", action='store_true',
                    help="Slowly reduce the watermark scale to the target value over half the training period")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device(args.device_index if use_cuda else "cpu")
torch.cuda.set_device(device)

transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.CenterCrop(args.image_size),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])

aug_transforms = transforms.RandomChoice([augmentation.RandomResizedCrop((args.image_size, args.image_size), scale=(0.8, 1.0)),
                                          augmentation.RandomHorizontalFlip(1),
                                          augmentation.RandomVerticalFlip(1),
                                          augmentation.RandomErasing(p=1, scale=(0.02, 0.1)),
                                          augmentation.RandomAffine(degrees=45.0, p=1.),
                                          augmentation.RandomGaussianBlur((5, 5), (0.001, 1.0), p=1.)
                                          ])

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
    data_set = Flower102Dataset(dataset_root=args.dataset_root + "/102flowers_128",
                                 transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))
elif args.dataset == "celeba":
    from Dataloader import CelebAHQDataset
    data_set = CelebAHQDataset(dataset_root=args.dataset_root + "/CelebAHQ",
                                target_attribute=args.data_attribute,
                                transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))
else:
    ValueError("Dataset not defined")

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# Get a test image batch from the test_loader to visualise the reconstruction quality etc
dataiter = iter(test_loader)
test_images, _ = next(dataiter)

watermark_net = Watermarker(channels=test_images.shape[1],
                            img_size=args.image_size,
                            dim=args.wm_ch_multi,
                            dim_mults=args.wm_block_widths,
                            num_watermarks=args.num_wm,
                            watermark_scale=args.watermark_scale,
                            blur_kernel_size=args.blur_kernel_size,
                            blur_sigma=args.blur_sigma).to(device)

if args.image_size < 64:
    from Decoder import DecoderResNet18Sml as Decoder
elif args.image_size < 128:
    from Decoder import DecoderResNet18 as Decoder
else:
    from Decoder import DecoderResNet34 as Decoder

decoder = Decoder(num_outputs=args.num_wm).to(device)

# Setup optimizer
params = [*watermark_net.parameters(), *decoder.parameters()]
optimizer = optim.Adam(params, lr=args.lr)

# Flag to check if model has exploded
model_is_good = True
# AMP Scaler
scaler = torch.cuda.amp.GradScaler()
max_iter = args.nepoch * len(train_loader)

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
save_file_name = args.model_name + "_ims_" + str(args.image_size) + "_emb_" + str(args.num_wm)
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
            scale_max_iter = max_iter/8
            starting_scale = 0.5 - args.watermark_scale
            wm_scale = max(0, starting_scale * ((scale_max_iter - current_iter)/scale_max_iter)) + args.watermark_scale
        else:
            wm_scale = args.watermark_scale

        images = images.to(device)
        bs, c, h, w = images.shape

        with torch.cuda.amp.autocast():
            # Randomly sample batch of codes
            ones_spot = torch.randint(args.num_wm, (bs, ), device=device)
            codes = F.one_hot(ones_spot, args.num_wm).float()

            model_output = watermark_net(images, wm_index=ones_spot, wm_scale=wm_scale)

            img_cat = torch.cat((model_output["image_out"], images), 0)

            # Simulate pre-processing steps
            decoder_input = aug_transforms(img_cat)

            # Simulate Diffusion learning steps
            random_sample = torch.randn_like(decoder_input)
            alpha = torch.rand(bs * 2, 1, 1, 1, device=device) * 0.05 + 0.95
            decoder_input = alpha.sqrt() * decoder_input + (1 - alpha).sqrt() * random_sample

            decoder_output = decoder(decoder_input)

            img_loss = F.mse_loss(model_output["image_out"], images)
            code_loss = F.cross_entropy(decoder_output["decoded"][:bs], ones_spot)
            no_code_loss = F.cross_entropy(decoder_output["decoded"][bs:],
                                           (1/args.num_wm) * torch.ones_like(codes))

            detector_target = torch.cat((torch.ones(bs, 1, device=device),
                                         torch.zeros(bs, 1, device=device)), 0)

            loss = code_loss + no_code_loss

        scaler.scale(loss/args.accum_steps).backward()

        if (i + 1) % args.accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            data_logger["img_loss"].append(img_loss.item())
            data_logger["code_loss"].append(code_loss.item())
            data_logger["no_code_loss"].append(no_code_loss.item())

            pred_detect_out = decoder_output["decoded"][:bs].argmax(1)
            data_logger["wm_accuracy"].append((pred_detect_out == ones_spot).float().mean().item())
            data_logger["wm_scale"].append(wm_scale)

        if (current_iter + 1) % args.save_interval == 0:
            watermark_net.eval()
            decoder.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    img_cat = torch.cat((images.cpu(),
                                         model_output["image_out"].cpu(),
                                         torch.clamp(decoder_input[:bs].cpu(), -1, 1),
                                         model_output["watermark"].cpu()), 2).float()

                    vutils.save_image(img_cat,
                                      "%s/%s/%s_train.png" % (args.save_dir, "Results", save_file_name),
                                      normalize=True)

                    bs = test_images.shape[0]
                    ones_spot = torch.randint(args.num_wm, (bs,), device=device)
                    codes = F.one_hot(ones_spot, args.num_wm).float()

                    model_output = watermark_net(test_images.to(device), wm_index=ones_spot, wm_scale=wm_scale)
                    wm_img = ((torch.clamp(model_output["image_out"], -1, 1) + 1) * 127.5).round()
                    wm_img = aug_transforms(wm_img/127.5 - 1)
                    img_cat = torch.cat((wm_img, test_images.to(device)), 0)

                    decoder_output = decoder(img_cat)
                    pred_detect_out = decoder_output["decoded"][:bs].argmax(1)
                    data_logger["test_wm_accuracy"].append((pred_detect_out == ones_spot).float().mean().item())

                    # detector_out = (torch.sigmoid(decoder_output["detector"]) > 0.5).float()
                    # detector_target = torch.cat((torch.ones(bs, 1, device=device),
                    #                              torch.zeros(bs, 1, device=device)), 0)
                    # data_logger["test_detector_accuracy"].append((detector_out == detector_target).float().mean().item())

                    # Test Watermark variance with fixed watermark code
                    ones_spot = torch.zeros_like(ones_spot)
                    codes = F.one_hot(ones_spot, args.num_wm).float()
                    model_output = watermark_net(test_images.to(device), wm_index=ones_spot, wm_scale=wm_scale)
                    img_cat = torch.cat((test_images.cpu(),
                                         wm_img.cpu(),
                                         model_output["watermark"].cpu()), 2).float()

                    vutils.save_image(img_cat,
                                      "%s/%s/%s_test.png" % (args.save_dir, "Results", save_file_name),
                                      normalize=True)
                    data_logger["wm_var"].append(model_output["watermark"].var(0).mean().item())

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
