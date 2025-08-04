import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import datasets, transforms
from tqdm import tqdm
# from PGD.models import resnet

import torchvision.transforms.functional as F
import os

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_resnet50():
    model = resnet50()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, 10)
    return model


def fgsm_attack(epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbation = epsilon * sign_data_grad
    return perturbation


# Load model
model = get_resnet50()  # CIFAR-10 has 10 classes

# Load checkpoint
checkpoint = torch.load("./PGD7/ckpt_199.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# model = resnet.resnet50()
# checkpoint = torch.load('weights/resnet50.pt', map_location=device)
# model.load_state_dict(checkpoint)
# model = model.to(device)
# model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Create output directory
output_dir = "at_pgd7_fgsm_outputs"
os.makedirs(output_dir, exist_ok=True)

# Attack parameters
epsilon = 0.03
global_idx = 0  # Global image index

print("Starting FGSM attack and saving images...")
for batch_idx, (images, labels) in enumerate(tqdm(test_loader)):
    images, labels = images.to(device), labels.to(device)
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data
    perturbations = fgsm_attack(epsilon, data_grad)
    adv_images = torch.clamp(images + perturbations, 0, 1)

    for i in range(images.size(0)):
        # global_idx = batch_idx * test_loader.batch_size + i

        # Save original image
        orig_img = F.to_pil_image(images[i].detach().cpu())
        orig_img.save(os.path.join(output_dir, f"{global_idx:04d}_original.png"))

        # Save perturbation image (grayscale)
        pert_img = F.to_pil_image(perturbations[i].detach().cpu())
        pert_img.save(os.path.join(output_dir, f"{global_idx:04d}_perturbation.png"))

        # Save adversarial image
        adv_img = F.to_pil_image(adv_images[i].detach().cpu())
        adv_img.save(os.path.join(output_dir, f"{global_idx:04d}_adversarial.png"))

        global_idx += 1
