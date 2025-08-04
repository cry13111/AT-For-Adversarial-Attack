import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--device', type=str, default='cuda', help='Device used for training')
parser.add_argument("--output", type=str, help='output path')
parser.add_argument("--steps", default=20, type=int, help='pgd_steps')

args = parser.parse_args()
output = args.output

if not os.path.exists(output):
    os.makedirs(output)

device = args.device
# === Parameter Settings ===
epsilon = 8 / 255
alpha = 2 / 255
pgd_steps = args.steps  # 👈 Change to 10, 20, or 40 for different training settings
num_epochs = 200
batch_size = 128
lr = 0.1

# === Data Loading ===
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


# === PGD Adversarial Example Generation ===
def pgd_attack(model, images, labels, eps, alpha, steps):
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    ori = images.clone().detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.data
        images = images + alpha * grad.sign()
        images = torch.max(torch.min(images, ori + eps), ori - eps)
        images = torch.clamp(images, 0, 1).detach()

    return images


def evaluate(model, testloader, epsilon, alpha, pgd_steps, device):
    model.eval()
    clean_total = 0
    adv_total = 0
    clean_correct = 0
    adv_correct = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # Adversarial samples must be generated in requires_grad mode
        adv_images = pgd_attack(model, images, labels, eps=epsilon, alpha=alpha, steps=pgd_steps)

        with torch.no_grad():
            # Prediction on clean images
            clean_output = model(images)
            _, pred_clean = clean_output.max(1)
            clean_correct += pred_clean.eq(labels).sum().item()
            clean_total += labels.size(0)

            # Prediction on adversarial images
            adv_output = model(adv_images)
            _, pred_adv = adv_output.max(1)
            adv_correct += pred_adv.eq(labels).sum().item()
            adv_total += labels.size(0)

    clean_acc = 100. * clean_correct / clean_total
    adv_acc = 100. * adv_correct / adv_total

    return clean_acc, adv_acc


# === Model Definition ===
def get_resnet50():
    model = resnet50()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # CIFAR-10 has small input size, no need for maxpool
    model.fc = nn.Linear(2048, 10)
    return model


model = get_resnet50().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

# === Adversarial Training ===
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    clean_total = 0
    adv_total = 0
    clean_correct = 0
    adv_correct = 0
    with tqdm(total=len(trainset), desc=f"[Epoch: {epoch + 1}]") as pbar:
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Generate PGD adversarial samples
            adv_images = pgd_attack(model, images, labels, eps=epsilon, alpha=alpha, steps=pgd_steps)

            # Forward pass
            adv_outputs = model(adv_images)
            clean_output = model(images)
            loss = (criterion(adv_outputs, labels) + criterion(clean_output, labels)) / 2

            # Backpropagation + Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.update(images.size(0))

            _, predicted = clean_output.max(1)
            clean_total += labels.size(0)
            clean_correct += predicted.eq(labels).sum().item()

            _, predicted = adv_outputs.max(1)
            adv_total += labels.size(0)
            adv_correct += predicted.eq(labels).sum().item()

    lr_scheduler.step()
    print(f"[Epoch {epoch + 1}/{num_epochs}] train_Loss: {total_loss / len(trainloader):.4f}")
    print(f"[Epoch {epoch + 1}/{num_epochs}] train_clean_Acc: {100 * clean_correct / clean_total:.2f}%")
    print(f"[Epoch {epoch + 1}/{num_epochs}] train_adv_Acc: {100 * adv_correct / adv_total:.2f}%")

    test_clean_acc, test_adv_acc = evaluate(model, testloader, epsilon, alpha, pgd_steps, device)
    print(f"[Test] clean_Acc: {test_clean_acc:.2f}%")
    print(f"[Test] adv_Acc:   {test_adv_acc:.2f}%")

    if (epoch + 1) % 20 == 0:
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(trainloader),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "train_clean_Acc": 100 * clean_correct / clean_total,
            "train_adv_Acc": 100 * adv_correct / adv_total,
            "test_clean_Acc": test_clean_acc,
            "test_adv_Acc": test_adv_acc,
        }, f"./{output}/ckpt_{epoch}.pth")
