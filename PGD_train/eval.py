import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from PIL import Image
from tqdm import tqdm

from models import resnet, vgg, densenet, googlenet, inception, mobilenetv2

# CIFAR-10 normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

# Load CIFAR-10 test set true labels
cifar10 = datasets.CIFAR10(root='./data', train=False, download=True)
true_labels = [label for _, label in cifar10]  # Length: 10000

# All model names
model_names = ['vgg', 'resnet', 'densenet', 'googlenet', 'mobilenetv2', 'inception']

model_configs = {
    'resnet':      (resnet, 'resnet50'),
    'vgg':         (vgg, 'vgg19_bn'),
    'densenet':    (densenet, 'densenet121'),
    'googlenet':   (googlenet, 'googlenet'),
    'mobilenetv2': (mobilenetv2, 'mobilenet_v2'),
    'inception':   (inception, 'inception_v3'),
}

# Directory of adversarial images
adv_img_dir = "./at_pgd7_fgsm_outputs"  # Replace with your adversarial sample path

# Result storage
results = {}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, model_name in enumerate(model_names):
    # 1. Load model architecture
    module, func_name = model_configs[model_name]
    print(f"\n==> Evaluating model: {func_name}")
    model = getattr(module, func_name)()
    model.load_state_dict(torch.load(f"./weights/{func_name}.pt", map_location=device))
    model.to(device)
    model.eval()

    total = 0
    success = 0  # Number of successfully attacked samples

    for idx in tqdm(range(10000)):
        img_path = os.path.join(adv_img_dir, f"{idx:04d}_adversarial.png")
        if not os.path.exists(img_path):
            continue  # Skip missing images

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1,3,32,32]

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            pred_label = output.argmax(dim=1).item()

        # Compare with true label
        true_label = true_labels[idx]
        if pred_label != true_label:
            success += 1  # Misclassification indicates successful attack
        total += 1

    success_rate = success / total * 100
    results[func_name] = success_rate
    print(f"✅ Attack Success Rate on {func_name}: {success_rate:.2f}%")

# Print overall results
print("\n📊 Attack Summary:")
for name, asr in results.items():
    print(f"{name:15s}: {asr:.2f}%")
