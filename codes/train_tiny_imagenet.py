import argparse
import csv
import os
import urllib.request
import zipfile
from functools import partial
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ==========================================
# 1. DATASET SETUP: TINY-IMAGENET-200
# ==========================================
def prepare_tiny_imagenet(data_dir="./data"):
    dataset_path = os.path.join(data_dir, "tiny-imagenet-200")
    if os.path.exists(dataset_path):
        return dataset_path

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print("Downloading Tiny-ImageNet-200 dataset...")
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Reorganize validation directory for torchvision ImageFolder compatibility
    val_dir = os.path.join(dataset_path, "val")
    val_img_dir = os.path.join(val_dir, "images")
    val_annotations = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_annotations):
        print("Organizing Tiny-ImageNet validation directory...")
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name, class_name = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_img_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    os.rename(src, dst)
        if os.path.exists(val_img_dir):
            os.rmdir(val_img_dir)
        os.remove(val_annotations)

    return dataset_path


# ==========================================
# 2. ACTIVATIONS (ORIGINAL, HARDWARE LUT & FORMULAS)
# ==========================================
class Tanhexp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.exp(x))


class HardwareLUTActivation(nn.Module):
    def __init__(self, csv_filename: str = "verilog_results.csv", column_idx: int = 3):
        super().__init__()
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"Could not find hardware CSV file: {csv_filename}")

        x_vals, y_vals = [], []
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                x_vals.append(float(int(row[0]) / 1024.0))
                y_vals.append(float(int(row[column_idx]) / 1024.0))

        self.register_buffer("x_grid", torch.tensor(x_vals, dtype=torch.float32))
        self.register_buffer("y_grid", torch.tensor(y_vals, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x_clamped = torch.clamp(x, min=self.x_grid[0], max=self.x_grid[-1])
        indices = torch.searchsorted(self.x_grid, x_clamped.contiguous()) - 1
        indices = torch.clamp(indices, min=0, max=len(self.x_grid) - 2)

        x0, x1 = self.x_grid[indices], self.x_grid[indices + 1]
        y0, y1 = self.y_grid[indices], self.y_grid[indices + 1]

        t = (x_clamped - x0) / (x1 - x0 + 1e-8)
        return y0 + t * (y1 - y0)


class HardwareSwish(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "verilog_results.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=1)

class HardwareGELU(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "verilog_results.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=2)

class HardwareTanhexp(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "verilog_results.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=3)


class UnifiedSwishTanhexpApprox(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        neg_poly = (x * torch.pow(x + 4.0, 2)) / 32.0
        return torch.where(x < -4.0, 0.0, torch.where(x < 0.0, neg_poly, x))


class UnifiedGELUApprox(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        neg_poly = (x * torch.pow(x + 3.0, 2)) / 32.0
        return torch.where(x < -3.0, 0.0, torch.where(x < 0.0, neg_poly, x))

class PolyApproxSwishTanhexp(nn.Module):
    """Kim et al. Polynomial Approximation for Swish and Tanhexp (K2 = 4.0)"""
    def forward(self, x: Tensor) -> Tensor:
        neg_poly = (x * torch.pow(x + 4.0, 2)) / 32.0
        # torch.where(condition, value_if_true, value_if_false)
        return torch.where(x < -4.0, 0.0, torch.where(x < 0.0, neg_poly, x))

class PolyApproxGELU(nn.Module):
    """Kim et al. Polynomial Approximation for GELU (K2 = 3.0)"""
    def forward(self, x: Tensor) -> Tensor:
        neg_poly = (x * torch.pow(x + 3.0, 2)) / 32.0
        return torch.where(x < -3.0, 0.0, torch.where(x < 0.0, neg_poly, x))

# --- CGP Classes ---
class CGPHardwareSwish(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "cgp_results2.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=1)

class CGPHardwareGELU(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "cgp_results2.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=2)

class CGPHardwareTanhexp(HardwareLUTActivation):
    def __init__(self, csv_filename: str = "cgp_results2.csv"):
        super().__init__(csv_filename=csv_filename, column_idx=3)
        
# ==========================================
# 3. RESNET ARCHITECTURE
# ==========================================
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, act_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if act_layer is None: act_layer = nn.ReLU

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act2(out)


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: list[int], num_classes: int = 200, norm_layer: Optional[Callable[..., nn.Module]] = None, act_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if act_layer is None: act_layer = nn.ReLU
        self._norm_layer = norm_layer
        self._act_layer = act_layer
        self.inplanes = 64

        # Adapted initial conv for 64x64 Tiny-ImageNet
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act1 = act_layer()
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), self._norm_layer(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=self._norm_layer, act_layer=self._act_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, act_layer=self._act_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ==========================================
# 4. MOBILENET V2 ARCHITECTURE
# ==========================================
class Conv2dNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None, act_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), norm_layer(out_channels)]
        if act_layer is not None: layers.append(act_layer())
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None, act_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))
        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer))
        layers.extend([
            Conv2dNormActivation(hidden_dim, hidden_dim, stride=stride, norm_layer=norm_layer, act_layer=act_layer),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup) if norm_layer else nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 200, norm_layer: Optional[Callable[..., nn.Module]] = None, act_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        if act_layer is None: act_layer = nn.ReLU6

        input_channel = 32
        last_channel = 1280
        setting = [[1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]

        features = [Conv2dNormActivation(3, input_channel, stride=1, norm_layer=norm_layer, act_layer=act_layer)]
        for t, c, n, s in setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, c, stride, expand_ratio=t, norm_layer=norm_layer, act_layer=act_layer))
                input_channel = c
        features.append(Conv2dNormActivation(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer, act_layer=act_layer))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(last_channel, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return self.classifier(torch.flatten(x, 1))


# ==========================================
# 5. ACCURACY EVALUATION
# ==========================================
def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size).item() for k in topk]


# ==========================================
# 6. MAIN EXECUTION WITH LOGGING
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Tiny-ImageNet Benchmark")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--act", type=str, default="Tanhexp", choices=[
        "Tanhexp", "HardwareTanhexp", "PolyTanhexp", "CGPTanhexp",
        "SiLU", "HardwareSwish", "PolySwish", "CGPSwish",
        "GELU", "HardwareGELU", "PolyGELU", "CGPGELU",
        "ReLU", "ReLU6"
    ])
    args = parser.parse_args()

    # Mapping Activations
    act_map = {
        "Tanhexp": Tanhexp, 
        "HardwareTanhexp": HardwareTanhexp,
        "PolyTanhexp": PolyApproxSwishTanhexp,
        "CGPTanhexp": CGPHardwareTanhexp,
        "SiLU": nn.SiLU, 
        "HardwareSwish": HardwareSwish,
        "PolySwish": PolyApproxSwishTanhexp,
        "CGPSwish": CGPHardwareSwish,
        "GELU": nn.GELU, 
        "HardwareGELU": HardwareGELU,
        "PolyGELU": PolyApproxGELU,
        "CGPGELU": CGPHardwareGELU,
        "ReLU": nn.ReLU, 
        "ReLU6": nn.ReLU6
    }
    selected_act = act_map[args.act]

    # Setup Logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_{args.model}_{args.act}.txt")
    log_file = open(log_file_path, "w")

    def print_log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    print_log(f"==================================================")
    print_log(f" MODEL: {args.model} | ACTIVATION: {args.act} | DATASET: Tiny-ImageNet-200")
    print_log(f"==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using device: {device}")

    # Prepare Tiny-ImageNet Data
    dataset_path = prepare_tiny_imagenet()
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform_train)
    valset = torchvision.datasets.ImageFolder(root=os.path.join(dataset_path, "val"), transform=transform_val)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate Model
    if args.model == "resnet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200, act_layer=selected_act)
    else:
        model = MobileNetV2(num_classes=200, act_layer=selected_act)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1 if args.model == "resnet18" else 0.05, momentum=0.9, weight_decay=5e-4)
    epochs = 200
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print_log("\n--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Evaluation
        model.eval()
        top1_sum, top5_sum, total_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                top1, top5 = accuracy(outputs, targets, topk=(1, 5))
                batch_size = targets.size(0)
                top1_sum += top1 * batch_size
                top5_sum += top5 * batch_size
                total_samples += batch_size

        epoch_top1 = top1_sum / total_samples
        epoch_top5 = top5_sum / total_samples

        # Remove the 'if' condition to log every single epoch
        print_log(f"Epoch [{epoch}/{epochs}] - Loss: {running_loss/len(trainloader):.4f} | "
                  f"Test Top-1 Acc: {epoch_top1:.2f}% | Test Top-5 Acc: {epoch_top5:.2f}%")

    log_file.close()

if __name__ == "__main__":
    main()

