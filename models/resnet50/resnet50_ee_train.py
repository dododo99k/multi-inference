import argparse
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ee_models import *
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def check_features(sub_layers):
    for i, module in enumerate(list(sub_layers.modules())):
        try:
            final_features = module.num_features
        except:
            pass
    # print('final_features: ',final_features)
    return final_features

def calculate_loss(exit_outputs, target):
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    for i, output in enumerate(exit_outputs):
        loss += loss_fn(output, target)
    return loss



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_imagenet_loaders(data_root, batch_size, val_batch_size, num_workers, pin_memory):
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"ImageNet data not found under {data_root}. Expected 'train' and 'val' sub-directories.")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(train_dataset.classes)
    return train_loader, val_loader, num_classes


def load_base_resnet50(base_model_path, use_pretrained, num_classes):
    checkpoint_path = base_model_path
    default_checkpoint = os.path.join('weights', 'resnet50.pth')
    if checkpoint_path is None and os.path.isfile(default_checkpoint):
        checkpoint_path = default_checkpoint

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            model = models.resnet50(weights=None)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if isinstance(state_dict, dict):
                state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                              for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        print(f'Loaded base ResNet-50 weights from {checkpoint_path}')
    else:
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        if weights:
            print('Initialized base ResNet-50 from torchvision pretrained weights.')
        else:
            print('Initialized base ResNet-50 with random weights.')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_one_epoch(model, dataloader, optimizer, device, epoch, print_freq):
    model.train()
    running_loss = 0.0
    window_loss = 0.0
    epoch_start = time.perf_counter()
    window_start = epoch_start
    non_blocking = device.type == 'cuda'

    for batch_idx, (inputs, targets) in enumerate(dataloader, 1):
        inputs = inputs.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = calculate_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        window_loss += loss.item()

        if batch_idx % print_freq == 0:
            elapsed = time.perf_counter() - window_start
            print(f'[Epoch {epoch + 1}, Batch {batch_idx}] loss: {window_loss / print_freq:.3f}, '
                  f'time for {print_freq} batches: {elapsed:.1f}s')
            window_loss = 0.0
            window_start = time.perf_counter()

    epoch_time = time.perf_counter() - epoch_start
    avg_loss = running_loss / max(len(dataloader), 1)
    print(f'Epoch {epoch + 1} completed. Avg loss: {avg_loss:.4f}, epoch time: {epoch_time:.1f}s')


def evaluate_early_exits(model, dataloader, device):
    model.eval()
    non_blocking = device.type == 'cuda'
    accuracies = {}

    with torch.no_grad():
        for ramp_id in range(len(model.exit_fcs)):
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                outputs = model(images, ramp_id)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100.0 * correct / total if total else 0.0
            accuracies[f'exit_{ramp_id}'] = accuracy
            print(f'Ramp id {ramp_id}: accuracy on validation set {accuracy:.2f}%')

        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            outputs = model(images, ramp=None)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total else 0.0
        accuracies['full'] = accuracy
        print(f'Full ResNet-50 head accuracy (no early exit): {accuracy:.2f}%')

    return accuracies


def save_model(model, output_path):
    model_dir = os.path.dirname(output_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model, output_path)
    print(f'Saved trained model to {output_path}')


def main():
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='Train the early-exit ResNet-50 on ImageNet.')
    parser.add_argument('--imagenet-root', type=str, default='./dataset/imagenet',
                        help='Root directory to ImageNet with train/ and val/ sub-directories.')
    parser.add_argument('--epochs', type=int, default=90, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=256, help='Mini-batch size for training.')
    parser.add_argument('--val-batch-size', type=int, default=256, help='Mini-batch size for validation.')
    parser.add_argument('--workers', type=int, default=16, help='Number of dataloader worker processes.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for SGD.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for SGD.')
    parser.add_argument('--step-size', type=int, default=30, help='StepLR step size.')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma.')
    parser.add_argument('--device', type=str, default=default_device,
                        help='Device to use, e.g. cuda:0 or cpu. Defaults to first CUDA device if available.')
    parser.add_argument('--base-model', type=str, default=None,
                        help='Optional path to a ResNet-50 checkpoint used to build the early-exit model.')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use torchvision pretrained weights when no checkpoint is provided (default).')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Do not load torchvision pretrained weights when no checkpoint is provided.')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--output', type=str, default='./weights/resnet50_EE.pth',
                        help='Where to store the trained early-exit model.')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='How often (in iterations) to print training stats.')
    args = parser.parse_args()
    if args.device != 'cpu' and not torch.cuda.is_available():
        print('CUDA requested but not available, falling back to CPU.')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    pin_memory = device.type == 'cuda'
    train_loader, val_loader, num_classes = build_imagenet_loaders(
        args.imagenet_root,
        args.batch_size,
        args.val_batch_size,
        args.workers,
        pin_memory,
    )

    print(f'Loaded ImageNet data. Train samples: {len(train_loader.dataset)}, '
          f'Val samples: {len(val_loader.dataset)}, Classes: {num_classes}')

    base_model = load_base_resnet50(args.base_model, args.pretrained, num_classes)
    ee_model = EarlyExitResNet50(base_model, num_classes)
    ee_model.to(device)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, ee_model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(f'Start training for {args.epochs} epochs on {device}.')
    for epoch in range(args.epochs):
        train_one_epoch(ee_model, train_loader, optimizer, device, epoch, args.print_freq)
        scheduler.step()

    print('Finished Training')
    evaluate_early_exits(ee_model, val_loader, device)
    save_model(ee_model, args.output)


if __name__ == '__main__':
    main()
