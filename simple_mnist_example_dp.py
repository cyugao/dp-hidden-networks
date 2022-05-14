# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
# Adapted from https://github.com/allenai/hidden-networks/blob/master/simple_mnist_example.py
from __future__ import print_function
import argparse
import os
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

from opacus.privacy_engine import PrivacyEngine
from opacus.grad_sample import register_grad_sampler
from opacus.utils.tensor_utils import unfold2d

args = None


class GetSubnetSlow(autograd.Function):
    @staticmethod
    def forward(ctx, scores, sparsity):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - sparsity) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


def percentile(t, q):
    k = 1 + round(0.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, sparsity):
        k_val = percentile(scores, sparsity * 100)
        out = torch.where(scores < k_val, 0.0, 1.0)
        out.requires_grad = True
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores, args.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores, args.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


@register_grad_sampler(SupermaskLinear)
def compute_linear_grad_sample(layer, activations, backprops):
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    gs = torch.einsum("n...v,n...u,vu->nvu", backprops, activations, layer.weight)
    ret = {layer.scores: gs}
    return ret


@register_grad_sampler(SupermaskConv)
def compute_conv_grad_sample(layer, activations, backprops):
    """
    Computes per sample gradients for convolutional layers
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]
    # get activations and backprops in shape depending on the Conv layer
    activations = unfold2d(
        activations,
        kernel_size=layer.kernel_size,
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
    )
    backprops = backprops.reshape(n, -1, activations.shape[-1])
    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    weight = layer.weight.reshape(layer.out_channels, -1)
    grad_sample = torch.einsum("noq,npq,op->nop", backprops, activations, weight)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)

    ret = {layer.scores: grad_sample.view(shape)}
    # if layer.bias is not None:
    #     ret[layer.bias] = torch.sum(backprops, dim=2)
    return ret


# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SupermaskConv(1, 32, 3, 1, bias=False)
        self.conv2 = SupermaskConv(32, 64, 3, 1, bias=False)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = SupermaskLinear(9216, 128, bias=False)
        self.fc2 = SupermaskLinear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if args.log_interval != -1 and batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, criterion, test_loader, log=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = correct / len(test_loader.dataset)
    if log:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * test_acc,
            )
        )
    return test_loss, test_acc


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0005,
        metavar="M",
        help="Weight decay (default: 0.0005)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--data", type=str, default="../data", help="Location to store data"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.5, help="how sparse is each layer"
    )
    parser.add_argument(
        "--no-dp", action="store_true", default=False, help="disables DP training"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            os.path.join(args.data, "mnist"),
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            os.path.join(args.data, "mnist"),
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.no_dp:
        privacy_engine = None
    else:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )
    log_root = "runs/mnist"
    os.makedirs(log_root, exist_ok=True)
    log_filename = str(args.sparsity) + ("_dp" if not args.no_dp else "")
    # Store training losses and accuracy
    train_losses = []
    train_accs = []

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, criterion, test_loader)
        train_loss, train_acc = test(model, device, criterion, train_loader, log=False)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(log_root, log_filename + ".pt"))

    eps = privacy_engine.get_epsilon(1e-5) if privacy_engine else None

    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "eps": eps,
    }
    print(f"sparsity={args.sparsity}, dp={not args.no_dp}: test_acc={test_acc:.4f}")

    with open(os.path.join(log_root, log_filename + ".json"), "w") as f:
        json.dump(history, f)

    return privacy_engine, model, optimizer, train_loader


if __name__ == "__main__":
    privacy_engine, model, optimizer, train_loader = main()
