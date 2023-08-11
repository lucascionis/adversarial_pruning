import os
from args import parse_args

import torch
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
import pandas as pd

from models.wrn_cifar import wrn_28_4 as hydra_wrn_28_4
from models.vgg_cifar import vgg16_bn as hydra_vgg16
from data.cifar import CIFAR10
from utils.model import prepare_model

from autoattack import AutoAttack as AA

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = {
    'wrn284': (
        'hydra_pretrained/adversarial_training/wrn284_cifar/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/90/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/95/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/99/model_best_dense.pth.tar'
    ),
    'vgg16': (
        'hydra_pretrained/adversarial_training/vgg16_cifar/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/90/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/95/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/99/model_best_dense.pth.tar'
    )
}

model_to_net = {
    'wrn284': hydra_wrn_28_4,
    'vgg16': hydra_vgg16
}

sparsities = (0, 90, 95, 99)


def compute_accuracy(net, test_loader, test_images=1000):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            if total >= test_images:
                break

            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct // total
    print(f'Accuracy of the {net.__class__.__name__} on {total} test images: {acc} %')
    return acc


def accuracy(model, samples, labels):
    preds = model(samples)
    acc = (preds.argmax(dim=1) == labels).float().mean()
    return acc.item()


def main():
    args.batch_size = args.test_batch_size = 50
    test_images = 1000
    rb_samples = 50
    rb_batch_size = 25

    # Load data
    print("->Retrieving the dataset...")
    dataset = CIFAR10(args=args)
    train_loader, test_loader, testset = dataset.data_loaders()

    aa_dataloader = DataLoader(testset, batch_size=rb_samples, shuffle=False)
    aa_images, aa_labels = next(iter(aa_dataloader))

    # Creating data lists
    test_data = {}

    for model_name in pretrained:
        print("\n")
        # iterate through checkpoints
        model = model_to_net[model_name](
            conv_layer=Conv2d,
            linear_layer=Linear,
            init_type='kaiming_normal',
            num_classes=10
        )

        if model_name not in test_data:
            test_data[model_name] = {}

        for i, chk_path in enumerate(pretrained[model_name]):
            print(f"\n->Loading the {model_name}/{sparsities[i]} model...")
            test_data[model_name][sparsities[i]/100] = {}

            checkpoint = torch.load(chk_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval()

            # Clean-acc evaluation
            print(f"->Evaluating clean accuracy on {test_images} test images...")
            acc = compute_accuracy(model, test_loader, test_images)
            test_data[model_name][sparsities[i] / 100]['clean acc'] = acc

            # Auto-Attack evaluation
            print("->Evaluating robustness...")
            model_adv = AA(model, norm='Linf', eps=8 / 255, version='standard', device=device)

            model_adv.attacks_to_run = model_adv.attacks_to_run = ['apgd-ce']
            x_adv = model_adv.run_standard_evaluation(aa_images, aa_labels, bs=rb_batch_size)

            robust_acc = accuracy(model, x_adv, aa_labels)
            print(f"->Robust accuracy: {robust_acc*100:.2f}")
            test_data[model_name][sparsities[i] / 100]['AA robust'] = robust_acc

    flat_data = {}
    for outer_key, inner_dict in test_data.items():
        for inner_key, value in inner_dict.items():
            flat_data.setdefault(inner_key, {})[outer_key] = value

    test_data_df = pd.DataFrame(flat_data)
    test_data_df.to_csv('./test_pretrained.csv')


if __name__ == '__main__':
    main()
