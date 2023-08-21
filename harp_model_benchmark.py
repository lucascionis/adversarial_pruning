import os
import datetime
from args import parse_args

import torch
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from models.resnet_cifar import resnet18 as harp_resnet18
from models.resnet_cifar import resnet18
from models.resnet import ResNet50 as resnet50
from models.vgg_cifar import vgg16_bn as vgg16
from data.cifar import CIFAR10
from data.svhn import SVHN
from data.imagenet import imagenet

from attacks.fmn_opt import FMNOpt

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = {
    'resnet18': (
        'harp_pretrained/resnet18/CIFAR10/pgd/pretrain/latest_exp/checkpoint/model_best.pth.tar',
    ),
    'resnet18_svhn': (
        'harp_pretrained/resnet18/SVHN/pgd/pretrain/latest_exp/checkpoint/model_best.pth.tar',
    ),
    'vgg16': (
        'harp_pretrained/vgg16_bn/CIFAR10/pgd/pretrain/latest_exp/checkpoint/model_best.pth.tar',
    ),
    'vgg16_svhn': (
        'harp_pretrained/vgg16_bn/SVHN/pgd/pretrain/latest_exp/checkpoint/model_best.pth.tar',
    ),
    'resnet50': (
        'harp_pretrained/ResNet50/imagenet/normalize/pgd/pretrain/latest_exp/checkpoint/model_best.pth.tar',
    )
}

model_to_net = {
    'resnet18': resnet18,
    'resnet18_svhn': resnet18,
    'resnet50': resnet50,
    'vgg16': vgg16,
    'vgg16_svhn': vgg16
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
    args.batch_size = args.test_batch_size = 10
    args.test_fmn = True

    test_images = 200
    attack_samples = 10
    attack_batch_size = 5

    # Load data
    print("->Retrieving the dataset...")
    cifar10 = CIFAR10(args=args)
    svhn = SVHN(args=args)

    train_loader, test_loader, testset = cifar10.data_loaders()
    svhn_train_loader, svhn_test_loader, svhn_testset = svhn.data_loaders()

    if 'resnet50' in pretrained.keys():
        imagenet_ds = imagenet(args=args)
        imgnet_train_loader, imgnet_test_loader, imgnet_testset = imagenet_ds.data_loaders()

    # Creating data lists
    test_data = {}

    for model_name in pretrained:
        print("\n")
        model = model_to_net[model_name](
            conv_layer=Conv2d,
            linear_layer=Linear,
            init_type='kaiming_normal',
            num_classes=10
            #mean=torch.Tensor([0.4914, 0.4822, 0.4465]),
            #std=torch.Tensor([0.2023, 0.1994, 0.2010])
        )

        if model_name not in test_data:
            test_data[model_name] = {}

        # iterate through chechkpoints
        for i, chk_path in enumerate(pretrained[model_name]):
            print(f"\n->Loading the {model_name}/{sparsities[i]} model...")
            test_data[model_name][sparsities[i]] = {}

            checkpoint = torch.load(chk_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval().to(device)

            if model_name == 'resnet50':
                try:
                    train_loader, test_loader, testset = imgnet_train_loader, imgnet_test_loader, imgnet_testset
                except Exception as e:
                    print("Error loading imagenet dataset")
                    print(e)

            if '_svhn' in model_name:
                try:
                    train_loader, test_loader, testset = svhn_train_loader, svhn_test_loader, svhn_testset
                except Exception as e:
                    print("Error loading svhn dataset")
                    print(e)

            # Clean-acc evaluation
            print(f"->Evaluating clean accuracy on {test_images} test images...")
            acc = compute_accuracy(model, test_loader, test_images)
            test_data[model_name][sparsities[i]]['clean acc'] = acc

            if args.test_fmn:
                print("->Evaluating robustness with FMN...")
                steps = 100

                optimizer = 'SGD'
                scheduler = 'CosineAnnealingLR'

                optimizer_config = {
                    'lr': 10,
                    'momentum': 0.9
                }
                scheduler_config = {}

                if scheduler == 'MultiStepLR':
                    milestones = len(scheduler_config['milestones'])
                    scheduler_config['milestones'] = np.linspace(0, steps, milestones)

                if scheduler == 'CosineAnnealingLR':
                    scheduler_config['T_max'] = steps

                if scheduler == 'CosineAnnealingWarmRestarts':
                    scheduler_config['T_0'] = steps // 2

                fmn_opt = FMNOpt(
                    model=model.eval().to(device),
                    dataset=testset,
                    norm='inf',
                    steps=steps,
                    batch_size=attack_batch_size,
                    batch_number=attack_samples//attack_batch_size,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    optimizer_config=optimizer_config,
                    scheduler_config=scheduler_config,
                    device=device
                )

                fmn_opt.run()
                robust_acc = accuracy(model, fmn_opt.attack_data[-1]['best_adv'], fmn_opt.attack_data[-1]['labels'])
                print(f"->FMN robust accuracy: {robust_acc * 100:.2f}")
                test_data[model_name][sparsities[i]]['AA robust'] = robust_acc

                # Get the current date and time
                current_datetime = datetime.datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
                data_path = os.path.join('harp_fmn_attack_data', f'{model_name}_{sparsities[i]}_{formatted_datetime}')

                print('-> Saving FMN data...')
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                # Save the data
                for j in range(fmn_opt.batch_number):
                    if not os.path.exists(f'{data_path}/{j}'):
                        os.mkdir(f'{data_path}/{j}')
                    for data in fmn_opt.attack_data[j]:
                        torch.save(fmn_opt.attack_data[j][data], f'{data_path}/{j}/{data}.pt')

    flat_data = {}
    for outer_key, inner_dict in test_data.items():
        for inner_key, value in inner_dict.items():
            flat_data.setdefault(inner_key, {})[outer_key] = value

    test_data_df = pd.DataFrame(flat_data)
    test_data_df.to_csv('./test_pretrained.csv')


if __name__ == '__main__':
    main()