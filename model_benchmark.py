import os
import datetime
from args import parse_args

import torch
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from models.wrn_cifar import wrn_28_4 as hydra_wrn_28_4
from models.vgg_cifar import vgg16_bn as hydra_vgg16
from data.cifar import CIFAR10

from autoattack import AutoAttack as AA
from attacks.fmn_opt import FMNOpt

args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = {
    'vgg16': (
        'hydra_pretrained/adversarial_training/vgg16_cifar/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/90/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/95/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/vgg16_cifar/99/model_best_dense.pth.tar'
    ),
    'wrn284': (
        'hydra_pretrained/adversarial_training/wrn284_cifar/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/90/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/95/model_best_dense.pth.tar',
        'hydra_pruned/adversarial_training/wrn284_cifar/99/model_best_dense.pth.tar'
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
    args.batch_size = args.test_batch_size = 10
    args.test_autoattack = False
    args.test_fmn = True

    test_images = 1000
    attack_samples = 100
    attack_batch_size = 50

    # Load data
    print("->Retrieving the dataset...")
    dataset = CIFAR10(args=args)
    train_loader, test_loader, testset = dataset.data_loaders()

    # Creating data lists
    test_data = {}

    for model_name in pretrained:
        print("\n")
        model = model_to_net[model_name](
            conv_layer=Conv2d,
            linear_layer=Linear,
            init_type='kaiming_normal',
            num_classes=10
        )

        if model_name not in test_data:
            test_data[model_name] = {}

        # iterate through chechkpoints
        for i, chk_path in enumerate(pretrained[model_name]):
            print(f"\n->Loading the {model_name}/{sparsities[i]} model...")
            test_data[model_name][sparsities[i]] = {}

            checkpoint = torch.load(chk_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval().to(device)

            # Clean-acc evaluation
            print(f"->Evaluating clean accuracy on {test_images} test images...")
            acc = compute_accuracy(model, test_loader, test_images)
            test_data[model_name][sparsities[i]]['clean acc'] = acc

            # Auto-Attack evaluation
            if args.test_autoattack:
                print("->Evaluating robustness with AA...")
                model_adv = AA(model, norm='Linf', eps=8 / 255, version='standard', device=device)
                model_adv.attacks_to_run = model_adv.attacks_to_run = ['apgd-ce']

                aa_dataloader = DataLoader(testset, batch_size=attack_samples, shuffle=False)
                aa_images, aa_labels = next(iter(aa_dataloader))
                x_adv = model_adv.run_standard_evaluation(aa_images, aa_labels, bs=attack_batch_size)

                robust_acc = accuracy(model, x_adv, aa_labels)
                print(f"->AA robust accuracy: {robust_acc*100:.2f}")
                test_data[model_name][sparsities[i]]['AA robust'] = robust_acc
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

                fmn_opt.run(log=True)
                robust_acc = accuracy(model, fmn_opt.attack_data[-1]['best_adv'], aa_labels)
                print(f"->FMN robust accuracy: {robust_acc * 100:.2f}")
                test_data[model_name][sparsities[i]]['AA robust'] = robust_acc

                # Get the current date and time
                current_datetime = datetime.datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
                data_path = os.path.join('fmn_attack_data', f'{model_name}_{sparsities[i]}_{formatted_datetime}')

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
