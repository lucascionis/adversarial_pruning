import os
import datetime
from args import parse_args

import torch
from torch.nn import Conv2d, Linear
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from attacks.fmn_opt import FMNOpt

from torchvision.models import resnet50 as torch_resnet50
from CHITA.models.mobilenet import mobilenet
from CHITA.utils.main_utils import imagenet_get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = {
    'mobilenetv1': (
        'chita_pretrained/mobilenetv1/75/mobilenetv1_sparsity_75_best.pth',
        'chita_pretrained/mobilenetv1/89/mobilenetv1_sparsity_89_best.pth'
    ),
    'resnet50': (
        'chita_pretrained/resnet50/90/resnet50_sparsity_90_best.pth',
        'chita_pretrained/resnet50/95/resnet50_sparsity_95_best.pth',
        'chita_pretrained/resnet50/98/resnet50_sparsity_98_best.pth'
    )
}

model_to_net = {
    'mobilenetv1': mobilenet,
    'resnet50': torch_resnet50,
}

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
    test_images = 200
    batch_size = test_batch_size = 10
    attack_samples = 10
    attack_batch_size = 5

    # Load data
    print("->Retrieving the dataset...")
    train_dataset, test_dataset = imagenet_get_datasets("./datasets/imagenet")
    '''
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True
    )
    '''

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Creating data lists
    test_data = {}

    for model_name in pretrained:
        print("\n")

        model = model_to_net[model_name]
        if model_name == 'resnet50':
            model = model(weights=None)
        else:
            model = model()

        if model_name not in test_data:
            test_data[model_name] = {}

        # iterate through chechkpoints
        for i, chk_path in enumerate(pretrained[model_name]):
            sparsity = int(chk_path.split('/')[-2])
            print(f"\n->Loading the {model_name}/{sparsity} model...")
            test_data[model_name][sparsity] = {}

            #checkpoint = torch.load(chk_path, map_location=device)
            state_trained = torch.load(chk_path, map_location=torch.device('cpu'))['model_state_dict']
            new_state_trained = model.state_dict()
            for k in state_trained:
                key = k[7:]
                if key in new_state_trained:
                    new_state_trained[key] = state_trained[k].view(new_state_trained[key].size())
                else:
                    print('Missing key', key)
            model.load_state_dict(new_state_trained, strict=False)
            model.eval().to(device)

            # Clean-acc evaluation
            print(f"->Evaluating clean accuracy on {test_images} test images...")
            acc = compute_accuracy(model, test_loader, test_images)
            test_data[model_name][sparsity]['clean acc'] = acc

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
                dataset=test_dataset,
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
            test_data[model_name][sparsity]['AA robust'] = robust_acc

            # Get the current date and time
            current_datetime = datetime.datetime.now()
            formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M")
            data_path = os.path.join('chita_fmn_attack_data', f'{model_name}_{sparsity}_{formatted_datetime}')

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
    test_data_df.to_csv('./chita_test_models.csv')


if __name__ == '__main__':
    main()
