import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# plt.style.use(['science','ieee'])

data_dir = pathlib.Path('harp_fmn_attack_data')
models = tuple({'_'.join(f.name.split('_')[:2])
                for f in data_dir.glob('*') if f.is_dir()})
models = sorted(models)

figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
for idx, model in enumerate(models):
    model_dir = list(data_dir.glob(f'{model}*'))[0]
    batches = [f for f in model_dir.glob('*')]

    best_advs = None
    inputs = None
    for batch in batches:
        _best_advs = torch.load(batch / 'best_adv.pt', map_location='cpu')
        _inputs = torch.load(batch / 'inputs.pt', map_location='cpu')

        if best_advs is None:
            best_advs = _best_advs if best_advs is None else torch.cat([best_advs, _best_advs])
        if inputs is None:
            inputs = _inputs if inputs is None else torch.cat([inputs, _inputs])

    norms = (best_advs - inputs).flatten(1).norm(torch.inf, dim=1)
    pert_sizes = torch.linspace(0, 0.2, 1000).unsqueeze(1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax = axes.flatten()[idx]
    # ax[i, j].scatter(8/255, aa_acc[models_dict[model]], label='AA', marker='+', color='green', zorder=3)
    # TODO: add AA point as the baseline/reference
    ax.plot(pert_sizes, norms, color='#3D5A80')
    ax.set_title(model)
    ax.grid(True)

    custom_xticks = np.linspace(0, 0.2, 5)
    ax.set_xticks(custom_xticks)

    closest_index = np.abs(pert_sizes - 8 / 255).argmin()
    closest_value = pert_sizes[closest_index]
    closest_norm = norms[closest_index]

    ax.axvline(x=8/255, color='#5DA271', linewidth=1)
    ax.scatter(closest_value, closest_norm, color='#EE6C4D', marker='*', label='8/255', zorder=3, s=20)
    ax.text(closest_value + 0.01, closest_norm, f'{closest_norm:.2f}', fontsize=14, verticalalignment='center', color='#EE6C4D')


    ax.legend()

figure.text(0.5, 0.04, r'Perturbation $||\delta||$', ha='center')
figure.text(0.08, 0.5, 'Robust Accuracy', va='center', rotation='vertical')


plt.savefig(f"harp_fmn_attack_exps_500samples_500steps.pdf", bbox_inches='tight')
