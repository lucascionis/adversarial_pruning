import pathlib
import torch
import matplotlib.pyplot as plt
import scienceplots

# plt.style.use(['science','ieee'])

data_dir = pathlib.Path('fmn_attack_data')
models = tuple({'_'.join(f.name.split('_')[:2])
                for f in data_dir.glob('*') if f.is_dir()})
models = sorted(models)

figure, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
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
    pert_sizes = torch.linspace(0, 0.5, 10000).unsqueeze(1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax = axes.flatten()[idx]
    # ax[i, j].scatter(8/255, aa_acc[models_dict[model]], label='AA', marker='+', color='green', zorder=3)
    # TODO: add AA point as the baseline/reference
    ax.plot(pert_sizes, norms)
    ax.set_title(model)
    ax.grid(True)

figure.text(0.5, 0.04, r'Perturbation $\delta$', ha='center')
figure.text(0.08, 0.5, 'Robust Accuracy', va='center', rotation='vertical')


plt.savefig(f"hydra_fmn_attack_exps.pdf", bbox_inches='tight')
