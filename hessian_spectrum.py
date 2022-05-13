from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from hessian_eigenthings import compute_hessian_eigenthings, compute_hessian_eigenthings_gan
from rs_gan import networks
from rs_gan import datasets
import hessian_eigenthings.density as density_lib
import pickle
import os
import copy

num_eigenthings = 100  # compute top 20 eigenvalues/eigenvectors
model = 'dis'
use_gpu = True
mode = 'power_iter'
norm = False
epochs = [0, 20000, 40000, 60000, 80000, 100000]


def get_gloss(dis_fake, dis_real, type='log'):
    if type == 'log':
        scalar = torch.FloatTensor([0]).to(dis_fake)
        z = dis_real - dis_fake
        z_star = torch.max(z, scalar.expand_as(z)).to(dis_fake)
        return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()
    elif type == 'hinge':
        return (F.relu(1 + (dis_real - dis_fake))).mean()


def get_dloss(dis_fake, dis_real, type='log'):
    if type == 'log':
        scalar = torch.FloatTensor([0]).to(dis_fake)
        z = dis_fake - dis_real
        z_star = torch.max(z, scalar.expand_as(z)).to(dis_fake)
        return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()
    elif type == 'hinge':
        return (F.relu(1 + (dis_fake - dis_real))).mean()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


for epoch in epochs:
    print(model, norm, epoch)
    args = dict(dataset='cifar', structure='resnet', losstype='log', batch_size=128, image_size=32,
                input_dim=128, num_iters=100000, num_features=256, bottleneck=False, g_lr=0.0002,
                d_lr=0.0002, beta1=0.5, beta2=0.999, g_freq=1, d_freq=1, lr_decay_start=50000,
                print_freq=100, plot_freq=1000, save_freq=1000, seed=3, ema_trick=False, reload=epoch,
                norm=norm)

    args = dotdict(args)

    if norm:
        save_path = 'experiment/with_batch_norm/models/'
    else:
        save_path = 'experiment/without_batch_norm/models/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    netG, netD = networks.getGD_SN(args.structure, args.dataset, args.image_size, args.num_features,
                                   dim_z=args.input_dim, bottleneck=args.bottleneck, norm=args.norm)

    if args.ema_trick:
        ema_netG_9999 = copy.deepcopy(netG)

    if args.reload > -1:
        netG.load_state_dict(torch.load(save_path + 'G_epoch{}.pth'.format(args.reload),
                                        map_location=torch.device(device)))
        netD.load_state_dict(torch.load(save_path + 'D_epoch{}.pth'.format(args.reload),
                                        map_location=torch.device(device)))
        if args.ema_trick:
            ema_netG_9999.load_state_dict(
                torch.load(save_path + 'emaG0.9999_epoch{}.pth'.format(args.reload), map_location=torch.device('cpu')))

    # netG = netG.to(device)
    # netD = netD.to(device)

    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=args.batch_size)

    eigenvals, eigenvecs = compute_hessian_eigenthings_gan(netG, netD, model, loader, get_gloss, get_dloss,
                                                           args.input_dim, num_eigenthings, use_gpu=use_gpu,
                                                           mode=mode,
                                                           )

    density, grids = density_lib.tridiag_to_density(eigenvals, eigenvecs, grid_len=10000, sigma_squared=1e-3)

    if not os.path.exists('results'):
        os.mkdir('results')

    pickle.dump({'density': density, 'grids': grids},
                open(f'results/model_{model}_norm_{norm}_epoch_{epoch}.pkl', 'wb'))
