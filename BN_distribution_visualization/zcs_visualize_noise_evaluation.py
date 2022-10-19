from ast import arg
import setGPU
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# from models.resnet import ResNet18, test
# from models.resnet_multi_bn import multi_bn_resnet18
# import sys
# sys.path.append("../")
from resnet_multi_bn_default_pgd import multi_bn_resnet18

# from models.resnet_bn_change_AlwaysUseBNofCE import ResNet18_always_use_bn_of_ce
# MODEL_LIST = {
#     "resnet18": ResNet18,
#     "multi_bn_resnet18": multi_bn_resnet18,
#     "resnet18_always_use_bn_of_ce": ResNet18_always_use_bn_of_ce,

# }

from torch.autograd import Variable

import shutil
import glob
from datetime import datetime

# import wandb
# import apex.amp as amp

import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

# from training_method_bn_default_pgd import train, eval_robust, eval_test

def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pgray = output.max(1, keepdim=True)[1]
            correct += pgray.eq(target.view_as(pgray)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def eval_test(model, device, test_loader, epoch, test_bn="pgd"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if test_bn:
                output = model(data, test_bn)
            else:
                output = model(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pgray = output.max(1, keepdim=True)[1]
            correct += pgray.eq(target.view_as(pgray)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: loss: {:.4f}, Clean accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    # writer.add_scalar(f'Accuracy/test_{test_bn}',test_accuracy , epoch)
    return test_loss, test_accuracy

# def adjust_learning_rate(optimizer, epoch, lr):
#     """decrease the learning rate"""
#     lr = lr
#     if epoch >= 75:
#         lr = lr * 0.1
#     if epoch >= 90:
#         lr = lr * 0.01
#     if epoch >= 100:
#         lr = lr * 0.001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, lr):
    """decrease the learning rate"""
    lr = lr
    if epoch >= 100:
        lr = lr * 0.1
    if epoch >= 105:
        lr = lr * 0.01
    # if epoch >= 100:
    #     lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_fast_55(optimizer, epoch, lr):
    """decrease the learning rate"""
    lr = lr
    if epoch >= 50:
        lr = lr * 0.1
    if epoch >= 52:
        lr = lr * 0.01
    # if epoch >= 100:
    #     lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_epoch100_decay_7590(optimizer, epoch, lr):
    """decrease the learning rate"""
    if epoch >= 75:
        lr = lr * 0.1
    if epoch >= 90:
        lr = lr * 0.01
    if epoch >= 100:
        lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=110, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--weight-decay', '--wd', default=5e-4,
                        type=float, metavar='W')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # parser.add_argument('--epsilon', default=0.031,
    #                     help='perturbation 8/255')
    parser.add_argument('--epsilon', default=8, type=int,
                        help='perturbation')
    parser.add_argument('--num-steps', default=10, type=int,
                        help='perturb number of steps')
    # parser.add_argument('--step-size', default=0.007,
    #                     help='perturb step size 2/255')
    parser.add_argument('--step-size', default=2, type=int,
                        help='perturb step size')
    parser.add_argument('--beta', default=6.0, type=float,
                        help='regularization, i.e., 1/lambda in TRADES')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                        help='directory of model for saving checkpoint')
    parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                        help='save frequency')

    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')

    # parser.add_argument('--model', default='resnet18', choices=MODEL_LIST.keys(), type=str, help='Selecte training model.')

    BN_LAYER_LIST = ["vanilla", "oct_adv", "oct_clean", "oct_all", "none", "GN", "IN", "remove_affine"]
    parser.add_argument('--norm_layer', default='vanilla', choices=BN_LAYER_LIST, type=str, help='Selecte batch normalizetion layer in model.')

    parser.add_argument('--dual_bn', action='store_true', default=False,
                        help='create multi bn layer')

    SUPPORT_METHOD = ["trades_vanilla", "trades_dual_bn", "trades_cat", "Madry_loss", "Madry_cat_loss", "Madry_dual_bn", "Madry_mixture_bn", "Hybrid_single_bn", "Finetune_BN_on_clean"]
    parser.add_argument('--training_method', default='resnet18', choices=SUPPORT_METHOD, type=str, help='Selecte training method.')

    # auto mixed precision argument
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')

    LRDECAY = ["vanilla", "fast55", "decay7590"]
    parser.add_argument('--lr_decay', default='vanilla', choices=LRDECAY, type=str, help='Output directory')


    # resume training from a ckpt file
    parser.add_argument('--resume_dir', type=str, default="./", help='resume training')




    # testing mode selection
    DRADWING_MODE = ["saved", "batch"]
    parser.add_argument('--drawing_mode', default='saved', choices=DRADWING_MODE, type=str)

    # argument for batch statistics
    FORWARD_DATA = ["clean", "ae_adv", "ae_clean", "noise"]
    parser.add_argument('--forward_data', default='clean', choices=FORWARD_DATA, type=str)

    parser.add_argument('--bn_name', default='pgd', type=str)


    parser.add_argument('--noise_size', default=0.0, type=float, help='noise scale')


    args = parser.parse_args()

    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.


    # random seed setting
    import random
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



    # settings

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='/dev/shm', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='/dev/shm', train=False, download=True, transform=transform_test)
    testset.data = testset.data[:1000]
    testset.targets = testset.targets[:1000]

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    from resnet_multi_bn_default_pgd import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all

    if args.norm_layer == "vanilla":

        # set momentum value in BN layer to 1, in order to get current mini-batch statistics
        def get_no_current_minibatch_stat_bn(channel):
            return nn.BatchNorm2d(channel, momentum=1)
        norm_layer = get_no_current_minibatch_stat_bn

        # norm_layer = nn.BatchNorm2d

    elif args.norm_layer == "oct_adv":
        # norm_layer = CrossTrainingBN_adv
        def get_no_current_minibatch_stat_bn(channel):
            return CrossTrainingBN_adv(channel, momentum=1)
        norm_layer = get_no_current_minibatch_stat_bn

    elif args.norm_layer == "oct_clean":
        # norm_layer = CrossTrainingBN_clean
        def get_no_current_minibatch_stat_bn(channel):
            return CrossTrainingBN_clean(channel, momentum=1)
        norm_layer = get_no_current_minibatch_stat_bn

    elif args.norm_layer == "oct_all":
        # norm_layer = CrossTrainingBN_all
        def get_no_current_minibatch_stat_bn(channel):
            return CrossTrainingBN_all(channel, momentum=1)
        norm_layer = get_no_current_minibatch_stat_bn

    elif args.norm_layer == "none":
        norm_layer = nn.Identity
    elif args.norm_layer == "GN":
        norm_layer = nn.GroupNorm

    elif args.norm_layer == "IN":
        norm_layer = nn.InstanceNorm2d

    elif args.norm_layer == "remove_affine":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, affine=False, momentum=1)

        norm_layer = get_no_affine_bn


    model_for_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=["pgd", "normal"]).cuda()
    model_state_dict = torch.load(args.resume_dir)
    model_for_test.load_state_dict(model_state_dict)

    fig_size = (10, 4)

    target_layer = [("layer1.1.bn2.bn_list.0.running_mean"), 
                    ("layer1.0.bn1.bn_list.0.running_mean"),

                    ("layer1.1.bn2.bn_list.0.running_var"),
                    ("layer1.0.bn1.bn_list.0.running_var"),

                    # ("layer1.1.bn2.bn_list.1.running_mean"), 
                    # ("layer1.0.bn1.bn_list.1.running_mean"),

                    # ("layer1.1.bn2.bn_list.1.running_var"),
                    # ("layer1.0.bn1.bn_list.1.running_var"),

                    ]

    saving_stat = get_saving_stat(torch.load(args.resume_dir), target_layer)


    # target_layer = [("layer1.1.bn2.bn_list.0.weight"), 
    #                 ("layer1.1.bn2.bn_list.0.bias"), 
    #                 ("layer1.1.bn2.bn_list.1.weight"), 
    #                 ("layer1.1.bn2.bn_list.1.bias"), 
    #                 ]
    # saving_gamma_beta = get_saving_stat(torch.load(args.resume_dir), target_layer)
    # shuffle_index = torch.randperm(saving_gamma_beta[0].size(0))
    # figure_data = {
    #     r"AP$_{adv}$": saving_gamma_beta[0],
    #     r"AP$_{clean}$": saving_gamma_beta[2],
    # }

    # plt.figure(figsize=fig_size)    
    # plt.subplot(2,1,1)
    # draw_figure(figure_data,  ylabel=r"$\gamma$", legend=True,color_list = ["royalblue", "orange"])

    # figure_data = {
    #     r"AP$_{adv}$": saving_gamma_beta[1],
    #     r"AP$_{clean}$": saving_gamma_beta[3],
    # }
    # plt.subplot(2,1,2)

    # draw_figure(figure_data,  ylabel=r"$\beta$", legend=True, color_list = ["royalblue", "orange"])
    # # plt.tight_layout()

    # plt.savefig(f"BN_distribution_visualization/zcs_figure/zcs_madry_cross_evaluation_beta_gamma_.png", bbox_inches="tight")


    target_layer = [("layer1.1.bn2.bn_list.0.running_mean"), 
                    ("layer1.0.bn1.bn_list.0.running_mean"),

                    ("layer1.1.bn2.bn_list.0.running_var"),
                    ("layer1.0.bn1.bn_list.0.running_var"),
                    ]

    adv_sample_forward_stats = get_forward_data_stat(multi_bn_resnet18(norm_layer=norm_layer, bn_names=["pgd", "normal"], state_dict_dir=args.resume_dir).cuda(), test_loader, "ae_adv", args.num_steps, args.epsilon, args.step_size, "pgd", target_layer)
    noise_sample_forward_stats = get_forward_data_stat(multi_bn_resnet18(norm_layer=norm_layer, state_dict_dir=args.resume_dir).cuda(), test_loader, "noise", args.num_steps, args.epsilon, args.step_size, "pgd", target_layer, noise_size=args.noise_size)

    figure_data = {
        r"NS$_{clean}$":saving_stat[0],
        r"NS$_{adv}$":adv_sample_forward_stats[0],
        r"NS$_{noise}$":noise_sample_forward_stats[0],
        # r"NS$_{clean}^{clean}}$":saving_stat[4],
        # r"NS$_{adv}^{clean}$":ae_adv_forward_stat_bn_normal[0],
    }


    shuffle_index = torch.randperm(saving_stat[0].size(0))
    plt.figure(figsize=fig_size)    
    plt.subplot(2,1,1)
    draw_figure(figure_data,  ylabel=r"$\mu$", legend=True,  color_list = ["royalblue", "lightsteelblue", "orange", "wheat"])

    figure_data = {
        r"NS$_{clean}$":saving_stat[2],
        r"NS$_{adv}$":adv_sample_forward_stats[2],
        r"NS$_{noise}$":noise_sample_forward_stats[2],
        # r"NS$_{clean}^{clean}}$":saving_stat[4],
        # r"NS$_{adv}^{clean}$":ae_adv_forward_stat_bn_normal[0],
    }

    # figure_data = {
    #     r"running$_{adv}$,affine$_{adv}$": saving_stat[2],
    #     r"CE,static$_{clean}$,affince$_{pgd}$": clean_forward_stat_bn_pgd[2],
    #     r"AE,static$_{adv}$,affince$_{adv}$": ae_adv_forward_stat_bn_pgd[2],
    #     r"running$_{clean}$,affine$_{clean}$": saving_stat[6],
    #     r"CE,static$_{clean}$,affince$_{clean}$": clean_forward_stat_bn_normal[2],
    #     r"AE,static$_{adv}$,affince$_{clean}$": ae_adv_forward_stat_bn_normal[2],
    # }
    plt.subplot(2,1,2)

    draw_figure(figure_data, ylabel=r"$\sigma$", color_list = ["royalblue", "lightsteelblue", "orange", "wheat"])
    # plt.tight_layout()

    plt.savefig(f"BN_distribution_visualization/zcs_figure/zcs_noise_evaluation_mean_std_layer1.png", bbox_inches="tight")





    # figure_data = {
    #     "running pgd": saving_stat[0],
    #     "pgd forward clean": clean_forward_stat_bn_pgd[0],
    #     "pgd forward ae_adv": ae_adv_forward_stat_bn_pgd[0],
    #     "pgd forward ae_clean": ae_clean_forward_stat_bn_pgd[0],
    #     "running normal": saving_stat[4],
    #     "normal forward clean": clean_forward_stat_bn_normal[0],
    #     "normal forward ae_adv": ae_adv_forward_stat_bn_normal[0],
    #     "normal forward ae_clean": ae_clean_forward_stat_bn_normal[0],
    # }

    # shuffle_index = torch.randperm(saving_stat[0].size(0))

    # draw_figure(figure_data, f"BN_distribution_visualization/figure/mean_all.png", shuffle_index=shuffle_index)


    # figure_data = {
    #     "running pgd": saving_stat[2],
    #     "pgd forward clean": clean_forward_stat_bn_pgd[2],
    #     "pgd forward ae_adv": ae_adv_forward_stat_bn_pgd[2],
    #     "pgd forward ae_clean": ae_clean_forward_stat_bn_pgd[2],
    #     "running normal": saving_stat[6],
    #     "normal forward clean": clean_forward_stat_bn_normal[2],
    #     "normal forward ae_adv": ae_adv_forward_stat_bn_normal[2],
    #     "normal forward ae_clean": ae_clean_forward_stat_bn_normal[2],
    # }

    # draw_figure(figure_data, f"BN_distribution_visualization/figure/var_all.png", shuffle_index=shuffle_index)




def get_forward_data_stat(model_for_test, test_loader, forward_data, num_steps, epsilon, step_size, bn_name, target_layer, noise_size=0):

    """
    # target_layer: list(tuple()), [(mean1, mean2, ...)]


    """


    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()

        # generating forward data
        # use clean image
        if forward_data == "clean":
            forward_data = data

        # use adversarial example generating by bn pgd
        elif forward_data == "ae_adv":
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            
            model_for_test.eval()
            for _ in range(num_steps):
                x_adv.requires_grad_()

                output = model_for_test(x_adv, "pgd")
                loss_ce = F.cross_entropy(output, target)
                # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                grad = torch.autograd.grad(loss_ce, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
            forward_data = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        # use adversarial example generating by bn normal
        elif forward_data == "ae_clean":
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            
            model_for_test.eval()
            for _ in range(num_steps):
                x_adv.requires_grad_()

                output = model_for_test(x_adv, "normal")
                loss_ce = F.cross_entropy(output, target)
                # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                grad = torch.autograd.grad(loss_ce, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
            forward_data = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        elif  forward_data == "noise":   
            current_noise_size = epsilon * noise_size
            input_noise = torch.zeros_like(data).cuda()
            input_noise.uniform_(-current_noise_size, current_noise_size)

            forward_data = data + input_noise
        
        # get current batch statistics
        model_for_test.train()
        # import ipdb; ipdb.set_trace()
        model_for_test(forward_data, bn_name)
        break

    
    model_state_dict = model_for_test.state_dict()


    target_stat = []
    for layers in target_layer:
        target_stat.append(model_state_dict[layers].data)

    del model_for_test

    return target_stat


def get_saving_stat(model_state_dict, target_layer):

    target_stat = []
    for layers in target_layer:
        target_stat.append(model_state_dict[layers].data)

    return target_stat


def draw_figure(draw_data, ylabel=None, legend=None, color_list = ["silver", "gray", "deepskyblue", "royalblue"]):
    """
    draw_data: {name: value}
    """

    # data to plot
    n_groups = 20


    # create plot
    # plt.figure(figsize=(10, 5))    
    index = np.arange(n_groups)

    bar_width = 0.8/len(draw_data)
    opacity = 0.8
    


    for i, (layer_name, layer_data) in enumerate(draw_data.items()):
        
        # print(layer_data.size())
        layer_data = layer_data[-n_groups:].cpu()
        # layer_data = layer_data[::step_size].cpu()

        # layer_data = layer_data[shuffle_index][:n_groups].cpu()

        plt.bar(index + bar_width * i, layer_data, bar_width,
        alpha=opacity,
        label=layer_name, color=color_list[i])
    if legend:
        lgnd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0,numpoints=1,fontsize=14)
    # plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.xlabel('channel', fontsize=15)
    if ylabel:
        plt.ylabel(ylabel, fontsize=15)
    # plt.title('Scores by person')
    plt.xticks([])
    # plt.legend(fontsize=14)

    # plt.tight_layout()

    # plt.savefig(f"BN_distribution_visualization/figure/batch_statistics-forward_data_{args.forward_data}-{model_name}_{pgd}.png")
    # plt.savefig(file_name)

if __name__ == '__main__':
    main()
