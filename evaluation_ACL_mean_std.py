from __future__ import print_function
import types
import setGPU
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *

# from trades import trades_loss
# from utils.seed_everything import seed_everything
# from utils.generate_adv import generate_adv
# from utils.file_backup import file_backup

# from madry import madry_loss, generate_adv
# from attack_method.dual_bn_Madry_original_pgd_code_based import dual_bn_madry_loss, generate_adv
# from attack_method.dual_bn_SameNetwork import dual_bn_madry_loss, generate_adv

import wandb
import torchattacks

# def change_bn_mode(model, bn_mode):
#     if bn_mode == "normal":
#         model.apply(change_bn_mode_normal)
    
#     elif bn_mode == "pgd":
#         model.apply(change_bn_mode_pgd)
    
#     else:
#         raise

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
# parser.add_argument('experiment', type=str)

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=16,type=int,
                    help='perturbation')
parser.add_argument('--num-steps', default=5,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=4, type=int,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')


BN_LAYER_LIST = ["vanilla", "oct_adv", "oct_clean", "oct_all", "none", "GN", "IN", "LN", "remove_affine", "Disentangling_LP"]
parser.add_argument('--norm_layer', default='vanilla', choices=BN_LAYER_LIST, type=str, help='Selecte batch normalizetion layer in model.')


parser.add_argument('--learn_adv', action='store_true')
parser.add_argument('--learn_clean', action='store_true')

parser.add_argument('--eval_pgd', action='store_true')
parser.add_argument('--eval_aa', action='store_true')


parser.add_argument('--ACL', action='store_true')



BN_NAME = ["pgd", "normal"]
parser.add_argument('--bn_name', default='pgd', choices=BN_NAME)


parser.add_argument('--checkpoint_file')

args = parser.parse_args()

args.epsilon = args.epsilon / 255.
args.step_size = args.step_size / 255.


# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(args.seed)
# seed_everything(args.seed)
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

# testset.data = testset.data[:1000]
# testset.targets = testset.targets[:1000]


test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# wandb.init(name=args.experiment, project="TRADES_change", entity="kaistssl")
# wandb.config.update(args)
# RUN_ID = wandb.run.id
# TRAIN_STEP = 0

def train(args, model, device, train_loader, optimizer, epoch, bn_name):
    # global TRAIN_STEP
    model.train()
    # loss_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # TRAIN_STEP += 1
        data, target = data.to(device), target.to(device)

        # optimizer.zero_grad()

        model(data, bn_name)


def learn_adv_bn(args, model, device, train_loader, optimizer, epoch, bn_name):
    # global TRAIN_STEP
    model.train()
    # loss_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # TRAIN_STEP += 1
        data, target = data.to(device), target.to(device)

        # optimizer.zero_grad()
        data_adv = generate_adv(model, data, target, optimizer, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=10, beta=1, distance='l_inf', bn_name=bn_name)


        model(data_adv, bn_name)

def generate_adv(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                bn_name="pgd"):
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                #                        F.softmax(model(x_natural), dim=1))
                # if dualBN == 0:
                loss_kl = F.cross_entropy(model(x_adv, bn_name), y)
                # elif dualBN == 1:
                #   loss_kl = F.cross_entropy(model(x_adv,'pgd'), y)
                
            # import ipdb; ipdb.set_trace()
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv



def eval_test(model, device, test_loader, epoch, bn_name="pgd"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, bn_name)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Clean Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy

def eval_robust(model, device, test_loader,epoch,optimizer, bn_name):
    model.eval()
    test_loss = 0
    correct = 0


    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data_adv = generate_adv(model, data, target, optimizer, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=20, beta=1, distance='l_inf', bn_name=bn_name)

            output = model(data_adv, bn_name)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('BN name: {}. Test: (robust) loss: {:.4f}, Robust accuracy: {}/{} ({:.4f}%)'.format(
        bn_name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy


def eval_autoattack(model, device, test_loader,epoch,optimizer, bn_name):
    model.eval()
    test_loss = 0
    correct = 0
    # change_bn_mode(model, bn_name)

    autoattack = torchattacks.AutoAttack(model, eps=args.epsilon)

    total_tested = 0

    with torch.no_grad():
        for data, target in test_loader:
            total_tested += data.size(0)
            data, target = data.to(device), target.to(device)
            # data_adv = generate_adv(model, data, target, optimizer, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=20, beta=1, distance='l_inf', bn_name=bn_name)
            data_adv = autoattack(data, target)
            output = model(data_adv)

            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            if total_tested > 1000:
                break
    test_loss /= total_tested
    print('Auto Attack :BN name: {}. Test: (robust) loss: {:.4f}, Robust accuracy: {}/{} ({:.4f}%)'.format(
        bn_name, test_loss, correct, total_tested,
        100. * correct / total_tested))

    test_accuracy = correct / total_tested

    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    # init model, ResNet18() can be also used here for training
    # model = WideResNet().to(device)
    # from models.resnet import ResNet18
    # from models.resnet_bn_change import ResNet18
    # model = ResNet18().to(device)

    # from models.resnet_multi_bn import resnet18

    # model = resnet18(bn_names=["normal", "pgd"]).to(device)

    # model = ResNet18DualBN().to(device)

    # checkpoint_file = "log_files/baselineexperiment-Madry_loss/baselineexperiment-Madry_loss-Madry_loss--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083614_2hgpv9hd/modle-epoch110.pt"
    # checkpoint_file = "log_files/baselineexperiment-vanilla_trades/baselineexperiment-vanilla_trades-trades_vanilla-beta_6.0--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220118083134_2qftcsmi/modle-epoch110.pt"
    
    # checkpoint_file = "/workspace/ssd2_4tb/ssl/Projects/TRADES/log_files/baselineexperiment-Madry_dual_bn/baselineexperiment-Madry_dual_bn-Madry_dual_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220119085022_2ecqsui1/model-best_normal-epoch40.pt"
    # from models.resnet_multi_bn import multi_bn_resnet18
    
    
    # checkpoint_file = "log_files/pgd_default-baselineexperiment-MART/pgd_default-baselineexperiment-MART-MART--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220121134334_3thpiutk/modle-epoch110.pt"
    # checkpoint_file = "log_files/pgd_default-baselineexperiment-Madry_mixture_bn/pgd_default-baselineexperiment-Madry_mixture_bn-Madry_mixture_bn--epochs_110-weight_decay_0.0005-lr_max_0.1-epsilon_0.06274509803921569-num_steps_10-step_size_0.00784313725490196-seed_0-/20220120063559_1cmh754s/modle-epoch110.pt"
    # from models.resnet_multi_bn_default_pgd import multi_bn_resnet18


    checkpoint_file = args.checkpoint_file
    print(checkpoint_file)
    # from models.resnet_multi_bn import multi_bn_resnet18
    # model = multi_bn_resnet18().to(device)
    # state_dict = torch.load(checkpoint_file)
    # model.load_state_dict(state_dict)

    # import ipdb; ipdb.set_trace()

    from models.resnet_multi_bn_default_pgd import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all, Disentangling_LP, multi_bn_resnet18

    if args.norm_layer == "vanilla":
        norm_layer = nn.BatchNorm2d
    elif args.norm_layer == "oct_adv":
        norm_layer = CrossTrainingBN_adv
    elif args.norm_layer == "oct_clean":
        norm_layer = CrossTrainingBN_clean
    elif args.norm_layer == "oct_all":
        norm_layer = CrossTrainingBN_all

    elif args.norm_layer == "none":
        norm_layer = nn.Identity
    elif args.norm_layer == "GN":
        norm_layer = nn.GroupNorm

    elif args.norm_layer == "IN":
        norm_layer = nn.InstanceNorm2d

    elif args.norm_layer == "LN":
        norm_layer = nn.LayerNorm


    elif args.norm_layer == "remove_affine":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, affine=False)

        norm_layer = get_no_affine_bn

    elif args.norm_layer == "Disentangling_LP":
        norm_layer = Disentangling_LP


    if not args.ACL:
        bn_names = ["pgd", "normal"]
        model = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names).cuda()
        state_dict = torch.load(checkpoint_file)
        model.load_state_dict(state_dict)
    else:
        bn_names = ["pgd", "normal"]
        from models.ACL_resnet import resnet18
        model = resnet18(bn_names=bn_names, num_classes=10).cuda()
        state_dict = torch.load(checkpoint_file)["state_dict"]

        # we need to swap the bn layer since previously ACL define dual bn as ["normal", "pgd"]
        for layer_name, _ in state_dict.items():
            if "bn_list.0" in layer_name:
                temp_weight = state_dict[layer_name].data

                state_dict[layer_name].data = state_dict[layer_name.replace("bn_list.0", "bn_list.1")].data
                state_dict[layer_name.replace("bn_list.0", "bn_list.1")].data = temp_weight

        model.load_state_dict(state_dict)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    metrics = {}
    bn_name = args.bn_name
    print(f"test on bn: {bn_name}")

    # we enforce the default bn used for testing.
    # then the bn name put with the input data will have no influence
    model.forward_bn = bn_name

    if args.eval_aa:
        test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
        metrics["AA_loss"] = test_loss
        metrics["AA_acc"] = test_accuracy * 100

    if args.eval_pgd:
        test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
        metrics["robust_loss"] = test_loss
        metrics["robust_acc"] = test_accuracy * 100

    test_loss, test_accuracy = eval_test(model, device, test_loader, 0, bn_name)
    metrics["clean_loss"] = test_loss
    metrics["clean_acc"] = test_accuracy * 100

    print(metrics)



    if args.learn_adv:
        for epoch in range(1, 2):
            # adjust learning rate for SGD
            adjust_learning_rate(optimizer, epoch)

            metrics = {"epoch":epoch}

            print('================================================================')
            # eval_train(model, device, train_loader)

            print("learn adv bn")
            learn_adv_bn(args, model, device, train_loader, optimizer, epoch, bn_name)

            if args.eval_aa:
                test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
                metrics["AA_loss"] = test_loss
                metrics["AA_acc"] = test_accuracy * 100

            if args.eval_pgd:
                test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
                metrics["robust_loss"] = test_loss
                metrics["robust_acc"] = test_accuracy * 100

            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, bn_name)
            metrics["clean_loss_normal"] = test_loss
            metrics["clean_acc_normal"] = test_accuracy * 100
            
            print('================================================================')
            print(metrics)

    if args.learn_clean:
        for epoch in range(1, 2):
            # adjust learning rate for SGD
            adjust_learning_rate(optimizer, epoch)

            metrics = {"epoch":epoch}

            print('================================================================')
            # eval_train(model, device, train_loader)

            print("learn clean bn")
            train(args, model, device, train_loader, optimizer, epoch, bn_name)
            # test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, bn_name)
            # metrics["robust_loss_normal"] = test_loss
            # metrics["robust_acc_normal"] = test_accuracy * 100

            # # test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
            # # metrics["AA_loss"] = test_loss
            # # metrics["AA_acc"] = test_accuracy * 100
            if args.eval_aa:
                test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
                metrics["AA_loss"] = test_loss
                metrics["AA_acc"] = test_accuracy * 100

            if args.eval_pgd:
                test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
                metrics["robust_loss"] = test_loss
                metrics["robust_acc"] = test_accuracy * 100

            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, bn_name)
            metrics["clean_loss_normal"] = test_loss
            metrics["clean_acc_normal"] = test_accuracy * 100
            print('================================================================')
            print(metrics)


if __name__ == '__main__':
    main()
