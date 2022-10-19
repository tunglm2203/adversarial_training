from __future__ import print_function
import types
import setGPU
import os
import argparse
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
import logging
from datetime import datetime
from tqdm import tqdm
# from trades import trades_loss
# from utils.seed_everything import seed_everything
# from utils.generate_adv import generate_adv
# from utils.file_backup import file_backup

# from madry import madry_loss, generate_adv
# from attack_method.dual_bn_Madry_original_pgd_code_based import dual_bn_madry_loss, generate_adv
# from attack_method.dual_bn_SameNetwork import dual_bn_madry_loss, generate_adv

import wandb
import torchattacks
from models.resnet_multi_bn_default_pgd import multi_bn_resnet18
from training_method_bn_default_pgd import generate_adv, eval_robust
from autoattack import AutoAttack
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
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')

parser.add_argument('--test-num-steps', default=10, type=int,
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


parser.add_argument('--dual_bn', action='store_true', default=False)
parser.add_argument('--swap_dual_bn', action='store_true', default=False)


BN_LAYER_LIST = ["vanilla", 
                    "oct_adv", 
                    "oct_clean", 
                    "oct_all", 
                    "none", 
                    "GN", 
                    "IN", 
                    "LN", 
                    "remove_affine", 
                    "Disentangling_LP", 
                    "vanilla_momentum1", 
                    "vanilla_momentum0.9",
                    "BN2dStrongerAT",
                    "Disentangling_StatP",
                    
                    ]
parser.add_argument('--norm_layer', default='vanilla', choices=BN_LAYER_LIST, type=str, help='Selecte batch normalizetion layer in model.')


parser.add_argument('--learn_adv', action='store_true')
parser.add_argument('--learn_clean', action='store_true')

parser.add_argument('--eval_pgd', action='store_true')
parser.add_argument('--eval_aa', action='store_true')
parser.add_argument('--eval_base', action='store_true')

parser.add_argument('--ACL', action='store_true')


parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--dataset_dir', default='./', type=str, help='dataset')


BN_NAME = ["pgd", "normal"]
parser.add_argument('--bn_name', default='pgd', choices=BN_NAME)


parser.add_argument('--checkpoint_file')

args = parser.parse_args()


if "0.06274509803921569" in args.checkpoint_file:
    args.epsilon = 16 / 255.

elif "0.03137254901960784" in args.checkpoint_file:
    args.epsilon = 8 / 255.
else:    
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

# # setup data loader
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])
# trainset = torchvision.datasets.CIFAR10(root='/dev/shm', train=True, download=True, transform=transform_train)

# # trainset.data = trainset.data[:1000]
# # trainset.targets = trainset.targets[:1000]


# train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
# testset = torchvision.datasets.CIFAR10(root='/dev/shm', train=False, download=True, transform=transform_test)

# # testset.data = testset.data[:100]
# # testset.targets = testset.targets[:100]


# test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

if args.dataset == "cifar10":
    num_classes = 10
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

elif args.dataset == "imagenette":
    num_classes = 10

    stats = ((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225)) 
    train_tfms = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.RandomCrop(224, padding=4, padding_mode='reflect'), 
                            transforms.RandomHorizontalFlip(), 
                            # tt.RandomRotate
                            # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                            # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            transforms.ToTensor(), ])
    valid_tfms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    
    trainset = datasets.ImageFolder(args.dataset_dir+'/train', train_tfms)
    testset = datasets.ImageFolder(args.dataset_dir+'/val', valid_tfms)
    

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

else:
    raise "dataset not support!"


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
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"learn bn {bn_name}")):
        # TRAIN_STEP += 1
        data, target = data.to(device), target.to(device)

        # optimizer.zero_grad()
        data_adv = generate_adv(model, data, target, optimizer, step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, bn_name=bn_name)


        model(data_adv, bn_name)



def eval_test(model, device, test_loader, epoch, bn_name="pgd"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # import ipdb; ipdb.set_trace()
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


    checkpoint_file = args.checkpoint_file
    print(checkpoint_file)

    # time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # logfile = "/".join(checkpoint_file.split("/")[:-1]) + f"/testing_log_{time_stamp}.log"
    
    logfile = os.path.join(os.path.dirname(checkpoint_file), "testing_log.log")
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile
        )
    logger.info(args)
    logger.info(f"Load ckptfile: {checkpoint_file}")

    from models.resnet_multi_bn_default_pgd import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all, Disentangling_LP, BN2dStrongerAT, Disentangling_StatP

    if args.norm_layer == "vanilla":
        norm_layer = nn.BatchNorm2d

    elif args.norm_layer == "vanilla_momentum1":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, momentum=1)
        norm_layer = get_no_affine_bn

    elif args.norm_layer == "vanilla_momentum0.9":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, momentum=0.9)
        norm_layer = get_no_affine_bn

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

    elif args.norm_layer == "BN2dStrongerAT":
        norm_layer = BN2dStrongerAT

    elif args.norm_layer == "Disentangling_StatP":
        norm_layer = Disentangling_StatP


    # import ipdb; ipdb.set_trace()
    bn_names = ["pgd", "normal"]
    model = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names, num_classes=num_classes).cuda()
    if args.dataset == "cifar10":
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        model.maxpool = nn.Identity()

    model.cuda()
    state_dict = torch.load(checkpoint_file)
    if args.swap_dual_bn:
        print("swap dual bn adv and clean...")
        logger.info(f"swap dual bn adv and clean...")
        
        for_save_swap_dual_bn_dir = os.path.join(os.path.dirname(checkpoint_file), "swap_dual_bn")
        os.makedirs(for_save_swap_dual_bn_dir, exist_ok=True)
        args.checkpoint_file = os.path.join(for_save_swap_dual_bn_dir, os.path.basename(checkpoint_file))

        for key, value in state_dict.items():

            if ("bn_list.0.weight" in key) or ("bn_list.0.bias" in key):
                temp_value = copy.deepcopy(state_dict[key])
                state_dict[key] = state_dict[key.replace("bn_list.0", "bn_list.1")]
                state_dict[key.replace("bn_list.0", "bn_list.1")] = temp_value

            # if "bn_list.0.bias" in key:
            #     temp_value = copy.deepcopy(state_dict["bn_list.0.bias"])
            #     state_dict["bn_list.0.bias"] = state_dict["bn_list.1.bias"]
            #     state_dict["bn_list.1.bias"] = temp_value


    if "oct" in args.checkpoint_file:
        strict = False
        new_state_dict = {}

        for key, value in state_dict.items():
            if "gamma" in key:
                key = key.replace("gamma", "weight")
                value = torch.squeeze(value)
                new_state_dict[key] = value
            if "beta" in key:
                value = torch.squeeze(value)
                
                key = key.replace("beta", "bias")
            new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=strict)
        
        
    else:
        strict = True
        # import ipdb; ipdb.set_trace()
        model.load_state_dict(state_dict, strict=strict)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    metrics = {}
    bn_name = args.bn_name
    print(f"test on bn: {bn_name}")
    logger.info(f"test on bn: {bn_name}")
    

    # we enforce the default bn used for testing.
    # then the bn name put with the input data will have no influence
    
    model.forward_bn_name = bn_name
    

    if args.eval_base:

        logger.info(f"Basic evaluation...")

        test_loss, test_accuracy = eval_test(model, device, test_loader, 0, bn_name)
        metrics["clean_loss"] = test_loss
        metrics["clean_acc"] = test_accuracy * 100
 
        test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer=None, epsilon=args.epsilon, perturb_steps=args.test_num_steps)
        metrics["robust_loss"] = test_loss
        metrics["robust_acc"] = test_accuracy * 100

        log_path = os.path.join(os.path.dirname(checkpoint_file), "eval_aa_eval_base.log")

        adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)

        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        with torch.no_grad():
            adversary.run_standard_evaluation(x_test, y_test, bs=256)


        print(metrics)
        for key, val in metrics.items():
            logger.info(f"{key}: {val}")


        if args.dual_bn:
            if bn_name == "pgd":
                bn_name = "normal"
            else:
                bn_name = "pgd"
            print(f"test on bn: {bn_name}")
            logger.info(f"test on bn: {bn_name}")
            
            model.forward_bn_name = bn_name

            logger.info(f"Basic evaluation...")

            test_loss, test_accuracy = eval_test(model, device, test_loader, 0, bn_name)
            metrics["clean_loss"] = test_loss
            metrics["clean_acc"] = test_accuracy * 100

            if args.eval_pgd:
                # test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
                test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer=None, epsilon=args.epsilon, perturb_steps=args.test_num_steps)
                metrics["robust_loss"] = test_loss
                metrics["robust_acc"] = test_accuracy * 100

            log_path = os.path.join(os.path.dirname(checkpoint_file), f"eval_aa_eval_base_{bn_name}.log")
            
            adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)
            
            l = [x for (x, y) in test_loader]
            x_test = torch.cat(l, 0)
            l = [y for (x, y) in test_loader]
            y_test = torch.cat(l, 0)

            with torch.no_grad():
                x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)

            print(metrics)
            for key, val in metrics.items():
                logger.info(f"{key}: {val}")


    if args.learn_adv:
        logger.info(f"learn adv bn...")

        for epoch in range(1):
            # adjust learning rate for SGD
            metrics = {"epoch":epoch}

            print('================================================================')
            # eval_train(model, device, train_loader)

            print("learn adv bn")
            learn_adv_bn(args, model, device, train_loader, None, epoch, bn_name)

            # if args.eval_pgd:
                # test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
            test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer=None, epsilon=args.epsilon, perturb_steps=args.test_num_steps)

            metrics["robust_loss"] = test_loss
            metrics["robust_acc"] = test_accuracy * 100

            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, bn_name)
            metrics["clean_loss_normal"] = test_loss
            metrics["clean_acc_normal"] = test_accuracy * 100
            print('================================================================')
            print(metrics)   

        log_path = os.path.join(os.path.dirname(checkpoint_file), "eval_aa_eval_learn_adv.log")        
        adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)
        
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        with torch.no_grad():
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)



        for key, val in metrics.items():
            logger.info(f"{key}: {val}")

    if args.learn_clean:
        logger.info(f"learn clean bn...")

        for epoch in range(1):
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
            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, bn_name)
            metrics["clean_loss_normal"] = test_loss
            metrics["clean_acc_normal"] = test_accuracy * 100
            # # test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
            # # metrics["AA_loss"] = test_loss
            # # metrics["AA_acc"] = test_accuracy * 100

            # if args.eval_pgd:
                # test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, bn_name)
            test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer=None, epsilon=args.epsilon, perturb_steps=args.test_num_steps)

            metrics["robust_loss"] = test_loss
            metrics["robust_acc"] = test_accuracy * 100

            # if args.eval_aa:
            #     test_loss, test_accuracy = eval_autoattack(model, device, test_loader, 0, optimizer, "normal")
            #     metrics["AA_loss"] = test_loss
            #     metrics["AA_acc"] = test_accuracy * 100

        log_path = os.path.join(os.path.dirname(checkpoint_file), "eval_aa_eval_learn_clean.log")
        adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)
        
        l = [x for (x, y) in test_loader]
        x_test = torch.cat(l, 0)
        l = [y for (x, y) in test_loader]
        y_test = torch.cat(l, 0)

        with torch.no_grad():
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)


            print('================================================================')
            print(metrics)
        for key, val in metrics.items():
            logger.info(f"{key}: {val}")

if __name__ == '__main__':
    main()
