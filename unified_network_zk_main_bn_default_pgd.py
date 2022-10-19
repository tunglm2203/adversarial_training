import setGPU
import copy
import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from models.wideresnet import *
# from models.resnet import ResNet18, test
# from models.resnet_multi_bn import multi_bn_resnet18
from models.resnet_multi_bn_default_pgd import multi_bn_resnet18, multi_bn_resnet50

# from models.resnet_bn_change_AlwaysUseBNofCE import ResNet18_always_use_bn_of_ce
# MODEL_LIST = {
#     "resnet18": ResNet18,
#     "multi_bn_resnet18": multi_bn_resnet18,
#     "resnet18_always_use_bn_of_ce": ResNet18_always_use_bn_of_ce,

# }


import shutil
import glob
from datetime import datetime

import wandb
# import apex.amp as amp
import torchattacks

import logging
logger = logging.getLogger(__name__)


from training_method_bn_default_pgd import train, eval_robust, eval_test
from autoattack import AutoAttack


def eval_autoattack(model, device, test_loader,epsilon, bn_name):
    model.eval()
    test_loss = 0
    correct = 0
    # change_bn_mode(model, bn_name)

    model.forward_bn = bn_name

    autoattack = torchattacks.AutoAttack(model, eps=epsilon)

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


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


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
    parser.add_argument('--lr_clean', type=float, default=0.1, metavar='LR',
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

    parser.add_argument('--clean_bn_range', default=-1, type=float)

    # parser.add_argument('--model', default='resnet18', choices=MODEL_LIST.keys(), type=str, help='Selecte training model.')

    BN_LAYER_LIST = ["vanilla", 
                        "oct_adv", 
                        "oct_clean", 
                        "oct_all", 
                        "dual_bn_oct_clean",
                        "none", 
                        "GN", 
                        "IN", 
                        "LN", 
                        "remove_affine", 
                        "Disentangling_LP", 
                        "vanilla_momentum1", 
                        "BN2dStrongerAT",
                        "BN2dStrongerAT_momentum1",
                        "Disentangling_StatP",
                        "CrossTraining_DualBN_swap",
                        ]
    parser.add_argument('--norm_layer', default='vanilla', choices=BN_LAYER_LIST, type=str, help='Selecte batch normalizetion layer in model.')

    parser.add_argument('--dual_bn', action='store_true', default=False,
                        help='create multi bn layer')

    SUPPORT_METHOD = ["trades_vanilla", 
                        "trades_dual_bn",
                        "trades_cat", 
                        "Madry_loss", 
                        "Madry_cat_loss", 
                        "Madry_dual_bn", 
                        "Madry_mixture_bn", 
                        "Madry_dual_bn_fix_cnn_clean_bn",
                        "Madry_dual_bn_fix_cnn_clean_affine",
                        "Hybrid_single_bn", 
                        "Hybrid_cat",
                        "Hybrid_dual_bn_oct_clean",
                        "Finetune_BN_on_clean",
                        "MART",
                        "MART_dual_bn",
                        "MART_cat",
                        "trades_ae2gt_vanilla",
                        "trades_ae2gt_dual_bn",
                        "trades_ae2gt_cat",

                        "trades_bat_loss",
                        "trades_mid_step5_loss"
                        ]
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
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--resume_dir', type=str, default="./", help='resume training')

    parser.add_argument('--wandb_project', type=str, default="CrossTraining")
    parser.add_argument('--wandb_entity', type=str, default="tunglm")

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--dataset_dir', default='../data', type=str, help='dataset directory')

    parser.add_argument('--architecture', default='resnet18', type=str, help='network')

    args = parser.parse_args()

    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.


    ################################################################
    # logging directory
    ################################################################
    args.out_dir = "BAT-" + args.out_dir 
    saving_prefix = args.out_dir

    wandb.init(name=saving_prefix, project=args.wandb_project, entity=args.wandb_entity)
    wandb.config.update(args)
    RUN_ID = wandb.run.id

    if args.training_method == "trades_vanilla":
        args.out_dir += f"-trades_vanilla-beta_{args.beta}-"
    elif args.training_method == "trades_dual_bn":
        # assert args.model == "multi_bn_resnet18"
        args.out_dir += f"-trades_dual_bn-"
    elif args.training_method == "trades_cat":
        # assert args.model == "multi_bn_resnet18"
        args.out_dir += f"-trades_cat-"

    else:
        args.out_dir += f"-{args.training_method}-"


    args.out_dir = args.out_dir + f"-epochs_{args.epochs}-weight_decay_{args.weight_decay}-lr_max_{args.lr}-epsilon_{args.epsilon}-num_steps_{args.num_steps}-step_size_{args.step_size}-seed_{args.seed}-"
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    args.out_dir = os.path.join(saving_prefix, args.out_dir, f"{time_stamp}_{RUN_ID}")


    args.out_dir = os.path.join("./log_files", args.out_dir) 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    ################################################################
    # logging directory
    ################################################################

    # random seed setting
    import random
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



    ################################################################
    # training code saving
    ################################################################
    pathname = "./*.py"
    files = glob.glob(pathname, recursive=True)

    for file in files:
        dest_fpath = os.path.join( args.out_dir, "code", file.split("/")[-1])
        try:
            shutil.copy(file, dest_fpath)
        except IOError as io_err:
            os.makedirs(os.path.dirname(dest_fpath))
            shutil.copy(file, dest_fpath)
    
    shutil.copytree("./models", os.path.join(args.out_dir, "code", "models"))
    ################################################################
    # training code saving
    ################################################################


    # settings

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # setup data loader
    
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
    

    from models.resnet_multi_bn_default_pgd import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all, Disentangling_LP, BN2dStrongerAT, Disentangling_StatP, BN2dStrongerATMomentum1, CrossTraining_DualBN_swap

    if args.norm_layer == "vanilla":
        norm_layer = nn.BatchNorm2d

    elif args.norm_layer == "vanilla_momentum1":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, momentum=1)
        norm_layer = get_no_affine_bn

    elif args.norm_layer == "oct_adv":
        norm_layer = CrossTrainingBN_adv
    elif args.norm_layer == "oct_clean":
        norm_layer = CrossTrainingBN_clean
    elif args.norm_layer == "oct_all":
        norm_layer = CrossTrainingBN_all
    elif args.norm_layer == "dual_bn_oct_clean":
        norm_layer = "dual_bn_oct_clean"
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

    elif args.norm_layer == "BN2dStrongerAT_momentum1":
        norm_layer = BN2dStrongerATMomentum1

    elif args.norm_layer == "Disentangling_StatP":
        norm_layer = Disentangling_StatP

    elif args.norm_layer == "CrossTraining_DualBN_swap":
        norm_layer = CrossTraining_DualBN_swap

    bn_names = ["pgd", "normal"]
    if args.architecture == "resnet18":
        architecture = multi_bn_resnet18
    elif args.architecture == "resnet50":
        architecture = multi_bn_resnet50
    
    else:
        raise "Not supportted architecture!"

    # num_classes =1000
    model = architecture(norm_layer=norm_layer, bn_names=bn_names, num_classes=num_classes)

    if args.dataset == "cifar10":
        # model.conv1 = nn.Conv2d(
        #     3, 64, kernel_size=3, stride=1, padding=2, bias=False
        # )
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    model.cuda()

    if args.resume:
        load_model_state_dict = torch.load(args.resume_dir)

        model.load_state_dict(load_model_state_dict)
        
        # Specific things for some method
        if args.training_method == "Finetune_BN_on_clean":

            for name, param in model.named_parameters():

                if "bn" not in name:
                    param.requires_grad = False

    # model = WideResNet().to(device)
   
    if args.training_method == "Madry_dual_bn_fix_cnn_clean_bn" or args.training_method =="Madry_dual_bn_fix_cnn_clean_affine":
        clean_affine_params = []
        model_params = []
        for name, param in model.named_parameters():
            if "bn_list.1" in name:
                clean_affine_params += [{'params': [param]}]
            else:
                model_params += [{'params': [param]}]
        optimizer = optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer_clean_affine = optim.SGD(clean_affine_params, lr=args.lr_clean, momentum=args.momentum, weight_decay=args.weight_decay)

    elif args.training_method == "Madry_mixture_bn":
        if args.clean_bn_range >= 0:
            model_params = []
            for name, param in model.named_parameters():
                if "bn_list.1" in name:
                    if args.clean_bn_range > 0 :
                        pass
                    else:
                        continue
                if "bn_list.0" in name:
                    continue

                model_params += [{'params': [param]}]            
            optimizer = optim.SGD(model_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # # do not update bn pgd
    # if args.clean_bn_range > 0:
    #     print("========Do not update bn pgd========")
    #     for name, param in model.named_parameters():
    #         if "bn_list.0" in name:
    #             param.requires_grad = False
    #             print(name)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    # model, optimizer = amp.initialize(model, optimizer, **amp_args)

    # test_loss, test_accuracy = eval_robust(model, device, test_loader, 0, optimizer, args.epsilon, test_bn="normal")

    best_robustness_normal = 0
    best_robustness_pgd = 0

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        if args.lr_decay == "vanilla":
            adjust_learning_rate(optimizer, epoch, args.lr)
        elif args.lr_decay == "fast55":
            adjust_learning_rate_fast_55(optimizer, epoch, args.lr)
        elif args.lr_decay == "decay7590":
            adjust_learning_rate_epoch100_decay_7590(optimizer, epoch, args.lr)
        
        if args.training_method == "Madry_dual_bn_fix_cnn_clean_bn" or args.training_method =="Madry_dual_bn_fix_cnn_clean_affine":
            if args.lr_decay == "vanilla":
                adjust_learning_rate(optimizer_clean_affine, epoch, args.lr_clean)
            elif args.lr_decay == "fast55":
                adjust_learning_rate_fast_55(optimizer_clean_affine, epoch, args.lr_clean)
            elif args.lr_decay == "decay7590":
                adjust_learning_rate_epoch100_decay_7590(optimizer_clean_affine, epoch, args.lr_clean)

        metrics = {"epoch":epoch}
        metrics["lr"] = optimizer.state_dict()['param_groups'][0]['lr']
        if args.training_method == "Madry_dual_bn_fix_cnn_clean_bn" or args.training_method =="Madry_dual_bn_fix_cnn_clean_affine":
            metrics["lr_clean"] = optimizer_clean_affine.state_dict()['param_groups'][0]['lr']

        # adversarial training
        if args.training_method == "Madry_dual_bn_fix_cnn_clean_bn" or args.training_method =="Madry_dual_bn_fix_cnn_clean_affine":
            loss = train(args, model, device, train_loader, optimizer, epoch, optimizer_clean_affine)
        else:
            loss = train(args, model, device, train_loader, optimizer, epoch)
        metrics["train_loss"] = loss

        # evaluation on natural examples
        print(f'=============================={RUN_ID}==================================')


        test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="pgd", perturb_steps=10)
        metrics["robust_loss_bn_pgd"] = test_loss
        metrics["robust_acc_bn_pgd"] = test_accuracy * 100
        if test_accuracy > best_robustness_pgd:
            best_robustness_pgd = test_accuracy
            # save checkpoint
            torch.save(model.state_dict(),
                        os.path.join(args.out_dir, 'model-best_pgd-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                        os.path.join(args.out_dir, 'opt-best_pgd-epoch{}.tar'.format(epoch)))

        test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="pgd")
        metrics["clean_loss_bn_pgd"] = test_loss
        metrics["clean_acc_bn_pgd"] = test_accuracy * 100

        if args.dual_bn:

            test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=10)
            metrics["robust_loss_bn_normal"] = test_loss
            metrics["robust_acc_bn_normal"] = test_accuracy * 100
            if test_accuracy > best_robustness_normal:
                best_robustness_normal = test_accuracy
                # save checkpoint
                torch.save(model.state_dict(),
                            os.path.join(args.out_dir, 'model-best_normal-epoch{}.pt'.format(epoch)))
                torch.save(optimizer.state_dict(),
                            os.path.join(args.out_dir, 'opt-best_normal-epoch{}.tar'.format(epoch)))


            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="normal")
            metrics["clean_loss_bn_normal"] = test_loss
            metrics["clean_acc_bn_normal"] = test_accuracy * 100



        print(f'================================={RUN_ID}===============================')
        wandb.log(metrics)
        for val, key in metrics.items():
            logger.info(f"{val}: {key:.3f}:")
            
    # save final epoch checkpoint
    torch.save(model.state_dict(),
                os.path.join(args.out_dir, 'modle-epoch{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(),
                os.path.join(args.out_dir, 'opt-epoch{}.tar'.format(epoch)))


    logger.info('\nTraining finished!')
    logger.info(f'Best robust on pgd bn is {best_robustness_normal:.3f}, last epoch is {metrics["robust_acc_bn_pgd"]:.3f}')
    if args.dual_bn:

        logger.info(f'Best robust on normal bn is {best_robustness_normal:.3f}, last epoch is {metrics["robust_acc_bn_normal"]:.3f}')

    logger.info('\nFinal epoch modle testing.')


    # Remove mixed precision testing
    # testset = torchvision.datasets.CIFAR10(root='/dev/shm', train=False, download=True, transform=transform_test)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    if args.dataset == "cifar10":
        num_classes = 10

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

        testset = torchvision.datasets.CIFAR10(root='/dev/shm', train=False, download=True, transform=transform_test)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    elif args.dataset == "imagenette":
        num_classes = 10

        valid_tfms = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        
        testset = datasets.ImageFolder(args.dataset_dir+'/val', valid_tfms)
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names, num_classes=num_classes).cuda()

    if args.dataset == "cifar10":
        model_test.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        model_test.maxpool = nn.Identity()
    model_test.cuda()

    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    logger.info('\nClean testing.')
    metrics = {}
    
    test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="pgd")
    metrics["clean_loss_bn_pgd"] = test_loss
    metrics["clean_acc_bn_pgd"] = test_accuracy * 100

    if args.dual_bn:
        test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="normal")
        metrics["clean_loss_bn_normal"] = test_loss
        metrics["clean_acc_bn_normal"] = test_accuracy * 100
        

    perturb_steps = 10
    test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="pgd", perturb_steps=perturb_steps)
    metrics["robust_pgd10_loss_bn_pgd"] = test_loss
    metrics["robust_pgd10_acc_bn_pgd"] = test_accuracy * 100
    if args.dual_bn:

        test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=perturb_steps)
        metrics["robust_pgd10_loss_bn_normal"] = test_loss
        metrics["robust_pgd10_acc_bn_normal"] = test_accuracy * 100

    perturb_steps = 20
    test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="pgd", perturb_steps=perturb_steps)
    metrics["robust_pgd20_loss_bn_pgd"] = test_loss
    metrics["robust_pgd20_acc_bn_pgd"] = test_accuracy * 100
    if args.dual_bn:

        test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=perturb_steps)
        metrics["robust_pgd20_loss_bn_normal"] = test_loss
        metrics["robust_pgd20_acc_bn_normal"] = test_accuracy * 100

    logger.info('Remove mixed precision testing')
    for val, key in metrics.items():
        logger.info(f"{val}: {key:.3f}:")

    # log_path = os.path.join(args.out_dir, "eval_aa_eval_after_training_bn_pgd.log")
    # adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)

    # l = [x for (x, y) in test_loader]
    # x_test = torch.cat(l, 0)
    # l = [y for (x, y) in test_loader]
    # y_test = torch.cat(l, 0)

    # with torch.no_grad():
    #     adversary.run_standard_evaluation(x_test, y_test, bs=256)

    # if args.dual_bn:

    #     test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=perturb_steps)
    #     metrics["robust_pgd20_loss_bn_normal"] = test_loss
    #     metrics["robust_pgd20_acc_bn_normal"] = test_accuracy * 100

    #     log_path = os.path.join(args.out_dir, "eval_aa_eval_after_training_bn_pgd.log")
    #     adversary = AutoAttack(model, norm="Linf", eps=args.epsilon, log_path=log_path, version="standard", seed=args.seed)

    #     l = [x for (x, y) in test_loader]
    #     x_test = torch.cat(l, 0)
    #     l = [y for (x, y) in test_loader]
    #     y_test = torch.cat(l, 0)

    #     with torch.no_grad():
    #         adversary.run_standard_evaluation(x_test, y_test, bs=256)




def generate_adv(model,
                x_natural,
                y,
                optimizer=None,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                bn_name="pgd",
                ):

    model.eval()

    loss = nn.CrossEntropyLoss()
    adv_images = x_natural.clone().detach()

    # if self.random_start:
        # Starting at a uniformly random point
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(perturb_steps):
        adv_images.requires_grad = True

        # import ipdb; ipdb.set_trace()
        if bn_name:
            # with torch.cuda.amp.autocast(): outputs = model(adv_images, bn_name)
            outputs = model(adv_images, bn_name)
        else:
            # with torch.cuda.amp.autocast(): outputs = model(adv_images)
            outputs = model(adv_images)

        # Calculate loss
        cost = loss(outputs, y)

        # Update adversarial images
        # if optimizer:
        #     with amp.scale_loss(cost, optimizer) as scaled_loss:
        #         grad = torch.autograd.grad(scaled_loss, adv_images,
        #                             retain_graph=False, create_graph=False)[0]
        # else:
        #     grad = torch.autograd.grad(cost, adv_images,
        #                             retain_graph=False, create_graph=False)[0]

        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]


        adv_images = adv_images.detach() + step_size*grad.sign()
        delta = torch.clamp(adv_images - x_natural, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(x_natural + delta, min=0, max=1).detach()

    return adv_images

if __name__ == '__main__':
    main()
