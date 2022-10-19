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

from models.wideresnet import *
# from models.resnet import ResNet18, test
from models.resnet_multi_bn import multi_bn_resnet18
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
import apex.amp as amp

import logging
logger = logging.getLogger(__name__)


from training_method import train, eval_robust, eval_test

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

    BN_LAYER_LIST = ["vanilla", "oct_adv", "oct_clean", "oct_all", "none", "GN", "IN"]
    parser.add_argument('--norm_layer', default='vanilla', choices=BN_LAYER_LIST, type=str, help='Selecte batch normalizetion layer in model.')

    parser.add_argument('--dual_bn', action='store_true', default=False,
                        help='create multi bn layer')

    SUPPORT_METHOD = ["trades_vanilla", "trades_dual_bn", "trades_cat", "Madry_loss", "Madry_cat_loss", "Madry_dual_bn"]
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


    args = parser.parse_args()

    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.


    ################################################################
    # logging directory
    ################################################################
    saving_prefix = args.out_dir

    wandb.init(name=saving_prefix, project="CrossTraining", entity="kaistssl")
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



    # init model, ResNet18() can be also used here for training
    # MODEL_class = MODEL_LIST[args.model]
    # # model = MODEL_class(**args.__dict__)
    # if args.training_method in ["trades_vanilla", "trades_dual_bn"]:
    #     model = MODEL_class().cuda()
    # elif args.training_method in ["trades_oct"]:
    #     from models.resnet_multi_bn import CrossTrainingBN
    #     model = MODEL_class(norm_layer=CrossTrainingBN).cuda()
    from models.resnet_multi_bn import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all

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




    if args.dual_bn:
        bn_names = ["pgd", "normal"]
    else:
        bn_names = ["normal"]

    bn_names = ["pgd", "normal"]
    model = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names).cuda()


    # model = WideResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, optimizer = amp.initialize(model, optimizer, **amp_args)

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



        metrics = {"epoch":epoch}
        # adversarial training
        loss = train(args, model, device, train_loader, optimizer, epoch)
        metrics["train_loss"] = loss

        # evaluation on natural examples
        print(f'=============================={RUN_ID}==================================')

        if args.training_method == "Madry_dual_bn":
            # Update pgd bn gama and beta
            model_state_dict = model.state_dict()

            for layer, weight  in model_state_dict.items():
                if "bn_list.1.weight" in layer:
                    model_state_dict[layer.replace("bn_list.1.weight", "bn_list.0.weight")].data = model_state_dict[layer].data
                if "bn_list.1.bias" in layer:
                    model_state_dict[layer.replace("bn_list.1.bias", "bn_list.0.bias")].data = model_state_dict[layer].data

            model.load_state_dict(model_state_dict)


        # if args.model in ["resnet18"]:
        if not args.dual_bn:

            # eval_train(model, device, train_loader)

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

        else:
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

            test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="pgd")
            metrics["clean_loss_bn_pgd"] = test_loss
            metrics["clean_acc_bn_pgd"] = test_accuracy * 100

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
    logger.info(f'Best robust on normal bn is {best_robustness_normal:.3f}, last epoch is {metrics["robust_acc_bn_normal"]:.3f}')
    if args.dual_bn:
        logger.info(f'Best robust on pgd bn is {best_robustness_normal:.3f}, last epoch is {metrics["robust_acc_bn_pgd"]:.3f}')


    logger.info('\nFinal epoch modle testing.')


    # Remove mixed precision testing
    testset = torchvision.datasets.CIFAR10(root='/dev/shm', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names).cuda()
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    logger.info('\nClean testing.')
    metrics = {}
    
    test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="normal")
    metrics["clean_loss_bn_normal"] = test_loss
    metrics["clean_acc_bn_normal"] = test_accuracy * 100
    if args.dual_bn:
        test_loss, test_accuracy = eval_test(model, device, test_loader, epoch, test_bn="pgd")
        metrics["clean_loss_bn_pgd"] = test_loss
        metrics["clean_acc_bn_pgd"] = test_accuracy * 100


    perturb_steps = 10
    test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=perturb_steps)
    metrics["robust_pgd10_loss_bn_normal"] = test_loss
    metrics["robust_pgd10_acc_bn_normal"] = test_accuracy * 100

    if args.dual_bn:
        test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="pgd", perturb_steps=perturb_steps)
        metrics["robust_pgd10_loss_bn_pgd"] = test_loss
        metrics["robust_pgd10_acc_bn_pgd"] = test_accuracy * 100

    perturb_steps = 20
    test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="normal", perturb_steps=perturb_steps)
    metrics["robust_pgd20_loss_bn_normal"] = test_loss
    metrics["robust_pgd20_acc_bn_normal"] = test_accuracy * 100

    if args.dual_bn:
        test_loss, test_accuracy = eval_robust(model, device, test_loader, epoch, optimizer, args.epsilon, test_bn="pgd", perturb_steps=perturb_steps)
        metrics["robust_pgd20_loss_bn_pgd"] = test_loss
        metrics["robust_pgd20_acc_bn_pgd"] = test_accuracy * 100


    logger.info('Remove mixed precision testing')
    for val, key in metrics.items():
        logger.info(f"{val}: {key:.3f}:")


if __name__ == '__main__':
    main()
