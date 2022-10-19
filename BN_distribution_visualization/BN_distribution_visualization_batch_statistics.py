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


import shutil
import glob
from datetime import datetime

# import wandb
import apex.amp as amp

import logging
logger = logging.getLogger(__name__)


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
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
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
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            break
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
    parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--resume_dir', type=str, default="./", help='resume training')



    args = parser.parse_args()

    args.epsilon = args.epsilon / 255.
    args.step_size = args.step_size / 255.


    ################################################################
    # logging directory
    ################################################################
    args.out_dir = "pgd_default-" + args.out_dir 
    saving_prefix = args.out_dir

    # wandb.init(name=saving_prefix, project="CrossTraining", entity="kaistssl")
    # wandb.config.update(args)
    # RUN_ID = wandb.run.id

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

    # args.out_dir = os.path.join(saving_prefix, args.out_dir, f"{time_stamp}_{RUN_ID}")


    # args.out_dir = os.path.join("./log_files", args.out_dir) 
    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)
    
    # logfile = os.path.join(args.out_dir, 'output.log')
    # if os.path.exists(logfile):
    #     os.remove(logfile)

    # logging.basicConfig(
    #     format='[%(asctime)s] - %(message)s',
    #     datefmt='%Y/%m/%d %H:%M:%S',
    #     level=logging.INFO,
    #     filename=logfile)
    # logger.info(args)

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
    # pathname = "./*.py"
    # files = glob.glob(pathname, recursive=True)

    # for file in files:
    #     dest_fpath = os.path.join( args.out_dir, "code", file.split("/")[-1])
    #     try:
    #         shutil.copy(file, dest_fpath)
    #     except IOError as io_err:
    #         os.makedirs(os.path.dirname(dest_fpath))
    #         shutil.copy(file, dest_fpath)
    
    # shutil.copytree("./models", os.path.join(args.out_dir, "code", "models"))
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
    from resnet_multi_bn_default_pgd import CrossTrainingBN_clean, CrossTrainingBN_adv, CrossTrainingBN_all, __saving_bn_batch_statistics__

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

    elif args.norm_layer == "remove_affine":
        def get_no_affine_bn(channel):
            return nn.BatchNorm2d(channel, affine=False)

        norm_layer = get_no_affine_bn


    bn_names = ["pgd", "normal"]
    model = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names).cuda()



    if args.resume:
        load_model_state_dict = torch.load(args.resume_dir)

        model.load_state_dict(load_model_state_dict)
        
        # Specific things for some method
        if args.training_method == "Finetune_BN_on_clean":

            for name, param in model.named_parameters():

                if "bn" not in name:
                    param.requires_grad = False
            # @torch.no_grad()
            # def init_weights(m: nn.Module):
            #     print(m)
            #     if type(m) != nn.BatchNorm2d:
            #         m.freeze
            #         m.weight.fill_(1.0)
            #         print(m.weight)


    # target_layer = [("layer3.1.bn2.bn_list.0.running_mean", "layer3.1.bn2.bn_list.0.running_var"), 
    #                 ("layer3.0.bn1.bn_list.0.running_mean", "layer3.0.bn1.bn_list.0.running_var"),
    #                 ]

    # if args.dual_bn:
    #     target_layer.extend([("layer3.1.bn2.bn_list.1.running_mean", "layer3.1.bn2.bn_list.1.running_var"), 
    #                 ("layer3.0.bn1.bn_list.1.running_mean", "layer3.0.bn1.bn_list.1.running_var")
    #                 ])

    model_name = args.resume_dir.split("/")[1].split("-")[-1]


    if args.dual_bn:
        # target_layer = [("layer3.1.bn2.bn_list.0.running_mean", "layer3.1.bn2.bn_list.1.running_mean"), 
        #                 ("layer3.0.bn1.bn_list.0.running_mean", "layer3.0.bn1.bn_list.1.running_mean"),

        #                 ("layer3.1.bn2.bn_list.0.running_var", "layer3.1.bn2.bn_list.1.running_var"),
        #                 ("layer3.0.bn1.bn_list.0.running_var", "layer3.0.bn1.bn_list.1.running_var"),

        #                 ]

        model_for_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=["pgd", "normal"]).cuda()
        model_state_dict = torch.load(args.resume_dir)

        model_for_test.load_state_dict(model_state_dict, strict=False)


        # import ipdb; ipdb.set_trace()

        target_layer = [("layer1.1.bn2.bn_list.0.running_mean", "layer1.1.bn2.bn_list.1.running_mean"), 
                        ("layer1.0.bn1.bn_list.0.running_mean", "layer1.0.bn1.bn_list.1.running_mean"),

                        ("layer1.1.bn2.bn_list.0.running_var", "layer1.1.bn2.bn_list.1.running_var"),
                        ("layer1.0.bn1.bn_list.0.running_var", "layer1.0.bn1.bn_list.1.running_var"),

                        ]

        test_loss, test_accuracy = eval_test(model_for_test, device, test_loader, 0, test_bn="pgd")

        print(__saving_bn_batch_statistics__)

        test_loss, test_accuracy = eval_test(model_for_test, device, test_loader, 0, test_bn="normal")


        model_state_dict = torch.load(args.resume_dir)
        import matplotlib.pyplot as plt

        for (pgd, normal) in target_layer:
            
            pgd_tensor = model_state_dict[pgd]
            normal_tensor = model_state_dict[normal]

            # data to plot
            n_groups = 20
            pgd_tensor = pgd_tensor[:n_groups].cpu()
            normal_tensor = normal_tensor[:n_groups].cpu()

            # create plot
            fig, ax = plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.35
            opacity = 0.8

            rects1 = plt.bar(index, pgd_tensor, bar_width,
            alpha=opacity,
            color='b',
            label='pgd_tensor')

            rects2 = plt.bar(index + bar_width, normal_tensor, bar_width,
            alpha=opacity,
            color='g',
            label='normal_tensor')

            # plt.xlabel('Person')
            # plt.ylabel('Scores')
            # plt.title('Scores by person')
            # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.savefig(f"BN_distribution_visualization/figure/{model_name}_{pgd}.png")

    else:


        # model_state_dict = model.state_dict()
        model_for_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=["pgd", "normal", "adv"]).cuda()
        model_state_dict = torch.load(args.resume_dir)

        model_for_test.load_state_dict(model_state_dict, strict=False)

        model_state_dict = model_for_test.state_dict()

        # copy gama and beta
        for layer, weight  in model_state_dict.items():
            if "bn_list.1.weight" in layer:
                model_state_dict[layer.replace("bn_list.1.weight", "bn_list.0.weight")].data = model_state_dict[layer].data
                model_state_dict[layer.replace("bn_list.1.weight", "bn_list.2.weight")].data = model_state_dict[layer].data

            if "bn_list.1.bias" in layer:
                model_state_dict[layer.replace("bn_list.1.bias", "bn_list.0.bias")].data = model_state_dict[layer].data
                model_state_dict[layer.replace("bn_list.1.bias", "bn_list.2.bias")].data = model_state_dict[layer].data

        model_for_test.load_state_dict(model_state_dict, strict=False)
        
        model_for_test.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # learn clean bn
            model_for_test(data, "pgd")

            # learn adv bn
            model_for_test.eval()
            batch_size = len(data)
            # generate adversarial example
            x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            for _ in range(10):
                x_adv.requires_grad_()
                # with torch.enable_grad():
                    # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                    #                        F.softmax(model(data), dim=1))
                output = model_for_test(x_adv, "normal")
                loss_ce = F.cross_entropy(output, target)
                # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                grad = torch.autograd.grad(loss_ce, [x_adv],
                                        retain_graph=False, create_graph=False)[0]

                x_adv = x_adv.detach() + args.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, data - args.epsilon), data + args.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

            model_for_test.train()

            model_for_test(x_adv, "adv")



        # target_layer = [("layer3.1.bn2.bn_list.0.running_mean", "layer3.1.bn2.bn_list.1.running_mean", "layer3.1.bn2.bn_list.2.running_mean"), 
        #                 ("layer3.0.bn1.bn_list.0.running_mean", "layer3.0.bn1.bn_list.1.running_mean", "layer3.0.bn1.bn_list.2.running_mean"),

        #                 ("layer3.1.bn2.bn_list.0.running_var", "layer3.1.bn2.bn_list.1.running_var", "layer3.1.bn2.bn_list.2.running_var"),
        #                 ("layer3.0.bn1.bn_list.0.running_var", "layer3.0.bn1.bn_list.1.running_var", "layer3.0.bn1.bn_list.2.running_var"),

        #                 ]


        target_layer = [("layer1.1.bn2.bn_list.0.running_mean", "layer1.1.bn2.bn_list.1.running_mean", "layer1.1.bn2.bn_list.2.running_mean"), 
                        ("layer1.0.bn1.bn_list.0.running_mean", "layer1.0.bn1.bn_list.1.running_mean", "layer1.0.bn1.bn_list.2.running_mean"),

                        ("layer1.1.bn2.bn_list.0.running_var", "layer1.1.bn2.bn_list.1.running_var", "layer1.1.bn2.bn_list.2.running_var"),
                        ("layer1.0.bn1.bn_list.0.running_var", "layer1.0.bn1.bn_list.1.running_var", "layer1.0.bn1.bn_list.2.running_var"),

                        ]

        # model_state_dict = torch.load(args.resume_dir)
        model_state_dict = model_for_test.state_dict()

        import matplotlib.pyplot as plt

        # because of this vanilla trades is trainded with old implementation, we need to change its name. previous default bn is normal
        for (normal, pgd, adv) in target_layer:
            
            pgd_tensor = model_state_dict[pgd]
            normal_tensor = model_state_dict[normal]
            adv_tensor = model_state_dict[adv]


            # data to plot
            n_groups = 20
            pgd_tensor = pgd_tensor[:n_groups].cpu()
            normal_tensor = normal_tensor[:n_groups].cpu()
            adv_tensor = adv_tensor[:n_groups].cpu()


            # create plot
            fig, ax = plt.subplots()
            index = np.arange(n_groups)
            bar_width = 0.25
            opacity = 0.8


            rects1 = plt.bar(index, pgd_tensor, bar_width,
            alpha=opacity,
            color='b',
            label='original_tensor')

            rects2 = plt.bar(index + bar_width, normal_tensor, bar_width,
            alpha=opacity,
            color='g',
            label='normal_tensor')
            
            rects2 = plt.bar(index + 2*bar_width, adv_tensor, bar_width,
            alpha=opacity,
            color='r',
            label='adv_tensor')

            # plt.xlabel('Person')
            # plt.ylabel('Scores')
            # plt.title('Scores by person')
            # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
            plt.legend()

            plt.tight_layout()
            plt.show()

            plt.savefig(f"BN_distribution_visualization/figure/{model_name}_{pgd}.png")


    # else:


    #     # model_state_dict = model.state_dict()


    #     model_state_dict = torch.load(args.resume_dir)

    #     # copy gama and beta
    #     for layer, weight  in model_state_dict.items():
    #         if "bn_list.1.weight" in layer:
    #             model_state_dict[layer.replace("bn_list.1.weight", "bn_list.0.weight")].data = model_state_dict[layer].data
    #         if "bn_list.1.bias" in layer:
    #             model_state_dict[layer.replace("bn_list.1.bias", "bn_list.0.bias")].data = model_state_dict[layer].data

    #     model_for_test = multi_bn_resnet18(norm_layer=norm_layer, bn_names=bn_names).cuda()

    #     model_for_test.load_state_dict(model_state_dict)
        
    #     model_for_test.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         data, target = data.to(device), target.to(device)

    #         model_for_test(data, "pgd")


    #     target_layer = [("layer3.1.bn2.bn_list.0.running_mean", "layer3.1.bn2.bn_list.1.running_mean"), 
    #                     ("layer3.0.bn1.bn_list.0.running_mean", "layer3.0.bn1.bn_list.1.running_mean"),

    #                     ("layer3.1.bn2.bn_list.0.running_var", "layer3.1.bn2.bn_list.1.running_var"),
    #                     ("layer3.0.bn1.bn_list.0.running_var", "layer3.0.bn1.bn_list.1.running_var"),

    #                     ]


    #     # model_state_dict = torch.load(args.resume_dir)
    #     model_state_dict = model_for_test.state_dict()

    #     import matplotlib.pyplot as plt

    #     for (pgd, normal) in target_layer:
            
    #         pgd_tensor = model_state_dict[pgd]
    #         normal_tensor = model_state_dict[normal]

    #         # data to plot
    #         n_groups = 20
    #         pgd_tensor = pgd_tensor[:n_groups].cpu()
    #         normal_tensor = normal_tensor[:n_groups].cpu()

    #         # create plot
    #         fig, ax = plt.subplots()
    #         index = np.arange(n_groups)
    #         bar_width = 0.35
    #         opacity = 0.8


    #         # because of this vanilla trades is trainded with old implementation, we need to change its name. previous default bn is normal
    #         rects1 = plt.bar(index, pgd_tensor, bar_width,
    #         alpha=opacity,
    #         color='b',
    #         label='normal_tensor')

    #         rects2 = plt.bar(index + bar_width, normal_tensor, bar_width,
    #         alpha=opacity,
    #         color='g',
    #         label='pgd_tensor')

    #         # plt.xlabel('Person')
    #         # plt.ylabel('Scores')
    #         # plt.title('Scores by person')
    #         # plt.xticks(index + bar_width, ('A', 'B', 'C', 'D'))
    #         plt.legend()

    #         plt.tight_layout()
    #         plt.show()

    #         plt.savefig(f"BN_distribution_visualization/figure/{model_name}_{pgd}.png")
    

    return


if __name__ == '__main__':
    main()
