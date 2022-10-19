
# from trades import trades_loss
from ossaudiodev import openmixer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import apex.amp as amp

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    loss_total = 0
    n = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if args.training_method == "trades_vanilla":
            # calculate robust loss
            loss = trades_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta)

        elif args.training_method == "trades_dual_bn":
            # calculate robust loss
            loss = trades_dual_bn_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta)

        elif args.training_method == "trades_cat":
            # calculate robust loss
            loss = trades_cat_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta)

        elif args.training_method == "Madry_loss":
            # calculate robust loss
            loss = Madry_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps)

        elif args.training_method == "Madry_cat_loss":
            # calculate robust loss
            loss = Madry_cat_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps)

        elif args.training_method == "Madry_dual_bn":
            # calculate robust loss
            loss = Madry_dual_bn(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps)


        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # loss.backward()
        optimizer.step()

        loss_total += loss.item()*n
        n += target.size(0)

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    return loss_total/n


def Madry_dual_bn(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                distance='l_inf'):
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            # with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                #                        F.softmax(model(x_natural), dim=1))
            output = model(x_adv, "normal")
            loss_ce = F.cross_entropy(output, y)
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            with amp.scale_loss(loss_ce, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv, "normal")
    loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))

    # loss = loss_natural + beta * loss_robust
    loss = loss_natural

    # Update pgd bn mean and std
    model(x_natural, "pgd")
    
    return loss


def Madry_cat_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                distance='l_inf'):
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            # with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                #                        F.softmax(model(x_natural), dim=1))
            output = model(x_adv)
            loss_ce = F.cross_entropy(output, y)
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            with amp.scale_loss(loss_ce, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    x_cat = torch.cat([x_adv,x_natural])
    logits_all = model(x_cat)
    bs = x_adv.size(0)
    logits_adv = logits_all[:bs]

    logits = logits_adv
    loss = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))

    # loss = loss_natural + beta * loss_robust
    # loss = loss_natural

    return loss



def Madry_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                distance='l_inf'):
    # define KL-loss
    # criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            # with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                #                        F.softmax(model(x_natural), dim=1))
            output = model(x_adv)
            loss_ce = F.cross_entropy(output, y)
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            with amp.scale_loss(loss_ce, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
    #                                                 F.softmax(model(x_natural), dim=1))

    # loss = loss_natural + beta * loss_robust
    loss = loss_natural

    return loss


def trades_cat_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            x_cat = torch.cat([x_adv,x_natural])
            # x_cat = x_adv

            # with torch.cuda.amp.autocast():
            logits_all = model(x_cat)

            # logits_all = model(x_cat.half())
            bs = x_adv.size(0)
            logits_adv = logits_all[:bs]
            logits_clean = logits_all[bs:2*bs]

            loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                    F.softmax(logits_clean, dim=1))
 
            with amp.scale_loss(loss_kl, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss

    x_cat = torch.cat([x_adv,x_natural])
    logits_all = model(x_cat)
    bs = x_adv.size(0)
    logits_adv = logits_all[:bs]
    logits_clean = logits_all[bs:2*bs]

    logits = logits_clean
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits_clean, dim=1))

    # logits = model(x_natural, "normal")
    # loss_natural = F.cross_entropy(logits, y)
    # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv, "pgd"), dim=1),
    #                                                 F.softmax(model(x_natural, "normal"), dim=1))
    
    loss = loss_natural + beta * loss_robust

    # print(loss)

    return loss



def trades_dual_bn_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv, "pgd"), dim=1),
                                       F.softmax(model(x_natural, "normal"), dim=1))
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            with amp.scale_loss(loss_kl, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]

            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural, "normal")
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv, "pgd"), dim=1),
                                                    F.softmax(model(x_natural, "normal"), dim=1))
    
    loss = loss_natural + beta * loss_robust

    # print(loss)

    return loss


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            # grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            with amp.scale_loss(loss_kl, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, [x_adv],
                                    retain_graph=False, create_graph=False)[0]


            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))

    # import ipdb; ipdb.set_trace()
    loss = loss_natural + beta * loss_robust
    return loss




def eval_robust(model, device, test_loader,epoch,optimizer, epsilon, test_bn="pgd", perturb_steps=20):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_adv = generate_adv(model, data, target, optimizer, step_size=epsilon/4, epsilon=epsilon, perturb_steps=perturb_steps, bn_name=test_bn)

        if test_bn :
            output = model(data_adv, test_bn)
        else:
            output = model(data_adv)

        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: (robust) loss: {:.4f}, Robust accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy


import torchattacks
def eval_robust_auto_attack(model, device, test_loader,epoch,optimizer, epsilon, test_bn="pgd"):
    model.eval()
    test_loss = 0
    correct = 0
    # with torch.no_grad():
    aa_generateor = torchattacks.AutoAttack(model, eps=epsilon)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # data_adv = generate_adv(model, data, target, optimizer, step_size=epsilon/4, epsilon=epsilon, perturb_steps=20, bn_name=test_bn)

        data_adv = aa_generateor(data, target)

        if test_bn :
            output = model(data_adv, test_bn)
        else:
            output = model(data_adv)

        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: (robust) loss: {:.4f}, Robust accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_accuracy = correct / len(test_loader.dataset)

    return test_loss, test_accuracy




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
        if optimizer:
            with amp.scale_loss(cost, optimizer) as scaled_loss:
                grad = torch.autograd.grad(scaled_loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]

        # grad = torch.autograd.grad(cost, adv_images,
        #                             retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + step_size*grad.sign()
        delta = torch.clamp(adv_images - x_natural, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(x_natural + delta, min=0, max=1).detach()

    return adv_images


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
    test_loss /= len(test_loader.dataset)
    print('Test: loss: {:.4f}, Clean accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    # writer.add_scalar(f'Accuracy/test_{test_bn}',test_accuracy , epoch)
    return test_loss, test_accuracy
