import torch
from matplotlib import pyplot as plt
from blackbox.simba import SimBA
import blackbox.utils as simba_utils
import math
import numpy as np
np.random.seed(0)


class ATTACKER:
    def __init__(self, type, epsilon, **kwargs):
        """ 
        type: str; FGSM / PGD
                Reference: https://adversarial-ml-tutorial.org/adversarial_examples/
        epsilon: max perturbation; l_infty norm
        """
        self.type = type
        self.epsilon = epsilon
        self.__dict__.update(kwargs)

    def attack(self, net, criterion, images, labels):
        """
            net: trained neural net work
            criterion: loss function
            images, labels: tensors being fed on to net work
        """
        # remove all existing gradients
        net.zero_grad()
        delta = torch.zeros_like(images, requires_grad=True)
        if self.type == 'PGD':
            for _ in range(self.num_iter):
                loss = criterion(net(images + delta), labels)
                loss.backward()
                # normalize the gradient to get better scaling on stepsize
                delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
                delta.grad.zero_()
            delta = delta.detach()
        elif self.type == 'FGSM':
            loss = criterion(net(images + delta), labels)
            loss.backward()
            delta = self.epsilon * delta.grad.detach().sign()
        else:
            raise ValueError(f"Invalid attack type:{self.type}")
        return delta


def fgsm_grad_sign(model, X, y, criterion):
    """ 
        Construct FGSM adversarial examples on the examples X.
        For given (image, label) pair, return the
        **sign** of gradient of loss function (criterion) with 
        respect to the image

        Reference: https://adversarial-ml-tutorial.org/adversarial_examples/
    """
    # remove all existing gradients
    model.zero_grad()
    perturb = torch.zeros_like(X, requires_grad=True)
    loss = criterion(model(X + perturb), y)
    loss.backward()
    return perturb.grad.detach().sign()


def fgsm_delta(model, X, y, criterion, epsilon):
    """ 
    Construct FGSM adversarial examples on the examples X
    epsilon: adversarial budget
    """
    grad_sign = fgsm_grad_sign(model, X, y, criterion)
    delta = epsilon * grad_sign
    return delta


def vis_fgsm_attack(net, images, labels, grad_sign, epsilon, figtype='gray',
                    inv_normalize=None, M=5, start=50, savename=None, figDir=None):
    """
        create fgsm attach on given images;
        the key parameter is epsilon, which controls the strength of attack
    """
    # generate attacks
    delta = epsilon * grad_sign
    images_adv = images + delta
    pred_clean = net(images).data.max(1)[1]
    pred_adv = net(images_adv).data.max(1)[1]
    f, ax = plt.subplots(M, 3, sharex=True, sharey=True, figsize=(3, M * 1.3))
    for i in range(M):
        for j in range(3):
            if j == 0:
                if figtype == 'gray':
                    ax[i][0].imshow(images[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][0].imshow(tensorToImg(images[i + start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_clean[i+start]}")
                plt.setp(title, color=('g' if pred_clean[i + start] == labels[i + start] else 'r'))
            elif j == 1:
                if figtype == 'gray':
                    ax[i][1].imshow(delta[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][1].imshow(tensorToImg(delta[i + start], inv_normalize))
                title = ax[i][j].set_title(r"$\delta$")
            else:
                if figtype == 'gray':
                    ax[i][2].imshow(images_adv[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][2].imshow(tensorToImg(images_adv[i + start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_adv[i+start]}")
                plt.setp(title, color=('g' if pred_adv[i + start] == labels[i + start] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{figDir}/{savename}.png')
        plt.close()


def fgsm_testacc(net, testloader, criterion, epsilon, use_cuda):
    """
        evaluate the performance of fgsm attacks
    """
    net.eval()
    correct = 0
    total = 0.0
    adv_examples = []
    adv_count = 0
    for images, labels in testloader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        # create adv examples
        delta_grad_sign = fgsm_grad_sign(net, images, labels, criterion)
        delta = epsilon * delta_grad_sign
        images_adv = images + delta
        pred_adv = net(images_adv).data.max(1)[1]
        pred_ori = net(images).data.max(1)[1]
        correct += pred_adv.eq(labels.data).sum().item()
        total += labels.size(0)
        # collect adv stats
        wrong_pred = (pred_adv != labels.data)
        # handel epsilon = 0.0 case
        if epsilon == 0.0:
            diff_pred = wrong_pred
        else:
            diff_pred = (pred_adv != pred_ori)
        adv_idx = wrong_pred & diff_pred
        adv_count += torch.sum(adv_idx).cpu()
        if len(adv_examples) < 5:
            img = images_adv[adv_idx]
            actual_label = labels[adv_idx]
            adv_pred_label = pred_adv[adv_idx]
            for i in range(min(5, torch.sum(adv_idx).cpu())):
                # (img, true_label, predicted_label)
                adv_examples.append((img[i].cpu(), actual_label[i].cpu(), adv_pred_label[i].cpu()))
    accuracy = correct / total
    return accuracy, adv_count, adv_examples[:5]


def tensorToImg(imgTensor, inv_normalize):
    inv_tensor = inv_normalize(imgTensor)
    inv_img = (inv_tensor * 255).clamp(0, 255)
    return inv_img.cpu().permute(1, 2, 0).clone().detach().to(torch.int32)


def vis_pgd_attack(net, images, labels, delta, figtype='gray',
                   inv_normalize=None, M=5, start=50, savename=None, figDir=None):
    """
        create fgsm attach on given images;
        the key parameter is epsilon, which controls the strength of attack
    """
    images_adv = images + delta
    pred_clean = net(images).data.max(1)[1]
    pred_adv = net(images_adv).data.max(1)[1]
    f, ax = plt.subplots(M, 3, sharex=True, sharey=True, figsize=(3, M * 1.3))
    for i in range(M):
        for j in range(3):
            if j == 0:
                if figtype == 'gray':
                    ax[i][0].imshow(images[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][0].imshow(tensorToImg(images[i + start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_clean[i+start]}")
                plt.setp(title, color=('g' if pred_clean[i + start] == labels[i + start] else 'r'))
            elif j == 1:
                if figtype == 'gray':
                    ax[i][1].imshow(delta[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][1].imshow(tensorToImg(delta[i + start], inv_normalize))
                title = ax[i][j].set_title(r"$\delta$")
            else:
                if figtype == 'gray':
                    ax[i][2].imshow(images_adv[i + start][0].cpu().numpy(), cmap="gray")
                else:
                    ax[i][2].imshow(tensorToImg(images_adv[i + start], inv_normalize))
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_adv[i+start]}")
                plt.setp(title, color=('g' if pred_adv[i + start] == labels[i + start] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{figDir}/{savename}.png')
        plt.close()


def pgd_delta(model, X, y, criterion, epsilon, alpha, num_iter):
    """ 
    Construct PGD adversarial examples on the examples X
    epsilon: adversarial budget
    alpha: constant stepsize along the gradient direction
    the gradient direction is scaled by its norm
    """
    delta = torch.zeros_like(X, requires_grad=True)
    for _ in range(num_iter):
        loss = criterion(model(X + delta), y)
        loss.backward()
        # normalize the gradient
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_testacc(net, testloader, criterion, epsilon, use_cuda, alpha=1e-2, num_iter=40):
    """
        evaluate the performance of fgsm attacks
    """
    net.eval()
    correct = 0
    total = 0.0
    adv_examples = []
    adv_count = 0
    for images, labels in testloader:
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        # create adv examples
        delta = pgd_delta(net, images, labels, criterion, epsilon, alpha, num_iter)
        images_adv = images + delta
        pred_adv = net(images_adv).data.max(1)[1]
        pred_ori = net(images).data.max(1)[1]
        correct += pred_adv.eq(labels.data).sum().item()
        total += labels.size(0)
        # collect adv stats
        wrong_pred = (pred_adv != labels.data)
        # handel epsilon = 0.0 case
        if epsilon == 0.0:
            diff_pred = wrong_pred
        else:
            diff_pred = (pred_adv != pred_ori)
        adv_idx = wrong_pred & diff_pred
        adv_count += torch.sum(adv_idx).cpu()
        if len(adv_examples) < 5:
            img = images_adv[adv_idx]
            actual_label = labels[adv_idx]
            adv_pred_label = pred_adv[adv_idx]
            for i in range(min(5, torch.sum(adv_idx).cpu())):
                # (img, true_label, predicted_label)
                adv_examples.append((img[i].cpu(), actual_label[i].cpu(), adv_pred_label[i].cpu()))
    accuracy = correct / total
    return accuracy, adv_count, adv_examples[:5]


def simba_cifa10(net, testset, num_imgs=None, batch_size=64, image_size=32, order='rand',
                 num_iters=-1, targeted=False, stride=7, epsilon=0.2,
                 linf_bound=0.0, pixel_attack=True, log_every=100, return_raw=False):
    """
        black box attacks; 
        assume the available information is the output from the softmax layer
        reference: https://github.com/cg563/simple-blackbox-attack
    """
    attacker = SimBA(net, 'cifar', image_size)
    freq_dims = image_size
    if not num_imgs:
        # number of test examples in cifar10
        num_imgs = len(testset)
    batch_size = min(batch_size, num_imgs)
    # get correctly predicted images
    images = torch.zeros(num_imgs, 3, image_size, image_size)
    labels = torch.zeros(num_imgs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[np.random.randint(0, len(testset) - 1)]
        preds[idx], _ = simba_utils.get_preds(net, images[idx], 'cifar', batch_size=batch_size)

    if order == 'rand':
        n_dims = 3 * freq_dims * freq_dims
    else:
        n_dims = 3 * image_size * image_size
    if num_iters > 0:
        max_iters = int(min(n_dims, num_iters))
    else:
        max_iters = int(n_dims)
    N = int(math.floor(float(num_imgs) / float(batch_size)))
    for i in range(N):
        print(f"work on {i+1:5d}/{N:5d}...")
        upper = min((i + 1) * batch_size, num_imgs)
        images_batch = images[(i * batch_size):upper]
        labels_batch = labels[(i * batch_size):upper]
        # replace true label with random target labels in case of targeted attack
        if targeted:
            labels_targeted = labels_batch.clone()
            while labels_targeted.eq(labels_batch).sum() > 0:
                labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
            labels_batch = labels_targeted
        adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
            images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound=linf_bound,
            order=order, targeted=targeted, pixel_attack=pixel_attack, log_every=log_every)
        if i == 0:
            all_adv = adv
            all_probs = probs
            all_succs = succs
            all_queries = queries
            all_l2_norms = l2_norms
            all_linf_norms = linf_norms
        else:
            all_adv = torch.cat([all_adv, adv], dim=0)
            all_probs = torch.cat([all_probs, probs], dim=0)
            all_succs = torch.cat([all_succs, succs], dim=0)
            all_queries = torch.cat([all_queries, queries], dim=0)
            all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
            all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
    result = {'adv': all_adv, 'probs': all_probs, 'succs': all_succs, 'queries': all_queries,
              'l2_norms': all_l2_norms, 'linf_norms': all_linf_norms}
    if return_raw:
        result['raw_images'] = images
        result['raw_labels'] = labels
    return result
