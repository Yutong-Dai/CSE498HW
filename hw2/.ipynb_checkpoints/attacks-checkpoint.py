import torch
from matplotlib import  pyplot as plt

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

def vis_fgsm_attack(net, images, labels, grad_sign, epsilon, M = 5, start=50, savename=None, figDir=None):
    """
        create fgsm attach on given images;
        the key parameter is epsilon, which controls the strength of attack
    """
    delta = epsilon * grad_sign
    images_adv = images + delta
    pred_clean = net(images).data.max(1)[1]
    pred_adv = net(images_adv).data.max(1)[1]
    f,ax = plt.subplots(M,3, sharex=True, sharey=True, figsize=(3,M*1.3))
    for i in range(M):
        for j in range(3):
            if j == 0:
                ax[i][0].imshow(images[i+start][0].cpu().numpy(), cmap="gray")
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_clean[i+start]}")
                plt.setp(title, color=('g' if pred_clean[i+start] == labels[i+start] else 'r'))
            elif j == 1:
                ax[i][1].imshow(delta[i+start][0].cpu().numpy(), cmap="gray")
                title = ax[i][j].set_title(r"$\delta$")
                # plt.setp(title)
            else:
                ax[i][2].imshow(images_adv[i+start][0].cpu().numpy(), cmap="gray")
                title = ax[i][j].set_title(f"{labels[i+start]} -> {pred_adv[i+start]}")
                plt.setp(title, color=('g' if pred_adv[i+start] == labels[i+start] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(f'{figDir}/mnist-cnn-{epsilon}.png')
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