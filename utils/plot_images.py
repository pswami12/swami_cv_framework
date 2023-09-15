import torch
import matplotlib.pyplot as plt
from .dataloader import dataloaders
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def plot_pred_images(data, classes, r=5,c=4):
    fig, axs = plt.subplots(r,c,figsize=(15,10))
    fig.tight_layout()
    # fig.suptitle(title)
    for i in range(r):
        for j in range(c):
            axs[i][j].axis('off')
            axs[i][j].set_title(f"Target: {classes[int(data[(i*c)+j]['target'])]}\nPred: {classes[int(data[(i*c)+j]['pred'])]}")
            axs[i][j].imshow(np.array(inverse_normalize(data[(i*c)+j]['data']).squeeze().cpu().permute(1,2,0)))
    plt.tight_layout()
    plt.show()


def inverse_normalize(tensor, mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)):
  # Not mul by 255 here
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def image_prediction(model, n=20,r=5,c=4, misclassified = True):
    model.eval()

    _, test_loader, classes, _ = dataloaders(val_batch_size=1)

    # print(classes)
    wrong = []
    right = []
    i, j = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).item()
            if (not correct) and i <= n - 1:
                wrong.append({
                    "data": data,
                    "target": target.item(),
                    "pred": pred.item()
                })
                i+=1
            elif j <= n - 1:
                right.append({
                    "data": data,
                    "target": target.item(),
                    "pred": pred.item()       
                })
                j+=1

    if misclassified:
        if len(wrong) >= n:
            plot_pred_images(wrong, classes, r, c)
        else:
            plot_pred_images(wrong, classes, r, int(len(wrong)/r))
    else:
        if len(right) >= n:
            plot_pred_images(right, classes, r, c)
        else:
            plot_pred_images(right, classes, r, int(len(right)/r))
