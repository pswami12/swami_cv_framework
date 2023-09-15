import torch

from models import *
from utils import *

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

model_name = "custom_animal"

def main(model_name = model_name, lr = 0.007, batch_size = 128, epochs = 10, norm = "batch", n = 10, resume = False, pretrained = False):
    SEED = 69
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n****************************************************************************\n")
    print(f"Device Used: {device}\n")
    print("\n****************************************************************************\n")

    model = train_model(model_name, resume, device, norm, epochs = epochs, batch_size = batch_size, learning_rate = lr, pretrained = pretrained)

    print("\n****************************************************************************\n")

    print("*****Correctly Classified Images*****\n")

    image_prediction(model, n=n, r=int(n/5), c=5, misclassified = False)

    print("\n****************************************************************************\n")

    print("*****Misclassified Images*****\n")

    image_prediction(model, n=n, r=int(n/5), c=5, misclassified = True)

    print("\n****************************************************************************\n")

if __name__ == "__main__":
    main()