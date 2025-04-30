import torch
import matplotlib.pyplot as plt
from Model_new.Model import UNeT
import numpy as np
import torchvision.transforms as T

def show_prediction(model,image,ground_truth):
    model.eval()
    with torch.no_grad():
        pred_mask = model(image)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.squeeze().cpu().numpy()

    image_np = image.squeeze().cpu().numpy()
    gt_mask_np = ground_truth.squeeze().cpu().numpy()

    fig,ax = plt.subplots(1,3,figsize=(12,4))

    ax[0].imshow(image_np,cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(gt_mask_np, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis("off")

    ax[2].imshow(pred_mask,cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    plt.show()

if __name__ == "__main__":
    model = UNeT(in_channels=1,num_layers=2,num_classes=1,attn=True)

    test_image = torch.randn(1,1,512,512)
    test_mask = torch.randint(0,2,(1,512,512))

    show_prediction(model,test_image,test_mask)
