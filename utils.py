import torch
import cv2
from torchvision.utils import save_image


DEVICE = torch.device('mps')
def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        # temp_img = torch.permute(temp_img,(1,2,0)).detach().cpu().numpy()
        # cv2.imwrite(folder+f"/y_gen_{epoch}.png", temp_img)
        #cv2.imwrite(folder + f"/input_{epoch}.png",x)
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
    gen.train()

