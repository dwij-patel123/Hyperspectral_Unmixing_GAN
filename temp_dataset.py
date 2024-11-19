import cv2
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from transforms import both_transform,transform_only_mask,transform_only_input


class MapDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path),dtype=np.float32)
        # input_image = cv2.resize(image[:, :600, :],(512,512))
        # target_image = cv2.resize(image[:, 600:, :],(512,512))

        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]
        # input_image = torch.tensor(input_image,dtype=torch.float32)
        # target_image = torch.tensor(target_image,dtype=torch.float32)
        # input_image = torch.permute(input_image,(2,0,1))
        # target_image = torch.permute(target_image,(2,0,1))
        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        input_image = transform_only_input(image=input_image)['image']
        target_image = transform_only_mask(image=target_image)['image']
        return input_image, target_image

#
if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        # cv2.imwrite("x.png",img_x)
        # cv2.imwrite("y.png",img_y)
        break





