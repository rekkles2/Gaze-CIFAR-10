import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms



from PIL import Image, ImageDraw
import numpy as np

def read_text_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            x, y = line.strip('()').split(', ')
            data.append((float(x), float(y)))
    data = np.array(data).reshape(-1, 2)
    return data

def extract_patches(image, coordinates, patch_size=17):
    patches = []
    half_size = patch_size // 2
    img_width, img_height = image.size

    for x, y in coordinates:

        left = int(x) - half_size
        right = int(x) + half_size + 1
        top = int(y) - half_size
        bottom = int(y) + half_size + 1


        valid_left = max(left, 0)
        valid_right = min(right, img_width)
        valid_top = max(top, 0)
        valid_bottom = min(bottom, img_height)


        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)


        patch_left = valid_left - left
        patch_right = valid_right - left
        patch_top = valid_top - top
        patch_bottom = valid_bottom - top


        if valid_left < valid_right and valid_top < valid_bottom:
            img_patch = np.array(image.crop((valid_left, valid_top, valid_right, valid_bottom)))
            patch[patch_top:patch_bottom, patch_left:patch_right] = img_patch

        patches.append(patch)

    return patches

def draw_coordinates_on_image(image_path, coordinates, output_image_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)


    point_radius = 5
    point_color = (255, 0, 0)


    for x, y in coordinates:
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)


    img.save(output_image_path)
    img.show()
    print(f"Image saved at： {output_image_path}")

def display_patches_in_grid(patches, patch_size=17, grid_size=13):
    new_image = Image.new('RGB', (patch_size * grid_size, patch_size * grid_size), (255, 255, 255))  # 白色背景

    for index, patch in enumerate(patches):
        if index >= grid_size * grid_size:
            break
        patch_image = Image.fromarray(patch)
        x_pos = (index % grid_size) * patch_size
        y_pos = (index // grid_size) * patch_size
        new_image.paste(patch_image, (x_pos, y_pos))

    return new_image





def read_split_image(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_text(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    flower_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".txt"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        images.sort()

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_data(root1,root2):
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_image(root1)
    train_text_path, train_text_label, val_text_path, val_text_label = read_split_text(root2)
    return train_images_path, train_images_label, val_images_path, val_images_label,train_text_path,  val_text_path

#读取txt
def read_text_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if line:
            x, y = line.strip('()').split(', ')
            data.append((float(x), float(y)))
    data = np.array(data).reshape(-1, 2)
    return data


class MyDataSet(Dataset):


    def __init__(self, images_path: list, images_class: list, transform=None,text_path=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.text_path = text_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.images_class[item]
        text_file = self.text_path[item]
        txt_data = read_text_file(text_file)
        txt_data=torch.from_numpy(txt_data.astype(np.float32))[:176,]
        return img, label,txt_data

    @staticmethod
    def collate_fn(batch):
        images, labels ,text= tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        texts = torch.stack(text, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels,texts



data_transform = {
        "train": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
