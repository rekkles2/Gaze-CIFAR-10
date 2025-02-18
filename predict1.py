import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from PIL import Image
from my_dataset import read_text_file

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from vit_model import vit_base_patch16_224_in21k as create_model


def read_text_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    data = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                line = line.strip('()')
                x, y = line.split(', ')
                data.append((float(x), float(y)))
            except ValueError as e:
                print(f"Warning: Skipping invalid line: {line}")
                continue


    data = np.array(data).reshape(-1, 2)
    return data


class ImageTextDataset(Dataset):
    def __init__(self, img_dir, txt_dir, transform=None):

        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.transform = transform
        self.classes = sorted(os.listdir(img_dir))

        self.img_paths = []
        self.txt_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_img_dir = os.path.join(self.img_dir, class_name)
            class_txt_dir = os.path.join(self.txt_dir, class_name)

            img_files = [f for f in os.listdir(class_img_dir) if f.endswith('.jpg')]
            txt_files = [f for f in os.listdir(class_txt_dir) if f.endswith('.txt')]

            for img_file, txt_file in zip(img_files, txt_files):
                img_path = os.path.join(class_img_dir, img_file)
                txt_path = os.path.join(class_txt_dir, txt_file)

                self.img_paths.append(img_path)
                self.txt_paths.append(txt_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        txt_path = self.txt_paths[idx]
        txt_data = read_text_file(txt_path)

        txt_data = torch.from_numpy(txt_data.astype(np.float32))[:176, :]

        label = self.labels[idx]

        return img, txt_data, label, img_path


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = r"\data\test data"
    txt_dir = r"\data\test data"

    dataset = ImageTextDataset(img_dir=img_dir, txt_dir=txt_dir, transform=data_transform)


    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=6)

    # create model
    model = create_model(num_classes=10, has_logits=False).to(device)
    # load model weights
    model_weight_path = "./weights/best_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()


    correct = 0
    total = 0



    def savetupian(input_path, output_folder, label):
        print(f"Save the error image：{input_path} to the label {label}")

        label_folder = os.path.join(output_folder, str(label))
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        image = Image.open(input_path)
        filename = os.path.basename(input_path)

        output_path = os.path.join(label_folder, filename)

        image.save(output_path)



    with torch.no_grad():
        output_folder = r"\data\the_error_image"

        pbar = tqdm(dataloader, desc="Evaluating", unit="batch", total=len(dataloader))

        total = 0
        correct = 0

        for batch_idx, (img, txt, label, img_paths) in enumerate(pbar):
            img, txt, label = img.to(device), txt.to(device), label.to(device)

            output = model(img, txt)

            _, predicted = torch.max(output, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

            # # Save the wrong image
            # for i in range(len(predicted)):
            #     if predicted[i] != label[i]:
            #         input_path = img_paths[i]
            #         savetupian(input_path, output_folder, label[i].item())


            acc = 100 * correct / total


            pbar.set_postfix(accuracy=f"{acc:.2f}%")

    print(f"CIFAR-10 Classification accuracy: {acc:.2f}%")
