这里是优化后的 `README.md`，改进了整体排版、可读性，并增强了美观性和一致性：

---

# Gaze-Guided Learning: Avoiding Shortcut Bias in Visual Classification  

## Overview  
This repository provides an implementation of **Gaze-Guided Learning**, a method designed to reduce shortcut bias in visual classification using eye-tracking data. The model is based on **Vision Transformer (ViT)** and trained on the **Gaze-CIFAR-10** dataset.  

<div align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/motivation.png" alt="Motivation Figure" width="600">
</div>  

---

## Pretrained Model  
The **pretrained ViT model** can be downloaded from the following link:  
🔗 [ViT Pretrained Model](https://drive.google.com/file/d/1FPUIYmZ4ooMbWByXUzBRNGLcrIYvNsxz/view?usp=drive_link)  

---

## Dataset  
The **Gaze-CIFAR-10** dataset can be downloaded from:  
📂 [Gaze-CIFAR-10 Dataset](https://drive.google.com/drive/folders/17zR9bIDWvb0FzSEgR2vXJIKo3w6wKDVB?usp=drive_link)  

<div align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/EX.png" alt="Dataset Example" width="500">
</div>  

---

## Training  
To **train** the model, use the following command:  
```bash
python train.py
```  

---

## Testing  
To **evaluate** the trained model, run:  
```bash
python predict1.py
```  

---

## Citation  
If you find this code or dataset useful in your research, please cite our work accordingly.  

---
