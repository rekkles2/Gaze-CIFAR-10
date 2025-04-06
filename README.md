# 👀 Gaze-Guided Learning: Avoiding Shortcut Bias in Visual Classification

> **[Project Page 🌐](https://szyyjl.github.io/eye_tracking_data.github.io/)**  
> If you find our dataset and code useful, please ⭐ star the repo and cite our paper!

---

## 📜 Abstract
Inspired by human visual attention, deep neural networks have widely adopted attention mechanisms to learn locally discriminative attributes for challenging visual classification tasks. However, existing approaches primarily emphasize the representation of such features while neglecting their precise localization, which often leads to misclassification caused by shortcut biases. This limitation becomes even more pronounced when models are evaluated on transfer or out-of-distribution datasets. In contrast, humans are capable of leveraging prior object knowledge to quickly localize and compare fine-grained attributes, a capability that is especially crucial in complex and high-variance classification scenarios. Motivated by this, we introduce **Gaze-CIFAR-10**, a human gaze time-series dataset, along with a **dual-sequence gaze encoder** that models the precise sequential localization of human attention on distinct local attributes. In parallel, a **Vision Transformer (ViT)** is employed to learn the sequential representation of image content. Through cross-modal fusion, our framework integrates human gaze priors with machine-derived visual sequences, effectively correcting inaccurate localization in image feature representations. Extensive qualitative and quantitative experiments demonstrate that gaze-guided cognitive cues significantly enhance classification accuracy.
![Motivation Figure](https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/motivation.png)
---

## 🧠 Method Overview
![Model](https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/model.jpeg)


---

## 📂 Dataset

The **Gaze-CIFAR-10** dataset can be downloaded from the following link:

👉 [**Gaze-CIFAR-10 Dataset**](https://drive.google.com/drive/folders/17zR9bIDWvb0FzSEgR2vXJIKo3w6wKDVB?usp=drive_link)

![Example Samples](https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/exp.png)

---

## 🧠 Pretrained Model

You can download the **pretrained ViT model** from:

📥 [ViT Pretrained Model](https://drive.google.com/file/d/1FPUIYmZ4ooMbWByXUzBRNGLcrIYvNsxz/view?usp=drive_link)

---

## 🚀 Training

To **train** the model, run the following command:

```bash
python train.py
```

---

## 🔍 Evaluation

To **evaluate** the trained model, use:

```bash
python predict1.py
```

---

## ⭐ Citation

> If our work is helpful, please consider citing us and starring the repository.  
> Thank you for supporting our research!
