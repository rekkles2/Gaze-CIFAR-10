<h1 align="center">👀 Gaze-Guided Learning: Avoiding Shortcut Bias in Visual Classification</h1>

<p align="center">
  <a href="https://szyyjl.github.io/eye_tracking_data.github.io/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?logo=google-chrome" />
  </a>
  <a href="https://github.com/rekkles2/Gaze-CIFAR-10">
    <img alt="GitHub Stars" src="https://img.shields.io/github/stars/rekkles2/Gaze-CIFAR-10?style=social" />
  </a>
</p>

<p align="center"><strong>⭐ If you find our dataset and code useful, please consider starring this repository and citing our paper!</strong></p>

---

## 📄 Abstract

> Inspired by human visual attention, deep neural networks have widely adopted attention mechanisms to learn locally discriminative attributes for challenging visual classification tasks. However, existing approaches primarily emphasize the representation of such features while neglecting their precise localization, which often leads to misclassification caused by shortcut biases. This limitation becomes even more pronounced when models are evaluated on transfer or out-of-distribution datasets. In contrast, humans are capable of leveraging prior object knowledge to quickly localize and compare fine-grained attributes, a capability that is especially crucial in complex and high-variance classification scenarios. Motivated by this, we introduce **Gaze-CIFAR-10**, a human gaze time-series dataset, along with a **dual-sequence gaze encoder** that models the precise sequential localization of human attention on distinct local attributes. In parallel, a **Vision Transformer (ViT)** is employed to learn the sequential representation of image content. Through cross-modal fusion, our framework integrates human gaze priors with machine-derived visual sequences, effectively correcting inaccurate localization in image feature representations. Extensive qualitative and quantitative experiments demonstrate that gaze-guided cognitive cues significantly enhance classification accuracy.

<p align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/motivation.png" alt="Motivation Figure" width="100%">
  <br/><em>Figure: A toy example illustrating shortcut bias: (a) DNNs attention versus (b) human gaze under limited data scale and diversity.</em>
</p>

---

## 🧠 Method Overview

<p align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/model.jpeg" alt="Model Architecture" width="100%">
  <br/><em>Figure: Gaze-guided cross-modal fusion network.</em>
</p>

---

## 📂 Dataset

The **Gaze-CIFAR-10** dataset can be downloaded from:

👉 [**Gaze-CIFAR-10 Dataset**](https://drive.google.com/drive/folders/17zR9bIDWvb0FzSEgR2vXJIKo3w6wKDVB?usp=drive_link)

<p align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/exp.png" alt="Example Samples" width="100%">
  <br/><em>Gaze data collection setup. (a) Overview of our data acquisition system. (b) Step 1: Reconstruct image resolution. Step 2: Participants freely view two randomly selected images from different categories. Step 3: One image is randomly re-sampled from the previously viewed categories and shown again for focused observation. Step 4: Gaze data is transmitted to the PC for processing.</em>
</p>

---

## 🧠 Pretrained Model

You can download the **pretrained ViT model** from:

📥 [ViT Pretrained Model](https://drive.google.com/file/d/1FPUIYmZ4ooMbWByXUzBRNGLcrIYvNsxz/view?usp=drive_link)

---

## 🚀 Training

Run the following command to **train** the model:

```bash
python train.py
```

---

## 🔍 Evaluation

Use the following command to **evaluate** the trained model:

```bash
python predict1.py
```


---

## 📈 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=rekkles2/Gaze-CIFAR-10&type=Date)](https://www.star-history.com/#rekkles2/Gaze-CIFAR-10&Date)
