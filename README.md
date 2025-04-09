
<h1 align="center">👀 Gaze-Guided Learning: Avoiding Shortcut Bias in Visual Classification</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2504.05583v1">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2504.05583v1-b31b1b?logo=arxiv&logoColor=white" />
  </a>
  <a href="https://szyyjl.github.io/eye_tracking_data.github.io/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-blue?logo=google-chrome" />
  </a>
  <a href="https://drive.google.com/drive/folders/17zR9bIDWvb0FzSEgR2vXJIKo3w6wKDVB?usp=drive_link">
    <img alt="Download Dataset" src="https://img.shields.io/badge/Dataset-GoogleDrive-brightgreen?logo=google-drive" />
  </a>
  <a href="https://paperswithcode.com/dataset/gaze-cifar-10">
    <img alt="Papers with Code" src="https://img.shields.io/badge/PaperWithCode-Gaze--CIFAR--10-orange?logo=paperswithcode" />
</p>

<p align="center"><strong>⭐ If you find our dataset and code useful, please consider starring this repository and citing our paper!</strong></p>

<details open>
<summary><strong>📋 BibTeX Citation (click to expand)</strong></summary>

```
@misc{li2025gazeguided,
    title={Gaze-Guided Learning: Avoiding Shortcut Bias in Visual Classification},
    author={Jiahang Li and Shibo Xue and Yong Su},
    year={2025},
    eprint={2504.05583},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

</details>

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

You can download the <strong>Gaze-CIFAR-10</strong> dataset from the following link:

👉 [**Gaze-CIFAR-10 Dataset**](https://drive.google.com/drive/folders/17zR9bIDWvb0FzSEgR2vXJIKo3w6wKDVB?usp=drive_link)

<p align="center">
  <img src="https://github.com/rekkles2/Gaze-CIFAR-10/blob/main/Figure/exp.png" alt="Example Samples" width="100%">
  <br/><em>Figure: Gaze data collection setup. (a) Overview of our data acquisition system. (b) Step 1: Reconstruct image resolution. Step 2: Participants freely view two randomly selected images from different categories. Step 3: One image is randomly re-sampled from the previously viewed categories and shown again for focused observation. Step 4: Gaze data is transmitted to the PC for processing.</em>
</p>

---

## 🧠 Pretrained Model

Download the pretrained **Vision Transformer (ViT)** model:

📥 [ViT Pretrained Model](https://drive.google.com/file/d/1FPUIYmZ4ooMbWByXUzBRNGLcrIYvNsxz/view?usp=drive_link)

---

## 🚀 Training

To train the model, run:

```bash
python train.py
```

---

## 🔍 Evaluation

To evaluate the trained model, run:

```bash
python predict1.py
```

---

## 📈 Star History


[![Star History Chart](https://api.star-history.com/svg?repos=rekkles2/Gaze-CIFAR-10&type=Date)](https://www.star-history.com/#rekkles2/Gaze-CIFAR-10&Date)
  


