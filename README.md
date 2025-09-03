# Diffusion Attribution Score (DAS)

This repository contains the official implementation for the paper **"[Diffusion Attribution Score: Evaluating Training Data Influence in Diffusion Model](https://iclr.cc/virtual/2025/poster/28556),"** accepted at **ICLR 2025**.

\[[arXiv](https://arxiv.org/abs/2410.18639)\]
\[[OpenReview](https://openreview.net/forum?id=kuutidLf6R)\]
---

## Introduction

This work introduces the Diffusion Attribution Score (DAS), a novel method for evaluating the influence of individual training data points on the output of a diffusion model. Our approach provides a powerful tool for understanding model behavior, debugging, and identifying the most impactful data for a given generation.

---

## Getting Started

To run the experiments, you'll need to set up the appropriate environment.

1.  **Clone the Repository:**

    ```bash
    git clone git@github.com:Jinxu-Lin/DAS.git
    ```

2.  **Create a Conda Environment:**

    ```bash
    conda create -n das python=3.11
    conda activate das
    ```

3.  **Install Dependencies:**
    Install all required packages from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```
---

## Reproducing Experiments

Our experiments were conducted on six datasets: CIFAR2, CIFAR10, ArtBench2, ArtBench5, ArtBench10, and CelebA. The following steps demonstrate how to reproduce the results using the **CIFAR2** dataset as an example.

1.  **Navigate to the Dataset Directory:**

    ```bash
    cd CIFAR2
    ```

2.  **Prepare the Dataset:**
    Run the `00_dataset.ipynb` notebook to generate the CIFAR2 dataset and prepare the necessary subsets for LDS.

3.  **Train the Diffusion Model:**
    Use the provided script to train a diffusion model.

    ```bash
    bash scripts/01_train.sh 1 0 18888
    ```

4.  **Generate Images:**
    After training, you can generate images using the generation script.

    ```bash
    bash scripts/02_gen.sh 0 0
    ```

5. **Compute Gradients:**
    Once we have the training and generated samples, we can calculate their gradients using the gradient script.

    ```bash
    bash scripts/03_grad.sh 0 0 train idx-train.pkl 0 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 train idx-train.pkl 1 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 train idx-train.pkl 2 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 train idx-train.pkl 3 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 train idx-train.pkl 4 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 val idx-val.pkl 0 mean uniform 10 4096
    bash scripts/03_grad.sh 0 0 gen idx-gen.pkl 0 mean uniform 10 4096
    ```

6. **Compute Error**

    To compute DAS, we need to compute the error of training set.

    ```
    bash scripts/04_eval.sh 0 0 train idx-train.pkl
    ```

7. **LDS Benchmark**

    Train 64 models corresponding to 64 subsets of the training set.

    ```
    bash scripts/05_ldstrain.sh 1 0 18888 0 63
    ```

    Evaluate the model outputs on the validation set

    ```
    bash scripts/06_ldseval.sh 0 0 val idx-val.pkl 0 63
    ```

    Evaluate the model outputs on the generation set

    ```
    bash scripts/06_ldseval.sh 0 0 gen idx-gen.pkl 0 63
    ```

8. **Compute the Scores**

    Compute LDS score for our DAS.
    ```
    bash scripts/07_score.sh val das mean 10 4096 uniform
    ```

---

## Bibtex

If you find this project useful in your research, please consider citing our paper:

```
@inproceedings{
lin2025diffusion,
title={Diffusion Attribution Score: Evaluating Training Data Influence in Diffusion Model},
author={Jinxu Lin and Linwei Tao and Minjing Dong and Chang Xu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=kuutidLf6R}
}
```

---

## Acknowledgement

Thanks to the authors of [Intriguing Properties of Data Attribution on Diffusion Models](https://sail-sg.github.io/D-TRAK) for providing their codes and LDS data repository.
This code is implement based on their code repository ([Code Link](https://github.com/sail-sg/D-TRAK)).
