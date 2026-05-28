<div align="center">

# $AGT^{AO}$: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality

<p>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/"><img alt="HuggingFace" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow"></a>
  <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://arxiv.org/abs/2602.01703"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2602.01703-b31b1b.svg?logo=arxiv"></a>
  <a href="https://2026.aclweb.org/"><img alt="Venue" src="https://img.shields.io/badge/ACL%202026-Findings-1f77b4?logo=googlescholar&logoColor=white"></a>
</p>

<p>
  <a href="https://arxiv.org/abs/2602.01703"><b>📄 Paper</b></a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2602.01703v1"><b>📥 PDF</b></a> &nbsp;|&nbsp;
  <a href="#-data-preparation"><b>📦 Data</b></a> &nbsp;|&nbsp;
  <a href="#-experimental-results"><b>📊 Results</b></a> &nbsp;|&nbsp;
  <a href="#-citation"><b>✍️ Citation</b></a>
</p>

<p>
  <em>Official PyTorch implementation of our <b>ACL 2026 Findings</b> paper.</em>
</p>

</div>

---

> [!IMPORTANT]
> 🎉 **News (May 2026):** Our paper has been **accepted to the Findings of ACL 2026**! Many thanks to our co-authors, reviewers, and the open-source community.

## 📖 Abstract

This repository contains the official implementation of our **ACL 2026 Findings** paper:
**"$AGT^{AO}$: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality."**

$AGT^{AO}$ is a novel LLM unlearning framework that simultaneously tackles two long-standing challenges:

- **Catastrophic forgetting** — collateral damage to retained knowledge and general utility.
- **Superficial forgetting** — hidden knowledge recovery under adversarial probing.

It achieves **state-of-the-art (SOTA)** performance on **TOFU**, **MUSE**, and **WMDP** benchmarks, while maintaining model utility and fluency.

<div align="center">
  <img src="arc.png" alt="AGT-AO Architecture" width="92%">
  <p><sub><b>Figure 1.</b> Overview of the $AGT^{AO}$ framework.</sub></p>
</div>

## 🔥 News

- **`[2026/05]`** 🎉 Our paper is **accepted to the Findings of ACL 2026**!
- **`[2026/02]`** 📦 Code and arXiv preprint released.
- **`[2026/01]`** 📝 Paper submitted to ACL 2026.

## 🚀 Key Features

- **🛡️ Adversarial Gating Training (AGT)** — A min-max game in the latent space that injects worst-case perturbations, ensuring *deep erasure* and resisting adversarial knowledge recovery.
- **⚖️ Adaptive Orthogonality (AO)** — A dynamic regularizer that projects forget gradients orthogonally to retain gradients *only when they conflict*, preserving model utility without sacrificing forgetting strength.
- **📈 Gradient-Norm-Based Gating (GBG)** — A curriculum-inspired gating mechanism that stabilizes adversarial training and avoids early-stage collapse.
- **🧪 Comprehensive Benchmarks** — Evaluated on **TOFU**, **MUSE**, and **WMDP** with multiple base LLMs (LLaMA-2, Gemma, Zephyr, ICLM).

<div align="center">
  <img src="GBG.png" alt="Gradient-Norm-Based Gating" width="62%">
  <p><sub><b>Figure 2.</b> Overview of the Gradient-Norm-Based Gating (GBG) mechanism.</sub></p>
</div>

## 🛠️ Installation

We recommend using `Conda` to manage the environment.

```bash
# 1. Create and activate the environment
conda create -n agt python=3.10 -y
conda activate agt

# 2. Install PyTorch (adjust CUDA version to match your driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
pip install -r requirements.txt
```

<details>
<summary><b>📋 Core dependencies (<code>requirements.txt</code>)</b></summary>

```text
transformers>=4.30.0
peft>=0.4.0
datasets
accelerate
scipy
hydra-core   # configuration management
wandb        # experiment logging
```

</details>

## 📂 Data Preparation

We support three major LLM-unlearning benchmarks:

| Benchmark | Domain | Description |
|:----------|:-------|:------------|
| **TOFU**  | Fictional Authors | Task of Fictitious Unlearning — synthetic author biographies. |
| **MUSE**  | News & Books      | Copyright unlearning on real-world text corpora. |
| **WMDP**  | Hazardous Knowledge | Cybersecurity / bio / chemistry safety unlearning. |

Download the datasets and arrange them as follows:

```text
data/
├── tofu/
├── muse/
└── wmdp/
```

## 🏃 Quick Start

### 1. Supported Models

`Llama-2-7b-chat` · `Gemma-2b-it` · `Zephyr-7b-beta` · `ICLM-7b`

### 2. Running $AGT^{AO}$

Reproduce the main TOFU results with `Llama-2-7b-chat` (hyperparameters follow Appendix A.3):

```bash
python baselines/unlearn.py \
    --method agt \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --dataset tofu \
    --lr 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --ao_gamma 1.0 \
    --tau_grad 2.0 \
    --warmup_steps 50 \
    --perturb_layer 10 \
    --inner_loop_steps 4
```

### 3. Key Hyperparameters

| Hyperparameter | Value / Definition | Notes |
|:---------------|:-------------------|:------|
| Learning rate | `1e-4` | AdamW |
| Batch size | `1` (× 8 grad-accum) | Effective batch = 8 |
| AO parameter $\gamma$ | `1` | Cosine-conflict exponent |
| Gradient threshold $\tau_{\text{grad}}$ | $\rho \cdot \lVert \nabla \mathcal{L}_{N_{\text{warmup}}}\rVert_2$ | Adaptive |
| Warm-up steps $N_{\text{warmup}}$ | `1 epoch` | GBG warm-up |
| Perturbation layer | `10` | "Semantic-Entry" layer |
| Inner-loop steps | `4` | From our ablation study |

## 📊 Method Details

### Adaptive Orthogonality (AO)

AO suppresses collateral damage by penalizing the cosine similarity between the **forget gradient** $g_f$ and the **retain gradient** $g_r$ **only when they conflict** (i.e., $g_f \cdot g_r < 0$):

$$
\mathcal{R}_{AO} \;=\; \mathbb{I}\bigl(g_f \cdot g_r < 0\bigr) \left(\frac{1 - \cos(g_f, g_r)}{2}\right)^{\gamma}
$$

### Adversarial Gating Training (AGT)

AGT casts unlearning as a **bi-level (min-max) optimization** that defends against latent perturbations $\delta$ at the semantic-entry layer:

$$
\min_{\theta}\;\max_{\lVert\delta\rVert_p \le \epsilon}\;\mathcal{L}_{\text{unlearn}}\bigl(h_f^{(l)} + \delta,\, h_r;\, \theta\bigr)
$$

This jointly enforces *deep erasure* of the forget set and *robustness* against adversarial probes, while AO and GBG keep training stable and utility-preserving.

## 🧪 Experimental Results

### TOFU Benchmark — *Llama-2-7b-chat*

We compare $AGT^{AO}$ against strong baselines: **GA**, **NPO**, **SimNPO**, **PGU**, **RMU**, and **LAT**.

> **Reading the table.** Results are averaged over **3 runs**.
> ↑ higher is better &nbsp;·&nbsp; ↓ lower is better &nbsp;·&nbsp; →0.5 closer to 0.5 is more ideal.
> **Bold** = best &nbsp;·&nbsp; <u>underline</u> = second best.

| Method            | Forget Quality ↑ | KUR ↓     | Model Utility ↑ | Fluency ↑ | PLR →0.5 |
|:------------------|:----------------:|:---------:|:---------------:|:---------:|:--------:|
| Target            | −46.91           | 0.91      | 0.59            | 0.87      | 0.98     |
| Retrain (oracle)  | 0.00             | 0.29      | 0.58            | 0.91      | 0.47     |
| GA                | −50.29           | 0.48      | 0.00            | 0.00      | 0.45     |
| GA + GDR          | −51.16           | 1.44      | 0.51            | 0.27      | 0.08     |
| GA + KLR          | −31.85           | 0.69      | 0.00            | 0.29      | 0.59     |
| NPO               | −19.78           | 0.30      | 0.00            | 0.02      | 0.40     |
| NPO + GDR         | −13.80           | 0.20      | 0.53            | 0.16      | 0.19     |
| NPO + KLR         | −30.51           | 0.38      | 0.45            | <u>0.89</u> | 0.81   |
| SimNPO + GDR      | −13.96           | 0.20      | 0.52            | 0.21      | 0.18     |
| PGU               | −15.39           | 0.23      | 0.47            | 0.83      | <u>0.55</u> |
| RMU               | −14.20           | 0.14      | 0.45            | 0.76      | 0.59     |
| LAT               | <u>−12.50</u>    | <u>0.05</u> | 0.41          | 0.70      | <u>0.55</u> |
| **$AGT^{AO}$ (Ours)** | **−9.43**    | **0.01**  | **0.59**        | **0.90**  | **0.53** |

#### Notes

- **Evaluation axes.** We report (i) *forgetting* via **Forget Quality** and **Knowledge Unlearning Ratio (KUR)**, (ii) *utility & quality* via **Model Utility** and **Fluency**, and (iii) *privacy* via **Privacy Leakage Rate (PLR)**.
- **Sub-metrics.** Composition and definitions of KUR and PLR sub-indicators are detailed in the paper appendix.
- **Takeaway.** $AGT^{AO}$ delivers the **best forgetting–utility trade-off**, with a PLR close to the ideal value of **0.5**, indicating that the unlearned model behaves statistically indistinguishably from a model that never saw the forget set.

## 📁 Repository Structure

```text
AGT-unlearning/
├── baselines/      # Unlearning methods (AGT, GA, NPO, SimNPO, PGU, RMU, LAT, ...)
├── evals/          # Evaluation pipelines for TOFU / MUSE / WMDP
├── arc.png         # Figure 1: framework overview
├── GBG.png         # Figure 2: gradient-norm-based gating
├── LICENSE         # MIT license
└── README.md
```

## ✍️ Citation

If you find our work useful, please consider citing:

```bibtex
@article{li2026textbf,
  title={{\textbf{AGT$^{AO}$}}: Robust and Stabilized LLM Unlearning via Adversarial Gating Training with Adaptive Orthogonality},
  author={Li, Pengyu and Zhang, Lingling and Gao, Zhitao and Wu, Yanrui and Dong, Yuxuan and Liu, Huan and Wei, Bifan and Liu, Jun},
  journal={arXiv preprint arXiv:2602.01703},
  year={2026}
}
```

## 🙏 Acknowledgements

This codebase builds upon several outstanding open-source projects. We sincerely thank their authors:

- [**TOFU**](https://github.com/locuslab/tofu) — A Task of Fictitious Unlearning.
- [**MUSE**](https://github.com/swj0419/muse_bench) — Machine Unlearning Six-way Evaluation.
- [**WMDP**](https://github.com/centerforaisafety/wmdp) — Weapons of Mass Destruction Proxy benchmark.

We also thank the broader LLM-unlearning community for fruitful discussions and feedback.

## 📜 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>⚠️ <em>Disclaimer: This repository is intended for academic research purposes only.</em></sub><br>
  <sub>Made with ❤️ for safer and more trustworthy LLMs.</sub>
</div>
