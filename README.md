# 2D-CrossScan Mamba: Enhancing State Space Models with Spatially Consistent Multi-Path 2D Information Propagation
<p align="center">
    <a href="https://aaai.org/Conferences/AAAI-26/">
        <img alt="AAAI 2026" src="https://img.shields.io/badge/AAAI-2026-blueviolet">
    </a>
    <a href="https://pytorch.org/">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.20+-ee4c2c">
    </a>
    <a href="https://github.com/longlong-yu/official-cross-scan">
        <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-green">
    </a>
    <a href="https://github.com/longlong-yu/official-cross-scan/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-yellowgreen">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#-introduction">Introduction</a> |
        <a href="#-key-contributions">Key Contributions</a> |
        <a href="#-updates">Updates</a> |
        <a href="#-getting-started">Getting Started</a> |
        <a href="#-acknowledgements">Acknowledgements</a> |
        <a href="#-contact">Contact</a>
    </p>
</h4>

## 🌀 Introduction
This repository contains the official implementation of the AAAI 2026 accepted paper:  
<p align="center"><b>2D-CrossScan Mamba: Enhancing State Space Models with Spatially Consistent Multi-Path 2D Information Propagation</b></p>

State Space Models (SSMs) like Mamba achieve excellent performance on sequence tasks but suffer from inherent 1D scanning limitations when applied to 2D image data. Existing adaptations (e.g., VMamba, 2DMamba) fail to align with spatial geometry or restrict to single-path propagation.  

We propose **2D-CrossScan**, a novel 2D-compatible scan framework that enables spatially consistent, multi-path hidden state propagation by integrating modified state equations over 2D neighborhoods. It addresses 1D-2D misalignment and boosts performance on visual tasks like object detection and semantic segmentation.

<p align="center">
    <img src="resources/images/fig_2.jpg" alt="2D-CrossScan Framework" width="80%"/>
</p>

---

## 🔍 Key Contributions
- Propose a **2D-CrossScan framework** that reformulates state equations to aggregate hidden states from spatial neighbors, ensuring propagation along true 2D shortest paths.
- Develop a **multi-path aggregation mechanism** with cross-directional subtraction to suppress redundancy from overlapping paths, efficient without factorial complexity.
- Design a **four-corner multi-directional scanning strategy** that fuses features with trainable weights, maximizing geometric symmetry and feature diversity.
- Achieve consistent improvements over SOTA SSM-based models on PANDA, COCO, ImageNet-1K, and ADE20K, with better spatial coherence verified by ERF and attention analyses.

---

## ✅ Updates
- (16/11/2025) Code release for 2D-CrossScan kernel.
- (08/11/2025) Paper accepted by AAAI 2026! 🎉🎉

---

## 🚀 Getting Started
### Setup
```bash
git clone https://github.com/longlong-yu/official-cross-scan.git
cd official-cross-scan/kernels/selective_nd_scan
pip install .

---

## 🙏 Acknowledgements

Our implementation of the 2D-CrossScan kernel is primarily inspired by [**Mamba**](https://github.com/state-spaces/mamba), [**VMamba**](https://github.com/MzeroMiko/VMamba), and [**2DMamba**](https://github.com/AtlasAnalyticsLab/2DMamba). Most experiments were conducted by adapting the codebases of [**SparseFormer**](https://github.com/liwenxi/SparseFormer) and [**VMamba**](https://github.com/MzeroMiko/VMamba).

---

<!-- ## 📄 Citation

If you find this work helpful, please consider citing:

```bibtex
@article{your2025paper,
  title     = {2D-CrossScan Mamba: Enhancing State Space Models with Spatially Consistent Multi-Path 2D Information Propagation},
  author    = {Your Name and Coauthors},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2025}
}
```

--- -->

## 📬 Contact

For questions or suggestions, please open an issue or contact us at: longlong.yu@hdu.edu.cn.