# OpenPheno: Phenotypic Bioactivity Prediction as Open-set Biological Assay Querying

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](链接论文PDF或ArXiv)
[![Dataset](https://img.shields.io/badge/Dataset-Broad--270-blue.svg)](#data-availability)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **OpenPheno**, as described in the paper *"Phenotypic Bioactivity Prediction as Open-set Biological Assay Querying"*. 

## 💡 Overview

The traditional drug discovery pipeline is bottlenecked by the need to design and execute bespoke biological assays for every new target. Current computational models are confined to "closed-set" paradigms, unable to generalize to entirely novel assays without target-specific training data. 

**OpenPheno** fundamentally redefines bioactivity prediction as an **open-set, visual-language question-answering (QA) task**. By integrating chemical structures (SMILES), universal phenotypic profiles (Cell Painting images), and natural language descriptions of biological assays, OpenPheno unlocks the **"profile once, predict many"** paradigm.

### ✨ Key Features
* **Zero-Shot Generalization:** Predicts compound bioactivity on entirely unseen biological assays using only natural language descriptions as queries.
* **Multimodal Architecture:** Employs a two-stage training strategy combining cross-modal contrastive alignment (CLIP) and cross-plate self-supervised consistency (DINO).
* **Assay Query Network (AQN):** Dynamically fuses compound features and assay descriptions to attend over image patch features, grounding linguistic queries in visual cellular phenotypes.
* **Robust Performance:** Achieves strong zero-shot performance (mean AUROC 0.75) on 54 unseen assays, exceeding supervised baselines trained with full labeled data.

![OpenPheno Overview](链接到您的Figure1图片,例如: ./assets/figure1.png)
*(Figure 1: Overview of OpenPheno for open-set bioactivity prediction.)*
