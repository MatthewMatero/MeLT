# MeLT: Message-Level Transformer

This repository contains code for our EMNLP 2021 Findings paper [MeLT: Message-Level Transformer with Masked Document Representations as Pre-Training for Stance Detection
](https://arxiv.org/pdf/2109.08113). This repo is activately maintained, if you have questions feel free to email the authors or leave an issue on GH. 

# About 

This work proposes a hierarchical transformer, built on top of distil-bert, that can directly encode sequences of messages within a user-level context. This hierarcical transformer is pre-trained
using a style of masked-language modeling applied to sequences of aggergated message vectors. Thus, turning the task into a masked-document modeling via reconstruction loss. The pre-trained transformer
was then applied to the downstream task of stance prediction using the SemEval-2016 task 6 data. The pre-training dataset is not open sourced. 

# Repo Structure

The code in this repostiory is used for constructing MeLT and pre-training it using the masked-document modeling task. Fine-tuning code is not supplied, as the novelty of the paper is focused on the construction and setup of MeLT itself. 
If there is a lot of interest in the fine-tuning code as well then I'll add it to this repostiory, but the application of MeLT should be much less complex than the pre-training. 

The *modeling* directory stores the class files for defining the necessary helper functions, transformer layers, attention calculations, and MeLT model. 
These are located in neural.py, encoder\_layers.py, attn.py, and encoder.py respectively. There is also a data\_handler.py which is used to load in the raw language data (from MySQL) 
and build a PyTorch dataloader batched by users. 

The root directory has a *main.py* which is used to load in the MeLT model and control training, testing, and hyperparameter tuning modes. 

# requirements

an envrionment.yml file is included that represents a conda envrionment of the libraries used for development of this project. At a high level you will need PyTorch (1.4), PyTorch Lightning(0.7.5), Pandas, and the standard numpy stack. 

# Cite
```
@article{matero2021melt,
  title={MeLT: Message-Level Transformer with Masked Document Representations as Pre-Training for Stance Detection},
  author={Matero, Matthew and Soni, Nikita and Balasubramanian, Niranjan and Schwartz, H Andrew},
  journal={arXiv preprint arXiv:2109.08113},
  year={2021}
}
```
