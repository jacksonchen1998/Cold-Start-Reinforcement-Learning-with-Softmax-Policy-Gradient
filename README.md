# Cold-Start-Reinforcement-Learning-with-Softmax-Policy-Gradient

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Paper](https://arxiv.org/abs/1709.09346)

This repository contains an implementation of the reinforcement learning method described in the paper "Cold-Start Reinforcement Learning with Softmax Policy Gradient" by Nan Ding and Radu Soricut from Google Inc. The method is based on a softmax value function that eliminates the need for warm-start training and sample variance reduction during policy updates.

## Method

[RNN Encoder Decoder](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb)

## Requirements

Create a conda environment using the following command:

```bash
conda create -n <env_name> python=3.9
```

Intsall the required packages using the following command:

```bash
conda install --file requirements.txt
```

## Program issues

In `pipeline.py`, change the following line if has an error:

```
AssertionError: Torch not compiled with CUDA enabled
```

Change

```py
z = torch.cat([z, zt_idx.cuda()[None]], dim=0) # (T, B) token id
```

to

```py
z = torch.cat([z, zt_idx[None]], dim=0) # (T, B) token id
```

## Experiment

### Summarization Task: Headline Generation

Dataset:
- Training: [English Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05)
- Testing: [DUC 2004](https://duc.nist.gov/duc2004/)

Evaluation: 
[ROUGE-L score](https://arxiv.org/abs/1803.01937)

### Automatic Image-Caption Generation

Dataset:
- Training / Validation: [Microsoft COCO](https://cocodataset.org/#home)
- Testing: [Microsoft COCO](https://cocodataset.org/#home)

Evaluation: 
[CIDer score](https://arxiv.org/abs/1411.5726) / ROUGE-L score

## Results

## Acknowledgements

We would like to thank Nan Ding and Radu Soricut for their valuable contributions to the field of reinforcement learning, and for making their paper available to the public. We also acknowledge the TensorFlow team for providing a powerful and flexible deep learning framework.

## Citation

```
@misc{20230615,
  author = {Chih-Chun Chen, Pin-Yen Liu, Po-Chuan Chen},
  title = {Cold-Start Reinforcement Learning with Softmax Policy Gradient},
  year = {2023},
  month = {06},
  note = {Version 1.0},
  howpublished = {GitHub},
  url = {https://github.com/jacksonchen1998/Cold-Start-Reinforcement-Learning-with-Softmax-Policy-Gradient}
}
```

```
@misc{ding2017coldstart,
      title={Cold-Start Reinforcement Learning with Softmax Policy Gradient}, 
      author={Nan Ding and Radu Soricut},
      year={2017},
      eprint={1709.09346},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
