# Cold-Start-Reinforcement-Learning-with-Softmax-Policy-Gradient

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Paper](https://arxiv.org/abs/1709.09346)

This repository contains an implementation of the reinforcement learning method described in the paper "Cold-Start Reinforcement Learning with Softmax Policy Gradient" by Nan Ding and Radu Soricut from Google Inc. The method is based on a softmax value function that eliminates the need for warm-start training and sample variance reduction during policy updates.

## Method

## Requirements

## Experiment

### Summarization Task: Headline Generation

- Dataset:
      - Training: [English Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05)
      - Testing: [DUC 2004](https://duc.nist.gov/duc2004/)
- Evaluation: [ROUGE-L score](https://arxiv.org/abs/1803.01937)

### Automatic Image-Caption Generation

- Dataset:
      - Training / Validation: [Microsoft COCO](https://cocodataset.org/#home)
      - Testing: [Microsoft COCO](https://cocodataset.org/#home)
- Evaluation: [CIDer score](https://arxiv.org/abs/1411.5726) / ROUGE-L score

## Results

## Acknowledgements

We would like to thank Nan Ding and Radu Soricut for their valuable contributions to the field of reinforcement learning, and for making their paper available to the public. We also acknowledge the TensorFlow team for providing a powerful and flexible deep learning framework.

## Reference


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