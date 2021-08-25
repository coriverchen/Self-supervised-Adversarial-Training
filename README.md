## Self-supervised Adversarial Training
[SAT](https://arxiv.org/abs/1911.06470)

### Prepared Work
Training a self-supervised model or download a pretrained self-supervised model. 

### Proposed method
1. Obtain the pretrained self-supervised model
2. Generating adversarial examples by PGD-KNN
3. Maximize the mutual information between the representations of clean examples and advesarial examples.


AMDIM is selected as the self-supervised model

For detail of the implementation, please refer to the jupyter notebook 'CIFAR_SAT_AMDIM'
