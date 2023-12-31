# BitNet-Transformer

This repository is a from scratch implementation of both the traditional transformer architecture from the 2017 paper "Attention is all you need" by Vaswani et al. at Google Brain as well as the BitNet transformer architecture from the 2023 "BitNet: Scaling 1-bit Transformers for Large Language Models" by Wang et al. at Microsoft Research. 

This implementation demonstrates how the quantized 'BitLinear' layer can be used instead of the 'nn.Linear' PyTorch layer to build a transformer with scaling performance during training on a machine translation task scales with training. 

This BitNet implementation has a few notable differences from the implementation described in the original BitNet paper. 



