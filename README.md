# PRISM

This repository contains the source code for **A Product Manifold Method for Feature Selection**, which has been accepted by KDD ’26. The overall architecture of our proposed method PRISM is shown as follows:

![Overall architecture](PRISM.png)

## Code Overview

1. `PRISM.py` is the core implementation of our feature selection algorithm PRISM. It calls key functions from `Functions.py`. The GPU-accelerated version, `PRISM_gpu.py`, utilizes the GPU for computation and relies on `Functions_gpu.py` for core functions.

2. The folder named `XOR-100` contains the experimental code for reproducing the XOR-100 problem mentioned in our paper (Sec. 4.1). Running `XOR-100/XOR-100.py` can get the experimental results of PRISM, and running `XOR-100/XOR-100-baseline.py` can get the experimental results of other baselines we compared.

3. The folder named `Prostate` contains the experimental code for reproducing the Prostate dataset mentioned in our paper. Running `Prostate/PRISM_Prostate_gpu.py` can get the experimental results of our algorithm PRISM.

6. It should be noted that the baseline method we compared is mainly implemented using functions in the `skfeature` library. The `skfeature` library needs to be installed before running the code file. The specific experimental settings are shown in Appendix C of our paper.




