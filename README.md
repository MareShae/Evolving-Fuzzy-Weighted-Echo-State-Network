# Evolving Fuzzy Weighted Echo State Network

## FWESN
The original Fuzzy Weighted Echo State Network (FWESN) proposed by [Yao Zhao and Yingshun Li](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2021.825526/full) [1] is a hybrid prediction model that improves upon the performance of Echo State Networks (ESN) by combining it with the concepts of the Tagaki-Sugeno models.

## eFWESN
The FWESN utilizes the fuzzy c-mean clustering method that produces a static representation of the training batch, and a mean square error that calculates the output weight, W_{out}, once. While efficient due to the generalization and stability of batch processing, it is unable to adapt to new information and requires retraining the entire system from scratch.

The main aim of this project is targeted changes that allow for online training. It does this by modifying two elements of the proposed FWESN including:
+ fuzzy c-mean → evolving membership via firing strength criteria, and
+ mean square error → recursive least squares.

![naive architecture](/assets/naive%20architecture.svg)

The *evolving membership* allows the **eFWESN** to self-adapt to dynamically changing pattern in a single pass, while the *recursive least square* fine-tunes the system.


## Resources
[1] Yao Z and Li Y (2022) Fuzzy-Weighted Echo State Networks. Front. Energy Res. 9:825526. doi: 10.3389/fenrg.2021.825526

[2] Li Y, Liu H and Gao H (2024) Online learning fuzzy echo state network with applications on redundant manipulators. Front. Neurorobot. 18:1431034. doi: 10.3389/fnbot.2024.1431034

[3] Gu, X., Han, J., Shen, Q. et al. Autonomous learning for fuzzy systems: a review. Artif Intell Rev 56, 7549–7595 (2023). https://doi.org/10.1007/s10462-022-10355-6

[4] GeeksforGeeks. (2025, July 23). Echo State Network – an overview. GeeksforGeeks. https://www.geeksforgeeks.org/machine-learning/echo-state-network-an-overview/

[5] K. S. T. R. Alves, Simpl_eTS: Simplified evolving Takagi-Sugeno, GitHub repository. https://github.com/kaikerochaalves/Simpl_eTS-Simplified-evolving-Takagi-Sugeno