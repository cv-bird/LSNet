# LSNet
From Optimization to Network: A Low-Rank and Sparse-Aware Deep Unfolding Framework for Infrared Small Target Detection

# Abstract
Infrared (IR) search and track systems are widely applied in aerospace and defense fields. Infrared small target detection (IRSTD) in heavy clouds and chaotic terrestrial environments remains a challenging task. Traditional physics-based models suffer from limited applicability in real-world vision tasks due to their reliance on manually tuned parameters and poor generalization. In contrast, deep neural networks, though impressive, inherently function as black boxes, lacking interpretability and transparency. Model-driven deep unfolding networks offer a promising alternative by fusing domain knowledge with learnable structures, enabling efficient parameter optimization through the unfolding process. To this end, we propose a low-rank and sparse-aware deep unfolding framework, termed LSNet. Specifically, the decomposition problem is reformulated as an implicit prior regularization model, and the iterative optimization process is unfolded into a deep neural network architecture. To avoid computationally expensive matrix operations, generalized sparse transformation and singular value thresholding are jointly learned through a State Space Model (SSM), which captures long-range dependencies, and a spatially-attentive network, which integrates fine-grained structural information. This design ensures both high generalization ability and interpretability, while enabling efficient small target detection. Both qualitative and quantitative experiments demonstrate that our proposed LSNet outperforms 14 recent benchmark algorithms on multiple public datasets.

<p align="center">
<img src="Figures/deep.png">
</p>



# Code
The official repository of From Optimization to Network: A Low-Rank and Sparse-Aware Deep Unfolding Framework for Infrared Small Target Detection.
  ```
The key package has been provided.
  ```

# Contact
If you have any questions, please contact Yongji Li (liyj328@mail2.sysu.edu.cn).


