This is code for the submission "A Nearly Optimal Single Loop Algorithm for Stochastic Bilevel Optimization 
under Unbounded Smoothness"
## Abstract
This paper studies the problem of stochastic bilevel optimization where the upper-level function is nonconvex with potentially unbounded smoothness and the lower-level function is strongly convex. This problem is motivated by meta-learning applied to sequential data, such as text classification using recurrent neural networks, where the smoothness constant of the upper-level loss function
scales linearly with the gradient norm and can be potentially unbounded. Existing algorithm crucially relies on the nested loop design, which requires significant tuning efforts and is not practical. In this paper, we address this issue by proposing a Single Loop bIlevel oPtimizer (SLIP). The proposed algorithm first updates the lower-level variable by a few steps of stochastic gradient descent, and then simultaneously updates the upper-level variable by normalized stochastic gradient descent with momentum and the lower-level variable by stochastic gradient descent. Under standard assumptions, we show that our algorithm finds an $\epsilon$-stationary point within $\widetilde{\mathcal{O}}(1/\epsilon^4)$\footnote{Here $\widetilde{\mathcal{O}}(\cdot)$ compresses logarithmic factors of $1/\epsilon$ and $1/\delta$, where $\delta\in(0,1)$ denotes the failure probability.} oracle calls of stochastic gradient or Hessian-vector product, both in expectation and with high probability. This complexity result is nearly optimal up to logarithmic factors without mean-square smoothness of the stochastic gradient oracle. Our proof relies on (i) a refined characterization and control of the lower-level variable and (ii) establishing a novel connection between bilevel optimization and stochastic optimization under distributional drift. Our experiments on various tasks show that our algorithm significantly outperforms strong baselines in bilevel optimization.

###Requirements
PyTorch >= v1.6.0.

Here we provide an examples for running the baselines and our algorithms on Hyper-representatioin. Download the dataset and put in the SLIP-main/data. For example, you could download snli_1.0 from https://nlp.stanford.edu/projects/snli/.
### preprocess data, 
This step creates meta tasks and save them.

    python task_create 

### Running code on Hyper-representation
    python main.py --methods maml      --data snli --epochs 20 --inner_update_lr 1e-2 --outer_update_lr 1e-2 --inner_update_step 5
    
    python main.py --methods anil      --data snli --epochs 20 --inner_update_lr 1e-2 --outer_update_lr 5e-2 --inner_update_step 5

    python main.py --methods stocbio   --data snli --epochs 20 --inner_update_lr 1e-2 --outer_update_lr 1e-2 --hessian_lr 1e-1 --inner_update_step 5

    python main.py --methods ttsa      --data snli --epochs 20 --inner_update_lr 2e-2 --outer_update_lr 1e-1 --hessian_lr 1e-1 --inner_update_step 1

    python main.py --methods f2sa      --data snli --epochs 20 --inner_update_lr 5e-2 --outer_update_lr 5e-2 --lamb 1e-2 --incre_step 0.1 --nu 5e-2 --inner_update_step 1

    python main.py --methods saba      --data snli --epochs 20 --inner_update_lr 5e-2 --outer_update_lr 5e-2 --beta 0.0 --nu 5e-2 --inner_update_step 1

    python main.py --methods ma_soba   --data snli --epochs 20 --inner_update_lr 5e-2 --outer_update_lr 5e-2 --beta 0.9 --nu 1e-2 --inner_update_step 1

    python main.py --methods bo_rep    --data snli --epochs 20 --inner_update_lr 5e-2 --outer_update_lr 1e-1 --grad_normalized  True --y_warm_start 3 --interval 2 --beta 0.9 --nu 1e-2 --inner_update_step 1

    python main.py --methods slip      --data snli --epochs 20 --inner_update_lr 5e-2 --outer_update_lr 1e-1 --grad_normalized  True --y_warm_start 3  --interval 1 --beta 0.9 --nu 1e-2 --inner_update_step 1




