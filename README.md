# Reinforcement Learning Papers
Related papers for Reinforcement Learning (we mainly focus on single-agent).

Since there are tens of thousands of new papers on reinforcement learning in each conference every year, we are only able to list those we read and consider as insightful.

## Contents 
- <a href="#Model Free Online">Model Free (Online) RL</a><br>
- <a href="#Model Based Online">Model Based (Online) RL</a><br>
- <a href="#Model Free Offline">(Model Free) Offline RL</a><br>
- <a href="#Model Based Offline">Model Based Offline RL</a><br>
- <a href="#Meta Reinforcement Learning">Meta RL</a><br>
- <a href="#Adversarial Reinforcement Learning">Adversarial RL</a><br>
- <a href="#Genaralization in RL">Genaralisation in RL</a><br>
- <a href="#Sequence Generation">RL as Sequence Generation</a><br>
- <a href="#Unsupervised RL">Unsupervised RL</a><br>
- <a href="#Lifelong RL">Lifelong RL</a><br>
- <a href="#Tutorial and Lesson">Tutorial and Lesson</a><br>

<a id='Model Free Online'></a>
## Model Free (Online) RL

### Classic Methods

|  Title | Method | Conference | on/off policy | Action Space | Policy | Description |
| ----  | ----   | ----       |   ----  | ----  |  ---- |  ---- | 
| [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf), [\[other link\]](http://www.kreimanlab.com/academia/classes/BAI/pdfs/MnihEtAlHassibis15NatureControlDeepRL.pdf) | DQN | Nature15 | off | Discrete | based on value function | use deep neural network to train q learning and reach the human level in the Atari games; mainly two trick: replay buffer for improving sample efficiency, decouple target network and behavior network |
| [Deep reinforcement learning with double q-learning](https://arxiv.org/pdf/1509.06461.pdf) | Double DQN | AAAI16 | off | Discrete | based on value function | find that the Q function in DQN may overestimate; decouple calculating q function and choosing action with two neural networks |
| [Dueling network architectures for deep reinforcement learning](https://arxiv.org/pdf/1511.06581.pdf) | Dueling DQN | ICML16 | off| Discrete | based on value function | use the same neural network to approximate q function and value function for calculating advantage function |
| [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) | Priority Sampling | ICLR16 | off | Discrete | based on value function | give different weights to the samples in the replay buffer (e.g. TD error) |
| [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf) | Rainbow | AAAI18 | off | Discrete | based on value function | combine different improvements to DQN: Double DQN, Dueling DQN, Priority Sampling, Multi-step learning, Distributional RL, Noisy Nets |
| [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | PG | NeurIPS99 | on/off | Continuous or Discrete | function approximation | propose Policy Gradient Theorem: how to calculate the gradient of the expected cumulative return to policy |
| ---- | AC/A2C | ---- | on/off | Continuous or Discrete | parameterized neural network | AC: replace the return in PG with q function approximator to reduce variance; A2C: replace the q function in AC with advantage function to reduce variance |
| [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) | A3C | ICML16 | on/off | Continuous or Discrete | parameterized neural network | propose three tricks to improve performance: (i) use different agents to interact with the environment; (ii) value function and policy share network parameters; (iii) modify the loss function (mse of value function + pg loss + policy entropy)|
| [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) | TRPO | ICML15 | on | Continuous or Discrete | parameterized neural network | introduce trust region to policy optimization for guaranteed monotonic improvement |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) | PPO | arxiv17 | on | Continuous or Discrete | parameterized neural network | replace the hard constraint of TRPO with a penalty by clipping the coefficient |
| [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) | DPG | ICML14 | off | Continuous | function approximation | consider deterministic policy for continuous action space and prove Deterministic Policy Gradient Theorem; use a stochastic behaviour policy for encouraging exploration |
| [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) | DDPG | ICLR16 | off | Continuous | parameterized neural network | adapt the ideas of DQN to DPG: (i) deep neural network function approximators, (ii) replay buffer, (iii) fix the target q function at each epoch |
| [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf) | TD3 | ICML18 | off | Continuous | parameterized neural network | adapt the ideas of Double DQN to DDPG: taking the minimum value between a pair of critics to limit overestimation |
| [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf) | SQL | ICML17 | off | main for Continuous | parameterized neural network | consider max-entropy rl and propose soft q iteration as well as soft q learning |
| [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf), [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf), [\[appendix\]](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf) | SAC | ICML18 | off | main for Continuous | parameterized neural network | base the theoretical analysis of SQL and extend soft q iteration (soft q evaluation + soft q improvement); reparameterize the policy and use two parameterized value functions; propose SAC |

### Current methods

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/pdf/2005.12729.pdf) | ---- | ICLR20 | show that the improvement of performance is related to code-level optimizations |
| [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) | ---- | ICLR21 | do a large scale empirical study to evaluate different tricks for on-policy algorithms on MuJoCo |
| [Randomized Ensemble Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/pdf/2101.05982.pdf) | REDQ | ICLR21 | consider three ingredients: (i) update q functions many times at every epoch; (ii) use an ensemble of Q functions; (iii) use the minimization across a random subset of Q functions from the ensemble for avoiding the overestimation; propose REDQ and achieve similar performance with model-based methods |
| [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) | MME | NeurIPS21 | find that SAC may fail in explore states with low entropy (arrive states with high entropy and increase their entropies); propose a max-min entropy framework to address this issue |
| [SO(2)-Equivariant Reinforcement Learning](https://arxiv.org/pdf/2203.04439.pdf) | Equi DQN, Equi SAC | ICLR22 Spotlight | consider to learn transformation-invariant policies and value functions; define and analyze group equivariant MDPs |
| [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431.pdf) | CoBERL | ICLR22 Spotlight | propose Contrastive BERT for RL (COBERL) that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency |
| [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/pdf?id=ZkC8wKoLbQ7) | InFeR | ICLR22 Spotlight | propose that deep RL agents lose some of their capacity to quickly fit new prediction tasks during training; propose InFeR to regularize a set of network outputs towards their initial values |
| [On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.01648.pdf) | ---- | ICLR22 Spotlight | consider lottery ticket hypothesis in deep reinforcement learning |
| [Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration](https://arxiv.org/pdf/2202.04628.pdf) | LOGO | ICLR22 Spotlight | consider the sparse reward challenges in RL; propose LOGO that exploits the offline demonstration data generated by a sub-optimal behavior policy; each step of LOGO contains a policy improvement step via TRPO and an additional policy guidance step by using the sub-optimal behavior policy |
| [Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation](https://arxiv.org/pdf/2201.01666.pdf) | IV-RL | ICLR22 Spotlight | analyze the sources of uncertainty in the supervision of modelfree DRL algorithms, and show that the variance of the supervision noise can be estimated with negative log-likelihood and variance ensembles |
| [Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning](https://arxiv.org/pdf/2201.09765.pdf) | GPM | ICLR22 Spotlight | focus on generating consistent actions for model-free RL, and borrow ideas from Model-based planning and action-repeat; use the policy to generate multi-step actions |
| [When should agents explore?](https://arxiv.org/pdf/2108.11811.pdf) | ---- | ICLR22 Spotlight | consider when to explore and propose to choose a heterogeneous mode-switching behavior policy |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that standard maximum entropy RL is robust to some disturbances in the dynamics and the reward function |
| [Maximizing Ensemble Diversity in Deep Reinforcement Learning](https://openreview.net/pdf?id=hjd-kcpDpf2) | MED-RL | ICLR22 |  |
| [Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities](https://openreview.net/pdf?id=zBOI9LFpESK) | AMBS | ICLR22 |  |
| [Large Batch Experience Replay](https://arxiv.org/pdf/2110.01528.pdf) |  LaBER | ICML22 oral | cast the replay buffer sampling problem as an importance sampling one for estimating the gradient and derive the theoretically optimal sampling distribution |
| Do Differentiable Simulators Give Better Gradients for Policy Optimization? || ICML22 oral ||
| Federated Reinforcement Learning: Communication-Efficient Algorithms and Convergence Analysis || ICML22 oral ||
| The Importance of Non-Markovianity in Maximum State Entropy Exploration | | ICML22 oral |  |
| An Analytical Update Rule for General Policy Optimization || ICML22 oral ||
| Align-RUDDER: Learning From Few Demonstrations by Reward Redistribution || ICML22 oral ||
| [Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/pdf/2201.12417.pdf) | ---- | ICML22 | aim to better understand the relationship between the Bellman error and the accuracy of value functions through theoretical analysis and empirical study; point out that the Bellman error is a poor replacement for value error, including (i) The magnitude of the Bellman error hides bias, (ii) Missing transitions breaks the Bellman equation |
| [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf) |  | ICML22 |  |
| Stabilizing Off-Policy Deep Reinforcement Learning from Pixels |  | ICML22 |  |
| Understanding Policy Gradient Algorithms: A Sensitivity-Based Approach |  | ICML22 |  |
| Mirror Learning: A Unifying Framework of Policy Optimisation || ICML22 ||
| Communicating via Maximum Entropy Reinforcement Learning || ICML22 ||
| Continuous Control with Action Quantization from Demonstrations || ICML22 ||
| Off-Policy Fitted Q-Evaluation with Differentiable Function Approximators: Z-Estimation and Inference Theory || ICML22 ||
| Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning || ICML22 ||
| Guarantees for Epsilon-Greedy Reinforcement Learning with Function Approximation || ICML22 ||
| Efficient Reinforcement Learning in Block MDPs: A Model-free Representation Learning approach || ICML22 ||
| On Well-posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-policy Evaluation || ICML22 ||
| A Temporal-Difference Approach to Policy Gradient Estimation || ICML22 ||
| Branching Reinforcement Learning || ICML22 ||
| The Primacy Bias in Deep Reinforcement Learning || ICML22 ||
| Stabilizing Q-learning with Linear Architectures for Provable Efficient Learning || ICML22 ||
| Optimizing Sequential Experimental Design with Deep Reinforcement Learning || ICML22 ||
| The Geometry of Robust Value Functions || ICML22 ||
| Direct Behavior Specification via Constrained Reinforcement Learning || ICML22 ||
| Utility Theory for Markovian Sequential Decision Making || ICML22 ||
| Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks || ICML22 ||
| Unifying Approximate Gradient Updates for Policy Optimization || ICML22 ||
| EqR: Equivariant Representations for Data-Efficient Reinforcement Learning || ICML22 ||
| Provable Reinforcement Learning with a Short-Term Memory || ICML22 ||
| Optimal Estimation of Off-Policy Policy Gradient via Double Fitted Iteration || ICML22 ||
| Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments || ICML22 ||
| Lagrangian Method for Q-Function Learning (with Applications to Machine Translation) || ICML22 ||
| Learning to Assemble with Large-Scale Structured Reinforcement Learning || ICML22 ||
| Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning || ICML22 ||
| Off-Policy Reinforcement Learning with Delayed Rewards || ICML22 ||
| Reachability Constrained Reinforcement Learning || ICML22 ||
| Improving Policy Optimization with Generalist-Specialist Learning || ICML22 ||


<a id='Model Based Online'></a>
## Model Based (Online) RL

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Value-Aware Loss Function for Model-based Reinforcement Learning](http://proceedings.mlr.press/v54/farahmand17a/farahmand17a-supp.pdf) | VAML | AISTATS17 | propose to train model by using the difference between TD error rather than KL-divergence |
| [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/pdf/1802.10592.pdf) | ME-TRPO | ICLR18 | analyze the behavior of vanilla MBRL methods with DNN; propose ME-TRPO with two ideas: (i) use an ensemble of models, (ii)  use likelihood ratio derivatives; significantly reduce the sample complexity compared to model-free methods |
| [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf) | MVE | ICML18 | use a dynamics model to simulate the short-term horizon and Q-learning to estimate the long-term value beyond the simulation horizon; use the trained model and the policy to estimate k-step value function for updating value function |
| [Iterative value-aware model learning](https://proceedings.neurips.cc/paper/2018/file/7a2347d96752880e3d58d72e9813cc14-Paper.pdf) | IterVAML | NeurIPS18 | replace e the supremum in VAML with the current estimate of the value function |
| [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/pdf/1807.01675.pdf) | STEVE | NeurIPS18 | an extension to MVE; only utilize roll-outs without introducing significant errors |
| [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf) | PETS | NeurIPS18 | propose PETS that incorporate uncertainty via an ensemble of bootstrapped models |
| [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/pdf/1807.03858.pdf)  | SLBO | ICLR19 | propose a novel algorithmic framework for designing and analyzing model-based RL algorithms with theoretical guarantees: provide a lower bound of the true return satisfying some properties s.t. optimizing this lower bound can actually optimize the true return |
| [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/pdf/1906.08253.pdf) | MBPO | NeurIPS19  | propose MBPO with monotonic model-based improvement; theoretically discuss how to choose k for model rollouts |
| [Model Based Reinforcement Learning for Atari](https://arxiv.org/pdf/1903.00374.pdf) | SimPLe | ICLR20 | first successfully handle ALE benchmark with model-based method with some designs: (i) deterministic Model; (ii) well-designed loss functions; (iii) scheduled sampling; (iv) stochastic Models |
| [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/pdf/1912.01603.pdf) | Dreamer | ICLR20 | solve long-horizon tasks from images purely by latent imagination; test in image-based MuJoCo |
| [Bidirectional Model-based Policy Optimization](https://arxiv.org/pdf/2007.01995.pdf) | BMPO | ICML20 | an extension to MBPO; consider both forward dynamics model and backward dynamics model |
| [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf)| M2AC | NeurIPS20 | an extension to MBPO; use model rollouts only when the model is confident |
| [MBRL-Lib: A Modular Library for Model-based Reinforcement Learning](https://arxiv.org/pdf/2104.10159.pdf) | ---- | arxiv21 | a codebase for MBRL |
| [Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193.pdf) | Dreamerv2 | ICLR21 | solve long-horizon tasks from images purely by latent imagination; test in image-based Atari |
| [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf) | EfficientZero | NeurIPS21 | first achieve super-human performance on Atari games with limited data; propose EfficientZero with three components: (i) use self-supervised learning to learn a temporally consistent environment model, (ii) learn the value prefix in an end-to-end manner, (iii) use the learned model to correct off-policy value targets |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | an extension to MBPO; automatically schedule the real data ratio as well as other hyperparameters for MBPO |
| [Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice](https://arxiv.org/pdf/2106.14080.pdf) | ---- | arxiv22 | bridge the gap in theory and practice of value-aware model learning (VAML) for model-based RL |
| [Model-Based Reinforcement Learning via Imagination with Derived Memory](https://openreview.net/pdf?id=jeATherHHGj) | IDM | NeurIPS21 |  |
| [Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/pdf/2204.01464.pdf) | VaGraM | ICLR22 Spotlight | consider the objective mismatch problem in MBRL; propose VaGraM by rescaling the MSE loss function with gradient information from the current value function estimate |
| [Constrained Policy Optimization via Bayesian World Models](https://arxiv.org/pdf/2201.09802.pdf) | LAMBDA | ICLR22 Spotlight | consider Bayesian model-based methods for CMDP |
| [On-Policy Model Errors in Reinforcement Learning](https://arxiv.org/pdf/2110.07985.pdf) | OPC | ICLR22 |  |
| [Generalised Policy Improvement with Geometric Policy Composition](https://arxiv.org/pdf/2206.08736.pdf) | GSPs | ICML22 oral |  |
| Denoised MDPs: Learning World Models Better Than the World Itself |  | ICML22 |  |
| Towards Adaptive Model-Based Reinforcement Learning || ICML22 ||
| [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://arxiv.org/pdf/2110.14565.pdf) | DreamerPro | ICML22 | consider reconstruction-free MBRL; propose to learn the prototypes from the recurrent states of the world model, thereby distilling temporal structures from past observations and actions into the prototypes. |
| Adaptive Model Design for Markov Decision Process || ICML22 ||




<a id='Model Free Offline'></a>
## (Model Free) Offline RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) | BCQ | ICML19 | show that off-policy methods perform badly because of extrapolation error; propose batch-constrained reinforcement learning: maximizing the return as well as minimizing the mismatch between the state-action visitation of the policy and the state-action pairs contained in the batch |
| [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) | CQL | NeurIPS20 | propose CQL with conservative q function, which is a lower bound of its true value, since standard off-policy methods will overestimate the value function |
| [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf) | ---- | arxiv20 | tutorial about methods, applications and open problems of offline rl |
| [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/pdf/2110.01548.pdf) |  | NeurIPS21 |  |
| [DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization](https://arxiv.org/pdf/2112.04716.pdf) | DR3 | ICLR22 Spotlight | consider the implicit regularization effect of SGD in RL; based on theoretical analyses, propose an explicit regularizer, called DR3, and combine with offline RL methods |
| [Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning ](https://arxiv.org/pdf/2202.11566.pdf) | PBRL | ICLR22 Spotlight | consider the distributional shift and extrapolation error in offline RL; propose PBRL with bootstrapping, for uncertainty quantification, and an OOD sampling method as a regularizer |
| [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://openreview.net/pdf?id=FLA55mBee6Q) | COptiDICE | ICLR22 Spotlight | consider offline constrained reinforcement learning; propose COptiDICE to directly optimize the distribution of state-action pair with contraints |
| [Offline Reinforcement Learning with In-sample Q-Learning](https://arxiv.org/pdf/2110.06169.pdf) | IQL | ICLR22 |  |
| Offline reinforcement learning with implicit Q-learning || ICLR22 ||
| Adversarially Trained Actor Critic for Offline Reinforcement Learning |  | ICML22 oral |  |
| Learning Bellman Complete Representations for Offline Policy Evaluation | | ICML22 oral | |
| Offline RL Policies Should Be Trained to be Adaptive | | ICML22 oral | |
| Pessimistic Q-Learning for Offline Reinforcement Learning: Towards Optimal Sample Complexity || ICML22 ||
| How to Leverage Unlabeled Data in Offline Reinforcement Learning? || ICML22 ||
| On the Role of Discount Factor in Offline Reinforcement Learning || ICML22 ||
| Model Selection in Batch Policy Optimization || ICML22 ||
| Koopman Q-learning: Offline Reinforcement Learning via Symmetries of Dynamics || ICML22 ||
| Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning || ICML22 ||
| Pessimism meets VCG: Learning Dynamic Mechanism Design via Offline Reinforcement Learning || ICML22 ||
| Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters || ICML22 ||
| Constrained Offline Policy Optimization || ICML22 ||



<a id='Model Based Offline'></a>
## Model Based Offline RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/pdf/2006.03647.pdf) | BREMEN | ICLR20 | propose deployment efficiency, to count the number of changes in the data-collection policy during learning (offline: 1, online: no limit); propose BERMEN with an ensemble of dynamics models for off-policy and offline rl |
| [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf) | MOPO | NeurIPS20 | observe that existing model-based RL algorithms can improve the performance of offline RL compared with model free RL algorithms; design MOPO by extending MBPO on uncertainty-penalized MDPs (new_reward = reward - uncertainty) |
| [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2005.05951.pdf) | MOReL | NeurIPS20 | present MOReL for model-based offline RL, including two steps: (a) learning a pessimistic MDP, (b) learning a near-optimal policy in this P-MDP |
| [Model-Based Offline Planning](https://arxiv.org/pdf/2008.05556.pdf) | MBOP | ICLR21 | learn a model for planning |
| [Representation Balancing Offline Model-Based Reinforcement Learning](https://openreview.net/pdf?id=QpNz8r_Ri2Y) | RepB-SDE | ICLR21 | focus on learning the representation for a robust model of the environment under the distribution shift and extend RepBM to deal with the curse of horizon; propose RepB-SDE framework for off-policy evaluation and offline rl |
| [Conservative Objective Models for Effective Offline Model-Based Optimization](https://arxiv.org/pdf/2107.06882.pdf) | COMs | ICML21 | consider offline model-based optimization (MBO, optimize an unknown function only with some samples); add a regularizer (resemble adversarial training methods) to the objective forlearning conservative objective models |
| [COMBO: Conservative Offline Model-Based Policy Optimization](https://arxiv.org/pdf/2102.08363v1.pdf) | COMBO | NeurIPS21 | try to optimize a lower bound of performance without considering uncertainty quantification; extend CQL with model-based methods|
| [Weighted Model Estimation for Offline Model-Based Reinforcement Learning](https://openreview.net/pdf?id=zdC5eXljMPy) | ---- | NeurIPS21 | address the covariate shift issue by re-weighting the model losses for different datapoints |
| [Revisiting Design Choices in Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2110.04135.pdf) | ---- | ICLR22 Spotlight | conduct a rigorous investigation into a series of these design choices for Model-based Offline RL |
| [Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage](https://arxiv.org/pdf/2107.06226.pdf) | CPPO | ICLR22 |  |
| [Pareto Policy Pool for Model-based Offline Reinforcement Learning](https://openreview.net/pdf?id=OqcZu8JIIzS) |  | ICLR22 |  |
| [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) | Diffuser | ICML22 oral | first design a denoising diffusion model for trajectory data and an associated probabilistic framework for behavior synthesis |
| Regularizing a Model-based Policy Stationary Distribution to Stabilize Offline Reinforcement Learning || ICML22 ||


<a id='Meta Reinforcement Learning'></a>
## Meta RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [RL2 : Fast reinforcement learning via slow reinforcement learning](https://arxiv.org/pdf/1611.02779.pdf) | RL2 | arxiv16 | view the learning process of the agent itself as an objective; structure the agent as a recurrent neural network to store past rewards, actions, observations and termination flags for adapting to the task at hand when deployed |
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://www.cs.utexas.edu/users/sniekum/classes/RL-F17/papers/Meta.pdf) | MAML | ICML17 | propose a general framework for different learning problems, including classification, regression andreinforcement learning; the main idea is to optimize the parameters to quickly adapt to new tasks (with a few steps of gradient descent) |
| [Meta reinforcement learning with latent variable gaussian processes](https://arxiv.org/pdf/1803.07551.pdf) | ---- | arxiv18 |  |
| [Learning to adapt in dynamic, real-world environments through meta-reinforcement learning](https://arxiv.org/pdf/1803.11347.pdf) | ReBAL, GrBAL | ICLR18 | consider learning online adaptation in the context of model-based reinforcement learning |
| [Meta-Learning by Adjusting Priors Based on Extended PAC-Bayes Theory](https://arxiv.org/pdf/1711.01244.pdf) | ---- | ICML18 | extend various PAC-Bayes bounds to meta learning |
| [Meta reinforcement learning of structured exploration strategies](https://arxiv.org/pdf/1802.07245.pdf) |  | NeurIPS18 |  |
| [Meta-learning surrogate models for sequential decision making](https://arxiv.org/pdf/1903.11907.pdf) |  | arxiv19 |  |
| [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/pdf/1903.08254.pdf) | PEARL | ICML19 | encode past tasks’ experience with probabilistic latent context and use inference network to estimate the posterior|
| [Fast context adaptation via meta-learning](https://arxiv.org/pdf/1810.03642.pdf) | CAVIA | ICML19 | propose CAVIA as an extension to MAML that is less prone to meta-overfitting, easier to parallelise, and more interpretable; partition the model parameters into two parts: context parameters and shared parameters, and only update the former one in the test stage |
| [Taming MAML: Efficient Unbiased Meta-Reinforcement Learning](http://proceedings.mlr.press/v97/liu19g/liu19g.pdf) |  | ICML19 |  |
| [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](http://proceedings.mlr.press/v100/yu20a/yu20a.pdf) | Meta World | CoRL19 | an envoriment for meta RL as well as multi-task RL |
| [Guided meta-policy search](https://arxiv.org/pdf/1904.00956.pdf) | GMPS | NeurIPS19 | consider the sample efficiency during the meta-training process by using supervised imitation learning; |
| [Meta-Q-Learning](https://arxiv.org/pdf/1910.00125.pdf) | MQL | ICLR20 | an off-policy algorithm for meta RL andbuilds upon three simple ideas: (i) Q Learning with context variable represented by pasttrajectories is competitive with SOTA; (ii) Multi-task objective is useful for meta RL; (iii) Past data from the meta-training replay buffer can be recycled |
| [Varibad: A very good method for bayes-adaptive deep RL via meta-learning](https://arxiv.org/pdf/1910.08348.pdf) | variBAD | ICLR20 | represent a single MDP M using a learned, low-dimensional stochastic latent variable m; jointly meta-train a variational auto-encoder that can infer the posterior distribution over m in a new task, and a policy that conditions on this posterior belief over MDP embeddings |
| [On the global optimality of modelagnostic meta-learning](https://arxiv.org/pdf/2006.13182.pdf), [ICML version](http://proceedings.mlr.press/v119/wang20b/wang20b-supp.pdf) | ---- | ICML20 | characterize the optimality gap of the stationary points attained by MAML for both rl and sl |
| [Meta-reinforcement learning robust to distributional shift via model identification and experience relabeling](https://arxiv.org/pdf/2006.07178.pdf) | MIER | arxiv20 |  |
| [FOCAL: Efficient fully-offline meta-reinforcement learning via distance metric learning and behavior regularization](https://arxiv.org/pdf/2010.01112.pdf) | FOCAL | ICLR21 | first consider offline meta-reinforcement learning; propose FOCAL based on PEARL |
| [Offline meta reinforcement learning with advantage weighting](https://arxiv.org/pdf/2008.06043.pdf) | MACAW | ICML21 | introduce the offline meta reinforcement learning problem setting; propose an optimization-based meta-learning algorithm named MACAW that uses simple, supervised regression objectives for both the inner and outer loop of meta-training |
| [Improving Generalization in Meta-RL with Imaginary Tasks from Latent Dynamics Mixture](https://arxiv.org/pdf/2105.13524.pdf) | LDM | NeurIPS21 | aim to train an agent that prepares for unseen test tasks during training, propose to train a policy on mixture tasks along with original training tasks for preventing the agent from overfitting the training tasks |
| [Unifying Gradient Estimators for Meta-Reinforcement Learning via Off-Policy Evaluation](https://arxiv.org/pdf/2106.13125.pdf) | ---- | NeurIPS21 | present a unified framework for estimating higher-order derivatives of value functions, based on the concept of off-policy evaluation, for gradient-based meta rl |
| [Generalization of Model-Agnostic Meta-Learning Algorithms: Recurring and Unseen Tasks](https://arxiv.org/pdf/2102.03832.pdf) | ---- | NeurIPS21 |  |
| [Offline Meta Learning of Exploration](https://arxiv.org/pdf/2008.02598.pdf), [Offline Meta Reinforcement Learning -- Identifiability Challenges and Effective Data Collection Strategies](https://openreview.net/pdf?id=IBdEfhLveS) | BOReL | NeurIPS21 |  |
| [On the Convergence Theory of Debiased Model-Agnostic Meta-Reinforcement Learning](https://arxiv.org/pdf/2002.05135.pdf) | SG-MRL | NeurIPS21 |  |
| [Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL](https://arxiv.org/pdf/2112.00901.pdf) | ---- | NeurIPS21 |  |
| [Generalization Bounds for Meta-Learning via PAC-Bayes and Uniform Stability](https://arxiv.org/pdf/2102.06589.pdf) | ---- | NeurIPS21 | provide generalization bound on meta-learning by combining PAC-Bayes thchnique and uniform stability |
| [Bootstrapped Meta-Learning](https://arxiv.org/pdf/2109.04504.pdf) | BMG | ICLR22 Oral | propose BMG to let the metalearner teach itself for tackling ill-conditioning problems and myopic metaobjectives in meta learning; BGM introduces meta-bootstrap to mitigate myopia and formulate the meta-objective in terms of minimising distance to control curvature |
| [Model-Based Offline Meta-Reinforcement Learning with Regularization](https://arxiv.org/pdf/2202.02929.pdf) | MerPO, RAC | ICLR22 | empirically point out that offline Meta-RL could be outperformed by offline single-task RL methods on tasks with good quality of datasets; consider how to learn an informative offline meta-policy in order to achieve the optimal tradeoff between “exploring” the out-of-distribution state-actions by following the meta-policy and “exploiting” the offline dataset by staying close to the behavior policy; propose MerPO which learns a meta-model for efficient task structure inference and an informative meta-policy for safe exploration of out-of-distribution state-actions |
| [Skill-based Meta-Reinforcement Learning](https://openreview.net/pdf?id=jeLW-Fh9bV) | SiMPL | ICLR22 | propose a method that jointly leverages (i) a large offline dataset of prior experience collected across many tasks without reward or task annotations and (ii) a set of meta-training tasks to learn how to quickly solve unseen long-horizon tasks. |
| [Hindsight Foresight Relabeling for Meta-Reinforcement Learning](https://arxiv.org/pdf/2109.09031.pdf) | HFR | ICLR22 | focus on improving the sample efficiency of the meta-training phase via data sharing; combine relabeling techniques with meta-RL algorithms in order to boost both sample efficiency and asymptotic performance |
| [CoMPS: Continual Meta Policy Search](https://arxiv.org/pdf/2112.04467.pdf) | CoMPS | ICLR22 | first formulate the continual meta-RL setting, where the agent interacts with a single task at a time and, once finished with a task, never interacts with it again |
| [Learning a subspace of policies for online adaptation in Reinforcement Learning](https://arxiv.org/pdf/2110.05169.pdf) | ---- | ICLR22 | consider the setting with just a single train environment; propose an approach where we learn a subspace of policies within the parameter space |
| [Model-based Meta Reinforcement Learning using Graph Structured Surrogate Models and Amortized Policy Search](https://arxiv.org/pdf/2102.08291.pdf) | GSSM | ICML22 | consider model-based meta reinforcement learning, which consists of dynamics model learning and policy optimization; develop a graph structured dynamics model with superior generalization capability across tasks|
| Meta-Learning Hypothesis Spaces for Sequential Decision-making || ICML22 ||
| Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning || ICML22 ||
| Transformers are Meta-Reinforcement Learners || ICML22 ||
| Offline Meta-Reinforcement Learning with Online Self-Supervision || ICML22 ||



<a id='Adversarial Reinforcement Learning'></a>
## Adversarial RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Adversarial Attacks on Neural Network Policies](https://arxiv.org/pdf/1702.02284.pdf) | ---- | ICLR 2017 workshop | first show that existing rl policies coupled with deep neural networks are vulnerable to adversarial noises in white-box and black-box settings | 
| [Delving into Adversarial Attacks on Deep Policies](https://arxiv.org/pdf/1705.06452.pdf) | ---- | ICLR 2017 workshop | show rl algorithms are vulnerable to adversarial noises; show adversarial training can improve robustness |
| [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702.pdf) | RARL | ICML17 | formulate the robust policy learning as a zero-sum, minimax objective function |
| [Stealthy and Efficient Adversarial Attacks against Deep Reinforcement Learning](https://arxiv.org/pdf/2005.07099.pdf) | Critical Point Attack, Antagonist Attack | AAAI20 |  critical point attack: build a model to predict the future environmental states and agent’s actions for attacking; antagonist attack: automatically learn a domain-agnostic model for attacking |
| [Safe Reinforcement Learning in Constrained Markov Decision Processes](https://arxiv.org/pdf/2008.06626.pdf) | SNO-MDP | ICML20 | explore and optimize Markov decision processes under unknown safety constraints |
| [Robust Deep Reinforcement Learning Against Adversarial Perturbations on State Observations](https://arxiv.org/pdf/2003.08938.pdf) | SA-MDP | NeurIPS20 | formalize adversarial attack on state observation as SA-MDP; propose some novel attack methods: Robust SARSA and Maximal Action Difference; propose a defence framework and some practical methods: SA-DQN, SA-PPO and SA-DDPG |
| [Robust Reinforcement Learning on State Observations with Learned Optimal Adversary](https://arxiv.org/pdf/2101.08452.pdf) | ATLA | ICLR21 | use rl algorithms to train an "optimal" adversary; alternatively train "optimal" adversary and robust agent |
| [Robust Deep Reinforcement Learning through Adversarial Loss](https://arxiv.org/pdf/2008.01976.pdf) | RADIAL-RL | NeurIPS21 | propose a robust rl framework, which penalizes the overlap between output bounds of actions; propose a more efficient evaluation method (GWC) to measure attack agnostic robustness | 
| [Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/pdf/2106.11420.pdf) | Policy Smoothing | ICLR22 | introduce randomized smoothing into RL; propose adaptive Neyman-Person Lemma |
| [CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing](https://arxiv.org/pdf/2106.09292.pdf) | CROP | ICLR22 | present a framework of Certifying Robust Policies for RL (CROP) against adversarial state perturbations with two certification criteria: robustness of per-state actions and lower bound of cumulative rewards; theoretically prove the certification radius; conduct experiments to provide certification for six empirically robust RL algorithms on Atari |
| [Policy Gradient Method For Robust Reinforcement Learning](https://arxiv.org/pdf/2205.07344.pdf) |  | ICML22 |  |
| SAUTE RL: Toward Almost Surely Safe Reinforcement Learning Using State Augmentation || ICML22 ||
| Constrained Variational Policy Optimization for Safe Reinforcement Learning || ICML22 ||
| Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum || ICML22 ||
| Distributionally Robust Q-Learning || ICML22 ||
| Robust Meta-learning with Sampling Noise and Label Noise via Eigen-Reptile || ICML22 ||
| DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck || ICML22 ||

<a id='Genaralization in RL'></a>
## Genaralisation in RL

| Title | Method | Conference | Description | 
| ----  | ----   | ----       |   ----  |
| [On the Generalization Gap in Reparameterizable Reinforcement Learning](https://arxiv.org/pdf/1905.12654.pdf) | ---- | ICML19 | theoretically provide guarantees on the gap between the expected and empirical return for both intrinsic and external errors in reparameterizable RL |
| [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/pdf/1812.02341.pdf) | CoinRun | ICML19 | introduce a new environment called CoinRun for generalisation in RL; empirically show L2 regularization, dropout, data augmentation and batch normalization can improve generalization in RL |
| [Investigating Generalisation in Continuous Deep Reinforcement Learning](https://arxiv.org/pdf/1902.07015.pdf) | ---- | arxiv19 | study generalisation in Deep RL for continuous control |
| [Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability](https://arxiv.org/pdf/2107.06277.pdf) | LEEP | NeurIPS21 | generalisation in RL induces implicit partial observability; propose LEEP to use an ensemble of policies to approximately learn the Bayes-optimal policy for maximizing test-time performance |
| [Automatic Data Augmentation for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf) | DrAC | NeurIPS21 | focus on automatic data augmentation based two novel regularization terms for the policy and value function |
| [When Is Generalizable Reinforcement Learning Tractable?](https://arxiv.org/pdf/2101.00300.pdf) | ---- | NeurIPS21 | propose Weak Proximity and Strong Proximity for theoretically analyzing the generalisation of RL |
| [A Survey of Generalisation in Deep Reinforcement Learning](https://arxiv.org/pdf/2111.09794.pdf) | ---- | arxiv21 | provide a unifying formalism and terminology for discussing different generalisation problems |
| [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) | CTRL | ICLR22 | consider zero-shot generalization (ZSG); use self-supervised learning to learn a representation across tasks |
| [The Role of Pretrained Representations for the OOD Generalization of RL Agents](https://arxiv.org/pdf/2107.05686.pdf) | ---- | ICLR22 |  |
| [Generalisation in Lifelong Reinforcement Learning through Logical Composition](https://openreview.net/pdf?id=ZOcX-eybqoL) | ---- | ICLR22 |  |
| [A Generalist Agent](https://arxiv.org/pdf/2205.06175.pdf) | Gato | arxiv22 | [slide](https://ml.cs.tsinghua.edu.cn/~chengyang/reading_meeting/Reading_Meeting_20220607.pdf) |

<a id='Sequence Generation'></a>
## RL as Sequence Generation

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (return-to-go, state, action, return-to-go, ...); there is not explicit optimization process; evaluate on Offline RL |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (s_0^0, ..., s_0^N, a_0^0, ..., a_0^M, r_0, ...); use beam search to inference; evaluate on imitation learning, goal-conditioned RL and Offline RL | 
| Online Decision Transformer |  | ICML22 oral |  |
| Prompting Decision Transformer for Few-shot Policy Generalization || ICML22 ||

<a id='Unsupervised RL'></a>
## Unsupervised RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/pdf/1802.06070.pdf) | DIAYN | ICLR19 | learn diverse skills in environments without any rewards by maximizing an information theoretic objective |
| [Decoupling representation learning from reinforcement learning](https://arxiv.org/pdf/2009.08319.pdf) | ATC | ICML21 | propose a new unsupervised task tailored to reinforcement learning named Augmented Temporal Contrast (ATC), which borrows ideas from Contrastive learning; benchmark several leading Unsupervised Learning algorithms by pre-training encoders on expert demonstrations and using them in RL agents|
| [Pretraining representations for data-efficient reinforcement learning](https://arxiv.org/pdf/2106.04799.pdf) | SGI | NeurIPS21 | consider to pretrian with unlabeled data and finetune on a small amount of task-specific data to improve the data efficiency of RL; employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL |
| [Understanding the World Through Action](https://arxiv.org/pdf/2110.12543.pdf) | ---- | CoRL21 | discusse how self-supervised reinforcement learning combined with offline RL can enable scalable representation learning |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | ---- | ICLR22 | show that unsupervised skill discovery algorithms based on mutual information maximization do not learn skills that are optimal for every possible reward function; provide a geometric perspective on some skill learning methods |
| The Unsurprising Effectiveness of Pre-Trained Vision Models for Control || ICML22 oral ||
| Reinforcement Learning with Action-Free Pre-Training from Videos || ICML22 ||


<a id='Lifelong RL'></a>
## Lifelong RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |



<a id='Tutorial and Lesson'></a>
## Tutorial and Lesson

| Tutorial and Lesson |
| ---- |
| [Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](https://d1wqtxts1xzle7.cloudfront.net/54674740/Reinforcement_Learning-with-cover-page-v2.pdf?Expires=1641130151&Signature=eYy7kmTVqTXFcANS-9GZJUyb86cDqKeh2QX8VvzjouEM-QSfuiCm1WHhP~bW5C57Mecj6en~YRoTvxekzU5lq~UaHSBoc-7xP8dXBp91shcwdfJ8M0LUkktpqcQjXQi7ZzhGn33qZeah0p8S06ARzjimF5coL5arvp9yANAsy4KigXSZwAZNXxksKwqUAult2QseLL~Bv1p2locjYahRzTuex3vMxdBLhT9HOGFF0qOdKYxsWiaITUKnVYl8AvePDHEEXgfmuqEfjqjF5p~FHOsYl3gEDZOvUp1eUzPg2~i0MQXY49nUpzsThL5~unTRIsYJiBghnkYl8py0r~UelQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| [Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) | 
| [Deep Reinforcement Learning, CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| [Deep Reinforcement Learning and Control, CMU 10703](https://katefvision.github.io/) |
| [RLChina](http://rlchina.org/topic/9) |
