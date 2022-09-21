# Reinforcement Learning Papers
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Related papers for Reinforcement Learning (we mainly focus on single-agent).

Since there are tens of thousands of new papers on reinforcement learning in each conference every year, we are only able to list those we read and consider as insightful.

## Contents 
* [Model Free (Online) RL](#Model-Free-Online)
    - [Classic Methods](#model-free-classic)
    - [Off-Policy Evaluation](#off-policy-evaluation)
    - [Soft RL](#soft-rl)
    - [Current methods](#current)
* [Model Based (Online) RL](#Model-Based-Online)
    - [Classic Methods](#model-based-classic)
    - [World Models](#dreamer)
    - [CodeBase](#model-based-code)
* [(Model Free) Offline RL](#Model-Free-Offline)
* [Model Based Offline RL](#Model-Based-Offline)
* [Meta RL](#Meta-RL)
* [Adversarial RL](#Adversarial-RL)
* [Genaralisation in RL](#Genaralization-in-RL)
* [RL as Sequence Generation](#Sequence-Generation)
* [Unsupervised RL](#Unsupervised-RL)
* [Lifelong RL](#Lifelong-RL)
* [Tutorial and Lesson](#Tutorial-and-Lesson)

<!-- - <a href="#Model-Free-Online">Model Free (Online) RL</a><br>
- <a href="#Model-Based-Online">Model Based (Online) RL</a><br>
- <a href="#Model-Free-Offline">(Model Free) Offline RL</a><br>
- <a href="#Model-Based-Offline">Model Based Offline RL</a><br>
- <a href="#Meta-RL">Meta RL</a><br>
- <a href="#Adversarial-RL">Adversarial RL</a><br>
- <a href="#Genaralization-in-RL">Genaralisation in RL</a><br>
- <a href="#Sequence-Generation">RL as Sequence Generation</a><br>
- <a href="#Unsupervised-RL">Unsupervised RL</a><br>
- <a href="#Lifelong-RL">Lifelong RL</a><br>
- <a href="#Tutorial-and-Lesson">Tutorial and Lesson</a><br> -->

<a id='Model-Free-Online'></a>
## Model Free (Online) RL
<!-- ## <span id='Model-Free-Online'>Model Free (Online) RL</span>
### <span id='classic'>Classic Methods</span> -->

<a id='model-free-classic'></a>
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

<a id='off-policy-evaluation'></a>
### Off-Policy Evaluation
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Weighted importance sampling for off-policy learning with linear function approximation](https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf) | WIS-LSTD | NeurIPS14 |  |
| [Importance Sampling Policy Evaluation with an Estimated Behavior Policy](https://arxiv.org/pdf/1806.01347.pdf) | RIS | ICML19 |  |
| [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf) |  | ICML22 |  |
| [Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning](https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf) | LDR2OPE | ICML22 ||
| [On Well-posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-policy Evaluation](https://proceedings.mlr.press/v162/chen22u/chen22u.pdf) |  | ICML22 ||


<a id='soft-rl'></a>
### Soft RL
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) | MME | NeurIPS21 | find that SAC may fail in explore states with low entropy (arrive states with high entropy and increase their entropies); propose a max-min entropy framework to address this issue |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that standard maximum entropy RL is robust to some disturbances in the dynamics and the reward function |
| The Importance of Non-Markovianity in Maximum State Entropy Exploration | | ICML22 oral |  |
| Communicating via Maximum Entropy Reinforcement Learning || ICML22 ||

<!-- ### <span id='current'>Current methods</span> -->
<a id='current'></a>
### Current methods

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2004.13649.pdf) | DrQ | ICLR20 | propsoe to apply data augmentation with model-free methods to reach state-of-the-art performance in image-pixels tasks |
| [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/pdf/2005.12729.pdf) | ---- | ICLR20 | show that the improvement of performance is related to code-level optimizations |
| [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) | ---- | ICLR21 | do a large scale empirical study to evaluate different tricks for on-policy algorithms on MuJoCo |
| [Mirror Descent Policy Optimization](https://arxiv.org/pdf/2005.09814.pdf) | MDPO | ICLR21 |  |
| [Randomized Ensemble Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/pdf/2101.05982.pdf) | REDQ | ICLR21 | consider three ingredients: (i) update q functions many times at every epoch; (ii) use an ensemble of Q functions; (iii) use the minimization across a random subset of Q functions from the ensemble for avoiding the overestimation; propose REDQ and achieve similar performance with model-based methods |
| [Generalizable Episodic Memory for Deep Reinforcement Learning](https://arxiv.org/pdf/2103.06469.pdf) | GEM | ICML21 | propose to integrate the generalization ability of neural networks and the fast retrieval manner of episodic memory |
| [SO(2)-Equivariant Reinforcement Learning](https://arxiv.org/pdf/2203.04439.pdf) | Equi DQN, Equi SAC | ICLR22 Spotlight | consider to learn transformation-invariant policies and value functions; define and analyze group equivariant MDPs |
| [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431.pdf) | CoBERL | ICLR22 Spotlight | propose Contrastive BERT for RL (COBERL) that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency |
| [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/pdf?id=ZkC8wKoLbQ7) | InFeR | ICLR22 Spotlight | propose that deep RL agents lose some of their capacity to quickly fit new prediction tasks during training; propose InFeR to regularize a set of network outputs towards their initial values |
| [On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.01648.pdf) | ---- | ICLR22 Spotlight | consider lottery ticket hypothesis in deep reinforcement learning |
| [Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration](https://arxiv.org/pdf/2202.04628.pdf) | LOGO | ICLR22 Spotlight | consider the sparse reward challenges in RL; propose LOGO that exploits the offline demonstration data generated by a sub-optimal behavior policy; each step of LOGO contains a policy improvement step via TRPO and an additional policy guidance step by using the sub-optimal behavior policy |
| [Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation](https://arxiv.org/pdf/2201.01666.pdf) | IV-RL | ICLR22 Spotlight | analyze the sources of uncertainty in the supervision of modelfree DRL algorithms, and show that the variance of the supervision noise can be estimated with negative log-likelihood and variance ensembles |
| [Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning](https://arxiv.org/pdf/2201.09765.pdf) | GPM | ICLR22 Spotlight | focus on generating consistent actions for model-free RL, and borrow ideas from Model-based planning and action-repeat; use the policy to generate multi-step actions |
| [When should agents explore?](https://arxiv.org/pdf/2108.11811.pdf) | ---- | ICLR22 Spotlight | consider when to explore and propose to choose a heterogeneous mode-switching behavior policy |
| [Maximizing Ensemble Diversity in Deep Reinforcement Learning](https://openreview.net/pdf?id=hjd-kcpDpf2) | MED-RL | ICLR22 |  |
| [Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities](https://openreview.net/pdf?id=zBOI9LFpESK) | AMBS | ICLR22 |  |
| [Large Batch Experience Replay](https://arxiv.org/pdf/2110.01528.pdf) |  LaBER | ICML22 oral | cast the replay buffer sampling problem as an importance sampling one for estimating the gradient and derive the theoretically optimal sampling distribution |
| [Do Differentiable Simulators Give Better Gradients for Policy Optimization?](https://arxiv.org/pdf/2202.00817.pdf) | ---- | ICML22 oral | consider whether differentiable simulators give better policy gradients; show some pitfalls of First-order estimates and propose alpha-order estimates |
| Federated Reinforcement Learning: Communication-Efficient Algorithms and Convergence Analysis || ICML22 oral ||
| [An Analytical Update Rule for General Policy Optimization](https://arxiv.org/pdf/2112.02045.pdf) | ---- | ICML22 oral | provide a tighter bound for truse-region methods |
| [Generalised Policy Improvement with Geometric Policy Composition](https://arxiv.org/pdf/2206.08736.pdf) | GSPs | ICML22 oral | propose the concept of geometric switching policy (GSP), i.e., we have a set of policies and will use them to take action in turn, for each policy, we sample a number from the geometric distribution and take this policy such number of steps; consider policy improvement over nonMarkov GSPs |
| [Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/pdf/2201.12417.pdf) | ---- | ICML22 | aim to better understand the relationship between the Bellman error and the accuracy of value functions through theoretical analysis and empirical study; point out that the Bellman error is a poor replacement for value error, including (i) The magnitude of the Bellman error hides bias, (ii) Missing transitions breaks the Bellman equation |
| [Adaptive Model Design for Markov Decision Process](https://proceedings.mlr.press/v162/chen22ab/chen22ab.pdf) | ---- | ICML22 | consider Regularized Markov Decision Process and formulate it as a bi-level problem |
| [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://proceedings.mlr.press/v162/cetin22a/cetin22a.pdf) | A-LIX | ICML22 | propose that temporal-difference learning with a convolutional encoder and lowmagnitude reward will cause instabilities, which is named catastrophic self-overfitting; propose to provide adaptive regularization to the encoder’s gradients that explicitly prevents the occurrence of catastrophic self-overfitting |
| [Understanding Policy Gradient Algorithms: A Sensitivity-Based Approach](https://proceedings.mlr.press/v162/wu22i/wu22i.pdf) | ---- | ICML22 | study PG from a perturbation perspective |
| [Mirror Learning: A Unifying Framework of Policy Optimisation](https://arxiv.org/pdf/2201.02373.pdf) | Mirror Learning | ICML22 | propose a novel unified theoretical framework named Mirror Learning to provide theoretical guarantees for General Policy Improvement (GPI) and Trust-Region Learning (TRL); propose an interesting, graph-theoretical perspective on mirror learning |
| [Continuous Control with Action Quantization from Demonstrations](https://proceedings.mlr.press/v162/dadashi22a/dadashi22a.pdf) | AQuaDem | ICML22 | leverag the prior of human demonstrations for reducing a continuous action space to a discrete set of meaningful actions; point out that using a set of actions rather than a single one (Behavioral Cloning) enables to capture the multimodality of behaviors in the demonstrations |
| [Off-Policy Fitted Q-Evaluation with Differentiable Function Approximators: Z-Estimation and Inference Theory](https://proceedings.mlr.press/v162/zhang22al/zhang22al.pdf) | ---- | ICML22 | analyze Fitted Q Evaluation (FQE) with general differentiable function approximators, including neural function approximations by using the Z-estimation theory |
| [A Temporal-Difference Approach to Policy Gradient Estimation](https://proceedings.mlr.press/v162/tosatto22a/tosatto22a.pdf) || ICML22 ||
| [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.07802.pdf) || ICML22 ||
| [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/pdf/2202.00821.pdf) |  | ICML22 | use DRL for solving the optimal design of sequential experiments |
| [The Geometry of Robust Value Functions](https://proceedings.mlr.press/v162/wang22k/wang22k.pdf) |  | ICML22 | study the geometry of the robust value space for the more general Robust MDPs |
| Direct Behavior Specification via Constrained Reinforcement Learning || ICML22 ||
| [Utility Theory for Markovian Sequential Decision Making](https://arxiv.org/pdf/2206.13637.pdf) | Affine-Reward MDPs | ICML22 | extend von Neumann-Morgenstern (VNM) utility theorem to decision making setting |
| [Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks](https://proceedings.mlr.press/v162/liang22c/liang22c.pdf) | MeanQ | ICML22 | consider variance reduction in Temporal-Difference Value Estimation; propose MeanQ to estimate target values by ensembling |
| Unifying Approximate Gradient Updates for Policy Optimization || ICML22 ||
| [EqR: Equivariant Representations for Data-Efficient Reinforcement Learning](https://proceedings.mlr.press/v162/mondal22a/mondal22a.pdf) || ICML22 ||
| [Provable Reinforcement Learning with a Short-Term Memory](https://proceedings.mlr.press/v162/efroni22a/efroni22a.pdf) || ICML22 ||
| Optimal Estimation of Off-Policy Policy Gradient via Double Fitted Iteration || ICML22 ||
| Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments || ICML22 ||
| Lagrangian Method for Q-Function Learning (with Applications to Machine Translation) || ICML22 ||
| Learning to Assemble with Large-Scale Structured Reinforcement Learning || ICML22 ||
| Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning || ICML22 ||
| Off-Policy Reinforcement Learning with Delayed Rewards || ICML22 ||
| Reachability Constrained Reinforcement Learning || ICML22 ||



<a id='Model-Based-Online'></a>
## Model Based (Online) RL

<a id='model-based-classic'></a>
### Classic Methods
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
| [Bidirectional Model-based Policy Optimization](https://arxiv.org/pdf/2007.01995.pdf) | BMPO | ICML20 | an extension to MBPO; consider both forward dynamics model and backward dynamics model |
| [Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2005.06800.pdf) | CaDM | ICML20 |  develop a context-aware dynamics model (CaDM) capable of generalizing across a distribution of environments with varying transition dynamics; introduce a backward dynamics model that predicts a previous state by utilizing a context latent vector |
| [A Game Theoretic Framework for Model Based Reinforcement Learning](https://arxiv.org/pdf/2004.07804.pdf) | PAL, MAL | ICML20 | develop a novel framework that casts MBRL as a game between a policy player and a model player; setup a Stackelberg game between the two players |
| [Planning to Explore via Self-Supervised World Models](https://arxiv.org/pdf/2005.05960.pdf) | Plan2Explore | ICML20 | propose a self-supervised reinforcement learning agent for addressing two challenges: quick adaptation and expected future novelty |
| [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf)| M2AC | NeurIPS20 | an extension to MBPO; use model rollouts only when the model is confident |
| [The LoCA Regret: A Consistent Metric to Evaluate Model-Based Behavior in Reinforcement Learning](https://arxiv.org/pdf/2007.03158.pdf) | LoCA | NeurIPS20 | propose LoCA to measure how quickly a method adapts its policy after the environment is changed from the first task to the second |
| [Generative Temporal Difference Learning for Infinite-Horizon Prediction](https://arxiv.org/pdf/2010.14496.pdf) | GHM, or gamma-model | NeurIPS20 | propose gamma-model to make long-horizon predictions without the need to repeatedly apply a single-step model |
| [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf) | EfficientZero | NeurIPS21 | first achieve super-human performance on Atari games with limited data; propose EfficientZero with three components: (i) use self-supervised learning to learn a temporally consistent environment model, (ii) learn the value prefix in an end-to-end manner, (iii) use the learned model to correct off-policy value targets |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | an extension to MBPO; automatically schedule the real data ratio as well as other hyperparameters for MBPO |
| [Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice](https://arxiv.org/pdf/2106.14080.pdf) | ---- | arxiv22 | bridge the gap in theory and practice of value-aware model learning (VAML) for model-based RL |
| [Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/pdf/2204.01464.pdf) | VaGraM | ICLR22 Spotlight | consider the objective mismatch problem in MBRL; propose VaGraM by rescaling the MSE loss function with gradient information from the current value function estimate |
| [Constrained Policy Optimization via Bayesian World Models](https://arxiv.org/pdf/2201.09802.pdf) | LAMBDA | ICLR22 Spotlight | consider Bayesian model-based methods for CMDP |
| [On-Policy Model Errors in Reinforcement Learning](https://arxiv.org/pdf/2110.07985.pdf) | OPC | ICLR22 | consider to combine real-world data and a learned model in order to get the best of both worlds; propose to exploit the real-world data for onpolicy predictions and use the learned model only to generalize to different actions; propose to use on-policy transition data on top of a separately learned model to enable accurate long-term predictions for MBRL |
| [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf) | Denoised MDP | ICML22 | divide information into four categories: controllable/uncontrollable (whether infected by the action) and reward-relevant/irrelevant (whether affects the return); propose to only consider information which is controllable and reward-relevant |
| [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/pdf/2203.04955.pdf) | TD-MPC | ICML22 | propose to use the model only to predice reward; use a policy to accelerate the planning |



<a id='dreamer'></a>
### World Models

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [World Models](https://arxiv.org/pdf/1803.10122.pdf) | world model | arxiv1803 | use an unsupervised manner to learn a compressed spatial and temporal representation of the environment and use the world model to train a very compact and simple policy for solving the required task |
| [Learning latent dynamics for planning from pixels](https://arxiv.org/pdf/1811.04551.pdf) | PlaNet | ICML19 | propose PlaNet to learn the environment dynamics from images; the dynamic model consists transition model, observation model, reward model and encoder; use the cross entropy method for selecting actions for planning |
| [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/pdf/1912.01603.pdf) | Dreamer | ICLR20 | solve long-horizon tasks from images purely by latent imagination; test in image-based MuJoCo; propose to use an agent to replace the control algorithm in the PlaNet |
| [Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12142.pdf) | BIRD | NeurIPS20 | propose to maximize the mutual information between imaginary and real trajectories so that the policy improvement learned from imaginary trajectories can be easily generalized to real trajectories |
| [Planning to Explore via Self-Supervised World Models](https://arxiv.org/pdf/2005.05960.pdf) | Plan2Explore | ICML20 | propose Plan2Explore to  self-supervised exploration and fast adaptation to new tasks |
| [Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193.pdf) | Dreamerv2 | ICLR21 | solve long-horizon tasks from images purely by latent imagination; test in image-based Atari |
| [Temporal Predictive Coding For Model-Based Planning In Latent Space](https://arxiv.org/pdf/2106.07156.pdf) | TPC | ICML21 | propose a temporal predictive coding approach for planning from high-dimensional observations and theoretically analyze its ability to prioritize the encoding of task-relevant information |
| [Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction](https://arxiv.org/pdf/2007.14535.pdf) | Dreaming | ICRA21 | propose a decoder-free extension of Dreamer since the autoencoding based approach often causes object vanishing|
| [Model-Based Reinforcement Learning via Imagination with Derived Memory](https://openreview.net/pdf?id=jeATherHHGj) | IDM | NeurIPS21 | hope to improve the diversity of imagination for model-based policy optimization with the derived memory; point out that current methods cannot effectively enrich the imagination if the latent state is disturbed by random noises |
| [Maximum Entropy Model-based Reinforcement Learning](https://arxiv.org/pdf/2112.01195.pdf) |  MaxEnt Dreamer | NeurIPS21 | create a connection between exploration methods and model-based reinforcement learning; apply maximum-entropy exploration for Dreamer |
| [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf) | DreamerPro | ICML22 | consider reconstruction-free MBRL; propose to learn the prototypes from the recurrent states of the world model, thereby distilling temporal structures from past observations and actions into the prototypes. |
| [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/pdf/2202.09481.pdf) | TransDreamer | arxiv2202 | replace the RNN in RSSM by a transformer |
| [DreamingV2: Reinforcement Learning with Discrete World Models without Reconstruction](https://arxiv.org/pdf/2203.00494.pdf) | Dreamingv2 | arxiv2203 |  |
| [Masked World Models for Visual Control](https://arxiv.org/pdf/2206.14244.pdf) | MWM | arxiv2206 | decouple visual representation learning and dynamics learning for visual model-based RL and use masked autoencoder to train visual representation |
| [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/pdf/2206.14176.pdf) | DayDreamer | arxiv2206 | apply Dreamer to 4 robots to learn online and directly in the real world, without any simulators |
| [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://proceedings.mlr.press/v162/wan22d/wan22d.pdf) | ---- | ICML22 | introduce an improved version of the LoCA setup and use it to evaluate PlaNet and Dreamerv2 |
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://arxiv.org/pdf/2203.13880.pdf) | APV | ICML22 | pre-train an action-free latent video prediction model using videos from different domains, and then fine-tune the pre-trained model on target domains |
| [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf) | IRIS | arxiv2209 | use a discrete autoencoder and an autoregressive Transformer to conduct World Models and significantly improve the data efficiency in Atari (2 hours of real-time experience); [\[Code\]](https://github.com/eloialonso/iris) |



<a id='model-based-code'></a>
### CodeBase

|  Title | Conference | Methods |  Github |
| ---- | ---- | ---- | ---- |
| [MBRL-Lib: A Modular Library for Model-based Reinforcement Learning](https://arxiv.org/pdf/2104.10159.pdf) | arxiv21 | MBPO,PETS,PlaNet | [link](https://github.com/facebookresearch/mbrl-lib) |



<a id='Model-Free-Offline'></a>
## (Model Free) Offline RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) | BCQ | ICML19 | show that off-policy methods perform badly because of extrapolation error; propose batch-constrained reinforcement learning: maximizing the return as well as minimizing the mismatch between the state-action visitation of the policy and the state-action pairs contained in the batch |
| [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) | CQL | NeurIPS20 | propose CQL with conservative q function, which is a lower bound of its true value, since standard off-policy methods will overestimate the value function |
| [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf) | ---- | arxiv20 | tutorial about methods, applications and open problems of offline rl |
| [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/pdf/2110.01548.pdf) |  | NeurIPS21 |  |
| [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/pdf/2106.06860.pdf) | TD3+BC | NeurIPS21 | propsoe to add a behavior cloning term to regularize the policy, and normalize the states over the dataset |
| [DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization](https://arxiv.org/pdf/2112.04716.pdf) | DR3 | ICLR22 Spotlight | consider the implicit regularization effect of SGD in RL; based on theoretical analyses, propose an explicit regularizer, called DR3, and combine with offline RL methods |
| [Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning ](https://arxiv.org/pdf/2202.11566.pdf) | PBRL | ICLR22 Spotlight | consider the distributional shift and extrapolation error in offline RL; propose PBRL with bootstrapping, for uncertainty quantification, and an OOD sampling method as a regularizer |
| [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://openreview.net/pdf?id=FLA55mBee6Q) | COptiDICE | ICLR22 Spotlight | consider offline constrained reinforcement learning; propose COptiDICE to directly optimize the distribution of state-action pair with contraints |
| [Offline Reinforcement Learning with Value-based Episodic Memory](https://openreview.net/pdf?id=RCZqv9NXlZ) | EVL, VEM | ICLR22 | present a new offline V -learning method to learn the value function through the trade-offs between imitation learning and optimal value learning; use a memory-based planning scheme to enhance advantage estimation and conduct policy learning in a regression manner |
| [Offline reinforcement learning with implicit Q-learning](https://arxiv.org/pdf/2110.06169.pdf) | IQL | ICLR22 | propose to learn an optimal policy with in-sample learning, without ever querying the values of any unseen actions |
| Adversarially Trained Actor Critic for Offline Reinforcement Learning |  | ICML22 oral |  |
| Learning Bellman Complete Representations for Offline Policy Evaluation | | ICML22 oral | |
| [Offline RL Policies Should Be Trained to be Adaptive](https://arxiv.org/pdf/2207.02200.pdf) | APE-V | ICML22 oral | show that learning from an offline dataset does not fully specify the environment; formally demonstrate the necessity of adaptation in offline RL by using the Bayesian formalism and to provide a practical algorithm for learning optimally adaptive policies; propose an ensemble-based offline RL algorithm that imbues policies with the ability to adapt within an episode |
| Pessimistic Q-Learning for Offline Reinforcement Learning: Towards Optimal Sample Complexity || ICML22 ||
| How to Leverage Unlabeled Data in Offline Reinforcement Learning? || ICML22 ||
| On the Role of Discount Factor in Offline Reinforcement Learning || ICML22 ||
| Model Selection in Batch Policy Optimization || ICML22 ||
| Koopman Q-learning: Offline Reinforcement Learning via Symmetries of Dynamics || ICML22 ||
| Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning || ICML22 ||
| Pessimism meets VCG: Learning Dynamic Mechanism Design via Offline Reinforcement Learning || ICML22 ||
| Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters || ICML22 ||
| Constrained Offline Policy Optimization || ICML22 ||



<a id='Model-Based-Offline'></a>
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


<a id='Meta-RL'></a>
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
| [Meta-Learning Hypothesis Spaces for Sequential Decision-making](https://arxiv.org/pdf/2202.00602.pdf) | Meta-KeL | ICML22 | argue that two critical capabilities of transformers, reason over long-term dependencies and present context-dependent weights from self-attention, compose the central role of a Meta-Reinforcement Learner; propose Meta-LeL for meta-learning the hypothesis space of a sequential decision task |
| Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning || ICML22 ||
| [Transformers are Meta-Reinforcement Learners](https://arxiv.org/pdf/2206.06614.pdf) | TrMRL | ICML22 | propose TrMRL, a memory-based meta-Reinforcement Learner which uses the transformer architecture to formulate the learning process; |
| Offline Meta-Reinforcement Learning with Online Self-Supervision || ICML22 ||



<a id='Adversarial-RL'></a>
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

<a id='Genaralization-in-RL'></a>
## Genaralisation in RL

| Title | Method | Conference | Description | 
| ----  | ----   | ----       |   ----  |
| [On the Generalization Gap in Reparameterizable Reinforcement Learning](https://arxiv.org/pdf/1905.12654.pdf) | ---- | ICML19 | theoretically provide guarantees on the gap between the expected and empirical return for both intrinsic and external errors in reparameterizable RL |
| [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/pdf/1812.02341.pdf) | CoinRun | ICML19 | introduce a new environment called CoinRun for generalisation in RL; empirically show L2 regularization, dropout, data augmentation and batch normalization can improve generalization in RL |
| [Network randomization: A simple technique for generalization in deep reinforcement learning](https://arxiv.org/pdf/1910.05396.pdf) |  | ICLR19 |  |
| [Investigating Generalisation in Continuous Deep Reinforcement Learning](https://arxiv.org/pdf/1902.07015.pdf) | ---- | arxiv19 | study generalisation in Deep RL for continuous control |
| [Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2005.06800.pdf) |  | ICML20 |  |
| [Augmented World Models Facilitate Zero-Shot Dynamics Generalization From a Single Offline Environment](https://arxiv.org/pdf/2104.05632.pdf) |  | ICML21 |  |
| [Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability](https://arxiv.org/pdf/2107.06277.pdf) | LEEP | NeurIPS21 | generalisation in RL induces implicit partial observability; propose LEEP to use an ensemble of policies to approximately learn the Bayes-optimal policy for maximizing test-time performance |
| [Automatic Data Augmentation for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf) | DrAC | NeurIPS21 | focus on automatic data augmentation based two novel regularization terms for the policy and value function |
| [When Is Generalizable Reinforcement Learning Tractable?](https://arxiv.org/pdf/2101.00300.pdf) | ---- | NeurIPS21 | propose Weak Proximity and Strong Proximity for theoretically analyzing the generalisation of RL |
| [A Survey of Generalisation in Deep Reinforcement Learning](https://arxiv.org/pdf/2111.09794.pdf) | ---- | arxiv21 | provide a unifying formalism and terminology for discussing different generalisation problems |
| [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) | CTRL | ICLR22 | consider zero-shot generalization (ZSG); use self-supervised learning to learn a representation across tasks |
| [The Role of Pretrained Representations for the OOD Generalization of RL Agents](https://arxiv.org/pdf/2107.05686.pdf) | ---- | ICLR22 |  |
| [Generalisation in Lifelong Reinforcement Learning through Logical Composition](https://openreview.net/pdf?id=ZOcX-eybqoL) | ---- | ICLR22 |  |
| [A Generalist Agent](https://arxiv.org/pdf/2205.06175.pdf) | Gato | arxiv22 | [slide](https://ml.cs.tsinghua.edu.cn/~chengyang/reading_meeting/Reading_Meeting_20220607.pdf) |
| [Learning Dynamics and Generalization in Reinforcement Learning](https://arxiv.org/pdf/2206.02126.pdf) | ---- | ICML22 | show theoretically that temporal difference learning encourages agents to fit non-smooth components of the value function early in training, and at the same time induces the second-order effect of discouraging generalization |
| [Improving Policy Optimization with Generalist-Specialist Learning](https://arxiv.org/pdf/2206.12984.pdf) | GSL | ICML22 | hope to utilize experiences from the specialists to aid the policy optimization of the generalist; propose the phenomenon “catastrophic ignorance” in multi-task learning |

<a id='Sequence-Generation'></a>
## RL as Sequence Generation

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (return-to-go, state, action, return-to-go, ...); there is not explicit optimization process; evaluate on Offline RL |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (s_0^0, ..., s_0^N, a_0^0, ..., a_0^M, r_0, ...); use beam search to inference; evaluate on imitation learning, goal-conditioned RL and Offline RL | 
| Online Decision Transformer |  | ICML22 oral |  |
| Prompting Decision Transformer for Few-shot Policy Generalization || ICML22 ||

<a id='Unsupervised-RL'></a>
## Unsupervised RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/pdf/1802.06070.pdf) | DIAYN | ICLR19 | learn diverse skills in environments without any rewards by maximizing an information theoretic objective |
| [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/pdf/2004.04136.pdf) | CURL | ICML20 | extracts high-level features from raw pixels using contrastive learning and performs offpolicy control on top of the extracted features |
| [Decoupling representation learning from reinforcement learning](https://arxiv.org/pdf/2009.08319.pdf) | ATC | ICML21 | propose a new unsupervised task tailored to reinforcement learning named Augmented Temporal Contrast (ATC), which borrows ideas from Contrastive learning; benchmark several leading Unsupervised Learning algorithms by pre-training encoders on expert demonstrations and using them in RL agents|
| [Pretraining representations for data-efficient reinforcement learning](https://arxiv.org/pdf/2106.04799.pdf) | SGI | NeurIPS21 | consider to pretrian with unlabeled data and finetune on a small amount of task-specific data to improve the data efficiency of RL; employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL |
| [Understanding the World Through Action](https://arxiv.org/pdf/2110.12543.pdf) | ---- | CoRL21 | discusse how self-supervised reinforcement learning combined with offline RL can enable scalable representation learning |
| [URLB: Unsupervised Reinforcement Learning Benchmark](https://arxiv.org/pdf/2110.15191.pdf) | URLB | NeurIPS21 | a benchmark for unsupervised reinforcement learning |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | ---- | ICLR22 | show that unsupervised skill discovery algorithms based on mutual information maximization do not learn skills that are optimal for every possible reward function; provide a geometric perspective on some skill learning methods |
| The Unsurprising Effectiveness of Pre-Trained Vision Models for Control || ICML22 oral ||


<a id='Lifelong-RL'></a>
## Lifelong RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |



<a id='Tutorial-and-Lesson'></a>
## Tutorial and Lesson

| Tutorial and Lesson |
| ---- |
| [Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](https://d1wqtxts1xzle7.cloudfront.net/54674740/Reinforcement_Learning-with-cover-page-v2.pdf?Expires=1641130151&Signature=eYy7kmTVqTXFcANS-9GZJUyb86cDqKeh2QX8VvzjouEM-QSfuiCm1WHhP~bW5C57Mecj6en~YRoTvxekzU5lq~UaHSBoc-7xP8dXBp91shcwdfJ8M0LUkktpqcQjXQi7ZzhGn33qZeah0p8S06ARzjimF5coL5arvp9yANAsy4KigXSZwAZNXxksKwqUAult2QseLL~Bv1p2locjYahRzTuex3vMxdBLhT9HOGFF0qOdKYxsWiaITUKnVYl8AvePDHEEXgfmuqEfjqjF5p~FHOsYl3gEDZOvUp1eUzPg2~i0MQXY49nUpzsThL5~unTRIsYJiBghnkYl8py0r~UelQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| [Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) | 
| [Deep Reinforcement Learning, CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| [Deep Reinforcement Learning and Control, CMU 10703](https://katefvision.github.io/) |
| [RLChina](http://rlchina.org/topic/9) |
