# Reinforcement Learning Papers
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Related papers for Reinforcement Learning (we mainly focus on single-agent).

Since there are tens of thousands of new papers on reinforcement learning at each conference every year, we are only able to list those we read and consider as insightful.

**We have added some ICLR23, ICML23 papers on RL**


## Contents 
* [Model Free (Online) RL](#Model-Free-Online)
    - [Classic Methods](#model-free-classic)
    - [Exploration](#exploration)
    - [Off-Policy Evaluation](#off-policy-evaluation)
    - [Soft RL](#soft-rl)
    - [Data Augmentation](#data-augmentation)
    - [Representation Learning](#Representation-RL)
    - [Current methods](#current)
* [Model Based (Online) RL](#Model-Based-Online)
    - [Classic Methods](#model-based-classic)
    - [World Models](#dreamer)
    - [CodeBase](#model-based-code)
* [(Model Free) Offline RL](#Model-Free-Offline)
    - [Current methods](#offline-current)
    - [Combined with Diffusion Models](#offline-diffusion)
* [Model Based Offline RL](#Model-Based-Offline)
* [Meta RL](#Meta-RL)
* [Adversarial RL](#Adversarial-RL)
* [Genaralisation in RL](#Genaralization-in-RL)
    - [Environments](#Gene-Environments)
    - [Methods](#Gene-Methods)
* [RL with Transformer](#Sequence-Generation)
* [Lifelong RL](#Lifelong-RL)
* [RL with LLM](#RL-LLM)
* [Tutorial and Lesson](#Tutorial-and-Lesson)

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

<a id='exploration'></a>
### Exploration
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/pdf/1705.05363.pdf) | ICM | ICML17 | propose that curiosity can serve as an intrinsic reward signal to enable the agent to explore its environment and learn skills when rewards are sparse; formulate curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model |
| [Automatic Intrinsic Reward Shaping for Exploration in Deep Reinforcement Learning](https://arxiv.org/pdf/2301.10886.pdf) | AIRS | ICML23 | select shaping function from a predefined set based on the estimated task return in real-time, providing reliable exploration incentives and alleviating the biased objective problem; develop a toolkit that provides highquality implementations of various intrinsic reward modules based on PyTorch |
| [Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments](https://arxiv.org/pdf/2211.10515.pdf) | Curiosity in Hindsight | ICML23 | consider exploration in stochastic environments; learn representations of the future that capture precisely the unpredictable aspects of each outcome—which we use as additional input for predictions, such that intrinsic rewards only reflect the predictable aspects of world dynamics |

<a id='off-policy-evaluation'></a>
### Off-Policy Evaluation
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Weighted importance sampling for off-policy learning with linear function approximation](https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf) | WIS-LSTD | NeurIPS14 |  |
| [Importance Sampling Policy Evaluation with an Estimated Behavior Policy](https://arxiv.org/pdf/1806.01347.pdf) | RIS | ICML19 |  |
| [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf) |  | ICML22 |  |
| [Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning](https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf) | LDR2OPE | ICML22 ||
| [On Well-posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-policy Evaluation](https://proceedings.mlr.press/v162/chen22u/chen22u.pdf) |  | ICML22 ||
| [A Unified Off-Policy Evaluation Approach for General Value Function](https://arxiv.org/pdf/2107.02711.pdf) || NeurIPS22 ||
| The Pitfalls of Regularizations in Off-Policy TD Learning || NeurIPS22 ||
| Off-Policy Evaluation for Action-Dependent Non-Stationary Environments || NeurIPS22 ||
| [Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions](https://arxiv.org/pdf/2210.13373.pdf) || NeurIPS22 ||
| [Off-Policy Evaluation with Policy-Dependent Optimization Response](https://arxiv.org/pdf/2202.12958.pdf) || NeurIPS22 ||
| Variational Latent Branching Model for Off-Policy Evaluation || ICLR23 ||
| [On the Reuse Bias in Off-Policy Reinforcement Learning](https://arxiv.org/pdf/2209.07074.pdf) | BIRIS | IJCAI23 | discuss the bias of off-policy evaluation due to reusing the replay buffer; derive a high-probability bound of the Reuse Bias; introduce the concept of stability for off-policy algorithms and provide an upper bound for stable off-policy algorithms |
| Improved Policy Evaluation for Randomized Trials of Algorithmic Resource Allocation || ICML23 ||
| An Instrumental Variable Approach to Confounded Off-Policy Evaluation || ICML23 ||
| Semiparametrically Efficient Off-Policy Evaluation in Linear Markov Decision Processes || ICML23 ||


<a id='soft-rl'></a>
### Soft RL

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) | MME | NeurIPS21 | find that SAC may fail in explore states with low entropy (arrive states with high entropy and increase their entropies); propose a max-min entropy framework to address this issue |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that standard maximum entropy RL is robust to some disturbances in the dynamics and the reward function |
| The Importance of Non-Markovianity in Maximum State Entropy Exploration | | ICML22 oral |  |
| Communicating via Maximum Entropy Reinforcement Learning || ICML22 ||

<a id='data-augmentation'></a>
### Data Augmentation
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Reinforcement Learning with Augmented Data](https://arxiv.org/pdf/2004.14990.pdf) | RAD | NeurIPS20 | propose first extensive study of general data augmentations for RL on both pixel-based and state-based inputs |
| [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2004.13649.pdf) | DrQ | ICLR21 Spotlight | propose to regularize the value function when applying data augmentation with model-free methods and reach state-of-the-art performance in image-pixels tasks |

<!-- | [Equivalence notions and model minimization in Markov decision processes](https://www.ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/givan-dean-greig.pdf) |  | Artificial Intelligence, 2003 |  |
| [Metrics for Finite Markov Decision Processes](https://arxiv.org/ftp/arxiv/papers/1207/1207.4114.pdf) || UAI04 ||
| [Bisimulation metrics for continuous Markov decision processes](https://www.normferns.com/assets/documents/sicomp2011.pdf) || SIAM Journal on Computing, 2011 ||
| [Scalable methods for computing state similarity in deterministic Markov Decision Processes](https://arxiv.org/pdf/1911.09291.pdf) || AAAI20 ||
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf) | DBC | ICLR21 || -->


<a id='Representation-RL'></a>
## Representation Learning

Note: representation learning with MBRL is in the part [World Models](#dreamer)

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/pdf/1802.06070.pdf) | DIAYN | ICLR19 | learn diverse skills in environments without any rewards by maximizing an information theoretic objective |
| [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/pdf/2004.04136.pdf) | CURL | ICML20 | extracts high-level features from raw pixels using contrastive learning and performs offpolicy control on top of the extracted features |
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf) | DBC | ICLR21 | propose using Bisimulation to learn robust latent representations which encode only the task-relevant information from observations |
| [Decoupling representation learning from reinforcement learning](https://arxiv.org/pdf/2009.08319.pdf) | ATC | ICML21 | propose a new unsupervised task tailored to reinforcement learning named Augmented Temporal Contrast (ATC), which borrows ideas from Contrastive learning; benchmark several leading Unsupervised Learning algorithms by pre-training encoders on expert demonstrations and using them in RL agents|
| [Reinforcement Learning with Prototypical Representations](https://arxiv.org/pdf/2102.11271.pdf) | Proto-RL | ICML21 | pre-train task-agnostic representations and prototypes on environments without downstream task information |
| [Pretraining representations for data-efficient reinforcement learning](https://arxiv.org/pdf/2106.04799.pdf) | SGI | NeurIPS21 | consider to pretrian with unlabeled data and finetune on a small amount of task-specific data to improve the data efficiency of RL; employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL |
| [Understanding the World Through Action](https://arxiv.org/pdf/2110.12543.pdf) | ---- | CoRL21 | discusse how self-supervised reinforcement learning combined with offline RL can enable scalable representation learning |
| [URLB: Unsupervised Reinforcement Learning Benchmark](https://arxiv.org/pdf/2110.15191.pdf) | URLB | NeurIPS21 | a benchmark for unsupervised reinforcement learning |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | ---- | ICLR22 | show that unsupervised skill discovery algorithms based on mutual information maximization do not learn skills that are optimal for every possible reward function; provide a geometric perspective on some skill learning methods |
| The Unsurprising Effectiveness of Pre-Trained Vision Models for Control || ICML22 oral ||
| a mixture of supervised and unsupervised reinforcement learning || NeurIPS22 ||
| [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568.pdf) | Contrastive  RL | NeurIPS22 | show (contrastive) representation learning methods can be cast as RL algorithms in their own right |
| [Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?](https://arxiv.org/pdf/2206.05266.pdf) | ---- | NeurIPS22 | conduct an extensive comparison of various self-supervised losses under the existing joint learning framework for pixel-based reinforcement learning in many environments from different benchmarks, including one real-world environment |
| [Unsupervised Reinforcement Learning with Contrastive Intrinsic Control](https://openreview.net/pdf?id=9HBbWAsZxFt) | CIC | NeurIPS22 | propose to maximize the mutual information between statetransitions and latent skill vectors |
| [Reinforcement Learning with Automated Auxiliary Loss Search](https://arxiv.org/pdf/2210.06041.pdf) | A2LS | NeurIPS22 | propose to automatically search top-performing auxiliary loss functions for learning better representations in RL; define a general auxiliary loss space of size 7.5 × 1020 based on the collected trajectory data and explore the space with an efficient evolutionary search strategy |
| [Mask-based Latent Reconstruction for Reinforcement Learning](https://arxiv.org/pdf/2201.12096.pdf) | MLR | NeurIPS22 | propose an effective self-supervised method to predict complete state representations in the latent space from the observations with spatially and temporally masked pixels |
| Choreographer: Learning and Adapting Skills in Imagination || ICLR23 Spotlight ||
| [Flow-based Recurrent Belief State Learning for POMDPs](https://proceedings.mlr.press/v162/chen22q/chen22q.pdf) | FORBES | ICML22 | incorporate normalizing flows into the variational inference to learn general continuous belief states for POMDPs |
| [Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training](https://arxiv.org/pdf/2210.00030.pdf) | VIP | ICLR23 Spotlight | cast representation learning from human videos as an offline goal-conditioned reinforcement learning problem; derive a self-supervised dual goal-conditioned value-function objective that does not depend on actions, enabling pre-training on unlabeled human videos |
| [Latent Variable Representation for Reinforcement Learning](https://arxiv.org/pdf/2212.08765.pdf) | ---- | ICLR23 | provide a representation view of the latent variable models for state-action value functions, which allows both tractable variational learning algorithm and effective implementation of the optimism/pessimism principle in the face of uncertainty for exploration |
| Spectral Decomposition Representation for Reinforcement Learning || ICLR23 ||
| Behavior Prior Representation learning for Offline Reinforcement Learning || ICLR23 ||
| Provable Unsupervised Data Sharing for Offline Reinforcement Learning || ICLR23 ||
| [Become a Proficient Player with Limited Data through Watching Pure Videos](https://openreview.net/pdf?id=Sy-o2N0hF4f) | FICC | ICLR23 | consider the setting where the pre-training data are action-free videos; introduce a two-phase training pipeline; pre-training phase: implicitly extract the hidden action embedding from videos and pre-train the visual representation and the environment dynamics network based on vector quantization; down-stream tasks: finetune with small amount of task data based on the learned models |
| Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels || ICML23 oral ||
| On the Importance of Feature Decorrelation for Unsupervised Representation Learning in Reinforcement Learning || ICML23 ||
| Bootstrapped Representations in Reinforcement Learning || ICML23 ||

<!-- ### <span id='current'>Current methods</span> -->
<a id='current'></a>
### Current methods

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) | Block MDP | ICML19 ||
| [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/pdf/2005.12729.pdf) | ---- | ICLR20 | show that the improvement of performance is related to code-level optimizations |
| [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) | ---- | ICLR21 | do a large scale empirical study to evaluate different tricks for on-policy algorithms on MuJoCo |
| [Mirror Descent Policy Optimization](https://arxiv.org/pdf/2005.09814.pdf) | MDPO | ICLR21 |  |
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf) | DBC | ICLR21 ||
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
| [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.07802.pdf) | primacy bias | ICML22 | find that deep RL agents incur a risk of overfitting to earlier experiences, which will negatively affect the rest of the learning process; propose a simple yet generally-applicable mechanism that tackles the primacy bias by periodically resetting a part of the agent |
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
| [Reinforcement Learning with Neural Radiance Fields](https://arxiv.org/pdf/2206.01634.pdf) | NeRF-RL | NeurIPS22 | propose to train an encoder that maps multiple image observations to a latent space describing the objects in the scene |
| [Recursive Reinforcement Learning](https://arxiv.org/pdf/2206.11430.pdf) || NeurIPS22 ||
| [Challenging Common Assumptions in Convex Reinforcement Learning](https://arxiv.org/pdf/2202.01511.pdf) || NeurIPS22 ||
| Explicable Policy Search || NeurIPS22 ||
| [On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting](https://arxiv.org/pdf/2206.00761.pdf) | ---- | NeurIPS22 | explore the theoretical connections between Reward Maximization (RM) and Distribution Matching (DM) |
| [When to Ask for Help: Proactive Interventions in Autonomous Reinforcement Learning](https://arxiv.org/pdf/2210.10765.pdf) || NeurIPS22 ||
| Adaptive Bio-Inspired Fish Simulation with Deep Reinforcement Learning || NeurIPS22 ||
| Reinforcement Learning in a Birth and Death Process: Breaking the Dependence on the State Space || NeurIPS22 ||
| [Discovered Policy Optimisation](https://arxiv.org/pdf/2210.05639.pdf) || NeurIPS22 ||
| Faster Deep Reinforcement Learning with Slower Online Network || NeurIPS22 ||
| exploration-guided reward shaping for reinforcement learning under sparse rewards || NeurIPS22 ||
| an adaptive deep rl method for non-stationary environments with piecewise stable context || NeurIPS22 ||
| [Large-Scale Retrieval for Reinforcement Learning](https://arxiv.org/pdf/2206.05314.pdf) || NeurIPS22 ||
| [Sustainable Online Reinforcement Learning for Auto-bidding](https://arxiv.org/pdf/2210.07006.pdf) || NeurIPS22 ||
| [LECO: Learnable Episodic Count for Task-Specific Intrinsic Reward](https://arxiv.org/pdf/2210.05409.pdf) || NeurIPS22 ||
| [DNA: Proximal Policy Optimization with a Dual Network Architecture](https://arxiv.org/pdf/2206.10027.pdf) || NeurIPS22 ||
| [Faster Deep Reinforcement Learning with Slower Online Network](https://assets.amazon.science/31/ca/0c09418b4055a7536ced1b218d72/faster-deep-reinforcement-learning-with-slower-online-network.pdf) | DQN Pro, Rainbow Pro | NeurIPS22 | incentivize the online network to remain in the proximity of the target network |
| [Online Reinforcement Learning for Mixed Policy Scopes](https://causalai.net/r84.pdf) || NeurIPS22 ||
| [ProtoX: Explaining a Reinforcement Learning Agent via Prototyping](https://arxiv.org/pdf/2211.03162.pdf) || NeurIPS22 ||
| [Hardness in Markov Decision Processes: Theory and Practice](https://arxiv.org/pdf/2210.13075.pdf) || NeurIPS22 ||
| [Robust Phi-Divergence MDPs](https://arxiv.org/pdf/2205.14202.pdf) || NeurIPS22 ||
| [On the convergence of policy gradient methods to Nash equilibria in general stochastic games](https://arxiv.org/pdf/2210.08857.pdf) || NeurIPS22 ||
| [A Unified Off-Policy Evaluation Approach for General Value Function](https://arxiv.org/pdf/2107.02711.pdf) || NeurIPS22 ||
| [Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning](https://arxiv.org/pdf/2111.14552.pdf) || NeurIPS22 ||
| Continuous Deep Q-Learning in Optimal Control Problems: Normalized Advantage Functions Analysis || NeurIPS22 ||
| [Parametrically Retargetable Decision-Makers Tend To Seek Power](https://arxiv.org/pdf/2206.13477.pdf) || NeurIPS22 ||
| [Batch size-invariance for policy optimization](https://arxiv.org/pdf/2110.00641.pdf) || NeurIPS22 ||
| [Trust Region Policy Optimization with Optimal Transport Discrepancies: Duality and Algorithm for Continuous Actions](https://arxiv.org/pdf/2210.11137.pdf) || NeurIPS22 ||
| Adaptive Interest for Emphatic Reinforcement Learning || NeurIPS22 ||
| [The Nature of Temporal Difference Errors in Multi-step Distributional Reinforcement Learning](https://arxiv.org/pdf/2207.07570.pdf) || NeurIPS22 ||
| [Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress](https://arxiv.org/pdf/2206.01626.pdf) | PVRL | NeurIPS22 | focus on reincarnating RL from any agent to any other agent; present reincarnating RL as an alternative workflow or class of problem settings, where prior computational work (e.g., learned policies) is reused or transferred between design iterations of an RL agent, or from one RL agent to another |
| [Bayesian Risk Markov Decision Processes](https://arxiv.org/pdf/2106.02558.pdf) || NeurIPS22 ||
| [Explainable Reinforcement Learning via Model Transforms](https://arxiv.org/pdf/2209.12006.pdf) || NeurIPS22 ||
| PDSketch: Integrated Planning Domain Programming and Learning || NeurIPS22 ||
| [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/pdf?id=OpC-9aBBVJe) | SR-SAC, SR-SPR | ICLR23 oral | show that fully or partially resetting the parameters of deep reinforcement learning agents causes better replay ratio scaling capabilities to emerge |
| [Guarded Policy Optimization with Imperfect Online Demonstrations](https://arxiv.org/pdf/2303.01728.pdf) | TS2C | ICLR23 Spotlight | h incorporate teacher intervention based on trajectory-based value estimation |
| [Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes](https://openreview.net/pdf?id=hWwY_Jq0xsN) | PW-Net | ICLR23 Spotlight | focus on making an “interpretable-by-design” deep reinforcement learning agent which is forced to use human-friendly prototypes in its decisions for making its reasoning process clear; train a “wrapper” model called PW-Net that can be added to any pre-trained agent, which allows them to be interpretable |
| Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning || ICLR23 Spotlight ||
| [DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems](https://arxiv.org/pdf/2206.00484.pdf) | DEP-RL | ICLR23 Spotlight | identify the DEP controller, known from the field of self-organizing behavior, to generate more effective exploration than other commonly used noise processes; first control the 7 degrees of freedom (DoF) human arm model with RL on a muscle stimulation level |
| [Efficient Deep Reinforcement Learning Requires Regulating Statistical Overfitting](https://arxiv.org/pdf/2304.10466.pdf) | AVTD | ICLR23 | propose a simple active model selection method (AVTD) that attempts to automatically select regularization schemes by hill-climbing on validation TD error |
| Replay Memory as An Empirical MDP: Combining Conservative Estimation with Experience Replay || ICLR23 ||
| [Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement](https://arxiv.org/pdf/1810.09103.pdf) | CCEM, GreedyAC | ICLR23 | propose to iteratively take the top percentile of actions, ranked according to the learned action-values; leverage theory for CEM to validate that CCEM concentrates on maximally valued actions across states over time |
| [Reward Design with Language Models](https://openreview.net/pdf?id=10uNUgI5Kl) | ---- | ICLR23 | explore how to simplify reward design by prompting a large language model (LLM) such as GPT-3 as a proxy reward function, where the user provides a textual prompt containing a few examples (few-shot) or a description (zero-shot) of the desired behavior |
| [Solving Continuous Control via Q-learning](https://arxiv.org/pdf/2210.12566.pdf) | DecQN | ICLR23 | combine value decomposition with bang-bang action space discretization to DQN to handle continuous control tasks; evaluate on DMControl, Meta World, and Isaac Gym |
| [Wasserstein Auto-encoded MDPs: Formal Verification of Efficiently Distilled RL Policies with Many-sided Guarantees](https://arxiv.org/pdf/2303.12558.pdf) | WAE-MDP | ICLR23 | minimize a penalized form of the optimal transport between the behaviors of the agent executing the original policy and the distilled policy |
| Quality-Similar Diversity via Population Based Reinforcement Learning  || ICLR23 ||
| [Human-level Atari 200x faster](https://arxiv.org/pdf/2209.07550.pdf) | MEME | ICLR23 | outperform the human baseline across all 57 Atari games in 390M frames; four key components: (1) an approximate trust region method which enables stable bootstrapping from the online network, (2) a normalisation scheme for the loss and priorities which improves robustness when learning a set of value functions with a wide range of scales, (3) an improved architecture employing techniques from NFNets in order to leverage deeper networks without the need for normalization layers, and (4) a policy distillation method which serves to smooth out the instantaneous greedy policy over time. |
| Policy Expansion for Bridging Offline-to-Online Reinforcement Learning  || ICLR23 ||
| [Improving Deep Policy Gradients with Value Function Search](https://arxiv.org/pdf/2302.10145.pdf) | VFS | ICLR23 | focus on improving value approximation and analyzing the effects on Deep PG primitives such as value prediction, variance reduction, and correlation of gradient estimates with the true gradient; show that value functions with better predictions improve Deep PG primitives, leading to better sample efficiency and policies with higher returns |
| [Memory Gym: Partially Observable Challenges to Memory-Based Agents](https://openreview.net/pdf?id=jHc8dCx6DDr) | Memory Gym | ICLR23 | a benchmark for challenging Deep Reinforcement Learning agents to memorize events across long sequences, be robust to noise, and generalize; consists of the partially observable 2D and discrete control environments Mortar Mayhem, Mystery Path, and Searing Spotlights; [\[code\]](https://github.com/MarcoMeter/drl-memory-gym/) |
| Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality || ICLR23 ||
| [Hybrid RL: Using both offline and online data can make RL efficient](https://arxiv.org/pdf/2210.06718.pdf) | Hy-Q | ICLR23 | focus on a hybrid setting named Hybrid RL, where the agent has both an offline dataset and the ability to interact with the environment; extend fitted Q-iteration algorithm |
| [POPGym: Benchmarking Partially Observable Reinforcement Learning](https://arxiv.org/pdf/2303.01859.pdf) | POPGym | ICLR23 | a two-part library containing (1) a diverse collection of 15 partially observable environments, each with multiple difficulties and (2) implementations of 13 memory model baselines; [\[code\]](https://github.com/proroklab/popgym) |
| [Critic Sequential Monte Carlo](https://arxiv.org/pdf/2205.15460.pdf) | CriticSMC | ICLR23 | combine sequential Monte Carlo with learned Soft-Q function heuristic factors |
| Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching || ICLR23 ||
| [Planning-oriented Autonomous Driving](https://arxiv.org/pdf/2212.10156.pdf) || CVPR23 best paper ||
| [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/pdf/2302.12902.pdf) | ReDo | ICML23 oral | understand the underlying reasons behind the loss of expressivity during the training of RL agents; demonstrate the existence of the dormant neuron phenomenon in deep RL; propose Recycling Dormant neurons (ReDo) to reduce the number of dormant neurons and maintain network expressivity during training |
| [Efficient RL via Disentangled Environment and Agent Representations](https://openreview.net/pdf?id=kWS8mpioS9) | SEAR | ICML23 oral | consider to build a representation that can disentangle a robotic agent from its environment for improving the learning efficiency for RL; augment the RL loss with an agent-centric auxiliary loss |
| On the Statistical Benefits of Temporal Difference Learning || ICML23 oral ||
| Warm-Start Actor-Critic: From Approximation Error to Sub-optimality Gap || ICML23 oral ||
| Reinforcement Learning from Passive Data via Latent Intentions || ICML23 oral ||
| Subequivariant Graph Reinforcement Learning in 3D Environments || ICML23 oral ||
| Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL || ICML23 oral ||
| [Reparameterized Policy Learning for Multimodal Trajectory Optimization](https://arxiv.org/pdf/2307.10710.pdf) | RPG | ICML23 oral | propose a principled framework that models the continuous RL policy as a generative model of optimal trajectories; present RPG to leverages the multimodal policy parameterization and learned world model to achieve strong exploration capabilities and high data efficiency |
| Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning || ICML23 oral ||
| Settling the Reward Hypothesis || ICML23 oral ||
| Information-Theoretic State Space Model for Multi-View Reinforcement Learning || ICML23 oral ||
| [Learning Belief Representations for Partially Observable Deep RL](https://openreview.net/pdf?id=4IzEmHLono) | Believer | ICML23 | decouple belief state modelling (via unsupervised learning) from policy optimization (via RL); propose a representation learning approach to capture a compact set of reward-relevant features of the state |
| Internally Rewarded Reinforcement Learning || ICML23 ||
| Active Policy Improvement from Multiple Black-box Oracles || ICML23 ||
| When is Realizability Sufficient for Off-Policy Reinforcement Learning? || ICML23 ||
| The Statistical Benefits of Quantile Temporal-Difference Learning for Value Estimation || ICML23 ||
| [Hyperparameters in Reinforcement Learning and How To Tune Them](https://arxiv.org/pdf/2306.01324.pdf) | ---- | ICML23 | Exploration of the hyperparameter landscape for commonly-used RL algorithms and environments; Comparison of different types of HPO methods on state-of-the-art RL algorithms and challenging RL environments |
| Simplified Temporal Consistency Reinforcement Learning || ICML23 ||
| Langevin Thompson Sampling with Logarithmic Communication: Bandits and Reinforcement Learning || ICML23 ||
| Correcting discount-factor mismatch in on-policy policy gradient methods || ICML23 ||
| Masked Trajectory Models for Prediction, Representation, and Control || ICML23 ||
| Off-Policy Average Reward Actor-Critic with Deterministic Policy Search || ICML23 ||
| TGRL: An Algorithm for Teacher Guided Reinforcement Learning || ICML23 ||
| Representation-Driven Reinforcement Learning || ICML23 ||
| LIV: Language-Image Representations and Rewards for Robotic Control || ICML23 ||
| Stein Variational Goal Generation for adaptive Exploration in Multi-Goal Reinforcement Learning || ICML23 ||
| Emergence of Adaptive Circadian Rhythms in Deep Reinforcement Learning || ICML23 ||
| Explaining Reinforcement Learning with Shapley Values || ICML23 ||
| Reinforcement Learning Can Be More Efficient with Multiple Rewards || ICML23 ||
| Jump-Start Reinforcement Learning || ICML23 ||
| Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning || ICML23 ||
| [Performative Reinforcement Learning](https://arxiv.org/pdf/2207.00046.pdf) | ---- | ICML23 | introduce the framework of performative reinforcement learning where the policy chosen by the learner affects the underlying reward and transition dynamics of the environment |
| Truncating Trajectories in Monte Carlo Reinforcement Learning || ICML23 ||
| ReLOAD: Reinforcement Learning with Optimistic Ascent-Descent for Last-Iterate Convergence in Constrained MDPs || ICML23 ||
| Low-Switching Policy Gradient with Exploration via Online Sensitivity Sampling || ICML23 ||
| Hyperbolic Diffusion Embedding and Distance for Hierarchical Representation Learning || ICML23 ||
| Non-stationary Reinforcement Learning under General Function Approximation || ICML23 ||
| Revisiting Domain Randomization via Relaxed State-Adversarial Policy Optimization || ICML23 ||
| Parallel $Q$-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation || ICML23 ||
| LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework || ICML23 ||
| Lower Bounds for Learning in Revealing POMDPs || ICML23 ||
| Graph Reinforcement Learning for Network Control via Bi-Level Optimization || ICML23 ||
| Stochastic Policy Gradient Methods: Improved Sample Complexity for Fisher-non-degenerate Policies || ICML23 ||
| Reinforcement Learning with History Dependent Dynamic Contexts || ICML23 ||
| Efficient Online Reinforcement Learning with Offline Data || ICML23 ||
| Variance Control for Distributional Reinforcement Learning || ICML23 ||
| Hindsight Learning for MDPs with Exogenous Inputs || ICML23 ||
| Behavior Contrastive Learning for Unsupervised Skill Discovery || ICML23 ||
| RLang: A Declarative Language for Describing Partial World Knowledge to Reinforcement Learning Agents || ICML23 ||
| Scalable Safe Policy Improvement via Monte Carlo Tree Search || ICML23 ||
| Bayesian Reparameterization of Reward-Conditioned Reinforcement Learning with Energy-based Models || ICML23 ||
| Understanding the Complexity Gains of Single-Task RL with a Curriculum || ICML23 ||
| PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient || ICML23 ||
| Variational Curriculum Reinforcement Learning for Unsupervised Discovery of Skills || ICML23 ||
| VIMA: Robot Manipulation with Multimodal Prompts || ICML23 ||
| Distilling Internet-Scale Vision-Language Models into Embodied Agents || ICML23 ||
| ContraBAR: Contrastive Bayes-Adaptive Deep RL || ICML23 ||
| On Many-Actions Policy Gradient || ICML23 ||
| Multi-task Hierarchical Adversarial Inverse Reinforcement Learning || ICML23 ||
| Cell-Free Latent Go-Explore || ICML23 ||
| Trustworthy Policy Learning under the Counterfactual No-Harm Criterion || ICML23 ||
| Reachability-Aware Laplacian Representation in Reinforcement Learning || ICML23 ||
| Interactive Object Placement with Reinforcement Learning || ICML23 ||
| Leveraging Offline Data in Online Reinforcement Learning || ICML23 ||
| Reinforcement Learning with General Utilities: Simpler Variance Reduction and Large State-Action Space || ICML23 ||
| Representations and Exploration for Deep Reinforcement Learning using Singular Value Decomposition || ICML23 ||
| CLUTR: Curriculum Learning via Unsupervised Task Representation Learning || ICML23 ||
| Controllability-Aware Unsupervised Skill Discovery || ICML23 ||
| Learning in POMDPs is Sample-Efficient with Hindsight Observability || ICML23 ||
| DoMo-AC: Doubly Multi-step Off-policy Actor-Critic Algorithm || ICML23 ||
| Reward-Mixing MDPs with Few Latent Contexts are Learnable || ICML23 ||
| Computationally Efficient PAC RL in POMDPs with Latent Determinism and Conditional Embeddings || ICML23 ||
| [Scaling Laws for Reward Model Overoptimization](https://openreview.net/attachment?id=bBLjms8nZE&name=pdf) | ---- | ICML23 | study overoptimization in the context of large language models fine-tuned as reward models trained to predict which of two options a human will prefer; study how the gold reward model score changes as we optimize against the proxy reward model using either reinforcement learning or best-of-n sampling |
| SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning || ICML23 ||
| Set-membership Belief State-based Reinforcement Learning for POMDPs || ICML23 ||
| Robust Satisficing MDPs || ICML23 ||
| Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling || ICML23 ||
| Quantum Policy Gradient Algorithm with Optimized Action Decoding || ICML23 ||
| For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal || ICML23 ||
| Model-Free Robust Average-Reward Reinforcement Learning || ICML23 ||
| Fair and Robust Estimation of Heterogeneous Treatment Effects for Policy Learning || ICML23 ||
| Trajectory-Aware Eligibility Traces for Off-Policy Reinforcement Learning || ICML23 ||
| Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons || ICML23 ||
| Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning || ICML23 ||
| [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/pdf/2305.19452.pdf) | BBF | ICML23 | rely on scaling the neural networks used for value estimation and a number of other design choices like resetting |


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
| [Models, Pixels, and Rewards: Evaluating Design Trade-offs in Visual Model-Based Reinforcement Learning](https://arxiv.org/pdf/2012.04603.pdf) | ---- | arXiv2012 | study a number of design decisions for the predictive model in visual MBRL algorithms, focusing specifically on methods that use a predictive model for planning |
| [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf) | EfficientZero | NeurIPS21 | first achieve super-human performance on Atari games with limited data; propose EfficientZero with three components: (i) use self-supervised learning to learn a temporally consistent environment model, (ii) learn the value prefix in an end-to-end manner, (iii) use the learned model to correct off-policy value targets |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | an extension to MBPO; automatically schedule the real data ratio as well as other hyperparameters for MBPO |
| [Model-Advantage and Value-Aware Models for Model-Based Reinforcement Learning: Bridging the Gap in Theory and Practice](https://arxiv.org/pdf/2106.14080.pdf) | ---- | arxiv22 | bridge the gap in theory and practice of value-aware model learning (VAML) for model-based RL |
| [Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/pdf/2204.01464.pdf) | VaGraM | ICLR22 Spotlight | consider the objective mismatch problem in MBRL; propose VaGraM by rescaling the MSE loss function with gradient information from the current value function estimate |
| [Constrained Policy Optimization via Bayesian World Models](https://arxiv.org/pdf/2201.09802.pdf) | LAMBDA | ICLR22 Spotlight | consider Bayesian model-based methods for CMDP |
| [On-Policy Model Errors in Reinforcement Learning](https://arxiv.org/pdf/2110.07985.pdf) | OPC | ICLR22 | consider to combine real-world data and a learned model in order to get the best of both worlds; propose to exploit the real-world data for onpolicy predictions and use the learned model only to generalize to different actions; propose to use on-policy transition data on top of a separately learned model to enable accurate long-term predictions for MBRL |
| [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/pdf/2203.04955.pdf) | TD-MPC | ICML22 | propose to use the model only to predice reward; use a policy to accelerate the planning |
| [Causal Dynamics Learning for Task-Independent State Abstraction](https://arxiv.org/pdf/2206.13452.pdf) |  | ICML22 |  |
| [Mismatched no More: Joint Model-Policy Optimization for Model-Based RL](https://arxiv.org/pdf/2110.02758.pdf) | MnM | NeurIPS22 | propose a model-based RL algorithm where the model and policy are jointly optimized with respect to the same objective, which is a lower bound on the expected return under the true environment dynamics, and becomes tight under certain assumptions |
| When to Update Your Model: Constrained Model-based Reinforcement Learning |  | NeurIPS22 |  |
| Bayesian Optimistic Optimization: Optimistic Exploration for Model-Based Reinforcement Learning || NeurIPS22 ||
| [Model-based Lifelong Reinforcement Learning with Bayesian Exploration](https://arxiv.org/pdf/2210.11579.pdf) || NeurIPS22 ||
| Plan to Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning || NeurIPS22 ||
| data-driven model-based optimization via invariant representation learning || NeurIPS22 ||
| [Reinforcement Learning with Non-Exponential Discounting](https://arxiv.org/pdf/2209.13413.pdf) | ---- | NeurIPS22 | propose a theory for continuous-time model-based reinforcement learning generalized to arbitrary discount functions; derive a Hamilton–Jacobi–Bellman (HJB) equation characterizing the optimal policy and describe how it can be solved using a collocation method |
| Making Better Decision by Directly Planning in Continuous Control || ICLR23 ||
| HiT-MDP: Learning the SMDP option framework on MDPs with Hidden Temporal Embeddings || ICLR23 ||
| Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning || ICLR23 ||
| [Simplifying Model-based RL: Learning Representations, Latent-space Models, and Policies with One Objective](https://arxiv.org/pdf/2209.08466.pdf) | ALM | ICLR23 | propose a single objective which jointly optimizes the policy, the latent-space model, and the representations produced by the encoder using the same objective: maximize predicted rewards while minimizing the errors in the predicted representations |
| [SpeedyZero: Mastering Atari with Limited Data and Time](https://openreview.net/pdf?id=Mg5CLXZgvLJ) | SpeedyZero | ICLR23 | a distributed RL system built upon EfficientZero with Priority Refresh and Clipped LARS; lead to human-level performances on the Atari benchmark within 35 minutes using only 300k samples |
| Investigating the role of model-based learning in exploration and transfer || ICML23 ||
| [STEERING : Stein Information Directed Exploration for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2301.12038.pdf) | STEERING | ICML23 |  |
| [Predictable MDP Abstraction for Unsupervised Model-Based RL](https://arxiv.org/pdf/2302.03921.pdf) | PMA | ICML23 | apply model-based RL on top of an abstracted, simplified MDP, by restricting unpredictable actions |
| The Virtues of Laziness in Model-based RL: A Unified Objective and Algorithms || ICML23 ||


<a id='dreamer'></a>
### World Models

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [World Models](https://arxiv.org/pdf/1803.10122.pdf), [\[NeurIPS version\]](https://proceedings.neurips.cc/paper/2018/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf) | World Models | NeurIPS18 | use an unsupervised manner to learn a compressed spatial and temporal representation of the environment and use the world model to train a very compact and simple policy for solving the required task |
| [Learning latent dynamics for planning from pixels](https://arxiv.org/pdf/1811.04551.pdf) | PlaNet | ICML19 | propose PlaNet to learn the environment dynamics from images; the dynamic model consists transition model, observation model, reward model and encoder; use the cross entropy method for selecting actions for planning |
| [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/pdf/1912.01603.pdf) | Dreamer | ICLR20 | solve long-horizon tasks from images purely by latent imagination; test in image-based MuJoCo; propose to use an agent to replace the control algorithm in the PlaNet |
| [Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning](https://arxiv.org/pdf/2010.12142.pdf) | BIRD | NeurIPS20 | propose to maximize the mutual information between imaginary and real trajectories so that the policy improvement learned from imaginary trajectories can be easily generalized to real trajectories |
| [Planning to Explore via Self-Supervised World Models](https://arxiv.org/pdf/2005.05960.pdf) | Plan2Explore | ICML20 | propose Plan2Explore to  self-supervised exploration and fast adaptation to new tasks |
| [Mastering Atari with Discrete World Models](https://arxiv.org/pdf/2010.02193.pdf) | Dreamerv2 | ICLR21 | solve long-horizon tasks from images purely by latent imagination; test in image-based Atari |
| [Temporal Predictive Coding For Model-Based Planning In Latent Space](https://arxiv.org/pdf/2106.07156.pdf) | TPC | ICML21 | propose a temporal predictive coding approach for planning from high-dimensional observations and theoretically analyze its ability to prioritize the encoding of task-relevant information |
| [Learning Task Informed Abstractions](https://arxiv.org/pdf/2106.15612.pdf) | TIA | ICML21 | introduce the formalism of Task Informed MDP (TiMDP) that is realized by training two models that learn visual features via cooperative reconstruction, but one model is adversarially dissociated from the reward signal |
| [Dreaming: Model-based Reinforcement Learning by Latent Imagination without Reconstruction](https://arxiv.org/pdf/2007.14535.pdf) | Dreaming | ICRA21 | propose a decoder-free extension of Dreamer since the autoencoding based approach often causes object vanishing|
| [Model-Based Reinforcement Learning via Imagination with Derived Memory](https://openreview.net/pdf?id=jeATherHHGj) | IDM | NeurIPS21 | hope to improve the diversity of imagination for model-based policy optimization with the derived memory; point out that current methods cannot effectively enrich the imagination if the latent state is disturbed by random noises |
| [Maximum Entropy Model-based Reinforcement Learning](https://arxiv.org/pdf/2112.01195.pdf) |  MaxEnt Dreamer | NeurIPS21 | create a connection between exploration methods and model-based reinforcement learning; apply maximum-entropy exploration for Dreamer |
| [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf) | DreamerPro | ICML22 | consider reconstruction-free MBRL; propose to learn the prototypes from the recurrent states of the world model, thereby distilling temporal structures from past observations and actions into the prototypes. |
| [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/pdf/2202.09481.pdf) | TransDreamer | arxiv2202 | replace the RNN in RSSM by a transformer |
| [DreamingV2: Reinforcement Learning with Discrete World Models without Reconstruction](https://arxiv.org/pdf/2203.00494.pdf) | Dreamingv2 | arxiv2203 | adopt both the discrete representation of DreamerV2 and the reconstruction-free objective of Dreaming |
| [Masked World Models for Visual Control](https://arxiv.org/pdf/2206.14244.pdf) | MWM | arxiv2206 | decouple visual representation learning and dynamics learning for visual model-based RL and use masked autoencoder to train visual representation |
| [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/pdf/2206.14176.pdf) | DayDreamer | arxiv2206 | apply Dreamer to 4 robots to learn online and directly in the real world, without any simulators |
| [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://proceedings.mlr.press/v162/wan22d/wan22d.pdf) | ---- | ICML22 | introduce an improved version of the LoCA setup and use it to evaluate PlaNet and Dreamerv2 |
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://arxiv.org/pdf/2203.13880.pdf) | APV | ICML22 | pre-train an action-free latent video prediction model using videos from different domains, and then fine-tune the pre-trained model on target domains |
| [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf) | Denoised MDP | ICML22 | divide information into four categories: controllable/uncontrollable (whether infected by the action) and reward-relevant/irrelevant (whether affects the return); propose to only consider information which is controllable and reward-relevant |
| [Iso-Dream: Isolating Noncontrollable Visual Dynamics in World Models](https://arxiv.org/pdf/2205.13817.pdf) | Iso-Dream | NeurIPS22 | consider noncontrollable dynamics independent of the action signals; encourage the world model to learn controllable and noncontrollable sources of spatiotemporal changes on isolated state transition branches; optimize the behavior of the agent on the decoupled latent imaginations of the world model |
| [Learning General World Models in a Handful of Reward-Free Deployments](https://arxiv.org/pdf/2210.12719.pdf) | CASCADE | NeurIPS22 | introduce the reward-free deployment efficiency setting to facilitate generalization (exploration should be task agnostic) and scalability (exploration policies should collect large quantities of data without costly centralized retraining); propose an information theoretic objective inspired by Bayesian Active Learning by specifically maximizing the diversity of trajectories sampled by the population through a novel cascading objective |
| [Learning Robust Dynamics through Variational Sparse Gating](https://arxiv.org/pdf/2210.11698.pdf) | VSG, SVSG, BBS | NeurIPS22 | consider to sparsely update the latent states at each step; develope a new partially-observable and stochastic environment, called BringBackShapes (BBS) |
| [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf) | IRIS | ICLR23 oral | use a discrete autoencoder and an autoregressive Transformer to conduct World Models and significantly improve the data efficiency in Atari (2 hours of real-time experience); [\[code\]](https://github.com/eloialonso/iris) |
| [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf) | TWM | ICLR23 | present a new autoregressive world model based on the Transformer-XL; obtain excellent results on the Atari 100k benchmark; [\[code\]](https://github.com/jrobine/twm) |
| [Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting](https://arxiv.org/pdf/2303.10144.pdf) | DUTD | ICLR23 | propose a new general method that dynamically adjusts the update to data (UTD) ratio during training based on underand overfitting detection on a small subset of the continuously collected experience not used for training; apply this method in DreamerV2 |
| [Evaluating Long-Term Memory in 3D Mazes](https://arxiv.org/pdf/2210.13383.pdf) | Memory Maze | ICLR23 | introduce the Memory Maze, a 3D domain of randomized mazes specifically designed for evaluating long-term memory in agents, including an online reinforcement learning benchmark, a diverse offline dataset, and an offline probing evaluation; [\[code\]](https://github.com/jurgisp/memory-maze) |
| [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104.pdf) | DreamerV3 | arxiv2301 | propose DreamerV3 to handle a wide range of domains, including continuous and discrete actions, visual and low-dimensional inputs, 2D and 3D worlds, different data budgets, reward frequencies, and reward scales|
| [Reward Informed Dreamer for Task Generalization in Reinforcement Learning](https://arxiv.org/pdf/2303.05092.pdf) | RID | arXiv2303 | propose Task Distribution Relevance to capture the relevance of the task distribution quantitatively; propose RID to use world models to improve task generalization via encoding reward signals into policies |
| [Posterior Sampling for Deep Reinforcement Learning](https://arxiv.org/pdf/2305.00477.pdf) | PSDRL | ICML23 | combine efficient uncertainty quantification over latent state space models with a specially tailored continual planning algorithm based on value-function approximation |
| Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators || ICML23 ||
| Learning Temporally Abstract World Models without Online Experimentation || ICML23 ||
| Go Beyond Imagination: Maximizing Episodic Reachability with World Models || ICML23 ||
| [Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling](https://arxiv.org/pdf/2301.12050.pdf) | DECKARD | ICML23 | hypothesize an Abstract World Model (AWM) over subgoals by few-shot prompting an LLM |
| Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum || ICML23 ||
| [Curious Replay for Model-based Adaptation](https://arxiv.org/pdf/2306.15934.pdf) | CR | ICML23 | aid model-based RL agent adaptation by prioritizing replay of experiences the agent knows the least about |
| [Multi-View Masked World Models for Visual Robotic Manipulation](https://arxiv.org/pdf/2302.02408.pdf) | MV-MWM | ICML23 | train a multi-view masked autoencoder that reconstructs pixels of randomly masked viewpoints and then learn a world model operating on the representations from the autoencoder |
| [Facing off World Model Backbones: RNNs, Transformers, and S4](https://arxiv.org/pdf/2307.02064.pdf) | S4WM | arXiv2307 | propose the first S4-based world model that can generate high-dimensional image sequences through latent imagination |



<a id='model-based-code'></a>
### CodeBase

|  Title | Conference | Methods |  Github |
| ---- | ---- | ---- | ---- |
| [MBRL-Lib: A Modular Library for Model-based Reinforcement Learning](https://arxiv.org/pdf/2104.10159.pdf) | arxiv21 | MBPO,PETS,PlaNet | [link](https://github.com/facebookresearch/mbrl-lib) |



<a id='Model-Free-Offline'></a>
## (Model Free) Offline RL

<a id='offline-current'></a>
### Current Methods

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
| DASCO: Dual-Generator Adversarial Support Constrained Offline Reinforcement Learning || NeurIPS22 ||
| [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/pdf/2202.06239.pdf) || NeurIPS22 ||
| [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/pdf/2205.13703.pdf) || NeurIPS22 ||
| Oracle Inequalities for Model Selection in Offline Reinforcement Learning || NeurIPS22 ||
| [Mildly Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.04745.pdf) || NeurIPS22 ||
| [A Policy-Guided Imitation Approach for Offline Reinforcement Learning](https://arxiv.org/pdf/2210.08323.pdf) || NeurIPS22 ||
| [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.08569.pdf) || NeurIPS22 ||
| [LobsDICE: Offline Learning from Observation via Stationary Distribution Correction Estimation](https://arxiv.org/pdf/2202.13536.pdf) || NeurIPS22 ||
| [Latent-Variable Advantage-Weighted Policy Optimization for Offline RL](https://arxiv.org/pdf/2203.08949.pdf) || NeurIPS22 ||
| [How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression](https://arxiv.org/pdf/2206.03023.pdf) || NeurIPS22 ||
| [NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning](https://arxiv.org/pdf/2102.00714.pdf) || NeurIPS22 ||
| [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/pdf/2206.01079.pdf) || NeurIPS22 ||
| [Bellman Residual Orthogonalization for Offline Reinforcement Learning](https://arxiv.org/pdf/2203.12786.pdf) || NeurIPS22 ||
| [Oracle Inequalities for Model Selection in Offline Reinforcement Learning](https://arxiv.org/pdf/2211.02016.pdf) || NeurIPS22 ||
| Offline Q-learning on Diverse Multi-Task Data Both Scales And Generalizes || ICLR23 oral ||
| Confidence-Conditioned Value Functions for Offline Reinforcement Learning || ICLR23 oral ||
| Extreme Q-Learning: MaxEnt RL without Entropy || ICLR23 oral ||
| Sparse Q-Learning: Offline Reinforcement Learning with Implicit Value Regularization || ICLR23 oral ||
| The In-Sample Softmax for Offline Reinforcement Learning || ICLR23 Spotlight ||
| Benchmarking Offline Reinforcement Learning on Real-Robot Hardware || ICLR23 Spotlight ||
| Decision S4: Efficient Sequence-Based RL via State Spaces Layers || ICLR23 ||
| Behavior Proximal Policy Optimization || ICLR23 ||
| Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward || ICLR23 ||
| Explaining RL Decisions with Trajectories  || ICLR23 ||
| User-Interactive Offline Reinforcement Learning || ICLR23 ||
| Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning || ICLR23 ||
| Offline RL for Natural Language Generation with Implicit Language Q Learning  || ICLR23 ||
| In-sample Actor Critic for Offline Reinforcement Learning  || ICLR23 ||
| Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting || ICLR23 ||
| Mind the Gap: Offline Policy Optimizaiton for Imperfect Rewards || ICLR23 ||
| [When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11027.pdf) | DOGE | ICLR23 | train a state-conditioned distance function that can be readily plugged into standard actor-critic methods as a policy constraint |
| MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations || ICLR23 ||
| Actor-Critic Alignment for Offline-to-Online Reinforcement Learning || ICML23 ||
| Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories || ICML23 ||
| Principled Offline RL in the Presence of Rich Exogenous Information || ICML23 ||
| Offline Meta Reinforcement Learning with In-Distribution Online Adaptation || ICML23 ||
| Policy Regularization with Dataset Constraint for Offline Reinforcement Learning || ICML23 ||
| Supported Trust Region Optimization for Offline Reinforcement Learning || ICML23 ||
| Constrained Decision Transformer for Offline Safe Reinforcement Learning || ICML23 ||
| PAC-Bayesian Offline Contextual Bandits With Guarantees || ICML23 ||
| Beyond Reward: Offline Preference-guided Policy Optimization || ICML23 ||
| Offline Reinforcement Learning with Closed-Form Policy Improvement Operators || ICML23 ||
| ChiPFormer: Transferable Chip Placement via Offline Decision Transformer || ICML23 ||
| Boosting Offline Reinforcement Learning with Action Preference Query || ICML23 ||


<a id='offline-diffusion'></a>
### Combined with Diffusion Models

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) | Diffuser | ICML22 oral | first propose a denoising diffusion model designed for trajectory data and an associated probabilistic framework for behavior synthesis; demonstrate that Diffuser has a number of useful properties and is particularly effective in offline control settings that require long-horizon reasoning and test-time flexibility |
| Is Conditional Generative Modeling all you need for Decision Making? || ICLR23 oral ||
| [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/pdf/2208.06193.pdf) | Diffusion-QL | ICLR23 | perform policy regularization using diffusion (or scorebased) models; utilize a conditional diffusion model to represent the policy |
| [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/pdf/2209.14548.pdf) | SfBC | ICLR23 | decouple the learned policy into two parts: an expressive generative behavior model and an action evaluation model |
| [AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/pdf/2302.01877.pdf) | AdaptDiffuser | ICML23 oral | propose AdaptDiffuser, an evolutionary planning method with diffusion that can self-evolve to improve the diffusion model hence a better planner, which can also adapt to unseen tasks |
| Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning | CEP | ICML23 ||
| MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL || ICML23 ||


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
| [Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief](https://arxiv.org/pdf/2210.06692.pdf) || NeurIPS22 ||
| [A Unified Framework for Alternating Offline Model Training and Policy Learning](https://arxiv.org/pdf/2210.05922.pdf) || NeurIPS22 ||
| [Bidirectional Learning for Offline Infinite-width Model-based Optimization](https://arxiv.org/pdf/2209.07507.pdf) || NeurIPS22 ||
| Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization || ICLR23 ||
| Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning || ICLR23 ||
| Efficient Offline Policy Optimization with a Learned Model  || ICLR23 ||
| Model-based Offline Reinforcement Learning with Count-based Conservatism || ICML23 ||
| Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning || ICML23 ||


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
| Distributional Meta-Gradient Reinforcement Learning || ICLR23 ||
| Simple Embodied Language Learning as a Byproduct of Meta-Reinforcement Learning || ICML23 ||



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
| [Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning](https://arxiv.org/pdf/2106.15860.pdf) | ---- | SCIS 2023 | summarize current optimization-based adversarial attacks in RL; propose a two-stage methods: train a deceptive policy and mislead the victim to imitate the deceptive policy |
| On the Robustness of Safe Reinforcement Learning under Observational Perturbations  || ICLR23 ||
| [Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation](https://arxiv.org/pdf/2206.05751.pdf) | Reward UAP, Trajectory UAP | PRL 2023 | extend universal adversarial perturbations into sequential decision and propose Reward UAP as well as Trajectory UAP via utilizing the dynamic; experiment in Embodied Vision Navigation tasks |
| Detecting Adversarial Directions in Deep Reinforcement Learning to Make Robust Decisions || ICML23 ||
| Robust Situational Reinforcement Learning in Face of Context Disturbances || ICML23 ||
| Adversarial Learning of Distributional Reinforcement Learning || ICML23 ||
| Towards Robust and Safe Reinforcement Learning with Benign Off-policy Data || ICML23 ||

<a id='Genaralization-in-RL'></a>
## Genaralisation in RL

<a id='Gene-Environments'></a>
### Environments

| Title | Method | Conference | Description | 
| ----  | ----   | ----       |   ----  |
| [Quantifying Generalization in Reinforcement Learning](https://arxiv.org/pdf/1812.02341.pdf) | CoinRun | ICML19 | introduce a new environment called CoinRun for generalisation in RL; empirically show L2 regularization, dropout, data augmentation and batch normalization can improve generalization in RL |
| [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/pdf/1912.01588.pdf) | Procgen Benchmark | ICML20 | introduce Procgen Benchmark, a suite of 16 procedurally generated game-like environments designed to benchmark both sample efficiency and generalization in reinforcement learning |

<a id='Gene-Methods'></a>
### Methods

| Title | Method | Conference | Description | 
| ----  | ----   | ----       |   ----  |
| [Towards Generalization and Simplicity in Continuous Control](https://arxiv.org/pdf/1703.02660.pdf) | ---- | NeurIPS17 | policies with simple linear and RBF parameterizations can be trained to solve a variety of widely studied continuous control tasks; training with a diverse initial state distribution induces more global policies with better generalization |
| [Universal Planning Networks](https://arxiv.org/pdf/1804.00645.pdf) | UPN | ICML18 |  study a model-based architecture that performs a differentiable planning computation in a latent space jointly learned with forward dynamics, trained end-to-end to encode what is necessary for solving tasks by gradient-based planning |
| [On the Generalization Gap in Reparameterizable Reinforcement Learning](https://arxiv.org/pdf/1905.12654.pdf) | ---- | ICML19 | theoretically provide guarantees on the gap between the expected and empirical return for both intrinsic and external errors in reparameterizable RL |
| [Investigating Generalisation in Continuous Deep Reinforcement Learning](https://arxiv.org/pdf/1902.07015.pdf) | ---- | arxiv19 | study generalisation in Deep RL for continuous control |
| [Generalization in Reinforcement Learning with Selective Noise Injection and Information Bottleneck](https://arxiv.org/pdf/1910.12911.pdf) | SNI | NeurIPS19 | consder regularization techniques relying on the injection of noise into the learned function for improving generalization; hope to maintain the regularizing effect of the injected noise and mitigate its adverse effects on the gradient quality |
| [Network randomization: A simple technique for generalization in deep reinforcement learning](https://arxiv.org/pdf/1910.05396.pdf) | Network Randomization | ICLR20 | introduce a randomized (convolutional) neural network that randomly perturbs input observations, which enables trained agents to adapt to new domains by learning robust features invariant across varied and randomized environments |
| [Observational Overfitting in Reinforcement Learning](https://arxiv.org/pdf/1912.02975.pdf) | observational overfitting | ICLR20 | discuss realistic instances where observational overfitting may occur and its difference from other confounding factors, and design a parametric theoretical framework to induce observational overfitting that can be applied to any underlying MDP |
| [Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning](https://arxiv.org/pdf/2005.06800.pdf) | CaDM | ICML20 | decompose the task of learning a global dynamics model into two stages: (a) learning a context latent vector that captures the local dynamics, then (b) predicting the next state conditioned on it |
| [Improving Generalization in Reinforcement Learning with Mixture Regularization](https://arxiv.org/pdf/2010.10814.pdf) | mixreg | NeurIPS20 | train agents on a mixture of observations from different training environments and imposes linearity constraints on the observation interpolations and the supervision (e.g. associated reward) interpolations |
| [Instance based Generalization in Reinforcement Learning](https://arxiv.org/pdf/2011.01089.pdf) | IPAE | NeurIPS20 | formalize the concept of training levels as instances and show that this instance-based view is fully consistent with the standard POMDP formulation; provide generalization bounds to the value gap in train and test environments based on the number of training instances, and use insights based on these to improve performance on unseen levels |
| [Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2101.05265.pdf) | PSM | ICLR21 | incorporate the inherent sequential structure in reinforcement learning into the representation learning process to improve generalization;  introduce a theoretically motivated policy similarity metric (PSM) for measuring behavioral similarity between states |
| [Generalization in Reinforcement Learning by Soft Data Augmentation](https://arxiv.org/pdf/2011.13389.pdf) | SODA | ICRA21 | imposes a soft constraint on the encoder that aims to maximize the mutual information between latent representations of augmented and non-augmented data, |
| [Augmented World Models Facilitate Zero-Shot Dynamics Generalization From a Single Offline Environment](https://arxiv.org/pdf/2104.05632.pdf) | AugWM | ICML21 | consider the setting named "dynamics generalization from a single offline environment" and concentrate on the zero-shot performance to unseen dynamics; propose dynamics augmentation for model based offline RL; propose a simple self-supervised context adaptation reward-free algorithm |
| [Decoupling Value and Policy for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2102.10330.pdf) | IDAAC | ICML21 | decouples the optimization of the policy and value function, using separate networks to model them; introduce an auxiliary loss which encourages the representation to be invariant to task-irrelevant properties of the environment |
| [Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability](https://arxiv.org/pdf/2107.06277.pdf) | LEEP | NeurIPS21 | generalisation in RL induces implicit partial observability; propose LEEP to use an ensemble of policies to approximately learn the Bayes-optimal policy for maximizing test-time performance |
| [Automatic Data Augmentation for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2006.12862.pdf) | DrAC | NeurIPS21 | focus on automatic data augmentation based two novel regularization terms for the policy and value function |
| [When Is Generalizable Reinforcement Learning Tractable?](https://arxiv.org/pdf/2101.00300.pdf) | ---- | NeurIPS21 | propose Weak Proximity and Strong Proximity for theoretically analyzing the generalisation of RL |
| [A Survey of Generalisation in Deep Reinforcement Learning](https://arxiv.org/pdf/2111.09794.pdf) | ---- | arxiv21 | provide a unifying formalism and terminology for discussing different generalisation problems |
| [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) | CTRL | ICLR22 | consider zero-shot generalization (ZSG); use self-supervised learning to learn a representation across tasks |
| [The Role of Pretrained Representations for the OOD Generalization of RL Agents](https://arxiv.org/pdf/2107.05686.pdf) | ---- | ICLR22 | train 240 representations and 11,520 downstream policies and systematically investigate their performance under a diverse range of distribution shifts; find that a specific representation metric that measures the generalization of a simple downstream proxy task reliably predicts the generalization of downstream RL agents under the broad spectrum of OOD settings considered here |
| [Generalisation in Lifelong Reinforcement Learning through Logical Composition](https://openreview.net/pdf?id=ZOcX-eybqoL) | ---- | ICLR22 | e leverage logical composition in reinforcement learning to create a framework that enables an agent to autonomously determine whether a new task can be immediately solved using its existing abilities, or whether a task-specific skill should be learned |
| [Local Feature Swapping for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2204.06355.pdf) | CLOP | ICLR22 | introduce a new regularization technique consisting of channel-consistent local permutations of the feature maps |
| [A Generalist Agent](https://arxiv.org/pdf/2205.06175.pdf) | Gato | arxiv2205 | [slide](https://ml.cs.tsinghua.edu.cn/~chengyang/reading_meeting/Reading_Meeting_20220607.pdf) |
| [Towards Safe Reinforcement Learning via Constraining Conditional Value at Risk](https://arxiv.org/pdf/2206.04436.pdf) | CPPO | IJCAI22 | find the connection between modifying observations and dynamics, which are structurally different |
| [CtrlFormer: Learning Transferable State Representation for Visual Control via Transformer](https://arxiv.org/pdf/2206.08883.pdf) | CtrlFormer | ICML22 | jointly learns self-attention mechanisms between visual tokens and policy tokens among different control tasks, where multitask representation can be learned and transferred without catastrophic forgetting |
| [Learning Dynamics and Generalization in Reinforcement Learning](https://arxiv.org/pdf/2206.02126.pdf) | ---- | ICML22 | show theoretically that temporal difference learning encourages agents to fit non-smooth components of the value function early in training, and at the same time induces the second-order effect of discouraging generalization |
| [Improving Policy Optimization with Generalist-Specialist Learning](https://arxiv.org/pdf/2206.12984.pdf) | GSL | ICML22 | hope to utilize experiences from the specialists to aid the policy optimization of the generalist; propose the phenomenon “catastrophic ignorance” in multi-task learning |
| [DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck](https://arxiv.org/pdf/2102.13268.pdf) | DRIBO | ICML22 | learn robust representations that encode only task-relevant information from observations based on the unsupervised multi-view setting; introduce a novel contrastive version of the Multi-View Information Bottleneck (MIB) objective for temporal data |
| [Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning](https://arxiv.org/pdf/2207.09081.pdf) | GRADER | NeurIPS22 | use the causal graph as a latent variable to reformulate the GCRL problem and then derive an iterative training framework from solving this problem |
| [Rethinking Value Function Learning for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2210.09960.pdf) | DCPG, DDCPG | NeurIPS22 | consider to train agents on multiple training environments to improve observational generalization performance; identify that the value network in the multiple-environment setting is more challenging to optimize; propose regularization methods that penalize large estimates of the value network for preventing overfitting |
| [Masked Autoencoding for Scalable and Generalizable Decision Making](https://arxiv.org/pdf/2211.12740.pdf) | MaskDP | NeurIPS22 | employ a masked autoencoder (MAE) to state-action trajectories for reinforcement learning (RL) and behavioral cloning (BC) and gain the capability of zero-shot transfer to new tasks |
| [Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning](https://arxiv.org/pdf/2212.08860.pdf) | PIE-G | NeurIPS22 | find that the early layers in an ImageNet pre-trained ResNet model could provide rather generalizable representations for visual RL |
| [GALOIS: Boosting Deep Reinforcement Learning via Generalizable Logic Synthesis](https://arxiv.org/pdf/2205.13728.pdf) || NeurIPS22 ||
| [Look where you look! Saliency-guided Q-networks for visual RL tasks](https://arxiv.org/pdf/2209.09203.pdf) | SGQN | NeurIPS22 | propose that a good visual policy should be able to identify which pixels are important for its decision; preserve this identification of important sources of information across images |
| [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/pdf/2301.07608.pdf) | AdA | arXiv 2301 | demonstrate that training an RL agent at scale leads to a general in-context learning algorithm that can adapt to open-ended novel embodied 3D problems as quickly as humans |
| [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/pdf/2210.14215.pdf) | AD | ICLR23 oral | propose Algorithm Distillation for distilling reinforcement learning (RL) algorithms into neural networks by modeling their training histories with a causal sequence model |
| Can Agents Run Relay Race with Strangers? Generalization of RL to Out-of-Distribution Trajectories || ICLR23 ||
| [Performance Bounds for Model and Policy Transfer in Hidden-parameter MDPs](https://openreview.net/pdf?id=sSt9fROSZRO) || ICLR23 | show that, given a fixed amount of pretraining data, agents trained with more variations are able to generalize better; find that increasing the capacity of the value and policy network is critical to achieve good performance |
| [Investigating Multi-task Pretraining and Generalization in Reinforcement Learning](https://openreview.net/pdf?id=sSt9fROSZRO) | ---- | ICLR23 |  find that, given a fixed amount of pretraining data, agents trained with more variations are able to generalize better; this advantage can still be present after fine-tuning for 200M environment frames than when doing zero-shot transfer |
| Priors, Hierarchy, and Information Asymmetry for Skill Transfer in Reinforcement Learning || ICLR23 ||
| [Cross-domain Random Pre-training with Prototypes for Reinforcement Learning](https://arxiv.org/pdf/2302.05614.pdf) | CRPTpro | arXiv2302 | use prototypical representation learning with a novel intrinsic loss to pre-train an effective and generic encoder across different domains |
| [Reward Informed Dreamer for Task Generalization in Reinforcement Learning](https://arxiv.org/pdf/2303.05092.pdf) | RID | arXiv2303 | propose Task Distribution Relevance to capture the relevance of the task distribution quantitatively; propose RID to use world models to improve task generalization via encoding reward signals into policies |
| [On the Power of Pre-training for Generalization in RL: Provable Benefits and Hardness](https://arxiv.org/pdf/2210.10464.pdf) || ICML23 oral ||
| [On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline](https://arxiv.org/pdf/2212.05749.pdf) || ICML23 ||
| Unsupervised Skill Discovery for Learning Shared Structures across Changing Environments || ICML23 ||
| An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning || ICML23 ||
| Guiding Pretraining in Reinforcement Learning with Large Language Models || ICML23 ||
| What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL? || ICML23 ||
| [The Benefits of Model-Based Generalization in Reinforcement Learning](https://openreview.net/pdf?id=Vue1ulwlPD) | ---- | ICML23 | provide theoretical and empirical insight into when, and how, we can expect data generated by a learned model to be useful |
| Multi-Environment Pretraining Enables Transfer to Action Limited Datasets || ICML23 ||
| Online Prototype Alignment for Few-shot Policy Transfer || ICML23 ||


<a id='Sequence-Generation'></a>
## RL with Transformer

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Stabilizing transformers for reinforcement learning](https://arxiv.org/pdf/1910.06764.pdf) | GTrXL | ICML20 | stabilizing training with a reordering of the layer normalization coupled with the addition of a new gating mechanism to key points in the submodules of the transformer |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (return-to-go, state, action, return-to-go, ...); there is not explicit optimization process; evaluate on Offline RL |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (s_0^0, ..., s_0^N, a_0^0, ..., a_0^M, r_0, ...); use beam search to inference; evaluate on imitation learning, goal-conditioned RL and Offline RL | 
| [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/pdf/2201.12122.pdf) | ChibiT | arxiv2201 | demonstrate that pre-training on autoregressively modeling natural language provides consistent performance gains when compared to the Decision Transformer on both the popular OpenAI Gym and Atari |
| [Online Decision Transformer](https://arxiv.org/pdf/2202.05607.pdf) | ODT | ICML22 oral | blends offline pretraining with online finetuning in a unified framework; use sequence-level entropy regularizers in conjunction with autoregressive modeling objectives for sample-efficient exploration and finetuning |
| Prompting Decision Transformer for Few-shot Policy Generalization || ICML22 ||
| [Multi-Game Decision Transformers](https://arxiv.org/pdf/2205.15241.pdf) | ---- | NeurIPS22 | show that a single transformer-based model trained purely offline can play a suite of up to 46 Atari games simultaneously at close-to-human performance |
| Bootstrapped Transformer for Offline Reinforcement Learning || NeurIPS22 ||
| Dichotomy of Control: Separating What You Can Control from What You Cannot  || ICLR23 oral ||
| Decision Transformer under Random Frame Dropping || ICLR23 ||
| Hyper-Decision Transformer for Efficient Online Policy Adaptation  || ICLR23 ||
| Preference Transformer: Modeling Human Preferences using Transformers for RL || ICLR23 ||
| On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning  || ICLR23 ||
| Future-conditioned Unsupervised Pretraining for Decision Transformer || ICML23 ||
| Emergent Agentic Transformer from Chain of Hindsight Experience || ICML23 ||


<a id='Lifelong-RL'></a>
## Continual / Lifelong RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| Revisiting Curiosity for Exploration in Procedurally Generated Environments  || ICLR23 ||


<a id='RL-LLM'></a>
## RL with LLM

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
