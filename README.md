# Reinforcement Learning Papers
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

Related papers for Reinforcement Learning (we mainly focus on single-agent).

Since there are tens of thousands of new papers on reinforcement learning at each conference every year, we are only able to list those we read and consider as insightful.

**We have added some ICLR22, ICML22, NeurIPS22, ICLR23, ICML23, NeurIPS23, ICLR24 papers on RL**
<!-- NeurIps24 page 71, ICLR24 page 31 -->


## Contents 
* [Model Free (Online) RL](#Model-Free-Online)
    - [Classic Methods](#model-free-classic)
    - [Exploration](#exploration)
    - [Representation Learning](#Representation-RL)
    - [Unsupervised Learning](#Unsupervised-RL)
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
* [RL with LLM](#RL-LLM)
* [Tutorial and Lesson](#Tutorial-and-Lesson)
* [ICLR22](#ICLR22)
* [ICML22](#ICML22)
* [NeurIPS22](#NeurIPS22)
* [ICLR23](#ICLR23)
* [ICML23](#ICML23)
* [NeurIPS23](#NeurIPS23)
* [ICLR24](#ICLR24)

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
| [Curiosity-Driven Exploration via Latent Bayesian Surprise](https://arxiv.org/pdf/2104.07495.pdf) | LBS | AAAI22 | apply Bayesian surprise in a latent space representing the agent’s current understanding of the dynamics of the system |
| [Automatic Intrinsic Reward Shaping for Exploration in Deep Reinforcement Learning](https://arxiv.org/pdf/2301.10886.pdf) | AIRS | ICML23 | select shaping function from a predefined set based on the estimated task return in real-time, providing reliable exploration incentives and alleviating the biased objective problem; develop a toolkit that provides highquality implementations of various intrinsic reward modules based on PyTorch |
| [Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments](https://arxiv.org/pdf/2211.10515.pdf) | Curiosity in Hindsight | ICML23 | consider exploration in stochastic environments; learn representations of the future that capture precisely the unpredictable aspects of each outcome—which we use as additional input for predictions, such that intrinsic rewards only reflect the predictable aspects of world dynamics |
| Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration || NeurIPS23 spotlight ||
| [MIMEx: Intrinsic Rewards from Masked Input Modeling](https://arxiv.org/pdf/2305.08932.pdf) | MIMEx | NeurIPS23 | propose that the mask distribution can be flexibly tuned to control the difficulty of the underlying conditional prediction task |

<!--<a id='off-policy-evaluation'></a>
### Off-Policy Evaluation
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Weighted importance sampling for off-policy learning with linear function approximation](https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf) | WIS-LSTD | NeurIPS14 |  |
| [Importance Sampling Policy Evaluation with an Estimated Behavior Policy](https://arxiv.org/pdf/1806.01347.pdf) | RIS | ICML19 |  |
| [On the Reuse Bias in Off-Policy Reinforcement Learning](https://arxiv.org/pdf/2209.07074.pdf) | BIRIS | IJCAI23 | discuss the bias of off-policy evaluation due to reusing the replay buffer; derive a high-probability bound of the Reuse Bias; introduce the concept of stability for off-policy algorithms and provide an upper bound for stable off-policy algorithms | 


<a id='soft-rl'></a>
### Soft RL

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) | MME | NeurIPS21 | find that SAC may fail in explore states with low entropy (arrive states with high entropy and increase their entropies); propose a max-min entropy framework to address this issue |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that standard maximum entropy RL is robust to some disturbances in the dynamics and the reward function | 


<a id='data-augmentation'></a>
### Data Augmentation
|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Reinforcement Learning with Augmented Data](https://arxiv.org/pdf/2004.14990.pdf) | RAD | NeurIPS20 | propose first extensive study of general data augmentations for RL on both pixel-based and state-based inputs |
| [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2004.13649.pdf) | DrQ | ICLR21 Spotlight | propose to regularize the value function when applying data augmentation with model-free methods and reach state-of-the-art performance in image-pixels tasks | -->

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
| [CURL: Contrastive Unsupervised Representations for Reinforcement Learning](https://arxiv.org/pdf/2004.04136.pdf) | CURL | ICML20 | extracts high-level features from raw pixels using contrastive learning and performs offpolicy control on top of the extracted features |
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf) | DBC | ICLR21 | propose using Bisimulation to learn robust latent representations which encode only the task-relevant information from observations |
| [Reinforcement Learning with Prototypical Representations](https://arxiv.org/pdf/2102.11271.pdf) | Proto-RL | ICML21 | pre-train task-agnostic representations and prototypes on environments without downstream task information |
| [Understanding the World Through Action](https://arxiv.org/pdf/2110.12543.pdf) | ---- | CoRL21 | discusse how self-supervised reinforcement learning combined with offline RL can enable scalable representation learning |
| [Flow-based Recurrent Belief State Learning for POMDPs](https://proceedings.mlr.press/v162/chen22q/chen22q.pdf) | FORBES | ICML22 | incorporate normalizing flows into the variational inference to learn general continuous belief states for POMDPs |
| [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568.pdf) | Contrastive RL | NeurIPS22 | show (contrastive) representation learning methods can be cast as RL algorithms in their own right |
| [Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?](https://arxiv.org/pdf/2206.05266.pdf) | ---- | NeurIPS22 | conduct an extensive comparison of various self-supervised losses under the existing joint learning framework for pixel-based reinforcement learning in many environments from different benchmarks, including one real-world environment |
| [Reinforcement Learning with Automated Auxiliary Loss Search](https://arxiv.org/pdf/2210.06041.pdf) | A2LS | NeurIPS22 | propose to automatically search top-performing auxiliary loss functions for learning better representations in RL; define a general auxiliary loss space of size 7.5 × 1020 based on the collected trajectory data and explore the space with an efficient evolutionary search strategy |
| [Mask-based Latent Reconstruction for Reinforcement Learning](https://arxiv.org/pdf/2201.12096.pdf) | MLR | NeurIPS22 | propose an effective self-supervised method to predict complete state representations in the latent space from the observations with spatially and temporally masked pixels |
| [Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training](https://arxiv.org/pdf/2210.00030.pdf) | VIP | ICLR23 Spotlight | cast representation learning from human videos as an offline goal-conditioned reinforcement learning problem; derive a self-supervised dual goal-conditioned value-function objective that does not depend on actions, enabling pre-training on unlabeled human videos |
| [Latent Variable Representation for Reinforcement Learning](https://arxiv.org/pdf/2212.08765.pdf) | ---- | ICLR23 | provide a representation view of the latent variable models for state-action value functions, which allows both tractable variational learning algorithm and effective implementation of the optimism/pessimism principle in the face of uncertainty for exploration |
| Spectral Decomposition Representation for Reinforcement Learning || ICLR23 ||
| [Become a Proficient Player with Limited Data through Watching Pure Videos](https://openreview.net/pdf?id=Sy-o2N0hF4f) | FICC | ICLR23 | consider the setting where the pre-training data are action-free videos; introduce a two-phase training pipeline; pre-training phase: implicitly extract the hidden action embedding from videos and pre-train the visual representation and the environment dynamics network based on vector quantization; down-stream tasks: finetune with small amount of task data based on the learned models |
| [Bootstrapped Representations in Reinforcement Learning](https://arxiv.org/pdf/2306.10171.pdf) | ---- | ICML23 | provide a theoretical characterization of the state representation learnt by temporal difference learning; find that this representation differs from the features learned by Monte Carlo and residual gradient algorithms for most transition structures of the environment in the policy evaluation setting |
| [Representation-Driven Reinforcement Learning](https://arxiv.org/pdf/2305.19922.pdf) | RepRL | ICML23 | reduce the policy search problem to a contextual bandit problem, using a mapping from policy space to a linear feature space |
| [Conditional Mutual Information for Disentangled Representations in Reinforcement Learning](https://arxiv.org/pdf/2305.14133.pdf) | CMID | NeurIPS23 spotlight | propose an auxiliary task for RL algorithms that learns a disentangled representation of high-dimensional observations with correlated features by minimising the conditional mutual information between features in the representation |

<a id='Unsupervised-RL'></a>
## Unsupervised Learning

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Variational Intrinsic Control](https://arxiv.org/pdf/1611.07507.pdf) | ---- | arXiv1611 | introduce a new unsupervised reinforcement learning method for discovering the set of intrinsic options available to an agent, which is learned by maximizing the number of different states an agent can reliably reach, as measured by the mutual information between the set of options and option termination states |
| [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/pdf/1802.06070.pdf) | DIAYN | ICLR19 | learn diverse skills in environments without any rewards by maximizing an information theoretic objective |
| Unsupervised Control Through Non-Parametric Discriminative Rewards || ICLR19 ||
| [Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/pdf/1907.01657.pdf) | DADS | ICLR20 | propose to learn low-level skills using model-free RL with the explicit aim of making model-based control easy |
| [Fast task inference with variational intrinsic successor features](https://arxiv.org/pdf/1906.05030.pdf) | VISR | ICLR20 ||
| [Decoupling representation learning from reinforcement learning](https://arxiv.org/pdf/2009.08319.pdf) | ATC | ICML21 | propose a new unsupervised task tailored to reinforcement learning named Augmented Temporal Contrast (ATC), which borrows ideas from Contrastive learning; benchmark several leading Unsupervised Learning algorithms by pre-training encoders on expert demonstrations and using them in RL agents|
| [Unsupervised Skill Discovery with Bottleneck Option Learning](https://arxiv.org/pdf/2106.14305.pdf) | IBOL | ICML21 | propose a novel skill discovery method with information bottleneck, which provides multiple benefits including learning skills in a more disentangled and interpretable way with respect to skill latents and being robust to nuisance information |
| [APS: Active Pretraining with Successor Features](https://arxiv.org/pdf/2108.13956.pdf) | APS | ICML21 | address the issues of APT and VISR by combining them together in a novel way |
| [Behavior From the Void: Unsupervised Active Pre-Training](https://arxiv.org/pdf/2103.04551.pdf) | APT | NeurIPS21 | propose a non-parametric entropy computed in an abstract representation space; compute the average of the Euclidean distance of each particle to its nearest neighbors for a set of samples |
| [Pretraining representations for data-efficient reinforcement learning](https://arxiv.org/pdf/2106.04799.pdf) | SGI | NeurIPS21 | consider to pretrian with unlabeled data and finetune on a small amount of task-specific data to improve the data efficiency of RL; employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL |
| [URLB: Unsupervised Reinforcement Learning Benchmark](https://arxiv.org/pdf/2110.15191.pdf) | URLB | NeurIPS21 | a benchmark for unsupervised reinforcement learning |
| [Discovering and Achieving Goals via World Models](https://arxiv.org/pdf/2110.09514.pdf) | LEXA | NeurIPS21 | unsupervised train both an explorer and an achiever policy via imagined rollouts in world models; after the unsupervised phase, solve tasks specified as goal images zero-shot without any additional learning |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | ---- | ICLR22 oral | show that unsupervised skill discovery algorithms based on mutual information maximization do not learn skills that are optimal for every possible reward function; provide a geometric perspective on some skill learning methods |
| [Lipschitz Constrained Unsupervised Skill Discovery](https://arxiv.org/pdf/2202.00914.pdf) | LSD | ICLR22 | argue that the MI-based skill discovery methods can easily maximize the MI objective with only slight differences in the state space; propose a novel objective based on a Lipschitz-constrained state representation function so that the objective maximization in the latent space always entails an increase in traveled distances (or variations) in the state space |
| [Learning more skills through optimistic exploration](https://arxiv.org/pdf/2107.14226.pdf) | DISDAIN | ICLR22 | derive an information gain auxiliary objective that involves training an ensemble of discriminators and rewarding the policy for their disagreement; the objective directly estimates the epistemic uncertainty that comes from the discriminator not having seen enough training examples|
| [A Mixture of Surprises for Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2210.06702.pdf) | MOSS | NeurIPS22 |  train one mixture component whose objective is to maximize the surprise and another whose objective is to minimize the surprise for handling the setting that the entropy of the environment’s dynamics may be unknown |
| [Unsupervised Reinforcement Learning with Contrastive Intrinsic Control](https://arxiv.org/pdf/2202.00161.pdf) | CIC | NeurIPS22 | propose to maximize the mutual information between statetransitions and latent skill vectors |
| [Unsupervised Skill Discovery via Recurrent Skill Training](https://openreview.net/pdf?id=sYDX_OxNNjh) | ReST | NeurIPS22 | encourage the latter trained skills to avoid entering the same states covered by the previous skills |
| [Choreographer: Learning and Adapting Skills in Imagination](https://arxiv.org/pdf/2211.13350.pdf) | Choreographer | ICLR23 Spotlight | decouples the exploration and skill learning processes; uses a meta-controller to evaluate and adapt the learned skills efficiently by deploying them in parallel in imagination |
| Provable Unsupervised Data Sharing for Offline Reinforcement Learning || ICLR23 ||
| Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality || ICLR23 ||
| [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels](https://arxiv.org/pdf/2209.12016.pdf) | Dyna-MPC | ICML23 oral | utilize unsupervised model-based RL for pre-training the agent; finetune downstream tasks via a task-aware finetuning strategy combined with a hybrid planner, Dyna-MPC |
| [On the Importance of Feature Decorrelation for Unsupervised Representation Learning in Reinforcement Learning](https://arxiv.org/pdf/2306.05637.pdf) | SimTPR | ICML23 | propose a novel URL framework that causally predicts future states while increasing the dimension of the latent manifold by decorrelating the features in the latent space |
| CLUTR: Curriculum Learning via Unsupervised Task Representation Learning || ICML23 ||
| [Controllability-Aware Unsupervised Skill Discovery](https://arxiv.org/pdf/2302.05103.pdf) | CSD | ICML23 | train a controllability-aware distance function based on the current skill repertoire and combine it with distance-maximizing skill discovery |
| [Behavior Contrastive Learning for Unsupervised Skill Discovery](https://arxiv.org/pdf/2305.04477.pdf) | BeCL | ICML23 | propose a novel unsupervised skill discovery method through contrastive learning among behaviors, which makes the agent produce similar behaviors for the same skill and diverse behaviors for different skills |
| Variational Curriculum Reinforcement Learning for Unsupervised Discovery of Skills || ICML23 ||
| [Learning to Discover Skills through Guidance](https://arxiv.org/pdf/2310.20178.pdf) | DISCO-DANCE | NeurIPS23 |  selects the guide skill that possesses the highest potential to reach unexplored states; guides other skills to follow guide skill; the guided skills are dispersed to maximize their discriminability in unexplored states |
| Creating Multi-Level Skill Hierarchies in Reinforcement Learning || NeurIPS23 ||
| Unsupervised Behavior Extraction via Random Intent Priors || NeurIPS23 ||
| [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://arxiv.org/pdf/2310.08887.pdf) | METRA | ICLR24 oral |  |
| Task Adaptation from Skills: Information Geometry, Disentanglement, and New Objectives for Unsupervised Reinforcement Learning | ---- |||


<!-- ### <span id='current'>Current methods</span> -->
<a id='current'></a>
### Current methods

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Weighted importance sampling for off-policy learning with linear function approximation](https://proceedings.neurips.cc/paper/2014/file/be53ee61104935234b174e62a07e53cf-Paper.pdf) | WIS-LSTD | NeurIPS14 |  |
| [Importance Sampling Policy Evaluation with an Estimated Behavior Policy](https://arxiv.org/pdf/1806.01347.pdf) | RIS | ICML19 |  |
| [Provably efficient RL with Rich Observations via Latent State Decoding](https://arxiv.org/pdf/1901.09018.pdf) | Block MDP | ICML19 ||
| [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/pdf/2005.12729.pdf) | ---- | ICLR20 | show that the improvement of performance is related to code-level optimizations |
| [Reinforcement Learning with Augmented Data](https://arxiv.org/pdf/2004.14990.pdf) | RAD | NeurIPS20 | propose first extensive study of general data augmentations for RL on both pixel-based and state-based inputs |
| [Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels](https://arxiv.org/pdf/2004.13649.pdf) | DrQ | ICLR21 Spotlight | propose to regularize the value function when applying data augmentation with model-free methods and reach state-of-the-art performance in image-pixels tasks |
| [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/pdf/2006.05990.pdf) | ---- | ICLR21 | do a large scale empirical study to evaluate different tricks for on-policy algorithms on MuJoCo |
| [Mirror Descent Policy Optimization](https://arxiv.org/pdf/2005.09814.pdf) | MDPO | ICLR21 |  |
| [Learning Invariant Representations for Reinforcement Learning without Reconstruction](https://arxiv.org/pdf/2006.10742.pdf) | DBC | ICLR21 ||
| [Randomized Ensemble Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/pdf/2101.05982.pdf) | REDQ | ICLR21 | consider three ingredients: (i) update q functions many times at every epoch; (ii) use an ensemble of Q functions; (iii) use the minimization across a random subset of Q functions from the ensemble for avoiding the overestimation; propose REDQ and achieve similar performance with model-based methods |
| [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/pdf/2108.13264.pdf) | ---- | NeurIPS21 oustandstanding paper | advocate for reporting interval estimates of aggregate performance and propose performance profiles to account for the variability in results, as well as present more robust and efficient aggregate metrics, such as interquartile mean scores, to achieve small uncertainty in results; [\[rliable\]](https://github.com/google-research/rliable/) |
| [Generalizable Episodic Memory for Deep Reinforcement Learning](https://arxiv.org/pdf/2103.06469.pdf) | GEM | ICML21 | propose to integrate the generalization ability of neural networks and the fast retrieval manner of episodic memory |
| [A Max-Min Entropy Framework for Reinforcement Learning](https://arxiv.org/pdf/2106.10517.pdf) | MME | NeurIPS21 | find that SAC may fail in explore states with low entropy (arrive states with high entropy and increase their entropies); propose a max-min entropy framework to address this issue |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that 
| [SO(2)-Equivariant Reinforcement Learning](https://arxiv.org/pdf/2203.04439.pdf) | Equi DQN, Equi SAC | ICLR22 Spotlight | consider to learn transformation-invariant policies and value functions; define and analyze group equivariant MDPs |
| [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431.pdf) | CoBERL | ICLR22 Spotlight | propose Contrastive BERT for RL (COBERL) that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency |
| [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/pdf?id=ZkC8wKoLbQ7) | InFeR | ICLR22 Spotlight | propose that deep RL agents lose some of their capacity to quickly fit new prediction tasks during training; propose InFeR to regularize a set of network outputs towards their initial values |
| [On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.01648.pdf) | ---- | ICLR22 Spotlight | consider lottery ticket hypothesis in deep reinforcement learning |
| [Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration](https://arxiv.org/pdf/2202.04628.pdf) | LOGO | ICLR22 Spotlight | consider the sparse reward challenges in RL; propose LOGO that exploits the offline demonstration data generated by a sub-optimal behavior policy; each step of LOGO contains a policy improvement step via TRPO and an additional policy guidance step by using the sub-optimal behavior policy |
| [Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation](https://arxiv.org/pdf/2201.01666.pdf) | IV-RL | ICLR22 Spotlight | analyze the sources of uncertainty in the supervision of modelfree DRL algorithms, and show that the variance of the supervision noise can be estimated with negative log-likelihood and variance ensembles |
| [Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning](https://arxiv.org/pdf/2201.09765.pdf) | GPM | ICLR22 Spotlight | focus on generating consistent actions for model-free RL, and borrow ideas from Model-based planning and action-repeat; use the policy to generate multi-step actions |
| [When should agents explore?](https://arxiv.org/pdf/2108.11811.pdf) | ---- | ICLR22 Spotlight | consider when to explore and propose to choose a heterogeneous mode-switching behavior policy |
| [Maximizing Ensemble Diversity in Deep Reinforcement Learning](https://openreview.net/pdf?id=hjd-kcpDpf2) | MED-RL | ICLR22 |  |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | ---- | ICLR22 | theoretically prove that standard maximum entropy RL is robust to some disturbances in the dynamics and the reward function |
| [Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities](https://openreview.net/pdf?id=zBOI9LFpESK) | AMBS | ICLR22 |  |
| [Large Batch Experience Replay](https://arxiv.org/pdf/2110.01528.pdf) | LaBER | ICML22 oral | cast the replay buffer sampling problem as an importance sampling one for estimating the gradient and derive the theoretically optimal sampling distribution |
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
| [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.07802.pdf) | primacy bias | ICML22 | find that deep RL agents incur a risk of overfitting to earlier experiences, which will negatively affect the rest of the learning process; propose a simple yet generally-applicable mechanism that tackles the primacy bias by periodically resetting a part of the agent |
| [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/pdf/2202.00821.pdf) |  | ICML22 | use DRL for solving the optimal design of sequential experiments |
| [The Geometry of Robust Value Functions](https://proceedings.mlr.press/v162/wang22k/wang22k.pdf) |  | ICML22 | study the geometry of the robust value space for the more general Robust MDPs |
| [Utility Theory for Markovian Sequential Decision Making](https://arxiv.org/pdf/2206.13637.pdf) | Affine-Reward MDPs | ICML22 | extend von Neumann-Morgenstern (VNM) utility theorem to decision making setting |
| [Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks](https://proceedings.mlr.press/v162/liang22c/liang22c.pdf) | MeanQ | ICML22 | consider variance reduction in Temporal-Difference Value Estimation; propose MeanQ to estimate target values by ensembling |
| Unifying Approximate Gradient Updates for Policy Optimization || ICML22 ||
| [Reinforcement Learning with Neural Radiance Fields](https://arxiv.org/pdf/2206.01634.pdf) | NeRF-RL | NeurIPS22 | propose to train an encoder that maps multiple image observations to a latent space describing the objects in the scene |
| [On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting](https://arxiv.org/pdf/2206.00761.pdf) | ---- | NeurIPS22 | explore the theoretical connections between Reward Maximization (RM) and Distribution Matching (DM) |
| [Faster Deep Reinforcement Learning with Slower Online Network](https://assets.amazon.science/31/ca/0c09418b4055a7536ced1b218d72/faster-deep-reinforcement-learning-with-slower-online-network.pdf) | DQN Pro, Rainbow Pro | NeurIPS22 | incentivize the online network to remain in the proximity of the target network |
| [Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress](https://arxiv.org/pdf/2206.01626.pdf) | PVRL | NeurIPS22 | focus on reincarnating RL from any agent to any other agent; present reincarnating RL as an alternative workflow or class of problem settings, where prior computational work (e.g., learned policies) is reused or transferred between design iterations of an RL agent, or from one RL agent to another |
| [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/pdf?id=OpC-9aBBVJe) | SR-SAC, SR-SPR | ICLR23 oral | show that fully or partially resetting the parameters of deep reinforcement learning agents causes better replay ratio scaling capabilities to emerge |
| [Guarded Policy Optimization with Imperfect Online Demonstrations](https://arxiv.org/pdf/2303.01728.pdf) | TS2C | ICLR23 Spotlight | h incorporate teacher intervention based on trajectory-based value estimation |
| [Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes](https://openreview.net/pdf?id=hWwY_Jq0xsN) | PW-Net | ICLR23 Spotlight | focus on making an “interpretable-by-design” deep reinforcement learning agent which is forced to use human-friendly prototypes in its decisions for making its reasoning process clear; train a “wrapper” model called PW-Net that can be added to any pre-trained agent, which allows them to be interpretable |
| [DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems](https://arxiv.org/pdf/2206.00484.pdf) | DEP-RL | ICLR23 Spotlight | identify the DEP controller, known from the field of self-organizing behavior, to generate more effective exploration than other commonly used noise processes; first control the 7 degrees of freedom (DoF) human arm model with RL on a muscle stimulation level |
| [Efficient Deep Reinforcement Learning Requires Regulating Statistical Overfitting](https://arxiv.org/pdf/2304.10466.pdf) | AVTD | ICLR23 | propose a simple active model selection method (AVTD) that attempts to automatically select regularization schemes by hill-climbing on validation TD error |
| [Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement](https://arxiv.org/pdf/1810.09103.pdf) | CCEM, GreedyAC | ICLR23 | propose to iteratively take the top percentile of actions, ranked according to the learned action-values; leverage theory for CEM to validate that CCEM concentrates on maximally valued actions across states over time |
| [Reward Design with Language Models](https://openreview.net/pdf?id=10uNUgI5Kl) | ---- | ICLR23 | explore how to simplify reward design by prompting a large language model (LLM) such as GPT-3 as a proxy reward function, where the user provides a textual prompt containing a few examples (few-shot) or a description (zero-shot) of the desired behavior |
| [Solving Continuous Control via Q-learning](https://arxiv.org/pdf/2210.12566.pdf) | DecQN | ICLR23 | combine value decomposition with bang-bang action space discretization to DQN to handle continuous control tasks; evaluate on DMControl, Meta World, and Isaac Gym |
| [Wasserstein Auto-encoded MDPs: Formal Verification of Efficiently Distilled RL Policies with Many-sided Guarantees](https://arxiv.org/pdf/2303.12558.pdf) | WAE-MDP | ICLR23 | minimize a penalized form of the optimal transport between the behaviors of the agent executing the original policy and the distilled policy |
| [Human-level Atari 200x faster](https://arxiv.org/pdf/2209.07550.pdf) | MEME | ICLR23 | outperform the human baseline across all 57 Atari games in 390M frames; four key components: (1) an approximate trust region method which enables stable bootstrapping from the online network, (2) a normalisation scheme for the loss and priorities which improves robustness when learning a set of value functions with a wide range of scales, (3) an improved architecture employing techniques from NFNets in order to leverage deeper networks without the need for normalization layers, and (4) a policy distillation method which serves to smooth out the instantaneous greedy policy over time. |
| [Improving Deep Policy Gradients with Value Function Search](https://arxiv.org/pdf/2302.10145.pdf) | VFS | ICLR23 | focus on improving value approximation and analyzing the effects on Deep PG primitives such as value prediction, variance reduction, and correlation of gradient estimates with the true gradient; show that value functions with better predictions improve Deep PG primitives, leading to better sample efficiency and policies with higher returns |
| [Memory Gym: Partially Observable Challenges to Memory-Based Agents](https://openreview.net/pdf?id=jHc8dCx6DDr) | Memory Gym | ICLR23 | a benchmark for challenging Deep Reinforcement Learning agents to memorize events across long sequences, be robust to noise, and generalize; consists of the partially observable 2D and discrete control environments Mortar Mayhem, Mystery Path, and Searing Spotlights; [\[code\]](https://github.com/MarcoMeter/drl-memory-gym/) |
| [Hybrid RL: Using both offline and online data can make RL efficient](https://arxiv.org/pdf/2210.06718.pdf) | Hy-Q | ICLR23 | focus on a hybrid setting named Hybrid RL, where the agent has both an offline dataset and the ability to interact with the environment; extend fitted Q-iteration algorithm |
| [POPGym: Benchmarking Partially Observable Reinforcement Learning](https://arxiv.org/pdf/2303.01859.pdf) | POPGym | ICLR23 | a two-part library containing (1) a diverse collection of 15 partially observable environments, each with multiple difficulties and (2) implementations of 13 memory model baselines; [\[code\]](https://github.com/proroklab/popgym) |
| [Critic Sequential Monte Carlo](https://arxiv.org/pdf/2205.15460.pdf) | CriticSMC | ICLR23 | combine sequential Monte Carlo with learned Soft-Q function heuristic factors |
| [Planning-oriented Autonomous Driving](https://arxiv.org/pdf/2212.10156.pdf) || CVPR23 best paper ||
| [On the Reuse Bias in Off-Policy Reinforcement Learning](https://arxiv.org/pdf/2209.07074.pdf) | BIRIS | IJCAI23 | discuss the bias of off-policy evaluation due to reusing the replay buffer; derive a high-probability bound of the Reuse Bias; introduce the concept of stability for off-policy algorithms and provide an upper bound for stable off-policy algorithms |
| [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/pdf/2302.12902.pdf) | ReDo | ICML23 oral | understand the underlying reasons behind the loss of expressivity during the training of RL agents; demonstrate the existence of the dormant neuron phenomenon in deep RL; propose Recycling Dormant neurons (ReDo) to reduce the number of dormant neurons and maintain network expressivity during training |
| [Efficient RL via Disentangled Environment and Agent Representations](https://openreview.net/pdf?id=kWS8mpioS9) | SEAR | ICML23 oral | consider to build a representation that can disentangle a robotic agent from its environment for improving the learning efficiency for RL; augment the RL loss with an agent-centric auxiliary loss |
| [On the Statistical Benefits of Temporal Difference Learning](https://arxiv.org/pdf/2301.13289.pdf) | ---- | ICML23 oral | provide crisp insight into the statistical benefits of TD |
| [Settling the Reward Hypothesis](https://arxiv.org/pdf/2212.10420.pdf) | ---- | ICML23 oral | provide a treatment of the reward hypothesis in both the setting that goals are the subjective desires of the agent and in the setting where goals are the objective desires of an agent designer |
| [Learning Belief Representations for Partially Observable Deep RL](https://openreview.net/pdf?id=4IzEmHLono) | Believer | ICML23 | decouple belief state modelling (via unsupervised learning) from policy optimization (via RL); propose a representation learning approach to capture a compact set of reward-relevant features of the state |
| [Internally Rewarded Reinforcement Learning](https://arxiv.org/pdf/2302.00270.pdf) | IRRL | ICML23 | study a class of reinforcement learning problems where the reward signals for policy learning are generated by an internal reward model that is dependent on and jointly optimized with the policy; theoretically derive and empirically analyze the effect of the reward function in IRRL and based on these analyses propose the clipped linear reward function |
| [Hyperparameters in Reinforcement Learning and How To Tune Them](https://arxiv.org/pdf/2306.01324.pdf) | ---- | ICML23 | Exploration of the hyperparameter landscape for commonly-used RL algorithms and environments; Comparison of different types of HPO methods on state-of-the-art RL algorithms and challenging RL environments |
| Langevin Thompson Sampling with Logarithmic Communication: Bandits and Reinforcement Learning || ICML23 ||
| [Correcting discount-factor mismatch in on-policy policy gradient methods](https://arxiv.org/pdf/2306.13284.pdf) | ---- | ICML23 | introduce a novel distribution correction to account for the discounted stationary distribution |
| [Reinforcement Learning Can Be More Efficient with Multiple Rewards](https://openreview.net/pdf?id=skDVsmXjPR) | ---- | ICML23 | theoretically analyze multi-reward extensions of action-elimination algorithms and prove more favorable instance-dependent regret bounds compared to their single-reward counterparts, both in multi-armed bandits and in tabular Markov decision processes |
| [Performative Reinforcement Learning](https://arxiv.org/pdf/2207.00046.pdf) | ---- | ICML23 | introduce the framework of performative reinforcement learning where the policy chosen by the learner affects the underlying reward and transition dynamics of the environment |
| [Reinforcement Learning with History Dependent Dynamic Contexts](https://arxiv.org/pdf/2302.02061.pdf) | DCMDPs | ICML23 | introduce DCMDPs, a novel reinforcement learning framework for history-dependent environments that handles non-Markov environments, where contexts change over time; derive an upper-confidence-bound style algorithm for logistic DCMDPs |
| [On Many-Actions Policy Gradient](https://arxiv.org/pdf/2210.13011.pdf) | MBMA | ICML23 | propose MBMA, an approach leveraging dynamics models for many-actions sampling in the context of stochastic policy gradient (SPG). which yields lower bias and comparable variance to SPG estimated from states in model-simulated rollouts |
| [Scaling Laws for Reward Model Overoptimization](https://openreview.net/attachment?id=bBLjms8nZE&name=pdf) | ---- | ICML23 | study overoptimization in the context of large language models fine-tuned as reward models trained to predict which of two options a human will prefer; study how the gold reward model score changes as we optimize against the proxy reward model using either reinforcement learning or best-of-n sampling |
| [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/pdf/2305.19452.pdf) | BBF | ICML23 | rely on scaling the neural networks used for value estimation and a number of other design choices like resetting |
| [Synthetic Experience Replay](https://arxiv.org/pdf/2303.06614.pdf) | SynthER | NeurIPS23 | utilize diffusion to augment data in the replay buffer; evaluate in both online RL and offline RL|

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
| [Reinforcement Learning with Non-Exponential Discounting](https://arxiv.org/pdf/2209.13413.pdf) | ---- | NeurIPS22 | propose a theory for continuous-time model-based reinforcement learning generalized to arbitrary discount functions; derive a Hamilton–Jacobi–Bellman (HJB) equation characterizing the optimal policy and describe how it can be solved using a collocation method |
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
| [Discovering and Achieving Goals via World Models](https://arxiv.org/pdf/2110.09514.pdf) | LEXA | NeurIPS21 | unsupervised train both an explorer and an achiever policy via imagined rollouts in world models; after the unsupervised phase, solve tasks specified as goal images zero-shot without any additional learning |
| [TransDreamer: Reinforcement Learning with Transformer World Models](https://arxiv.org/pdf/2202.09481.pdf) | TransDreamer | arxiv2202 | replace the RNN in RSSM by a transformer |
| [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf) | DreamerPro | ICML22 | consider reconstruction-free MBRL; propose to learn the prototypes from the recurrent states of the world model, thereby distilling temporal structures from past observations and actions into the prototypes. |
| [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://proceedings.mlr.press/v162/wan22d/wan22d.pdf) | ---- | ICML22 | introduce an improved version of the LoCA setup and use it to evaluate PlaNet and Dreamerv2 |
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://arxiv.org/pdf/2203.13880.pdf) | APV | ICML22 | pre-train an action-free latent video prediction model using videos from different domains, and then fine-tune the pre-trained model on target domains |
| [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf) | Denoised MDP | ICML22 | divide information into four categories: controllable/uncontrollable (whether infected by the action) and reward-relevant/irrelevant (whether affects the return); propose to only consider information which is controllable and reward-relevant |
| [DreamingV2: Reinforcement Learning with Discrete World Models without Reconstruction](https://arxiv.org/pdf/2203.00494.pdf) | Dreamingv2 | arxiv2203 | adopt both the discrete representation of DreamerV2 and the reconstruction-free objective of Dreaming |
| [Masked World Models for Visual Control](https://arxiv.org/pdf/2206.14244.pdf) | MWM | arxiv2206 | decouple visual representation learning and dynamics learning for visual model-based RL and use masked autoencoder to train visual representation |
| [DayDreamer: World Models for Physical Robot Learning](https://arxiv.org/pdf/2206.14176.pdf) | DayDreamer | arxiv2206 | apply Dreamer to 4 robots to learn online and directly in the real world, without any simulators |
| [Iso-Dream: Isolating Noncontrollable Visual Dynamics in World Models](https://arxiv.org/pdf/2205.13817.pdf) | Iso-Dream | NeurIPS22 | consider noncontrollable dynamics independent of the action signals; encourage the world model to learn controllable and noncontrollable sources of spatiotemporal changes on isolated state transition branches; optimize the behavior of the agent on the decoupled latent imaginations of the world model |
| [Learning General World Models in a Handful of Reward-Free Deployments](https://arxiv.org/pdf/2210.12719.pdf) | CASCADE | NeurIPS22 | introduce the reward-free deployment efficiency setting to facilitate generalization (exploration should be task agnostic) and scalability (exploration policies should collect large quantities of data without costly centralized retraining); propose an information theoretic objective inspired by Bayesian Active Learning by specifically maximizing the diversity of trajectories sampled by the population through a novel cascading objective |
| [Learning Robust Dynamics through Variational Sparse Gating](https://arxiv.org/pdf/2210.11698.pdf) | VSG, SVSG, BBS | NeurIPS22 | consider to sparsely update the latent states at each step; develope a new partially-observable and stochastic environment, called BringBackShapes (BBS) |
| [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf) | IRIS | ICLR23 oral | use a discrete autoencoder and an autoregressive Transformer to conduct World Models and significantly improve the data efficiency in Atari (2 hours of real-time experience); [\[code\]](https://github.com/eloialonso/iris) |
| [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf) | TWM | ICLR23 | present a new autoregressive world model based on the Transformer-XL; obtain excellent results on the Atari 100k benchmark; [\[code\]](https://github.com/jrobine/twm) |
| [Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting](https://arxiv.org/pdf/2303.10144.pdf) | DUTD | ICLR23 | propose a new general method that dynamically adjusts the update to data (UTD) ratio during training based on underand overfitting detection on a small subset of the continuously collected experience not used for training; apply this method in DreamerV2 |
| [Evaluating Long-Term Memory in 3D Mazes](https://arxiv.org/pdf/2210.13383.pdf) | Memory Maze | ICLR23 | introduce the Memory Maze, a 3D domain of randomized mazes specifically designed for evaluating long-term memory in agents, including an online reinforcement learning benchmark, a diverse offline dataset, and an offline probing evaluation; [\[code\]](https://github.com/jurgisp/memory-maze) |
| [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104.pdf) | DreamerV3 | arxiv2301 | propose DreamerV3 to handle a wide range of domains, including continuous and discrete actions, visual and low-dimensional inputs, 2D and 3D worlds, different data budgets, reward frequencies, and reward scales|
| [Task Aware Dreamer for Task Generalization in Reinforcement Learning](https://arxiv.org/pdf/2303.05092.pdf) | TAD | arXiv2303 | propose Task Distribution Relevance to capture the relevance of the task distribution quantitatively; propose TAD to use world models to improve task generalization via encoding reward signals into policies |
| [Reparameterized Policy Learning for Multimodal Trajectory Optimization](https://arxiv.org/pdf/2307.10710.pdf) | RPG | ICML23 oral | propose a principled framework that models the continuous RL policy as a generative model of optimal trajectories; present RPG to leverages the multimodal policy parameterization and learned world model to achieve strong exploration capabilities and high data efficiency |
| [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels](https://arxiv.org/pdf/2209.12016.pdf) | Dyna-MPC | ICML23 oral | utilize unsupervised model-based RL for pre-training the agent; finetune downstream tasks via a task-aware finetuning strategy combined with a hybrid planner, Dyna-MPC |
| [Posterior Sampling for Deep Reinforcement Learning](https://arxiv.org/pdf/2305.00477.pdf) | PSDRL | ICML23 | combine efficient uncertainty quantification over latent state space models with a specially tailored continual planning algorithm based on value-function approximation |
| [Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators](https://openreview.net/pdf?id=rDMAJECBM2) | TPX | ICML23 | propose Total Propagation X, the first composite gradient estimation algorithm using inverse variance weighting that is demonstrated to be applicable at scale; combine TPX with Dreamer |
| [Go Beyond Imagination: Maximizing Episodic Reachability with World Models](https://openreview.net/pdf?id=JsAMuzA9o2) | GoBI | ICML23 | combine the traditional lifelong novelty motivation with an episodic intrinsic reward that is designed to maximize the stepwise reachability expansion; apply learned world models to generate predicted future states with random actions |
| [Simplified Temporal Consistency Reinforcement Learning](https://arxiv.org/pdf/2306.09466.pdf) | TCRL | ICML23 | propose a simple representation learning approach relying only on a latent dynamics model trained by latent temporal consistency and it is sufficient for high-performance RL |
| [Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling](https://arxiv.org/pdf/2301.12050.pdf) | DECKARD | ICML23 | hypothesize an Abstract World Model (AWM) over subgoals by few-shot prompting an LLM |
| Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum || ICML23 ||
| [Curious Replay for Model-based Adaptation](https://arxiv.org/pdf/2306.15934.pdf) | CR | ICML23 | aid model-based RL agent adaptation by prioritizing replay of experiences the agent knows the least about |
| [Multi-View Masked World Models for Visual Robotic Manipulation](https://arxiv.org/pdf/2302.02408.pdf) | MV-MWM | ICML23 | train a multi-view masked autoencoder that reconstructs pixels of randomly masked viewpoints and then learn a world model operating on the representations from the autoencoder |
| [Facing off World Model Backbones: RNNs, Transformers, and S4](https://arxiv.org/pdf/2307.02064.pdf) | S4WM | NeurIPS23 | propose the first S4-based world model that can generate high-dimensional image sequences through latent imagination |



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
| [Offline RL Policies Should Be Trained to be Adaptive](https://arxiv.org/pdf/2207.02200.pdf) | APE-V | ICML22 oral | show that learning from an offline dataset does not fully specify the environment; formally demonstrate the necessity of adaptation in offline RL by using the Bayesian formalism and to provide a practical algorithm for learning optimally adaptive policies; propose an ensemble-based offline RL algorithm that imbues policies with the ability to adapt within an episode |
| [When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11027.pdf) | DOGE | ICLR23 | train a state-conditioned distance function that can be readily plugged into standard actor-critic methods as a policy constraint |
| [Jump-Start Reinforcement Learning](https://arxiv.org/pdf/2204.02372.pdf) | JSRL | ICML23 | consider the setting that employs two policies to solve tasks: a guide-policy, and an exploration-policy; bootstrap an RL algorithm by gradually “rolling in” with the guide-policy |


<a id='offline-diffusion'></a>
### Combined with Diffusion Models

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) | Diffuser | ICML22 oral | first propose a denoising diffusion model designed for trajectory data and an associated probabilistic framework for behavior synthesis; demonstrate that Diffuser has a number of useful properties and is particularly effective in offline control settings that require long-horizon reasoning and test-time flexibility |
| Is Conditional Generative Modeling all you need for Decision Making? || ICLR23 oral ||
| [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/pdf/2208.06193.pdf) | Diffusion-QL | ICLR23 | perform policy regularization using diffusion (or scorebased) models; utilize a conditional diffusion model to represent the policy |
| [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/pdf/2209.14548.pdf) | SfBC | ICLR23 | decouple the learned policy into two parts: an expressive generative behavior model and an action evaluation model |
| [AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/pdf/2302.01877.pdf) | AdaptDiffuser | ICML23 oral | propose AdaptDiffuser, an evolutionary planning method with diffusion that can self-evolve to improve the diffusion model hence a better planner, which can also adapt to unseen tasks |


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
| [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) | Diffuser | ICML22 oral | first design a denoising diffusion model for trajectory data and an associated probabilistic framework for behavior synthesis |
| [Learning Temporally Abstract World Models without Online Experimentation](https://openreview.net/pdf?id=YeTYJz7th5) | OPOSM | ICML23 | present an approach for simultaneously learning sets of skills and temporally abstract, skill-conditioned world models purely from offline data, enabling agents to perform zero-shot online planning of skill sequences for new tasks |


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
| [an adaptive deep rl method for non-stationary environments with piecewise stable context](https://arxiv.org/pdf/2212.12735.pdf) | SeCBAD | NeurIPS22 | introduce latent situational MDP with piecewise-stable context; jointly infer the belief distribution over latent context with the posterior over segment length and perform more accurate belief context inference with observed data within the current context segment |
| [Model-based Meta Reinforcement Learning using Graph Structured Surrogate Models and Amortized Policy Search](https://arxiv.org/pdf/2102.08291.pdf) | GSSM | ICML22 | consider model-based meta reinforcement learning, which consists of dynamics model learning and policy optimization; develop a graph structured dynamics model with superior generalization capability across tasks|
| [Meta-Learning Hypothesis Spaces for Sequential Decision-making](https://arxiv.org/pdf/2202.00602.pdf) | Meta-KeL | ICML22 | argue that two critical capabilities of transformers, reason over long-term dependencies and present context-dependent weights from self-attention, compose the central role of a Meta-Reinforcement Learner; propose Meta-LeL for meta-learning the hypothesis space of a sequential decision task |
| [Transformers are Meta-Reinforcement Learners](https://arxiv.org/pdf/2206.06614.pdf) | TrMRL | ICML22 | propose TrMRL, a memory-based meta-Reinforcement Learner which uses the transformer architecture to formulate the learning process; |
| [ContraBAR: Contrastive Bayes-Adaptive Deep RL](https://arxiv.org/pdf/2306.02418.pdf) | ContraBAR | ICML23 | investigate whether contrastive methods, like contrastive predictive coding, can be used for learning Bayes-optimal behavior |



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
| [Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning](https://arxiv.org/pdf/2106.15860.pdf) | ---- | SCIS 2023 | summarize current optimization-based adversarial attacks in RL; propose a two-stage methods: train a deceptive policy and mislead the victim to imitate the deceptive policy |
| [Consistent Attack: Universal Adversarial Perturbation on Embodied Vision Navigation](https://arxiv.org/pdf/2206.05751.pdf) | Reward UAP, Trajectory UAP | PRL 2023 | extend universal adversarial perturbations into sequential decision and propose Reward UAP as well as Trajectory UAP via utilizing the dynamic; experiment in Embodied Vision Navigation tasks |

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
| [Look where you look! Saliency-guided Q-networks for visual RL tasks](https://arxiv.org/pdf/2209.09203.pdf) | SGQN | NeurIPS22 | propose that a good visual policy should be able to identify which pixels are important for its decision; preserve this identification of important sources of information across images |
| [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/pdf/2301.07608.pdf) | AdA | arXiv 2301 | demonstrate that training an RL agent at scale leads to a general in-context learning algorithm that can adapt to open-ended novel embodied 3D problems as quickly as humans |
| [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/pdf/2210.14215.pdf) | AD | ICLR23 oral | propose Algorithm Distillation for distilling reinforcement learning (RL) algorithms into neural networks by modeling their training histories with a causal sequence model |
| [Performance Bounds for Model and Policy Transfer in Hidden-parameter MDPs](https://openreview.net/pdf?id=sSt9fROSZRO) || ICLR23 | show that, given a fixed amount of pretraining data, agents trained with more variations are able to generalize better; find that increasing the capacity of the value and policy network is critical to achieve good performance |
| [Investigating Multi-task Pretraining and Generalization in Reinforcement Learning](https://openreview.net/pdf?id=sSt9fROSZRO) | ---- | ICLR23 |  find that, given a fixed amount of pretraining data, agents trained with more variations are able to generalize better; this advantage can still be present after fine-tuning for 200M environment frames than when doing zero-shot transfer |
| [Cross-domain Random Pre-training with Prototypes for Reinforcement Learning](https://arxiv.org/pdf/2302.05614.pdf) | CRPTpro | arXiv2302 | use prototypical representation learning with a novel intrinsic loss to pre-train an effective and generic encoder across different domains |
| [Task Aware Dreamer for Task Generalization in Reinforcement Learning](https://arxiv.org/pdf/2303.05092.pdf) | TAD | arXiv2303 | propose Task Distribution Relevance to capture the relevance of the task distribution quantitatively; propose TAD to use world models to improve task generalization via encoding reward signals into policies |
| [The Benefits of Model-Based Generalization in Reinforcement Learning](https://openreview.net/pdf?id=Vue1ulwlPD) | ---- | ICML23 | provide theoretical and empirical insight into when, and how, we can expect data generated by a learned model to be useful |
| [Multi-Environment Pretraining Enables Transfer to Action Limited Datasets](https://arxiv.org/pdf/2211.13337.pdf) | ALPT | ICML23 | given n source environments with fully action labelled dataset, consider offline RL in the target environment with a small action labelled dataset and a large dataset without action labels; utilize inverse dynamics model to learn a representation that generalizes well to the limited action data from the target environment |


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


<a id='RL-LLM'></a>
## RL with LLM

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Can Wikipedia Help Offline Reinforcement Learning?](https://arxiv.org/pdf/2201.12122.pdf) || arXiv2201 ||
| [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning](https://arxiv.org/pdf/2302.02662.pdf) | GLAM | ICML23 | consider an agent using an LLM as a policy that is progressively updated as the agent interacts with the environment, leveraging online Reinforcement Learning to improve its performance to solve goals |


<a id='Tutorial-and-Lesson'></a>
## Tutorial and Lesson

| Tutorial and Lesson |
| ---- |
| [Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](https://d1wqtxts1xzle7.cloudfront.net/54674740/Reinforcement_Learning-with-cover-page-v2.pdf?Expires=1641130151&Signature=eYy7kmTVqTXFcANS-9GZJUyb86cDqKeh2QX8VvzjouEM-QSfuiCm1WHhP~bW5C57Mecj6en~YRoTvxekzU5lq~UaHSBoc-7xP8dXBp91shcwdfJ8M0LUkktpqcQjXQi7ZzhGn33qZeah0p8S06ARzjimF5coL5arvp9yANAsy4KigXSZwAZNXxksKwqUAult2QseLL~Bv1p2locjYahRzTuex3vMxdBLhT9HOGFF0qOdKYxsWiaITUKnVYl8AvePDHEEXgfmuqEfjqjF5p~FHOsYl3gEDZOvUp1eUzPg2~i0MQXY49nUpzsThL5~unTRIsYJiBghnkYl8py0r~UelQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| [Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) | 
| [Deep Reinforcement Learning, CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| [Deep Reinforcement Learning and Control, CMU 10703](https://katefvision.github.io/) |
| [RLChina](http://rlchina.org/topic/9) |

<a id='ICLR22'></a>
## ICLR22
| Paper | Type |
| ---- | ---- |
| [Bootstrapped Meta-Learning](https://arxiv.org/pdf/2109.04504.pdf) | oral |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | oral |
| [SO(2)-Equivariant Reinforcement Learning](https://arxiv.org/pdf/2203.04439.pdf) | spotlight |
| [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431.pdf) | spotlight |
| [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/pdf?id=ZkC8wKoLbQ7) | spotlight |
| [On Lottery Tickets and Minimal Task Representations in Deep Reinforcement Learning](https://arxiv.org/pdf/2105.01648.pdf) | spotlight |
| [Reinforcement Learning with Sparse Rewards using Guidance from Offline Demonstration](https://arxiv.org/pdf/2202.04628.pdf) | spotlight |
| [Sample Efficient Deep Reinforcement Learning via Uncertainty Estimation](https://arxiv.org/pdf/2201.01666.pdf) | spotlight |
| [Generative Planning for Temporally Coordinated Exploration in Reinforcement Learning](https://arxiv.org/pdf/2201.09765.pdf) | spotlight |
| [When should agents explore?](https://arxiv.org/pdf/2108.11811.pdf) | spotlight |
| [Revisiting Design Choices in Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2110.04135.pdf) | spotlight |
| [DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization](https://arxiv.org/pdf/2112.04716.pdf) | spotlight |
| [Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning ](https://arxiv.org/pdf/2202.11566.pdf) | spotlight |
| [COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation](https://openreview.net/pdf?id=FLA55mBee6Q) | spotlight |
| [Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/pdf/2204.01464.pdf) | spotlight |
| [Constrained Policy Optimization via Bayesian World Models](https://arxiv.org/pdf/2201.09802.pdf) | spotlight |
| [Cross-Trajectory Representation Learning for Zero-Shot Generalization in RL](https://arxiv.org/pdf/2106.02193.pdf) | poster |
| [The Role of Pretrained Representations for the OOD Generalization of RL Agents](https://arxiv.org/pdf/2107.05686.pdf) | poster |
| [Generalisation in Lifelong Reinforcement Learning through Logical Composition](https://openreview.net/pdf?id=ZOcX-eybqoL) | poster |
| [Local Feature Swapping for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2204.06355.pdf) | poster |
| [Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/pdf/2106.11420.pdf) | poster |
| [CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing](https://arxiv.org/pdf/2106.09292.pdf) | poster |
| [Model-Based Offline Meta-Reinforcement Learning with Regularization](https://arxiv.org/pdf/2202.02929.pdf) | poster |
| [Skill-based Meta-Reinforcement Learning](https://openreview.net/pdf?id=jeLW-Fh9bV) | poster |
| [Hindsight Foresight Relabeling for Meta-Reinforcement Learning](https://arxiv.org/pdf/2109.09031.pdf) | poster |
| [CoMPS: Continual Meta Policy Search](https://arxiv.org/pdf/2112.04467.pdf) | poster |
| [Learning a subspace of policies for online adaptation in Reinforcement Learning](https://arxiv.org/pdf/2110.05169.pdf) | poster |
| [Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage](https://arxiv.org/pdf/2107.06226.pdf) | poster |
| [Pareto Policy Pool for Model-based Offline Reinforcement Learning](https://openreview.net/pdf?id=OqcZu8JIIzS) | poster |
| [Offline Reinforcement Learning with Value-based Episodic Memory](https://openreview.net/pdf?id=RCZqv9NXlZ) | poster |
| [Offline reinforcement learning with implicit Q-learning](https://arxiv.org/pdf/2110.06169.pdf) | poster |
| [On-Policy Model Errors in Reinforcement Learning](https://arxiv.org/pdf/2110.07985.pdf) | poster |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | poster |
| [Maximizing Ensemble Diversity in Deep Reinforcement Learning](https://openreview.net/pdf?id=hjd-kcpDpf2) | poster |
| [Maximum Entropy RL (Provably) Solves Some Robust RL Problems ](https://arxiv.org/pdf/2103.06257.pdf) | poster |
| [Learning Generalizable Representations for Reinforcement Learning via Adaptive Meta-learner of Behavioral Similarities](https://openreview.net/pdf?id=zBOI9LFpESK) | poster |
| [Lipschitz Constrained Unsupervised Skill Discovery](https://arxiv.org/pdf/2202.00914.pdf) | poster |
| [Learning more skills through optimistic exploration](https://arxiv.org/pdf/2107.14226.pdf) | poster |

<a id='ICML22'></a>
## ICML22
| Paper | Type |
| ---- | ---- |
| [Online Decision Transformer](https://arxiv.org/pdf/2202.05607.pdf) | oral |
| The Unsurprising Effectiveness of Pre-Trained Vision Models for Control | oral |
| The Importance of Non-Markovianity in Maximum State Entropy Exploration | oral |
| [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/pdf/2205.09991.pdf) | oral |
| Adversarially Trained Actor Critic for Offline Reinforcement Learning | oral |
| Learning Bellman Complete Representations for Offline Policy Evaluation | oral |
| [Offline RL Policies Should Be Trained to be Adaptive](https://arxiv.org/pdf/2207.02200.pdf) | oral |
| [Large Batch Experience Replay](https://arxiv.org/pdf/2110.01528.pdf) | oral |
| [Do Differentiable Simulators Give Better Gradients for Policy Optimization?](https://arxiv.org/pdf/2202.00817.pdf) | oral |
| Federated Reinforcement Learning: Communication-Efficient Algorithms and Convergence Analysis | oral |
| [An Analytical Update Rule for General Policy Optimization](https://arxiv.org/pdf/2112.02045.pdf) | oral |
| [Generalised Policy Improvement with Geometric Policy Composition](https://arxiv.org/pdf/2206.08736.pdf) | oral |
| Prompting Decision Transformer for Few-shot Policy Generalization | poster |
| [CtrlFormer: Learning Transferable State Representation for Visual Control via Transformer](https://arxiv.org/pdf/2206.08883.pdf) | poster |
| [Learning Dynamics and Generalization in Reinforcement Learning](https://arxiv.org/pdf/2206.02126.pdf) | poster |
| [Improving Policy Optimization with Generalist-Specialist Learning](https://arxiv.org/pdf/2206.12984.pdf) | poster |
| [DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck](https://arxiv.org/pdf/2102.13268.pdf) | poster |
| [Policy Gradient Method For Robust Reinforcement Learning](https://arxiv.org/pdf/2205.07344.pdf) | poster |
| SAUTE RL: Toward Almost Surely Safe Reinforcement Learning Using State Augmentation | poster |
| Constrained Variational Policy Optimization for Safe Reinforcement Learning | poster |
| Robust Deep Reinforcement Learning through Bootstrapped Opportunistic Curriculum | poster |
| Distributionally Robust Q-Learning | poster |
| Robust Meta-learning with Sampling Noise and Label Noise via Eigen-Reptile | poster |
| DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck | poster |
| [Model-based Meta Reinforcement Learning using Graph Structured Surrogate Models and Amortized Policy Search](https://arxiv.org/pdf/2102.08291.pdf) | poster |
| [Meta-Learning Hypothesis Spaces for Sequential Decision-making](https://arxiv.org/pdf/2202.00602.pdf) | poster |
| Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning | poster |
| [Transformers are Meta-Reinforcement Learners](https://arxiv.org/pdf/2206.06614.pdf) | poster |
| Offline Meta-Reinforcement Learning with Online Self-Supervision | poster |
| Regularizing a Model-based Policy Stationary Distribution to Stabilize Offline Reinforcement Learning | poster |
| Pessimistic Q-Learning for Offline Reinforcement Learning: Towards Optimal Sample Complexity | poster |
| How to Leverage Unlabeled Data in Offline Reinforcement Learning? | poster |
| On the Role of Discount Factor in Offline Reinforcement Learning | poster |
| Model Selection in Batch Policy Optimization | poster |
| Koopman Q-learning: Offline Reinforcement Learning via Symmetries of Dynamics | poster |
| Robust Task Representations for Offline Meta-Reinforcement Learning via Contrastive Learning | poster |
| Pessimism meets VCG: Learning Dynamic Mechanism Design via Offline Reinforcement Learning | poster |
| Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters | poster |
| Constrained Offline Policy Optimization | poster |
| [DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](https://proceedings.mlr.press/v162/deng22a/deng22a.pdf) | poster |
| [Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods](https://proceedings.mlr.press/v162/wan22d/wan22d.pdf) | poster |
| [Reinforcement Learning with Action-Free Pre-Training from Videos](https://arxiv.org/pdf/2203.13880.pdf) | poster |
| [Denoised MDPs: Learning World Models Better Than the World Itself](https://arxiv.org/pdf/2206.15477.pdf) | poster |
| [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/pdf/2203.04955.pdf) | poster |
| [Causal Dynamics Learning for Task-Independent State Abstraction](https://arxiv.org/pdf/2206.13452.pdf) | poster |
| [Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/pdf/2201.12417.pdf) | poster |
| [Adaptive Model Design for Markov Decision Process](https://proceedings.mlr.press/v162/chen22ab/chen22ab.pdf) | poster |
| [Stabilizing Off-Policy Deep Reinforcement Learning from Pixels](https://proceedings.mlr.press/v162/cetin22a/cetin22a.pdf) | poster |
| [Understanding Policy Gradient Algorithms: A Sensitivity-Based Approach](https://proceedings.mlr.press/v162/wu22i/wu22i.pdf) | poster |
| [Mirror Learning: A Unifying Framework of Policy Optimisation](https://arxiv.org/pdf/2201.02373.pdf) | poster |
| [Continuous Control with Action Quantization from Demonstrations](https://proceedings.mlr.press/v162/dadashi22a/dadashi22a.pdf) | poster |
| [Off-Policy Fitted Q-Evaluation with Differentiable Function Approximators: Z-Estimation and Inference Theory](https://proceedings.mlr.press/v162/zhang22al/zhang22al.pdf) | poster | Evaluation (FQE) with general differentiable function approximators, including neural function approximations by using the Z-estimation theory |
| [A Temporal-Difference Approach to Policy Gradient Estimation](https://proceedings.mlr.press/v162/tosatto22a/tosatto22a.pdf) | poster |
| [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/pdf/2205.07802.pdf) | poster |
| [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/pdf/2202.00821.pdf) | poster |
| [The Geometry of Robust Value Functions](https://proceedings.mlr.press/v162/wang22k/wang22k.pdf) | poster |
| Direct Behavior Specification via Constrained Reinforcement Learning | poster |
| [Utility Theory for Markovian Sequential Decision Making](https://arxiv.org/pdf/2206.13637.pdf) | poster |
| [Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks](https://proceedings.mlr.press/v162/liang22c/liang22c.pdf) | poster |
| Unifying Approximate Gradient Updates for Policy Optimization | poster |
| [EqR: Equivariant Representations for Data-Efficient Reinforcement Learning](https://proceedings.mlr.press/v162/mondal22a/mondal22a.pdf) | poster |
| [Provable Reinforcement Learning with a Short-Term Memory](https://proceedings.mlr.press/v162/efroni22a/efroni22a.pdf) | poster |
| Optimal Estimation of Off-Policy Policy Gradient via Double Fitted Iteration | poster |
| Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments | poster |
| Lagrangian Method for Q-Function Learning (with Applications to Machine Translation) | poster |
| Learning to Assemble with Large-Scale Structured Reinforcement Learning | poster |
| Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning | poster |
| Off-Policy Reinforcement Learning with Delayed Rewards | poster |
| Reachability Constrained Reinforcement Learning | poster |
| [Flow-based Recurrent Belief State Learning for POMDPs](https://proceedings.mlr.press/v162/chen22q/chen22q.pdf) | poster |
| [Off-Policy Evaluation for Large Action Spaces via Embeddings](https://arxiv.org/pdf/2202.06317.pdf) | poster |
| [Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning](https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf) | poster |
| [On Well-posedness and Minimax Optimal Rates of Nonparametric Q-function Estimation in Off-policy Evaluation](https://proceedings.mlr.press/v162/chen22u/chen22u.pdf) | poster |
| Communicating via Maximum Entropy Reinforcement Learning | poster |

<a id='NeurIPS22'></a>
## NeurIPS22
| Paper | Type |
| ---- | ---- |
| [Multi-Game Decision Transformers](https://arxiv.org/pdf/2205.15241.pdf) | poster |
| Bootstrapped Transformer for Offline Reinforcement Learning | poster |
| [Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning](https://arxiv.org/pdf/2207.09081.pdf) | poster |
| [Rethinking Value Function Learning for Generalization in Reinforcement Learning](https://arxiv.org/pdf/2210.09960.pdf) | poster |
| [Masked Autoencoding for Scalable and Generalizable Decision Making](https://arxiv.org/pdf/2211.12740.pdf) | poster |
| [Pre-Trained Image Encoder for Generalizable Visual Reinforcement Learning](https://arxiv.org/pdf/2212.08860.pdf) | poster |
| [GALOIS: Boosting Deep Reinforcement Learning via Generalizable Logic Synthesis](https://arxiv.org/pdf/2205.13728.pdf) | poster |
| [Look where you look! Saliency-guided Q-networks for visual RL tasks](https://arxiv.org/pdf/2209.09203.pdf) | poster |
| [an adaptive deep rl method for non-stationary environments with piecewise stable context](https://arxiv.org/pdf/2212.12735.pdf) | poster |
| [Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief](https://arxiv.org/pdf/2210.06692.pdf) | poster |
| [A Unified Framework for Alternating Offline Model Training and Policy Learning](https://arxiv.org/pdf/2210.05922.pdf) | poster |
| [Bidirectional Learning for Offline Infinite-width Model-based Optimization](https://arxiv.org/pdf/2209.07507.pdf) | poster |
| DASCO: Dual-Generator Adversarial Support Constrained Offline Reinforcement Learning | poster |
| [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/pdf/2202.06239.pdf) | poster |
| [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/pdf/2205.13703.pdf) | poster |
| Oracle Inequalities for Model Selection in Offline Reinforcement Learning | poster |
| [Mildly Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.04745.pdf) | poster |
| [A Policy-Guided Imitation Approach for Offline Reinforcement Learning](https://arxiv.org/pdf/2210.08323.pdf) | poster |
| [Bootstrapped Transformer for Offline Reinforcement Learning](https://arxiv.org/pdf/2206.08569.pdf) | poster |
| [LobsDICE: Offline Learning from Observation via Stationary Distribution Correction Estimation](https://arxiv.org/pdf/2202.13536.pdf) | poster |
| [Latent-Variable Advantage-Weighted Policy Optimization for Offline RL](https://arxiv.org/pdf/2203.08949.pdf) | poster |
| [How Far I'll Go: Offline Goal-Conditioned Reinforcement Learning via f-Advantage Regression](https://arxiv.org/pdf/2206.03023.pdf) | poster |
| [NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning](https://arxiv.org/pdf/2102.00714.pdf) | poster |
| [When does return-conditioned supervised learning work for offline reinforcement learning?](https://arxiv.org/pdf/2206.01079.pdf) | poster |
| [Bellman Residual Orthogonalization for Offline Reinforcement Learning](https://arxiv.org/pdf/2203.12786.pdf) | poster |
| [Oracle Inequalities for Model Selection in Offline Reinforcement Learning](https://arxiv.org/pdf/2211.02016.pdf) | poster |
| [Mismatched no More: Joint Model-Policy Optimization for Model-Based RL](https://arxiv.org/pdf/2110.02758.pdf) | poster |
| When to Update Your Model: Constrained Model-based Reinforcement Learning | poster |
| Bayesian Optimistic Optimization: Optimistic Exploration for Model-Based Reinforcement Learning | poster |
| [Model-based Lifelong Reinforcement Learning with Bayesian Exploration](https://arxiv.org/pdf/2210.11579.pdf) | poster |
| Plan to Predict: Learning an Uncertainty-Foreseeing Model for Model-Based Reinforcement Learning | poster |
| data-driven model-based optimization via invariant representation learning | poster |
| [Reinforcement Learning with Non-Exponential Discounting](https://arxiv.org/pdf/2209.13413.pdf) | poster |
| [Reinforcement Learning with Neural Radiance Fields](https://arxiv.org/pdf/2206.01634.pdf) | poster |
| [Recursive Reinforcement Learning](https://arxiv.org/pdf/2206.11430.pdf) | poster |
| [Challenging Common Assumptions in Convex Reinforcement Learning](https://arxiv.org/pdf/2202.01511.pdf) | poster |
| Explicable Policy Search | poster |
| [On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting](https://arxiv.org/pdf/2206.00761.pdf)| poster |
| [When to Ask for Help: Proactive Interventions in Autonomous Reinforcement Learning](https://arxiv.org/pdf/2210.10765.pdf) | poster |
| Adaptive Bio-Inspired Fish Simulation with Deep Reinforcement Learning | poster |
| Reinforcement Learning in a Birth and Death Process: Breaking the Dependence on the State Space | poster |
| [Discovered Policy Optimisation](https://arxiv.org/pdf/2210.05639.pdf) | poster |
| Faster Deep Reinforcement Learning with Slower Online Network | poster |
| exploration-guided reward shaping for reinforcement learning under sparse rewards | poster |
| [Large-Scale Retrieval for Reinforcement Learning](https://arxiv.org/pdf/2206.05314.pdf) | poster |
| [Sustainable Online Reinforcement Learning for Auto-bidding](https://arxiv.org/pdf/2210.07006.pdf) | poster |
| [LECO: Learnable Episodic Count for Task-Specific Intrinsic Reward](https://arxiv.org/pdf/2210.05409.pdf) | poster |
| [DNA: Proximal Policy Optimization with a Dual Network Architecture](https://arxiv.org/pdf/2206.10027.pdf) | poster |
| [Faster Deep Reinforcement Learning with Slower Online Network](https://assets.amazon.science/31/ca/0c09418b4055a7536ced1b218d72/faster-deep-reinforcement-learning-with-slower-online-network.pdf) | poster |
| [Online Reinforcement Learning for Mixed Policy Scopes](https://causalai.net/r84.pdf) | poster |
| [ProtoX: Explaining a Reinforcement Learning Agent via Prototyping](https://arxiv.org/pdf/2211.03162.pdf) | poster |
| [Hardness in Markov Decision Processes: Theory and Practice](https://arxiv.org/pdf/2210.13075.pdf) | poster |
| [Robust Phi-Divergence MDPs](https://arxiv.org/pdf/2205.14202.pdf) | poster |
| [On the convergence of policy gradient methods to Nash equilibria in general stochastic games](https://arxiv.org/pdf/2210.08857.pdf) | poster |
| [A Unified Off-Policy Evaluation Approach for General Value Function](https://arxiv.org/pdf/2107.02711.pdf) | poster |
| [Robust On-Policy Sampling for Data-Efficient Policy Evaluation in Reinforcement Learning](https://arxiv.org/pdf/2111.14552.pdf) | poster |
| Continuous Deep Q-Learning in Optimal Control Problems: Normalized Advantage Functions Analysis | poster |
| [Parametrically Retargetable Decision-Makers Tend To Seek Power](https://arxiv.org/pdf/2206.13477.pdf) | poster |
| [Batch size-invariance for policy optimization](https://arxiv.org/pdf/2110.00641.pdf) | poster |
| [Trust Region Policy Optimization with Optimal Transport Discrepancies: Duality and Algorithm for Continuous Actions](https://arxiv.org/pdf/2210.11137.pdf) | poster |
| Adaptive Interest for Emphatic Reinforcement Learning | poster |
| [The Nature of Temporal Difference Errors in Multi-step Distributional Reinforcement Learning](https://arxiv.org/pdf/2207.07570.pdf) | poster |
| [Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress](https://arxiv.org/pdf/2206.01626.pdf) | poster |
| [Bayesian Risk Markov Decision Processes](https://arxiv.org/pdf/2106.02558.pdf) | poster |
| [Explainable Reinforcement Learning via Model Transforms](https://arxiv.org/pdf/2209.12006.pdf) | poster |
| PDSketch: Integrated Planning Domain Programming and Learning | poster |
| [Contrastive Learning as Goal-Conditioned Reinforcement Learning](https://arxiv.org/pdf/2206.07568.pdf) | poster |
| [Does Self-supervised Learning Really Improve Reinforcement Learning from Pixels?](https://arxiv.org/pdf/2206.05266.pdf) | poster |
| [Reinforcement Learning with Automated Auxiliary Loss Search](https://arxiv.org/pdf/2210.06041.pdf) | poster |
| [Mask-based Latent Reconstruction for Reinforcement Learning](https://arxiv.org/pdf/2201.12096.pdf) | poster |
| [Iso-Dream: Isolating Noncontrollable Visual Dynamics in World Models](https://arxiv.org/pdf/2205.13817.pdf) | poster |
| [Learning General World Models in a Handful of Reward-Free Deployments](https://arxiv.org/pdf/2210.12719.pdf) | poster |
| [Learning Robust Dynamics through Variational Sparse Gating](https://arxiv.org/pdf/2210.11698.pdf) | poster |
| [A Mixture of Surprises for Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2210.06702.pdf)  | poster |
| [Unsupervised Reinforcement Learning with Contrastive Intrinsic Control](https://arxiv.org/pdf/2202.00161.pdf)  | poster |
| [Unsupervised Skill Discovery via Recurrent Skill Training](https://openreview.net/pdf?id=sYDX_OxNNjh)  | poster |
| [A Unified Off-Policy Evaluation Approach for General Value Function](https://arxiv.org/pdf/2107.02711.pdf) | poster |
| The Pitfalls of Regularizations in Off-Policy TD Learning | poster |
| Off-Policy Evaluation for Action-Dependent Non-Stationary Environments | poster |
| [Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions](https://arxiv.org/pdf/2210.13373.pdf) | poster |
| [Off-Policy Evaluation with Policy-Dependent Optimization Response](https://arxiv.org/pdf/2202.12958.pdf) | poster |

<a id='ICLR23'></a>
## ICLR23
| Paper | Type |
| ---- | ---- |
| Dichotomy of Control: Separating What You Can Control from What You Cannot  | oral |
| [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/pdf/2210.14215.pdf) | oral |
| Is Conditional Generative Modeling all you need for Decision Making? | oral |
| Offline Q-learning on Diverse Multi-Task Data Both Scales And Generalizes | oral |
| Confidence-Conditioned Value Functions for Offline Reinforcement Learning | oral |
| Extreme Q-Learning: MaxEnt RL without Entropy | oral |
| Sparse Q-Learning: Offline Reinforcement Learning with Implicit Value Regularization | oral |
| [Transformers are Sample Efficient World Models](https://arxiv.org/pdf/2209.00588.pdf) | oral | 
| [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier](https://openreview.net/pdf?id=OpC-9aBBVJe) | oral |
| [Guarded Policy Optimization with Imperfect Online Demonstrations](https://arxiv.org/pdf/2303.01728.pdf) | spotlight |
| [Towards Interpretable Deep Reinforcement Learning with Human-Friendly Prototypes](https://openreview.net/pdf?id=hWwY_Jq0xsN) | spotlight | 
| Pink Noise Is All You Need: Colored Noise Exploration in Deep Reinforcement Learning | spotlight |
| [DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems](https://arxiv.org/pdf/2206.00484.pdf) | spotlight |
| The In-Sample Softmax for Offline Reinforcement Learning | spotlight |
| Benchmarking Offline Reinforcement Learning on Real-Robot Hardware | spotlight |
| [Choreographer: Learning and Adapting Skills in Imagination](https://arxiv.org/pdf/2211.13350.pdf) | spotlight | 
| [Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training](https://arxiv.org/pdf/2210.00030.pdf) | spotlight | 
| Decision Transformer under Random Frame Dropping | poster |
| Hyper-Decision Transformer for Efficient Online Policy Adaptation  | poster |
| Preference Transformer: Modeling Human Preferences using Transformers for RL | poster |
| On the Data-Efficiency with Contrastive Image Transformation in Reinforcement Learning  | poster |
| Can Agents Run Relay Race with Strangers? Generalization of RL to Out-of-Distribution Trajectories | poster |
| [Performance Bounds for Model and Policy Transfer in Hidden-parameter MDPs](https://openreview.net/pdf?id=sSt9fROSZRO) | poster |
| [Investigating Multi-task Pretraining and Generalization in Reinforcement Learning](https://openreview.net/pdf?id=sSt9fROSZRO) | poster |
| Priors, Hierarchy, and Information Asymmetry for Skill Transfer in Reinforcement Learning | poster |
| On the Robustness of Safe Reinforcement Learning under Observational Perturbations  | poster |
| Distributional Meta-Gradient Reinforcement Learning | poster |
| Conservative Bayesian Model-Based Value Expansion for Offline Policy Optimization | poster |
| Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning | poster |
| Efficient Offline Policy Optimization with a Learned Model  | poster |
| [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/pdf/2208.06193.pdf) | poster |
| [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/pdf/2209.14548.pdf) | poster |
| Decision S4: Efficient Sequence-Based RL via State Spaces Layers | poster |
| Behavior Proximal Policy Optimization | poster |
| Learning Achievement Structure for Structured Exploration in Domains with Sparse Reward | poster |
| Explaining RL Decisions with Trajectories | poster |
| User-Interactive Offline Reinforcement Learning | poster |
| Pareto-Efficient Decision Agents for Offline Multi-Objective Reinforcement Learning | poster |
| Offline RL for Natural Language Generation with Implicit Language Q Learning | poster |
| In-sample Actor Critic for Offline Reinforcement Learning | poster |
| Harnessing Mixed Offline Reinforcement Learning Datasets via Trajectory Weighting | poster |
| Mind the Gap: Offline Policy Optimizaiton for Imperfect Rewards | poster |
| [When Data Geometry Meets Deep Function: Generalizing Offline Reinforcement Learning](https://arxiv.org/pdf/2205.11027.pdf) | poster |
| MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations | poster |
| [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/pdf/2303.07109.pdf) | poster |
| [Dynamic Update-to-Data Ratio: Minimizing World Model Overfitting](https://arxiv.org/pdf/2303.10144.pdf) | poster |
| [Evaluating Long-Term Memory in 3D Mazes](https://arxiv.org/pdf/2210.13383.pdf) | poster |
| Making Better Decision by Directly Planning in Continuous Control | poster |
| HiT-MDP: Learning the SMDP option framework on MDPs with Hidden Temporal Embeddings | poster |
| Diminishing Return of Value Expansion Methods in Model-Based Reinforcement Learning | poster |
| [Simplifying Model-based RL: Learning Representations, Latent-space Models, and Policies with One Objective](https://arxiv.org/pdf/2209.08466.pdf) | poster |
| [SpeedyZero: Mastering Atari with Limited Data and Time](https://openreview.net/pdf?id=Mg5CLXZgvLJ) | poster |
| [Efficient Deep Reinforcement Learning Requires Regulating Statistical Overfitting](https://arxiv.org/pdf/2304.10466.pdf) | poster |
| Replay Memory as An Empirical MDP: Combining Conservative Estimation with Experience Replay | poster |
| [Greedy Actor-Critic: A New Conditional Cross-Entropy Method for Policy Improvement](https://arxiv.org/pdf/1810.09103.pdf) | poster |
| [Reward Design with Language Models](https://openreview.net/pdf?id=10uNUgI5Kl) | poster |
| [Solving Continuous Control via Q-learning](https://arxiv.org/pdf/2210.12566.pdf) | poster |
| [Wasserstein Auto-encoded MDPs: Formal Verification of Efficiently Distilled RL Policies with Many-sided Guarantees](https://arxiv.org/pdf/2303.12558.pdf) | poster |
| Quality-Similar Diversity via Population Based Reinforcement Learning | poster |
| [Human-level Atari 200x faster](https://arxiv.org/pdf/2209.07550.pdf) | poster |
| Policy Expansion for Bridging Offline-to-Online Reinforcement Learning  | poster |
| [Improving Deep Policy Gradients with Value Function Search](https://arxiv.org/pdf/2302.10145.pdf) | poster |
| [Memory Gym: Partially Observable Challenges to Memory-Based Agents](https://openreview.net/pdf?id=jHc8dCx6DDr) | poster |
| [Hybrid RL: Using both offline and online data can make RL efficient](https://arxiv.org/pdf/2210.06718.pdf) | poster |
| [POPGym: Benchmarking Partially Observable Reinforcement Learning](https://arxiv.org/pdf/2303.01859.pdf) | poster |
| [Critic Sequential Monte Carlo](https://arxiv.org/pdf/2205.15460.pdf) | poster |
| Revocable Deep Reinforcement Learning with Affinity Regularization for Outlier-Robust Graph Matching | poster |
| Provable Unsupervised Data Sharing for Offline Reinforcement Learning | poster |
| Discovering Policies with DOMiNO: Diversity Optimization Maintaining Near Optimality | poster |
| [Latent Variable Representation for Reinforcement Learning](https://arxiv.org/pdf/2212.08765.pdf) | poster |
| Spectral Decomposition Representation for Reinforcement Learning | poster |
| Behavior Prior Representation learning for Offline Reinforcement Learning | poster |
| [Become a Proficient Player with Limited Data through Watching Pure Videos](https://openreview.net/pdf?id=Sy-o2N0hF4f) | poster |
| Variational Latent Branching Model for Off-Policy Evaluation | poster |

<a id='ICML23'></a>
## ICML23
| Paper | Type |
| ---- | ---- |
| [On the Power of Pre-training for Generalization in RL: Provable Benefits and Hardness](https://arxiv.org/pdf/2210.10464.pdf) | oral |
| [AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners](https://arxiv.org/pdf/2302.01877.pdf) | oral |
| [Reparameterized Policy Learning for Multimodal Trajectory Optimization](https://arxiv.org/pdf/2307.10710.pdf) | oral |
| [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels](https://arxiv.org/pdf/2209.12016.pdf) | oral |
| [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://arxiv.org/pdf/2302.12902.pdf) | oral |
| [Efficient RL via Disentangled Environment and Agent Representations](https://openreview.net/pdf?id=kWS8mpioS9) | oral |
| [On the Statistical Benefits of Temporal Difference Learning](https://arxiv.org/pdf/2301.13289.pdf) | oral |
| Warm-Start Actor-Critic: From Approximation Error to Sub-optimality Gap | oral |
| Reinforcement Learning from Passive Data via Latent Intentions | oral |
| Subequivariant Graph Reinforcement Learning in 3D Environments | oral |
| Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL | oral |
| Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning | oral |
| [Settling the Reward Hypothesis](https://arxiv.org/pdf/2212.10420.pdf) | oral |
| Information-Theoretic State Space Model for Multi-View Reinforcement Learning | oral |
| [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels](https://arxiv.org/pdf/2209.12016.pdf) | oral |
| [Learning Belief Representations for Partially Observable Deep RL](https://openreview.net/pdf?id=4IzEmHLono) | poster |
| [Internally Rewarded Reinforcement Learning](https://arxiv.org/pdf/2302.00270.pdf) | poster |
| Active Policy Improvement from Multiple Black-box Oracles | poster |
| When is Realizability Sufficient for Off-Policy Reinforcement Learning? | poster |
| The Statistical Benefits of Quantile Temporal-Difference Learning for Value Estimation | poster |
| [Hyperparameters in Reinforcement Learning and How To Tune Them](https://arxiv.org/pdf/2306.01324.pdf) |  poster |
| Langevin Thompson Sampling with Logarithmic Communication: Bandits and Reinforcement Learning | poster |
| [Correcting discount-factor mismatch in on-policy policy gradient methods](https://arxiv.org/pdf/2306.13284.pdf) | poster |
| Masked Trajectory Models for Prediction, Representation, and Control | poster |
| Off-Policy Average Reward Actor-Critic with Deterministic Policy Search | poster |
| TGRL: An Algorithm for Teacher Guided Reinforcement Learning | poster |
| LIV: Language-Image Representations and Rewards for Robotic Control | poster |
| Stein Variational Goal Generation for adaptive Exploration in Multi-Goal Reinforcement Learning | poster |
| Emergence of Adaptive Circadian Rhythms in Deep Reinforcement Learning | poster |
| Explaining Reinforcement Learning with Shapley Values | poster |
| [Reinforcement Learning Can Be More Efficient with Multiple Rewards](https://openreview.net/pdf?id=skDVsmXjPR) | poster |
| [Performative Reinforcement Learning](https://arxiv.org/pdf/2207.00046.pdf) | poster |
| Truncating Trajectories in Monte Carlo Reinforcement Learning | poster |
| ReLOAD: Reinforcement Learning with Optimistic Ascent-Descent for Last-Iterate Convergence in Constrained MDPs | poster |
| Low-Switching Policy Gradient with Exploration via Online Sensitivity Sampling | poster |
| Hyperbolic Diffusion Embedding and Distance for Hierarchical Representation Learning | poster |
| Revisiting Domain Randomization via Relaxed State-Adversarial Policy Optimization | poster |
| Parallel $Q$-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation | poster |
| LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework | poster |
| Graph Reinforcement Learning for Network Control via Bi-Level Optimization | poster |
| Stochastic Policy Gradient Methods: Improved Sample Complexity for Fisher-non-degenerate Policies | poster |
| [Reinforcement Learning with History Dependent Dynamic Contexts](https://arxiv.org/pdf/2302.02061.pdf) | poster |
| Efficient Online Reinforcement Learning with Offline Data | poster |
| Variance Control for Distributional Reinforcement Learning | poster |
| Hindsight Learning for MDPs with Exogenous Inputs | poster |
| RLang: A Declarative Language for Describing Partial World Knowledge to Reinforcement Learning Agents | poster |
| Scalable Safe Policy Improvement via Monte Carlo Tree Search | poster |
| Bayesian Reparameterization of Reward-Conditioned Reinforcement Learning with Energy-based Models | poster |
| Understanding the Complexity Gains of Single-Task RL with a Curriculum | poster |
| PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient | poster |
| [On Many-Actions Policy Gradient](https://arxiv.org/pdf/2210.13011.pdf) | poster |
| Multi-task Hierarchical Adversarial Inverse Reinforcement Learning | poster |
| Cell-Free Latent Go-Explore | poster |
| Trustworthy Policy Learning under the Counterfactual No-Harm Criterion | poster |
| Reachability-Aware Laplacian Representation in Reinforcement Learning | poster |
| Interactive Object Placement with Reinforcement Learning | poster |
| Leveraging Offline Data in Online Reinforcement Learning | poster |
| Reinforcement Learning with General Utilities: Simpler Variance Reduction and Large State-Action Space | poster |
| DoMo-AC: Doubly Multi-step Off-policy Actor-Critic Algorithm | poster |
| [Scaling Laws for Reward Model Overoptimization](https://openreview.net/attachment?id=bBLjms8nZE&name=pdf) |  poster  |
| SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning | poster |
| Set-membership Belief State-based Reinforcement Learning for POMDPs | poster |
| Robust Satisficing MDPs | poster |
| Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling | poster |
| Quantum Policy Gradient Algorithm with Optimized Action Decoding | poster |
| For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal | poster |
| Model-Free Robust Average-Reward Reinforcement Learning | poster |
| Fair and Robust Estimation of Heterogeneous Treatment Effects for Policy Learning | poster |
| Trajectory-Aware Eligibility Traces for Off-Policy Reinforcement Learning | poster |
| Principled Reinforcement Learning with Human Feedback from Pairwise or K-wise Comparisons | poster |
| Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning | poster |
| [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/pdf/2305.19452.pdf) | poster |
| [Posterior Sampling for Deep Reinforcement Learning](https://arxiv.org/pdf/2305.00477.pdf) | poster |
| [Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators](https://openreview.net/pdf?id=rDMAJECBM2) | poster |
| [Go Beyond Imagination: Maximizing Episodic Reachability with World Models](https://openreview.net/pdf?id=JsAMuzA9o2) | poster |
| [Simplified Temporal Consistency Reinforcement Learning](https://arxiv.org/pdf/2306.09466.pdf) | poster |
| [Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling](https://arxiv.org/pdf/2301.12050.pdf) | poster |
| Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum | poster |
| [Curious Replay for Model-based Adaptation](https://arxiv.org/pdf/2306.15934.pdf) | poster |
| [Multi-View Masked World Models for Visual Robotic Manipulation](https://arxiv.org/pdf/2302.02408.pdf) | poster |
| [Automatic Intrinsic Reward Shaping for Exploration in Deep Reinforcement Learning](https://arxiv.org/pdf/2301.10886.pdf) | poster |
| [Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments](https://arxiv.org/pdf/2211.10515.pdf) | poster |
| Representations and Exploration for Deep Reinforcement Learning using Singular Value Decomposition | poster |
| [Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning](https://arxiv.org/pdf/2302.02662.pdf) | poster |
| Distilling Internet-Scale Vision-Language Models into Embodied Agents | poster |
| VIMA: Robot Manipulation with Multimodal Prompts | poster |
| Future-conditioned Unsupervised Pretraining for Decision Transformer | poster |
| Emergent Agentic Transformer from Chain of Hindsight Experience | poster |
| [The Benefits of Model-Based Generalization in Reinforcement Learning](https://openreview.net/pdf?id=Vue1ulwlPD) | poster |
| [Multi-Environment Pretraining Enables Transfer to Action Limited Datasets](https://arxiv.org/pdf/2211.13337.pdf) | poster |
| [On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline](https://arxiv.org/pdf/2212.05749.pdf) | poster |
| Unsupervised Skill Discovery for Learning Shared Structures across Changing Environments | poster |
| An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning | poster |
| Guiding Pretraining in Reinforcement Learning with Large Language Models | poster |
| What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL? | poster |
| Online Prototype Alignment for Few-shot Policy Transfer | poster |
| Detecting Adversarial Directions in Deep Reinforcement Learning to Make Robust Decisions | poster |
| Robust Situational Reinforcement Learning in Face of Context Disturbances | poster |
| Adversarial Learning of Distributional Reinforcement Learning | poster |
| Towards Robust and Safe Reinforcement Learning with Benign Off-policy Data | poster |
| Simple Embodied Language Learning as a Byproduct of Meta-Reinforcement Learning | poster |
| [ContraBAR: Contrastive Bayes-Adaptive Deep RL](https://arxiv.org/pdf/2306.02418.pdf) | poster |
| Model-based Offline Reinforcement Learning with Count-based Conservatism | poster |
| Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning | poster |
| [Learning Temporally Abstract World Models without Online Experimentation](https://openreview.net/pdf?id=YeTYJz7th5) | poster |
| Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning | poster |
| MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL | poster |
| Actor-Critic Alignment for Offline-to-Online Reinforcement Learning | poster |
| Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories | poster |
| Principled Offline RL in the Presence of Rich Exogenous Information | poster |
| Offline Meta Reinforcement Learning with In-Distribution Online Adaptation | poster |
| Policy Regularization with Dataset Constraint for Offline Reinforcement Learning | poster |
| Supported Trust Region Optimization for Offline Reinforcement Learning | poster |
| Constrained Decision Transformer for Offline Safe Reinforcement Learning | poster |
| PAC-Bayesian Offline Contextual Bandits With Guarantees | poster |
| Beyond Reward: Offline Preference-guided Policy Optimization | poster |
| Offline Reinforcement Learning with Closed-Form Policy Improvement Operators | poster |
| ChiPFormer: Transferable Chip Placement via Offline Decision Transformer | poster |
| Boosting Offline Reinforcement Learning with Action Preference Query | poster |
| [Jump-Start Reinforcement Learning](https://arxiv.org/pdf/2204.02372.pdf) | poster |
| Investigating the role of model-based learning in exploration and transfer | poster |
| [STEERING : Stein Information Directed Exploration for Model-Based Reinforcement Learning](https://arxiv.org/pdf/2301.12038.pdf) | poster |
| [Predictable MDP Abstraction for Unsupervised Model-Based RL](https://arxiv.org/pdf/2302.03921.pdf) | poster |
| The Virtues of Laziness in Model-based RL: A Unified Objective and Algorithms | poster |
| [On the Importance of Feature Decorrelation for Unsupervised Representation Learning in Reinforcement Learning](https://arxiv.org/pdf/2306.05637.pdf) | poster |
| CLUTR: Curriculum Learning via Unsupervised Task Representation Learning | poster |
| [Controllability-Aware Unsupervised Skill Discovery](https://arxiv.org/pdf/2302.05103.pdf) | poster |
| [Behavior Contrastive Learning for Unsupervised Skill Discovery](https://arxiv.org/pdf/2305.04477.pdf) | poster |
| Variational Curriculum Reinforcement Learning for Unsupervised Discovery of Skills | poster |
| [Bootstrapped Representations in Reinforcement Learning](https://arxiv.org/pdf/2306.10171.pdf) | poster |
| [Representation-Driven Reinforcement Learning](https://arxiv.org/pdf/2305.19922.pdf) | poster |
| Improved Policy Evaluation for Randomized Trials of Algorithmic Resource Allocation | poster |
| An Instrumental Variable Approach to Confounded Off-Policy Evaluation | poster |
| Semiparametrically Efficient Off-Policy Evaluation in Linear Markov Decision Processes | poster |
| [Automatic Intrinsic Reward Shaping for Exploration in Deep Reinforcement Learning](https://arxiv.org/pdf/2301.10886.pdf) | poster |
| [Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments](https://arxiv.org/pdf/2211.10515.pdf) | poster |

<a id='NeurIPS23'></a>
## NeurIPS23
| Paper | Type |
| ---- | ---- |
| Learning Generalizable Agents via Saliency-guided Features Decorrelation | oral |
| Understanding Expertise through Demonstrations: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning | oral |
| When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning | oral |
| DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models | oral |
| When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment | oral |
| Bridging RL Theory and Practice with the Effective Horizon | oral |
| SwiftSage: A Generative Agent with Fast and Slow Thinking for Complex Interactive Tasks | spotlight |
| RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability | spotlight |
| Maximize to Explore: One Objective Function Fusing Estimation, Planning, and Exploration | spotlight |
| [Conditional Mutual Information for Disentangled Representations in Reinforcement Learning](https://arxiv.org/pdf/2305.14133.pdf) | spotlight |
| Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL | spotlight |
| Double Gumbel Q-Learning | spotlight |
| Future-Dependent Value-Based Off-Policy Evaluation in POMDPs | spotlight |
| Supervised Pretraining Can Learn In-Context Reinforcement Learning | spotlight |
| Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning | spotlight |
| Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning | poster |
| Explore to Generalize in Zero-Shot RL | poster |
| Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies | poster |
| Reining Generalization in Offline Reinforcement Learning via Representation Distinction | poster |
| Contrastive Retrospection: honing in on critical steps for rapid learning and generalization in RL | poster |
| Doubly Robust Augmented Transfer for Meta-Reinforcement Learning | poster |
| Recurrent Hypernetworks are Surprisingly Strong in Meta-RL | poster |
| Parameterizing Non-Parametric Meta-Reinforcement Learning Tasks via Subtask Decomposition | poster |
| One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning | poster |
| Efficient Diffusion Policies For Offline Reinforcement Learning | poster |
| Learning to Influence Human Behavior with Offline Reinforcement Learning | poster |
| Design from Policies: Conservative Test-Time Adaptation for Offline Policy Optimization | poster |
| SafeDICE: Offline Safe Imitation Learning with Non-Preferred Demonstrations | poster |
| Constrained Policy Optimization with Explicit Behavior Density For Offline Reinforcement Learning | poster |
| Conservative State Value Estimation for Offline Reinforcement Learning | poster |
| Offline RL with Discrete Proxy Representations for Generalizability in POMDPs | poster |
| Context Shift Reduction for Offline Meta-Reinforcement Learning | poster |
| Mutual Information Regularized Offline Reinforcement Learning | poster |
| Recovering from Out-of-sample States via Inverse Dynamics in Offline Reinforcement Learning | poster |
| Percentile Criterion Optimization in Offline Reinforcement Learning | poster |
| Language Models Meet World Models: Embodied Experiences Enhance Language Models | poster |
| Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models | poster |
| [Facing off World Model Backbones: RNNs, Transformers, and S4](https://arxiv.org/pdf/2307.02064.pdf) | poster |
| Efficient Exploration in Continuous-time Model-based Reinforcement Learning | poster |
| Model-Based Reparameterization Policy Gradient Methods: Theory and Practical Algorithms | poster |
| [Learning to Discover Skills through Guidance](https://arxiv.org/pdf/2310.20178.pdf) | poster |
| Creating Multi-Level Skill Hierarchies in Reinforcement Learning | poster |
| Unsupervised Behavior Extraction via Random Intent Priors | poster |
| [MIMEx: Intrinsic Rewards from Masked Input Modeling](https://arxiv.org/pdf/2305.08932.pdf) | poster |
| [Synthetic Experience Replay](https://arxiv.org/pdf/2303.06614.pdf) | poster |
| f-Policy Gradients: A General Framework for Goal-Conditioned RL using f-Divergences | poster |
| Prediction and Control in Continual Reinforcement Learning | poster |
| Residual Q-Learning: Offline and Online Policy Customization without Value | poster |
| Small batch deep reinforcement learning | poster |
| Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs | poster |
| Is RLHF More Difficult than Standard RL? A Theoretical Perspective | poster |
| Reflexion: language agents with verbal reinforcement learning | poster |
| Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning | poster |
| Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning | poster |
| Direct Preference-based Policy Optimization without Reward Modeling | poster |
| Learning to Modulate pre-trained Models in RL | poster |
| Ignorance is Bliss: Robust Control via Information Gating | poster |
| Marginal Density Ratio for Off-Policy Evaluation in Contextual Bandits | poster |
| Model-Free Reinforcement Learning with the Decision-Estimation Coefficient | poster |
| Optimal and Fair Encouragement Policy Evaluation and Learning | poster |
| BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning | poster |
| Probabilistic Inference in Reinforcement Learning Done Right | poster |
| Reference-Based POMDPs | poster |
| Persuading Farsighted Receivers in MDPs: the Power of Honesty | poster |
| Distributional Policy Evaluation: a Maximum Entropy approach to Representation Learning | poster |
| Structured State Space Models for In-Context Reinforcement Learning | poster |
| An Alternative to Variance: Gini Deviation for Risk-averse Policy Gradient | poster |
| Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning | poster |
| PLASTIC: Improving Input and Label Plasticity for Sample Efficient Reinforcement Learning | poster |
| Hybrid Policy Optimization from Imperfect Demonstrations | poster |
| Policy Optimization in a Noisy Neighborhood: On Return Landscapes in Continuous Control | poster |
| Semantic HELM: A Human-Readable Memory for Reinforcement Learning | poster |
| A Definition of Continual Reinforcement Learning | poster |
| Fast Bellman Updates for Wasserstein Distributionally Robust MDPs | poster |
| Policy Gradient for Rectangular Robust Markov Decision Processes | poster |
| Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning | poster |
| Truncating Trajectories in Monte Carlo Policy Evaluation: an Adaptive Approach | poster |
| Model-Free Active Exploration in Reinforcement Learning | poster |
| Self-Supervised Reinforcement Learning that Transfers using Random Features | poster |
| FlowPG: Action-constrained Policy Gradient with Normalizing Flows | poster |
| Flexible Attention-Based Multi-Policy Fusion for Efficient Deep Reinforcement Learning | poster |
| ODE-based Recurrent Model-free Reinforcement Learning for POMDPs | poster |
| Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning | poster |
| SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning | poster |
| CaMP: Causal Multi-policy Planning for Interactive Navigation in Multi-room Scenes | poster |
| POMDP Planning for Object Search in Partially Unknown Environment | poster |
| Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective | poster |
| Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation | poster |
| A Long $N$-step Surrogate Stage Reward for Deep Reinforcement Learning | poster |
| State-Action Similarity-Based Representations for Off-Policy Evaluation | poster |
| Weakly Coupled Deep Q-Networks | poster |
| Large Language Models Are Semi-Parametric Reinforcement Learning Agents | poster |
| The Benefits of Being Distributional: Small-Loss Bounds for Reinforcement Learning | poster |
| Online Nonstochastic Model-Free Reinforcement Learning | poster |
| When is Agnostic Reinforcement Learning Statistically Tractable? | poster |
| Bayesian Risk-Averse Q-Learning with Streaming Observations | poster |
| Resetting the Optimizer in Deep RL: An Empirical Study | poster |
| Optimistic Exploration in Reinforcement Learning Using Symbolic Model Estimates | poster |
| Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms | poster |
| Regularity as Intrinsic Reward for Free Play | poster |
| TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning | poster |
| Policy Optimization for Continuous Reinforcement Learning | poster |
| Active Observing in Continuous-time Control | poster |
| Replicable Reinforcement Learning | poster |
| On the Importance of Exploration for Generalization in Reinforcement Learning | poster |
| Monte Carlo Tree Search with Boltzmann Exploration | poster |
| Iterative Reachability Estimation for Safe Reinforcement Learning | poster |
| Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design | poster |
| Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence? | poster |
| Inverse Dynamics Pretraining Learns Good Representations for Multitask Imitation | poster |
| Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach | poster |
| Contrastive Modules with Temporal Attention for Multi-Task Reinforcement Learning | poster |
| Sample-Efficient and Safe Deep Reinforcement Learning via Reset Deep Ensemble Agents | poster |
| Distributional Pareto-Optimal Multi-Objective Reinforcement Learning | poster |
| Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents | poster |
| Efficient Potential-based Exploration in Reinforcement Learning using Inverse Dynamic Bisimulation Metric | poster |
| Iteratively Learn Diverse Strategies with State Distance Information | poster |
| Accelerating Reinforcement Learning with Value-Conditional State Entropy Exploration | poster |
| Gradient Informed Proximal Policy Optimization | poster |
| The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model | poster |
| Optimal Treatment Allocation for Efficient Policy Evaluation in Sequential Decision Making | poster |

<a id='ICLR24'></a>
## ICLR24
| Paper | Type |
| ---- | ---- |
| [Predictive auxiliary objectives in deep RL mimic learning in the brain](https://openreview.net/pdf?id=agPpmEgf8C) | oral |
| [Pre-Training Goal-based Models for Sample-Efficient Reinforcement Learning](https://openreview.net/pdf?id=o2IEmeLL9r) | oral |
| [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://openreview.net/pdf?id=c5pwL0Soay) | oral |
| [ASID: Active Exploration for System Identification and Reconstruction in Robotic Manipulation](https://openreview.net/pdf?id=jNR6s6OSBT) | oral |
| [Mastering Memory Tasks with World Models](https://openreview.net/pdf?id=1vDArHJ68h) | oral |
| [Generalized Policy Iteration using Tensor Approximation for Hybrid Control](https://openreview.net/pdf?id=csukJcpYDe) | spotlight |
| [Selective Visual Representations Improve Convergence and Generalization for Embodied AI](https://openreview.net/pdf?id=kC5nZDU5zf) | spotlight |
| [AMAGO: Scalable In-Context Reinforcement Learning for Adaptive Agents](https://openreview.net/pdf?id=M6XWoEdmwf) | spotlight |
| [Confronting Reward Model Overoptimization with Constrained RLHF](https://openreview.net/pdf?id=gkfUvn0fLU) | spotlight |
| [Proximal Policy Gradient Arborescence for Quality Diversity Reinforcement Learning](https://openreview.net/pdf?id=TFKIfhvdmZ) | spotlight |
| [Improving Offline RL by Blending Heuristics](https://openreview.net/pdf?id=MCl0TLboP1) | spotlight |
| [Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making](https://openreview.net/pdf?id=af2c8EaKl8) | spotlight |
| [Tool-Augmented Reward Modeling](https://openreview.net/pdf?id=d94x0gWTUX) | spotlight |
| [Reward-Consistent Dynamics Models are Strongly Generalizable for Offline Reinforcement Learning](https://openreview.net/pdf?id=GSBHKiw19c) | spotlight |
| [Dual RL: Unification and New Methods for Reinforcement and Imitation Learning](https://openreview.net/pdf?id=xt9Bu66rqv) | spotlight |
| [Stabilizing Contrastive RL: Techniques for Robotic Goal Reaching from Offline Data](https://openreview.net/pdf?id=Xkf2EBj4w3) | spotlight |
| [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://openreview.net/pdf?id=TyFrPOKYXw) | spotlight |
| [Cross$Q$: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity](https://openreview.net/pdf?id=PczQtTsTIX) | spotlight |
| [Blending Imitation and Reinforcement Learning for Robust Policy Improvement](https://openreview.net/pdf?id=eJ0dzPJq1F) | spotlight |
| [Unlocking the Power of Representations in Long-term Novelty-based Exploration](https://openreview.net/pdf?id=OwtMhMSybu) | spotlight |
| [Spatially-Aware Transformers for Embodied Agents](https://openreview.net/pdf?id=Ts95eXsPBc) | spotlight |
| [Learning to Act without Actions](https://openreview.net/pdf?id=rvUq3cxpDF) | spotlight |
| [Towards Principled Representation Learning from Videos for Reinforcement Learning](https://openreview.net/pdf?id=3mnWvUZIXt) | spotlight |
| [TorchRL: A data-driven decision-making library for PyTorch](https://openreview.net/pdf?id=QxItoEAVMb) | spotlight |
| [Towards Robust Offline Reinforcement Learning under Diverse Data Corruption](https://openreview.net/pdf?id=5hAMmCU0bK) | spotlight |
| [Learning Hierarchical World Models with Adaptive Temporal Abstractions from Discrete Latent Dynamics](https://openreview.net/pdf?id=TjCDNssXKU) | spotlight |
| [Text2Reward: Reward Shaping with Language Models for Reinforcement Learning](https://openreview.net/pdf?id=tUM39YTRxH) | spotlight |
| [Robotic Task Generalization via Hindsight Trajectory Sketches](https://openreview.net/pdf?id=F1TKzG8LJO) | spotlight |
| [Submodular Reinforcement Learning](https://openreview.net/pdf?id=loYSzjSaAK) | spotlight |
| [Query-Policy Misalignment in Preference-Based Reinforcement Learning](https://openreview.net/pdf?id=UoBymIwPJR) | spotlight |
| [Kernel Metric Learning for In-Sample Off-Policy Evaluation of Deterministic RL Policies](https://openreview.net/pdf?id=plebgsdiiV) | spotlight |
| [GenSim: Generating Robotic Simulation Tasks via Large Language Models](https://openreview.net/pdf?id=OI3RoHoWAN) | spotlight |
| [Entity-Centric Reinforcement Learning for Object Manipulation from Pixels](https://openreview.net/pdf?id=uDxeSZ1wdI) | spotlight |
| [Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers](https://openreview.net/pdf?id=F5dhGCdyYh) | spotlight |
| [Addressing Signal Delay in Deep Reinforcement Learning](https://openreview.net/pdf?id=Z8UfDs4J46) | spotlight |
| [DrM: Mastering Visual Reinforcement Learning through Dormant Ratio Minimization](https://openreview.net/pdf?id=MSe8YFbhUE) | spotlight |
| [Task Adaptation from Skills: Information Geometry, Disentanglement, and New Objectives for Unsupervised Reinforcement Learning](https://openreview.net/pdf?id=zSxpnKh1yS) | spotlight |
| [$\mathcal{B}$-Coder: On Value-Based Deep Reinforcement Learning for Program Synthesis](https://openreview.net/pdf?id=fLf589bx1f) | spotlight |
| [Physics-Regulated Deep Reinforcement Learning: Invariant Embeddings](https://openreview.net/pdf?id=5Dwqu5urzs) | spotlight |
| [Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization](https://openreview.net/pdf?id=KOZu91CzbK) | spotlight |
| [Learning to Act from Actionless Videos through Dense Correspondences](https://openreview.net/pdf?id=Mhb5fpA1T0) | spotlight |
| [CivRealm: A Learning and Reasoning Odyssey in Civilization for Decision-Making Agents](https://openreview.net/pdf?id=UBVNwD3hPN) | spotlight |
| [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/pdf?id=Oxh5CstDJU) | spotlight |
| [Universal Humanoid Motion Representations for Physics-Based Control](https://openreview.net/pdf?id=OrOd8PxOO2) | spotlight |
| [Adaptive Rational Activations to Boost Deep Reinforcement Learning](https://openreview.net/pdf?id=g90ysX1sVs) | spotlight |
| [Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula](https://openreview.net/pdf?id=pFOoOdaiue) | spotlight |
| [Locality Sensitive Sparse Encoding for Learning World Models Online](https://openreview.net/pdf?id=i8PjQT3Uig) | poster |
| [On Representation Complexity of Model-based and Model-free Reinforcement Learning](https://openreview.net/pdf?id=3K3s9qxSn7) | poster |
| [Policy Rehearsing: Training Generalizable Policies for Reinforcement Learning](https://openreview.net/pdf?id=m3xVPaZp6Z) | poster |
| [What Matters to You? Towards Visual Representation Alignment for Robot Learning](https://openreview.net/pdf?id=CTlUHIKF71) | poster |
| [Improving Language Models with Advantage-based Offline Policy Gradients](https://openreview.net/pdf?id=ZDGKPbF0VQ) | poster |
| [Training Diffusion Models with Reinforcement Learning](https://openreview.net/pdf?id=YCWjhGrJFD) | poster |
| [The Trickle-down Impact of Reward Inconsistency on RLHF](https://openreview.net/pdf?id=MeHmwCDifc) | poster |
| [Maximum Entropy Model Correction in Reinforcement Learning](https://openreview.net/pdf?id=kNpSUN0uCc) | poster |
| [Tree Search-Based Policy Optimization under Stochastic Execution Delay](https://openreview.net/pdf?id=RaqZX9LSGA) | poster |
| [Offline RL with Observation Histories: Analyzing and Improving Sample Complexity](https://openreview.net/pdf?id=GnOLWS4Llt) | poster |
| [Understanding Hidden Context in Preference Learning: Consequences for RLHF](https://openreview.net/pdf?id=0tWTxYYPnW) | poster |
| [Eureka: Human-Level Reward Design via Coding Large Language Models](https://openreview.net/pdf?id=IEduRUO55F) | poster |
| [Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization](https://openreview.net/pdf?id=0t1O8ziRZp) | poster |
| [Score Models for Offline Goal-Conditioned Reinforcement Learning](https://openreview.net/pdf?id=oXjnwQLcTA) | poster |
| [Contrastive Difference Predictive Coding](https://openreview.net/pdf?id=0akLDTFR9x) | poster |
| [Hindsight PRIORs for Reward Learning from Human Preferences](https://openreview.net/pdf?id=NLevOah0CJ) | poster |
| [Reward Model Ensembles Help Mitigate Overoptimization](https://openreview.net/pdf?id=dcjtMYkpXx) | poster |
| [Safe Offline Reinforcement Learning with Feasibility-Guided Diffusion Model](https://openreview.net/pdf?id=j5JvZCaDM0) | poster |
| [Compositional Conservatism: A Transductive Approach in Offline Reinforcement Learning](https://openreview.net/pdf?id=HRkyLbBRHI) | poster |
| [Flow to Better: Offline Preference-based Reinforcement Learning via Preferred Trajectory Generation](https://openreview.net/pdf?id=EG68RSznLT) | poster |
| [PAE: Reinforcement Learning from External Knowledge for Efficient Exploration](https://openreview.net/pdf?id=R7rZUSGOPD) | poster |
| [Identifying Policy Gradient Subspaces](https://openreview.net/pdf?id=iPWxqnt2ke) | poster |
| [PARL: A Unified Framework for Policy Alignment in Reinforcement Learning](https://openreview.net/pdf?id=ByR3NdDSZB) | poster |
| [SafeDreamer: Safe Reinforcement Learning with World Models](https://openreview.net/pdf?id=tsE5HLYtYg) | poster |
| [Vanishing Gradients in Reinforcement Finetuning of Language Models](https://openreview.net/pdf?id=IcVNBR7qZi) | poster |
| [Goodhart's Law in Reinforcement Learning](https://openreview.net/pdf?id=5o9G4XF1LI) | poster |
| [Score Regularized Policy Optimization through Diffusion Behavior](https://openreview.net/pdf?id=xCRr9DrolJ) | poster |
| [Making RL with Preference-based Feedback Efficient via Randomization](https://openreview.net/pdf?id=Pe2lo3QOvo) | poster |
| [Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning](https://openreview.net/pdf?id=v8jdwkUNXb) | poster |
| [Contrastive Preference Learning: Learning from Human Feedback without Reinforcement Learning](https://openreview.net/pdf?id=iX1RjVQODj) | poster |
| [Privileged Sensing Scaffolds Reinforcement Learning](https://openreview.net/pdf?id=EpVe8jAjdx) | poster |
| [Learning Planning Abstractions from Language](https://openreview.net/pdf?id=3UWuFoksGb) | poster |
| [CrossLoco: Human Motion Driven Control of Legged Robots via Guided Unsupervised Reinforcement Learning](https://openreview.net/pdf?id=UCfz492fM8) | poster |
| [Efficient Dynamics Modeling in Interactive Environments with Koopman Theory](https://openreview.net/pdf?id=fkrYDQaHOJ) | poster |
| [Jumanji: a Diverse Suite of Scalable Reinforcement Learning Environments in JAX](https://openreview.net/pdf?id=C4CxQmp9wc) | poster |
| [Searching for High-Value Molecules Using Reinforcement Learning and Transformers](https://openreview.net/pdf?id=nqlymMx42E) | poster |
| [Privately Aligning Language Models with Reinforcement Learning](https://openreview.net/pdf?id=3d0OmYTNui) | poster |
| [The HIM Solution for Legged Locomotion: Minimal Sensors, Efficient Learning, and Substantial Agility](https://openreview.net/pdf?id=93LoCyww8o) | poster |
| [S$2$AC: Energy-Based Reinforcement Learning with Stein Soft Actor Critic](https://openreview.net/pdf?id=rAHcTCMaLc) | poster |
| [Replay across Experiments: A Natural Extension of Off-Policy RL](https://openreview.net/pdf?id=Nf4Lm6fXN8) | poster |
| [Piecewise Linear Parametrization of Policies: Towards Interpretable Deep Reinforcement Learning](https://openreview.net/pdf?id=hOMVq57Ce0) | poster |
| [Time-Efficient Reinforcement Learning with Stochastic Stateful Policies](https://openreview.net/pdf?id=5liV2xUdJL) | poster |
| [Open the Black Box: Step-based Policy Updates for Temporally-Correlated Episodic Reinforcement Learning](https://openreview.net/pdf?id=mnipav175N) | poster |
| [On Trajectory Augmentations for Off-Policy Evaluation](https://openreview.net/pdf?id=eMNN0wIyVw) | poster |
| [Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation](https://openreview.net/pdf?id=NxoFmGgWC9) | poster |
| [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://openreview.net/pdf?id=PXD3FAVHJT) | poster |
| [Delphic Offline Reinforcement Learning under Nonidentifiable Hidden Confounding](https://openreview.net/pdf?id=lUYY2qsRTI) | poster |
| [Prioritized Soft Q-Decomposition for Lexicographic Reinforcement Learning](https://openreview.net/pdf?id=c0MyyXyGfn) | poster |
| [The Curse of Diversity in Ensemble-Based Exploration](https://openreview.net/pdf?id=M3QXCOTTk4) | poster |
| [Off-Policy Primal-Dual Safe Reinforcement Learning](https://openreview.net/pdf?id=vy42bYs1Wo) | poster |
| [STARC: A General Framework For Quantifying Differences Between Reward Functions](https://openreview.net/pdf?id=wPhbtwlCDa) | poster |
| [Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning](https://openreview.net/pdf?id=AY6aM13gGF) | poster |
| [Discovering Temporally-Aware Reinforcement Learning Algorithms](https://openreview.net/pdf?id=MJJcs3zbmi) | poster |
| [Revisiting Data Augmentation in Deep Reinforcement Learning](https://openreview.net/pdf?id=EGQBpkIEuu) | poster |
| [Reward-Free Curricula for Training Robust World Models](https://openreview.net/pdf?id=eCGpNGDeNu) | poster |
| [CPPO: Continual Learning for Reinforcement Learning with Human Feedback](https://openreview.net/pdf?id=86zAUE80pP) | poster |
