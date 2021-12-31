# Reinforcement Learning Papers
Related papers for Reinforcement Learning (we mainly focus on single-agent).

Since there are tens of thousands of new papers on reinforcement learning in each conference every year, we are only able to list those we read and consider as insightful.

## Contents 
- <a href="#Model Free Online">Model Free (Online) RL</a><br>
- <a href="#Model Based Online">Model Based (Online) RL</a><br>
- <a href="#Model Free Offline">(Model Free) Offline RL</a><br>
- <a href="#Model Based Offline">Model Based Offline RL</a><br>
- <a href="#Sequence Generation">Sequence Generation</a><br>
- <a href="#Meta Reinforcement Learning">Meta RL</a><br>
- <a href="#Adversarial Reinforcement Learning">Adversarial RL</a><br>

<a id='Model Free Online'></a>
## Model Free (Online) RL

### Classic Methods
|  Title | Method | Conference | on/off policy | Action Space | Policy | Description |
| -----  | ----   | ----       |   ----  | ----  |  ---- |  ---- | 
| [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236.pdf), [\[other link\]](http://www.kreimanlab.com/academia/classes/BAI/pdfs/MnihEtAlHassibis15NatureControlDeepRL.pdf) | DQN | Nature15 | off | Discrete | based on value function | use deep neural network to train q learning and reach the human level in the Atari games; mainly two trick: replay buffer for improving sample efficiency, decouple target network and behavior network |
| [Deep reinforcement learning with double q-learning](https://arxiv.org/pdf/1509.06461.pdf) | Double DQN | AAAI16 | off | Discrete | based on value function | find that the Q function in DQN may overestimate; decouple calculating q function and choosing action with two neural networks |
| [Dueling network architectures for deep reinforcement learning](https://arxiv.org/pdf/1511.06581.pdf) | Dueling DQN | ICML16 | off| Discrete | based on value function | use the same neural network to approximate q function and value function for calculating advantage function |
| [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) | Priority Sampling | ICLR16 | off | Discrete | based on value function | give different weights to the samples in the replay buffer (e.g. TD error) |
| [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf) | Rainbow | AAAI18 | off | Discrete | based on value function | combine different improvements to DQN: Double DQN, Dueling DQN, Priority Sampling, Multi-step learning, Distributional RL, Noisy Nets |
| [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | PG | NeurIPS99 | on/off | Continuous or Discrete | function approximation | propose Policy Gradient Theorem: how to calculate the gradient of the expected cumulative return to policy |
| ---- | AC/A2C | ---- | on/off | Continuous or Discrete | parameterized neural network | AC replace the return in PG with q function approximator to reduce variance; A2C replace the q function in AC with advantage function to reduce variance |
| [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) | A3C | ICML16 | on/off | Continuous or Discrete | parameterized neural network | propose three tricks to improve performance: (i) use different agents to interact with the environment; (ii) value function and policy share network parameters; (iii) modify the loss function (mse of value function + pg loss + policy entropy)|
| [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) | TRPO | ICML15 | on | Continuous or Discrete | parameterized neural network | introduce trust region to policy optimization for guaranteed monotonic improvement |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) | PPO | arxiv17 | on | Continuous or Discrete | parameterized neural network | replace the hard constraint of TRPO with a penalty by clipping the coefficient |
| [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) | DPG | ICML14 | off | Continuous | function approximation | consider deterministic policy for continuous action space and prove Deterministic Policy Gradient Theorem; use a stochastic behaviour policy for encouraging exploration |
| [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf) | DDPG | ICLR16 | off | Continuous | parameterized neural network | adapt the ideas of DQN to DPG: (i) deep neural network function approximators, (ii) replay buffer, (iii) fix the target q function at each epoch |
| [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/pdf/1802.09477.pdf) | TD3 | ICML18 | off | Continuous | parameterized neural network | adapt the ideas of Double DQN to DDPG: taking the minimum value between a pair of critics to limit overestimation |
| [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf) | SQL | ICML17 | off | main for Continuous | parameterized neural network | consider max-entropy rl and propose soft q iteration as well as soft q learning |
| [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf), [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf), [\[appendix\]](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf) | SAC | ICML18 | off | main for Continuous | parameterized neural network | base the theoretical analysis of SQL and extend soft q iteration (soft q evaluation + soft q improvement); reparameterize the policy and use two parameterized value functions; propose SAC |

### current methods
|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| [Randomized Ensemble Double Q-Learning: Learning Fast Without a Model](https://arxiv.org/pdf/2101.05982.pdf) | REDQ | ICLR21 | consider three ingredients: (i) update q functions many times at every epoch; (ii) use an ensemble of Q functions; (iii) use the minimization across a random subset of Q functions from the ensemble for avoiding the overestimation; propose REDQ and achieve similar performance with model-based methods |

<a id='Model Based Online'></a>
## Model Based (Online) RL
### Model as Simulator

|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/pdf/1802.10592.pdf) | ME-TRPO | ICLR18 | |
| [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf) | MVE | ICML18 |  |
| [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/pdf/1807.01675.pdf) | STEVE | NeurIPS18 | |
| [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf) | PETS | NeurIPS18 | |
| [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/pdf/1807.03858.pdf)  | SLBO | ICLR19  | |
| [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/pdf/1906.08253.pdf) | MBPO | NeurIPS19  |  |
| [Bidirectional Model-based Policy Optimization](https://arxiv.org/pdf/2007.01995.pdf) | BMPO | ICML20  | an extension of MBPO; consider both forward dynamics model and backward dynamics model |
| [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf)| M2AC | NeurIPS20 | an extension of MBPO; use model rollouts only when the model is confident |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | an extension of MBPO; automatically schedule the real data ratio as well as other hyperparameters for MBPO |

|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| [Model Based Reinforcement Learning for Atari](https://arxiv.org/pdf/1903.00374.pdf) | | ICLR20 | |


<!-- ### Model for Planning


### Survey and Benchmark
|  Title |Conference | Description |
| -----  |----       |   ----  |
| [Survey of Model-Based Reinforcement Learning: Applications on Robotics](https://www.researchgate.net/profile/Athanasios-Polydoros/publication/312921419_Survey_of_Model-Based_Reinforcement_Learning_Applications_on_Robotics/links/59cec68baca2721f434effc6/Survey-of-Model-Based-Reinforcement-Learning-Applications-on-Robotics.pdf) | JIRS17 | |
| [Model-based Reinforcement Learning: A Survey](https://arxiv.org/pdf/2006.16712.pdf) | arxiv20 | |
| [Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/pdf/1907.02057.pdf) | arxiv19 | | -->

<a id='Model Free Offline'></a>
## (Model Free) Offline RL


|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) | BCQ | ICML19 | show that off-policy methods perform badly because of extrapolation error; propose batch-constrained reinforcement learning: maximizing the return as well as minimizing the mismatch between the state-action visitation of the policy and the state-action pairs contained in the batch |
| [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) | CQL | NeurIPS20 | propose CQL with conservative q function, which is a lower bound of its true value, since standard off-policy methods will overestimate the value function |
| [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf) | ---- | arxiv20 | tutorial about methods, applications and open problems of offline rl |


<a id='Model Based Offline'></a>
## Model Based Offline RL

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/pdf/2006.03647.pdf) | BREMEN | ICLR20 | propose deployment efficiency, to count the number of changes in the data-collection policy during learning (offline: 1, online: no limit); propose BERMEN with an ensemble of dynamics models for off-policy and offline rl |
| [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf) | MOPO | NeurIPS20 | observe that existing model-based RL algorithms can improve the performance of offline RL compared with model free RL algorithms; design MOPO by extending MBPO on uncertainty-penalized MDPs (new_reward = reward - uncertainty) |
| [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2005.05951.pdf) | MOReL | NeurIPS20 | present MOReL for model-based offline RL, including two steps: (a) learning a pessimistic MDP, (b) learning a near-optimal policy in this P-MDP |
| [Model-Based Offline Planning](https://arxiv.org/pdf/2008.05556.pdf) | MBOP | ICLR21 | learn a model for planning |
| [Representation Balancing Offline Model-Based Reinforcement Learning](https://openreview.net/pdf?id=QpNz8r_Ri2Y) | RepB-SDE | ICLR21 | focus on learning the representation for a robust model of the environment under the distribution shift and extend RepBM to deal with the curse of horizon; propose RepB-SDE framework for off-policy evaluation and offline rl |
| [Conservative Objective Models for Effective Offline Model-Based Optimization](https://arxiv.org/pdf/2107.06882.pdf) | COMs | ICML21 | consider offline model-based optimization (MBO, optimize an unknown function only with some samples); add a regularizer (resemble adversarial training methods) to the objective forlearning conservative objective models |
| [COMBO: Conservative Offline Model-Based Policy Optimization](https://arxiv.org/pdf/2102.08363v1.pdf) | COMBO | NeurIPS21 | try to optimize a lower bound of performance without considering uncertainty quantification; extend CQL with model-based methods|


<a id='Sequence Generation'></a>
## Sequence Generation

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (return-to-go, state, action, return-to-go, ...); there is not explicit optimization process; evaluate on Offline RL |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (s_0^0, ..., s_0^N, a_0^0, ..., a_0^M, r_0, ...); use beam search to inference; evaluate on imitation learning，goal-conditioned RL and Offline RL,  | 


<a id='Meta Reinforcement Learning'></a>
## Meta RL
|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://www.cs.utexas.edu/users/sniekum/classes/RL-F17/papers/Meta.pdf) | MAML | ICML17 | propose a general framework for different learningproblems, including classification, regression andreinforcement learning; the main idea is to optimize the parameters to quickly adapt to new tasks (with a few steps of gradient descent) |
| [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/pdf/1903.08254.pdf) | PEARL | ICML19 | encode past tasks’ experience with probabilistic latent context and use inference network to estimate the posterior|
| [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](http://proceedings.mlr.press/v100/yu20a/yu20a.pdf) | Meta World | CoRL19 | an envoriment for meta RL as well as multi-task RL |
| [Meta-Q-Learning](https://arxiv.org/pdf/1910.00125.pdf) | MQL | ICLR20 | an off-policy algorithm for meta RL andbuilds upon three simple ideas: (i) Q Learning with context variable represented by pasttrajectories is competitive with SOTA; (ii) Multi-task objective is useful for meta RL; (iii) Past data from the meta-training replay buffer can be recycled |


<a id='Adversarial Reinforcement Learning'></a>
## Adversarial RL

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Adversarial Attacks on Neural Network Policies](https://arxiv.org/pdf/1702.02284.pdf) | ---- | ICLR 2017 workshop | first show that existing rl policies coupled with deep neural networks are vulnerable to adversarial noises in white-box and black-box settings | 
| [Delving into Adversarial Attacks on Deep Policies](https://arxiv.org/pdf/1705.06452.pdf) | ---- | ICLR 2017 workshop | show rl algorithms are vulnerable to adversarial noises; show adversarial training can improve robustness |
| [Robust Adversarial Reinforcement Learning](https://arxiv.org/pdf/1703.02702.pdf) | RARL | ICML17 | formulate the robust policy learning as a zero-sum, minimax objective function |
| [Robust Deep Reinforcement Learning Against Adversarial Perturbations on State Observations](https://arxiv.org/pdf/2003.08938.pdf) | SA-MDP | NeurIPS20 | formalize adversarial attack on state observation as SA-MDP; propose some novel attack methods: Robust SARSA and Maximal Action Difference; propose a defence framework and some practical methods: SA-DQN, SA-PPO and SA-DDPG |
| [Robust Reinforcement Learning on State Observations with Learned Optimal Adversary](https://arxiv.org/pdf/2101.08452.pdf) | ATLA | ICLR21 | use rl algorithms to train an "optimal" adversary; alternatively train "optimal" adversary and robust agent |
| [Robust Deep Reinforcement Learning through Adversarial Loss](https://arxiv.org/pdf/2008.01976.pdf) | RADIAL-RL | NeurIPS21 | propose a robust rl framework, which penalizes the overlap between output bounds of actions; propose a more efficient evaluation method (GWC) to measure attack agnostic robustness | 
| [Policy Smoothing for Provably Robust Reinforcement Learning](https://arxiv.org/pdf/2106.11420.pdf) | Policy Smoothing | arxiv21 | introduce randomized smoothing into RL; propose adaptive Neyman-Person Lemma |
