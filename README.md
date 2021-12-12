# Reinforcement Learning Papers


Model Free (Online) RL
======
Based Methods
|  Title | Method | Conference | on/off policy | Action Space | Policy | Description |
| -----  | ----   | ----       |   ----  | ----  |  ---- |  ---- | 
| | DQN | | off | Discrete | | |
| | Dueling DQN | | off| Discrete | | |
| | Double DQN | | off | Discrete | | |
| | Priority Sampling | | off | Discrete | | |
| | Rainbow | | off | Discrete | | |
| [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | PG | NeurIPS99 | on/off |  | | propose Policy Gradient Theorem: how to calculate the gradient of the expected cumulative return to policy |
| [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) | TRPO | ICML15 | on | | | |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) | PPO | arxiv17 | on | | | |
| | A2C | | on/off ||  | |
| | A3C | | on/off ||  | |
| [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/pdf/1702.08165.pdf) | SQL | ICML17 | off | | parameterized neural network | consider max-entropy rl and propose soft q iteration as well as soft q learning |
| [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf), [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf), [appendix](http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b-supp.pdf) | SAC | ICML18 | off | | | |
|  | DPG | | off | | | |
|  | DDPG | | off | | | |
|  | TD3 | | off | | | |

current methods
|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| | REDQ | | |

Model Based (Online) RL
======
Model as Simulator

|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/pdf/1802.10592.pdf) | ME-TRPO | ICLR18 | |
| [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf) | MVE | ICML18 |  |
| [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/pdf/1807.01675.pdf) | STEVE | NeurIPS18 | |
| [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf) | PETS | NeurIPS18 | |
| [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/pdf/1807.03858.pdf)  | SLBO | ICLR19  | |
| [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/pdf/1906.08253.pdf) | MBPO | NeurIPS19  |  |
| [Bidirectional Model-based Policy Optimization](https://arxiv.org/pdf/2007.01995.pdf) | BMPO | ICML20  | extend MBPO for  |
| [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf)| M2AC | NeurIPS20 | extend MBPO for |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | extend MBPO for |

|  Title | Method | Conference |  Description |
| -----  | ----   | ----       |   ----  |
| [Model Based Reinforcement Learning for Atari](https://arxiv.org/pdf/1903.00374.pdf) | | ICLR20 | |


Model for Planning


Survey and Benchmark
|  Title |Conference | Description |
| -----  |----       |   ----  |
| [Survey of Model-Based Reinforcement Learning: Applications on Robotics](https://www.researchgate.net/profile/Athanasios-Polydoros/publication/312921419_Survey_of_Model-Based_Reinforcement_Learning_Applications_on_Robotics/links/59cec68baca2721f434effc6/Survey-of-Model-Based-Reinforcement-Learning-Applications-on-Robotics.pdf) | JIRS17 | |
| [Model-based Reinforcement Learning: A Survey](https://arxiv.org/pdf/2006.16712.pdf) | arxiv20 | |
| [Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/pdf/1907.02057.pdf) | arxiv19 | |


(Model Free) Offline RL
======

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) | BCQ | ICML19 | show that off-policy methods perform badly because of extrapolation error; propose batch-constrained reinforcement learning: maximizing the return as well as minimizing the mismatch between the state-action visitation of the policy and the state-action pairs contained in the batch |
| [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) | CQL | NeurIPS20 | propose CQL with conservative q function, which is a lower bound of its true value, since standard off-policy methods will overestimate the value function |



Model Based Offline RL
======

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization](https://arxiv.org/pdf/2006.03647.pdf) | BREMEN | ICLR20 | propose deployment efficiency, to count the number of changes in the data-collection policy during learning (offline: 1, online: no limit); propose BERMEN with an ensemble of dynamics models for off-policy and offline rl |
| [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf) | MOPO | NeurIPS20 | observe that existing model-based RL algorithms can improve the performance of offline RL compared with model free RL algorithms; design MOPO by extending MBPO on uncertainty-penalized MDPs (new_reward = reward - uncertainty) |
| [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2005.05951.pdf) | MOReL | NeurIPS20 | present MOReL for model-based offline RL, including two steps: (a) learning a pessimistic MDP, (b) learning a near-optimal policy in this P-MDP |
| [Model-Based Offline Planning](https://arxiv.org/pdf/2008.05556.pdf) | MOPO | ICLR21 | learn a model for planning |
| [Representation Balancing Offline Model-Based Reinforcement Learning](https://openreview.net/pdf?id=QpNz8r_Ri2Y) | RepB-SDE | ICLR21 | focus on learning the representation for a robust model of the environment under the distribution shift and extend RepBM to deal with the curse of horizon; propose RepB-SDE framework for off-policy evaluation and offline rl |
| [Conservative Objective Models for Effective Offline Model-Based Optimization](https://arxiv.org/pdf/2107.06882.pdf) | COMs | ICML21 | consider offline model-based optimization (MBO, optimize an unknown function only with some samples); add a regularizer (resemble adversarial training methods) to the objective forlearning conservative objective models |
| [COMBO: Conservative Offline Model-Based Policy Optimization](https://arxiv.org/pdf/2102.08363v1.pdf) | COMBO | NeurIPS21 | try to optimize a lower bound of performance without considering uncertainty quantification; extend CQL with model-based methods|


Sequence Generation
======
|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | | 



