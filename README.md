# Reinforcement Learning Papers


Model Free (Online) RL
======
Based Methods
|  Title | Method | Conference | on/off policy | Value Function | Policy | Description |
| -----  | ----   | ----       |   ----  |   ---- | ----  |   ---- | 
| | DQN | | off | | | |
| | Dueling DQN | | off | | | |
| | Double DQN | | off | | | |
| | Priority Sampling | | off | | | |
| | Rainbow | | off | | | |
| | PG | | on/off | | | |
| | TRPO | | on | | | |
| | PPO | | on | | | |
| | A2C | | on/off | | | |
| | A3C | | on/off | | | |
|  | SQL | | off | | | |
|  | SAC | | off | | | |
|  | DPG | | off | | | |
|  | DDPG | | off | | | |
|  | TD3 | | off | | | |

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
| | BCQ | ICML19 | |
| | CQL | NeurIPS20 | |



Model Based Offline RL
======

|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/pdf/2005.13239.pdf) | MOPO | NeurIPS20 | observe that existing model-based RL algorithms can improve the performance of offline RL compared with model free RL algorithms; design MOPO by extending MBPO on uncertainty-penalized MDPs (new_reward = reward - uncertainty) |
| [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2005.05951.pdf) | MOReL | NeurIPS20 | present MOReL for model-based offline RL, including two steps: (a) learning a pessimistic MDP, (b) learning a near-optimal policy in this P-MDP |
| [Model-Based Offline Planning](https://arxiv.org/pdf/2008.05556.pdf) | MOPO | ICLR21 | learn a model for planning |
| [Representation Balancing Offline Model-Based Reinforcement Learning](https://openreview.net/pdf?id=QpNz8r_Ri2Y) | RepB-SDE | ICLR21 | |
| [Conservative Objective Models for Effective Offline Model-Based Optimization](https://arxiv.org/pdf/2107.06882.pdf) | COMs | ICML21 | |
| [COMBO: Conservative Offline Model-Based Policy Optimization](https://arxiv.org/pdf/2102.08363v1.pdf) | COMBO | NeurIPS21 | try to optimize a lower bound of performance without considering uncertainty quantification; extend CQL with model-based methods|


Sequence Generation
======
|  Title | Method | Conference | Description |
| -----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | | 



