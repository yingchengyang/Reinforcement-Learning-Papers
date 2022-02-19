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
| [CoBERL: Contrastive BERT for Reinforcement Learning](https://arxiv.org/pdf/2107.05431.pdf) | CoBERL | ICLR22 | propose Contrastive BERT for RL (COBERL) that combines a new contrastive loss and a hybrid LSTM-transformer architecture to tackle the challenge of improving data efficiency |
| [SO(2)-Equivariant Reinforcement Learning](https://openreview.net/pdf?id=7F9cOhdvfk_) | Equi DQN, Equi SAC | ICLR22 |  |
| [Understanding and Preventing Capacity Loss in Reinforcement Learning](https://openreview.net/pdf?id=ZkC8wKoLbQ7) | InFeR | ICLR22 | propose that deep RL agents lose some of their capacity to quickly fit new prediction tasks during training; propose InFeR to regularize a set of network outputs towards their initial values |

<a id='Model Based Online'></a>
## Model Based (Online) RL

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Model-Ensemble Trust-Region Policy Optimization](https://arxiv.org/pdf/1802.10592.pdf) | ME-TRPO | ICLR18 | analyze the behavior of vanilla MBRL methods with DNN; propose ME-TRPO with two ideas: (i) use an ensemble of models, (ii)  use likelihood ratio derivatives; significantly reduce the sample complexity compared to model-free methods |
| [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf) | MVE | ICML18 | use a dynamics model to simulate the short-term horizon and Q-learning to estimate the long-term value beyond the simulation horizon; use the trained model and the policy to estimate k-step value function for updating value function |
| [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/pdf/1807.01675.pdf) | STEVE | NeurIPS18 | an extension to MVE; only utilize roll-outs without introducing significant errors |
| [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/pdf/1805.12114.pdf) | PETS | NeurIPS18 | propose PETS that incorporate uncertainty via an ensemble of bootstrapped models |
| [Algorithmic Framework for Model-based Deep Reinforcement Learning with Theoretical Guarantees](https://arxiv.org/pdf/1807.03858.pdf)  | SLBO | ICLR19 | propose a novel algorithmic framework for designing and analyzing model-based RL algorithms with theoretical guarantees: provide a lower bound of the true return satisfying some properties s.t. optimizing this lower bound can actually optimize the true return |
| [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/pdf/1906.08253.pdf) | MBPO | NeurIPS19  | propose MBPO with monotonic model-based improvement; theoretically discuss how to choose k for model rollouts |
| [Bidirectional Model-based Policy Optimization](https://arxiv.org/pdf/2007.01995.pdf) | BMPO | ICML20  | an extension to MBPO; consider both forward dynamics model and backward dynamics model |
| [Trust the Model When It Is Confident: Masked Model-based Actor-Critic](https://arxiv.org/pdf/2010.04893.pdf)| M2AC | NeurIPS20 | an extension to MBPO; use model rollouts only when the model is confident |
| [On Effective Scheduling of Model-based Reinforcement Learning](https://arxiv.org/pdf/2111.08550.pdf) | AutoMBPO | NeurIPS21 | an extension to MBPO; automatically schedule the real data ratio as well as other hyperparameters for MBPO |
| [Value Gradient weighted Model-Based Reinforcement Learning](https://openreview.net/pdf?id=4-D6CZkRXxI) | VaGraM | ICLR22 |  |

|  Title | Method | Conference |  Description |
| ----  | ----   | ----       |   ----  |
| [Model Based Reinforcement Learning for Atari](https://arxiv.org/pdf/1903.00374.pdf) | SimPLe | ICLR20 | first successfully handle ALE benchmark with model-based method with some designs: (i) deterministic Model; (ii) well-designed loss functions; (iii) scheduled sampling; (iv) stochastic Models |
| [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf) | EfficientZero | NeurIPS21 | first achieve super-human performance on Atari games with limited data; propose EfficientZero with three components: (i) use self-supervised learning to learn a temporally consistent environment model, (ii) learn the value prefix in an end-to-end manner, (iii) use the learned model to correct off-policy value targets |




<a id='Model Free Offline'></a>
## (Model Free) Offline RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/pdf/1812.02900.pdf) | BCQ | ICML19 | show that off-policy methods perform badly because of extrapolation error; propose batch-constrained reinforcement learning: maximizing the return as well as minimizing the mismatch between the state-action visitation of the policy and the state-action pairs contained in the batch |
| [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/pdf/2006.04779.pdf) | CQL | NeurIPS20 | propose CQL with conservative q function, which is a lower bound of its true value, since standard off-policy methods will overestimate the value function |
| [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/pdf/2005.01643.pdf) | ---- | arxiv20 | tutorial about methods, applications and open problems of offline rl |


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
| [Revisiting Design Choices in Model-Based Offline Reinforcement Learning](https://arxiv.org/pdf/2110.04135.pdf) | ---- | ICLR22 | conduct a rigorous investigation into a series of these design choices for Model-based Offline RL |


<a id='Meta Reinforcement Learning'></a>
## Meta RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [RL2 : Fast reinforcement learning via slow reinforcement learning](https://arxiv.org/pdf/1611.02779.pdf) | RL2 | ICLR17 | view the learning process of the agent itself as an objective; structure the agent as a recurrent neural network to store past rewards, actions, observations and termination flags for adapting to the task at hand when deployed |
| [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://www.cs.utexas.edu/users/sniekum/classes/RL-F17/papers/Meta.pdf) | MAML | ICML17 | propose a general framework for different learningproblems, including classification, regression andreinforcement learning; the main idea is to optimize the parameters to quickly adapt to new tasks (with a few steps of gradient descent) |
| [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/pdf/1903.08254.pdf) | PEARL | ICML19 | encode past tasks’ experience with probabilistic latent context and use inference network to estimate the posterior|
| [Fast context adaptation via meta-learning](https://arxiv.org/pdf/1810.03642.pdf) | CAVIA | ICML19 | propose CAVIA as an extension to MAML that is less prone to meta-overfitting, easier to parallelise, and more interpretable; partition the model parameters into two parts: context parameters and shared parameters, and only update the former one in the test stage |
| [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](http://proceedings.mlr.press/v100/yu20a/yu20a.pdf) | Meta World | CoRL19 | an envoriment for meta RL as well as multi-task RL |
| [Meta-Q-Learning](https://arxiv.org/pdf/1910.00125.pdf) | MQL | ICLR20 | an off-policy algorithm for meta RL andbuilds upon three simple ideas: (i) Q Learning with context variable represented by pasttrajectories is competitive with SOTA; (ii) Multi-task objective is useful for meta RL; (iii) Past data from the meta-training replay buffer can be recycled |
| [Varibad: A very good method for bayes-adaptive deep RL via meta-learning](https://arxiv.org/pdf/1910.08348.pdf) | variBAD | ICLR20 |  |
| [Improving Generalization in Meta-RL with Imaginary Tasks from Latent Dynamics Mixture](https://arxiv.org/pdf/2105.13524.pdf) | LDM | NeurIPS21 | aim to train an agent that prepares for unseen test tasks during training, propose to train a policy on mixture tasks along with original training tasks for preventing the agent from overfitting the training tasks |
| [Unifying Gradient Estimators for Meta-Reinforcement Learning via Off-Policy Evaluation](https://arxiv.org/pdf/2106.13125.pdf) | ---- | NeurIPS21 |  |
| [Bootstrapped Meta-Learning](https://arxiv.org/pdf/2109.04504.pdf) | BMG | ICLR22 | propose BMG to let the metalearner teach itself for tackling ill-conditioning problems and myopic metaobjectives in meta learning; BGM introduces meta-bootstrap to mitigate myopia and formulate the meta-objective in terms of minimising distance to control curvature |
| [Model-Based Offline Meta-Reinforcement Learning with Regularization](https://arxiv.org/pdf/2202.02929.pdf) | MerPO | ICLR22 |  |
| [Skill-based Meta-Reinforcement Learning](https://openreview.net/pdf?id=jeLW-Fh9bV) | SiMPL | ICLR22 |  |
| [Hindsight Foresight Relabeling for Meta-Reinforcement Learning](https://arxiv.org/pdf/2109.09031.pdf) | HFR | ICLR22 |  |
| [CoMPS: Continual Meta Policy Search](https://arxiv.org/pdf/2112.04467.pdf) | CoMPS | ICLR22 |  |
| [Learning a subspace of policies for online adaptation in Reinforcement Learning](https://arxiv.org/pdf/2110.05169.pdf) | ---- | ICLR22 |  |


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

<a id='Sequence Generation'></a>
## RL as Sequence Generation

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/pdf/2106.01345.pdf) | DT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (return-to-go, state, action, return-to-go, ...); there is not explicit optimization process; evaluate on Offline RL |
| [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/pdf/2106.02039.pdf) | TT | NeurIPS21 | regard RL as a sequence generation task and use transformer to generate (s_0^0, ..., s_0^N, a_0^0, ..., a_0^M, r_0, ...); use beam search to inference; evaluate on imitation learning, goal-conditioned RL and Offline RL | 

<a id='Unsupervised RL'></a>
## Unsupervised RL

|  Title | Method | Conference | Description |
| ----  | ----   | ----       |   ----  |
| [Decoupling representation learning from reinforcement learning](https://arxiv.org/pdf/2009.08319.pdf) | ATC | ICML21 | propose a new unsupervised task tailored to reinforcement learning named Augmented Temporal Contrast (ATC), which borrows ideas from Contrastive learning; benchmark several leading Unsupervised Learning algorithms by pre-training encoders on expert demonstrations and using them in RL agents|
| [Pretraining representations for data-efficient reinforcement learning](https://arxiv.org/pdf/2106.04799.pdf) | SGI | NeurIPS21 | consider to pretrian with unlabeled data and finetune on a small amount of task-specific data to improve the data efficiency of RL; employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL |
| [Understanding the World Through Action](https://arxiv.org/pdf/2110.12543.pdf) | ---- | CoRL21 | discusse how self-supervised reinforcement learning combined with offline RL can enable scalable representation learning |
| [The Information Geometry of Unsupervised Reinforcement Learning](https://arxiv.org/pdf/2110.02719.pdf) | ---- | ICLR22 | show that unsupervised skill discovery algorithms based on mutual information maximization do not learn skills that are optimal for every possible reward function; provide a geometric perspective on some skill learning methods |


<a id='Tutorial and Lesson'></a>
## Tutorial and Lesson

| Tutorial and Lesson |
| ---- |
| [Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](https://d1wqtxts1xzle7.cloudfront.net/54674740/Reinforcement_Learning-with-cover-page-v2.pdf?Expires=1641130151&Signature=eYy7kmTVqTXFcANS-9GZJUyb86cDqKeh2QX8VvzjouEM-QSfuiCm1WHhP~bW5C57Mecj6en~YRoTvxekzU5lq~UaHSBoc-7xP8dXBp91shcwdfJ8M0LUkktpqcQjXQi7ZzhGn33qZeah0p8S06ARzjimF5coL5arvp9yANAsy4KigXSZwAZNXxksKwqUAult2QseLL~Bv1p2locjYahRzTuex3vMxdBLhT9HOGFF0qOdKYxsWiaITUKnVYl8AvePDHEEXgfmuqEfjqjF5p~FHOsYl3gEDZOvUp1eUzPg2~i0MQXY49nUpzsThL5~unTRIsYJiBghnkYl8py0r~UelQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) |
| [Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) | 
| [Deep Reinforcement Learning, CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) |
| [Deep Reinforcement Learning and Control, CMU 10703](https://katefvision.github.io/) |
| [RLChina](http://rlchina.org/topic/9) |
