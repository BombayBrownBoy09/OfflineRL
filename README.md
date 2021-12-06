## Assignment 3 (building a pipeline for offline RL) - AIPI 530

Access my Medium Article on Reinforcement Learning here: 

https://medium.com/geekculture/reinforcement-learning-what-rewards-you-makes-you-stronger-9aa03ad9e0e

Project link: 
**Objective** Train CQL on d3rlpy dataset, and training OPE (FQE) for evaluation of trained policy
https://colab.research.google.com/drive/1A0c9bYn5NHVb8dAwej83kPr-xtxrTQQZ?usp=sharing

![image](https://user-images.githubusercontent.com/67541365/144878386-c89acac5-3038-4a7d-a963-05f710fe5bc1.png)

**TRAINED CQL PLOT** 

![image](https://user-images.githubusercontent.com/67541365/144878660-dad69a58-209d-4047-9bcd-52aaff6b3f52.png)

**TRAINED OPE(FQE) PLOTS**

### :zap: Most Practical RL Library Ever
- **offline RL**: d3rlpy supports state-of-the-art offline RL algorithms. Offline RL is extremely powerful when the online interaction is not feasible during training (e.g. robotics, medical).
- **online RL**: d3rlpy also supports conventional state-of-the-art online training algorithms without any compromising, which means that you can solve any kinds of RL problems only with `d3rlpy`.
- **advanced engineering**: d3rlpy is designed to implement the faster and efficient training algorithms. For example, you can train Atari environments with x4 less memory space and as fast as the fastest RL library.

### :beginner: Easy-To-Use API
- **zero-knowledge of DL library**: d3rlpy provides many state-of-the-art algorithms through intuitive APIs. You can become a RL engineer even without knowing how to use deep learning libraries.
- **scikit-learn compatibility**: d3rlpy is not only easy, but also completely compatible with scikit-learn API, which means that you can maximize your productivity with the useful scikit-learn's utilities.

### :rocket: Beyond State-Of-The-Art
- **distributional Q function**: d3rlpy is the first library that supports distributional Q functions in the all algorithms. The distributional Q function is known as the very powerful method to achieve the state-of-the-performance.
- **many tweek options**: d3rlpy is also the first to support N-step TD backup and ensemble value functions in the all algorithms, which lead you to the place no one ever reached yet.


## installation
d3rlpy supports Linux, macOS and Windows.

1. clone this repository using - https://github.com/BombayBrownBoy09/OfflineRL.git
2. Install - https://github.com/takuseno/d4rl-pybullet
3. !pip install Cython numpy  and !pip install -e .
4. Next we train CQl on default environment in d3rlpy (https://github.com/takuseno/d3rlpy) for 20 epochs folllowed by evaluation (True Q, Estimated Q and Average Reward)
5. Train OPE (FQE) and evaluate using estimated Q alongside True Q  for FQE

## Citation 

The paper is available [here](https://arxiv.org/abs/2111.03788).

d3rlpy [Documentation](https://d3rlpy.readthedocs.io/en/v0.91/)

[know more..](https://arxiv.org/abs/2111.03788)
```
@InProceedings{seno2021d3rlpy,
  author = {Takuma Seno, Michita Imai},
  title = {d3rlpy: An Offline Deep Reinforcement Library},
  booktitle = {NeurIPS 2021 Offline Reinforcement Learning Workshop},
  month = {December},
  year = {2021}
}
```

# Offline RL examples 
Here is a tutorial for a Toy Task - Line tracer for you to try

https://colab.research.google.com/drive/1QpMMqVByz0U--fh-x_po9-Dgat9am9YU?usp=sharing

Here, we:
a. Make an Original Environment (setup state and reward calculations)
b. Generate a dataset and visualize it by sampling random action
c. Setup and train CQL offline RL algorithm along with dataset from d3rlpy
d. Plot trajectory of RL model that traces the original line

Easy to use, right?
