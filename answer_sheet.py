
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Since Monte Carlo relies on full episode returns for updates, and gamma influences returns throughout the entire episode. In contrast, Q-Learning uses single-step updates, where gamma's impact is more localized, making Monte Carlo more sensitive to gamma changes"  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 6e-1
    b) 6e-2
    c) 6e-3
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.75
    b) 0.25
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which epsilon decay achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.75
    c) 0.001
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of  training when employing an exponential decay strategy
    with epsilon decay set to 0.990?
    a) 0.990
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Using an exploration fraction parameter in the decay strategy (linear decay) can make the agent more robust and adaptable to different environments than decay based on decay rate parameter (exponential) because it adapts the exploration rate relative to the total training time. Linear decay ensures adequate exploration in early stages while exploiting more as learning progresses. In contrast, exponential decay does not inherently adapt to the total training time, requiring careful tuning for each environment to balance exploration and exploitation. Thus, using an exploration fraction parameter makes the agent more adaptable to varying environments without the need for extensive tuning."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "In DQN, the loss does not behave like in typical supervised learning approaches mainly due to two reasons: non-stationary targets and exploration-exploitation trade-off. First, the non-stationary nature of the targets stems from the target network updates. In supervised learning, the targets are fixed, whereas in DQN, target Q-values are updated periodically using the target network. This continuous adaptation of targets makes the learning process less stable, resulting in a loss that does not steadily decrease. Second, the exploration-exploitation trade-off adds another layer of complexity to the learning process. The agent is not only learning from its past experiences but also constantly exploring new actions to improve its policy. This exploration leads to a more dynamic loss landscape, contributing to fluctuations in the loss during training."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Spikes in loss observed at regular intervals throughout training are mainly due to the periodic updates of target network. When target network is updated, the target Q-values for the same state-action pairs can change abruptly, leading to sudden increase in loss. As the target network is updated at a fixed frequency, these spikes in the loss can be observed at regular intervals. As target network becomes more stable and converges into good policy, the effect of these updates on loss become less prominent. However, spikes remain characteristic feature of DQN training due to continuous adaptation of targets and exploration-exploitation trade-off."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer



############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "We combined hyperparameter tuning and scheduling to optimize performance of our DDPG agents. Hyperparameter-tuning techniques: grid search, random search, Bayesian optimization are deployed to create a list of hyperparameter configurations for sweeping. For grid search, we generated grids of hyperparameter values using grid_search(). For random search, we implemented random_search() to sample hyperparameters from specified distributions like uniform, exponential. In addition, we used Bayesian optimization from scikit-optimize to efficiently search the hyperparameter space by building a probabilistic model of the objective function w.r.t. rewards, selecting the most promising hyperparameters for evaluation in 'train_ddpg_bayesian.py'. This approach encouraged exploration of under-explored regions while exploiting promising areas, converging to a near-optimal set of hyperparameters. For scheduling, we adapted learning rates and exploration noise during training. We applied exponential decay to policy and critic learning rates through schedule_hyperparameters() function. We adapted exploration noise by linearly decreasing standard deviation of Gaussian noise added to the actor's output over certain fraction of total timesteps, maintaining minimum value for remaining training period. These hyperparameter tuning and scheduling techniques enabled efficient exploration of the hyperparameter space, while balancing exploration and exploitation. This process facilitated larger updates and exploration during initial training, leading to improved overall performance for DDPG agents."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer