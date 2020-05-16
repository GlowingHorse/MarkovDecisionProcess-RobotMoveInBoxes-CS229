from World import World
import numpy as np
import pandas as pd
import copy


def construct_p(world, p=0.8, step=-0.04):
    # p是机器人跑偏的概率
    nstates = world.get_nstates()
    nrows = world.get_nrows()
    obsacle_index = world.get_stateobstacles()
    terminal_index = world.get_stateterminals()
    bad_index = obsacle_index + terminal_index
    rewards = np.array([step] * 4 + [0] + [step] * 4 + [1, -1] + [step])
    actions = ["N", "S", "E", "W"]
    transition_models = {}
    for action in actions:
        transition_model = np.zeros((nstates, nstates))
        for i in range(1, nstates + 1):
            if i not in bad_index:
                if action == "N":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        # 此处判断能否往右边走一格，i-1+nrows就是右边一格的列坐标
                        # 因为现在action是向上，那么如果可以往右边走一格
                        # 那么到达右边的概率应该是0.1(if p = 0.8)
                        transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                    else:
                        # 如果向右走失败，那么就在原始的格子不动
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        # 此处判断能否往左边走一格，i-1-nrows就是左边一格的列坐标
                        # 因为现在action是向上，那么如果可以往左边走一格
                        # 那么到达左边的概率应该是0.1(if p = 0.8)
                        transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                    else:
                        # 如果向左走失败，那么就在原始的格子不动
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        # i-1 % nrows是判断上面是否能够走
                        # 如果上面能走那么在状态转移概率的矩阵里面直接赋值p即可
                        transition_model[i - 1][i - 1 - 1] += p
                    else:
                        # 如果上面不能走那么仍然回到原位
                        # 给状态转移概率的矩阵对应位置加上概率
                        transition_model[i - 1][i - 1] += p
                if action == "S":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        # 此处判断能否往右边走一格，i-1+nrows就是右边一格的列坐标
                        # 因为现在action是向上，那么如果可以往右边走一格
                        # 那么到达右边的概率应该是0.1(if p = 0.8)
                        transition_model[i - 1][i + nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        # 此处判断能否往左边走一格，i-1-nrows就是左边一格的列坐标
                        # 因为现在action是向上，那么如果可以往左边走一格
                        # 那么到达左边的概率应该是0.1(if p = 0.8)
                        transition_model[i - 1][i - nrows - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        # 第一个判断条件把最后一行格子去掉了
                        # 第二个判断条件去掉了障碍格子
                        # 第三个没有起作用，因为最后一个格子12也在第一个判断条件被去掉了，判断能够向下走
                        transition_model[i - 1][i + 1 - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                if action == "E":
                    if i + nrows <= nstates and (i + nrows) not in obsacle_index:
                        # 判断能够向右走
                        transition_model[i - 1][i + nrows - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        # 判断能够向下走
                        transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        # 判断能否向上走
                        transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                if action == "W":
                    if 0 < i - nrows <= nstates and (i - nrows) not in obsacle_index:
                        # 判断能否向左走
                        transition_model[i - 1][i - nrows - 1] += p
                    else:
                        transition_model[i - 1][i - 1] += p
                    if 0 < i % nrows and (i + 1) not in obsacle_index and (i + 1) <= nstates:
                        # 判断能够向下走
                        transition_model[i - 1][i + 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
                    if (i - 1) % nrows > 0 and (i - 1) not in obsacle_index:
                        # 判断能否向上走
                        transition_model[i - 1][i - 1 - 1] += (1 - p) / 2
                    else:
                        transition_model[i - 1][i - 1] += (1 - p) / 2
            elif i in terminal_index:
                transition_model[i - 1][i - 1] = 1
        transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1), columns=range(1, nstates + 1))

    return transition_models, rewards


def max_action(transition_models, rewards, gamma, s, V, actions, terminal_ind):

    maxs = {key: 0 for key in actions}
    max_a = ""
    action_map = {k: v for k, v in zip(actions, [1, 3, 2, 4])}
    for action in actions:
        if s not in terminal_ind:
            # 对于每一个状态s，在不同的四个动作中for循环
            # 从而计算出在当前的状态转移概率+当前状态之下的期望和
            # 此处除了即时奖励，后面加上的就是奖励的期望和
            # 动作确定时对应状态的状态转移概率是一定的，即transition_models[action].loc[s, :].values
            # 然后V是上一次迭代时获得的不同状态的value
            # 通过这里的V和概率矩阵，来计算在状态s时，以动作action转移到下一状态sprime（课程中的用法）的期望和
            maxs[action] += rewards[s - 1] + gamma * np.dot(transition_models[action].loc[s, :].values, V)
        else:
            maxs[action] = rewards[s - 1]
    maxi = -10 ** 10
    for key in maxs:
        if maxs[key] > maxi:
            max_a = key
            maxi = maxs[key]
    return maxi, action_map[max_a]


def value_iteration(world, transition_models, rewards, gamma=1.0, theta=10 ** -4):

    nstates = world.get_nstates()
    terminal_ind = world.get_stateterminals()
    V = np.zeros((nstates, ))
    P = np.zeros((nstates, 1))
    actions = ["N", "S", "E", "W"]
    delta = theta + 1
    while delta > theta:
        # 迭代至收敛
        delta = 0

        # 每次把上次计算得到的value function结果赋值给小v
        v = copy.deepcopy(V)
        for s in range(1, nstates + 1):
            # 计算在状态s时，不同action中能够获得最大总收入期望和的action以及其对应的奖励结果
            V[s - 1], P[s - 1] = max_action(transition_models, rewards, gamma, s, v, actions, terminal_ind)
            delta = max(delta, np.abs(v[s - 1] - V[s - 1]))
    return V, P


def policy_iter(policy, world, transition_models, rewards, gamma=0.9, theta=10 ** -4):

    nstates = world.get_nstates()
    terminal_ind = world.get_stateterminals()
    # Initiate value function to zeros
    V = np.zeros((nstates,))
    a = ["N", "S", "E", "W"]
    while True:
        delta = 0
        # For each state, perform a backup
        for s in range(1, nstates + 1):
            v = 0
            # Look at the policy actions and their probabilities
            for action, action_prob in enumerate(policy[s-1]):
                action = a[action]
                # For each action, calculate total gain
                if s not in terminal_ind:
                    # 在状态s时，四个动作下，不同的下一个状态sprime总收入的期望和的累加
                    # 也就是四个动作的累加
                    v += rewards[s - 1] + action_prob * gamma * np.dot(transition_models[action].loc[s, :].values, V)
                else:
                    v = rewards[s - 1]
            delta = max(delta, np.abs(v - V[s-1]))
            V[s-1] = v
            # print(V[s-1])
        # Stop evaluating once the value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)


#  Helper function to calculate the value for all action in a given state
def lookfoword(s, V, transition_models, rewards, gamma=0.9):
    nActions = world.get_nactions()
    terminal_ind = world.get_stateterminals()
    A = np.zeros(nActions)
    a = ["N", "S", "E", "W"]
    for i, action in enumerate(a):
        # action = a[action]
        if s not in terminal_ind:
            A[i] += rewards[s - 1] + gamma * np.dot(transition_models[action].loc[s, :].values, V)
        else:
            A[i] = rewards[s - 1]
    return A


def policy_improvement(world, transition_models, rewards, gamma= 0.9):
    nstates = world.get_nstates()
    nActions = world.get_nactions()
    # Start with a uniform policy
    policy = np.ones((nstates, nActions)) / nActions

    while True:
        # Evaluate the current policy
        V = policy_iter(policy, world, transition_models, rewards, gamma)

        # Will be set to false if we make any changes to the policy
        policy_stable = True

        for s in range(1, nstates + 1):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s-1])

            # s是当前要处理的状态，根据这个状态来调整该状态时的策略policy
            # V是当前policy下的全部的状态的收入的结果
            # transition_models是状态转移概率
            # rewards是每个状态的即时奖励
            action_values = lookfoword(s, V, transition_models, rewards, gamma)
            # best_action的顺序是N S E W，这是在lookfoword里定义的
            best_action = np.argmax(action_values)

            # Greedily update the policy
            if chosen_a != best_action:
                policy_stable = False
            policy[s-1] = np.eye(nActions)[best_action]

        if policy_stable:
            return V, policy


if __name__ == "__main__":
    world = World()
    # world.plot()
    # world.plot_value([np.random.random() for i in range(12)])
    # world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))

    # # part c
    # transition_models, rewards = construct_p(world, step=-0.02)
    # V, P = value_iteration(world, transition_models, rewards, gamma=0.9)
    # world.plot_value(V)
    # world.plot_policy(P)

    # part e
    transition_models, rewards = construct_p(world, step=-0.02)
    V, P = policy_improvement(world, transition_models, rewards, gamma=0.9)
    P_new = np.zeros((world.get_nstates(), 1))
    where_one = np.nonzero(P)[1]
    for i in range(world.get_nstates()):
        if where_one[i] == 0:
            P_new[i, 0] = 1
        elif where_one[i] == 1:
            P_new[i, 0] = 3
        elif where_one[i] == 2:
            P_new[i, 0] = 2
        elif where_one[i] == 3:
            P_new[i, 0] = 4

    world.plot_value(V)
    world.plot_policy(P_new)
    print("Finished")
