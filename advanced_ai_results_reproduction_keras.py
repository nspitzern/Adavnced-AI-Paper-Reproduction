import keras
from keras.activations import relu, linear
from collections import deque

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import lunar_lander as lander

NEURONS_LIST = [64, 128, 256]


def replay_experiences():
    if len(memory) >= batch_size:
        sample_choices = np.array(memory, dtype=object)
        mini_batch_index = np.random.choice(len(sample_choices), batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        finishes = []
        for index in mini_batch_index:
            states.append(memory[index][0])
            actions.append(memory[index][1])
            next_states.append(memory[index][2])
            rewards.append(memory[index][3])
            finishes.append(memory[index][4])
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        finishes = np.array(finishes)
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        q_vals_next_state = model.predict_on_batch(next_states)
        q_vals_target = model.predict_on_batch(states)
        max_q_values_next_state = np.amax(q_vals_next_state, axis=1)
        q_vals_target[np.arange(batch_size), actions] = rewards + gamma * (max_q_values_next_state) * (1 - finishes)
        model.fit(states, q_vals_target, verbose=0)
        global epsilon
        if epsilon > min_eps:
            epsilon *= 0.996


with tf.device("gpu:0"):
    for NEURONS in NEURONS_LIST:
        epsilon = 1
        gamma = 0.99
        batch_size = 64
        # define memory replay buffer
        memory = deque(maxlen=1000000)
        min_eps = 0.01

        # define model
        learning_rate = 1e-3
        model = keras.Sequential()
        model.add(keras.layers.Dense(NEURONS, input_dim=8, activation=relu))
        model.add(keras.layers.Dense(NEURONS, activation=relu))
        model.add(keras.layers.Dense(4, activation=linear))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        env = lander.LunarLander()
        # env.seed(0)
        np.random.seed(0)
        num_episodes = 400
        scores = []
        avg_scores = []
        print(f'Number of neurons: {NEURONS}, LR: {learning_rate}, Epsilon start: {epsilon},'
              f' Minimum Epsilon: {min_eps}, Batch size: {batch_size}, Gamma (discount factor): {gamma}')
        for i in range(num_episodes+1):
            score = 0
            state = env.reset()
            finished = False
            # keep the model every 50 episodes
            if i != 0 and i % 50 == 0:
                model.save(f"./saved_models/model_{str(i)}_episodes_{NEURONS}_neurons.h5")
            # limit each episode to 3000 steps
            for j in range(3000):
                state = np.reshape(state, (1, 8))
                if np.random.random() <= epsilon:
                    action = np.random.choice(4)
                else:
                    action_values = model.predict(state)
                    action = np.argmax(action_values[0])

                next_state, reward, finished, metadata = env.step(action)
                next_state = np.reshape(next_state, (1, 8))
                memory.append((state, action, next_state, reward, finished))
                replay_experiences()
                score += reward
                state = next_state
                if finished:
                    break

            scores.append(score)
            avg_scores.append(np.mean(scores[-10:]))
            print("Episode = {}, Score = {}, Avg_Score = {}".format(i, score, np.mean(scores[-10:])))

        np.save(f'scores_neurons_{NEURONS}.npy', np.asarray(avg_scores))

results_64 = np.load('scores_neurons_64.npy')[:350]
results_128 = np.load('scores_neurons_128.npy')[:350]
results_256 = np.load('scores_neurons_256.npy')[:350]

episodes = range(len(results_64))

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax.plot(episodes, results_64,  label='DQN 64', color='red')
ax.plot(episodes, results_128, label='DQN 128', color='green')
ax.plot(episodes, results_256, label='DQN 256', color='blue')

plt.ylabel("Average Reward")
plt.xlabel("Iterations")
plt.title('Comparison of Different Hidden Neurons')
plt.legend()

plt.savefig('results.jpg')

plt.show()