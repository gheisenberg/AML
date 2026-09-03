from Evolution import Evolution
import numpy as np

if __name__ == "__main__":
    # Optional instructor reference run for the baseline environment.
    env = Evolution(render_mode="human", all_states=False, fps=8)

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    num_episodes = 4000
    max_steps_per_episode = 50

    learning_rate = 0.3
    discount_rate = 0.95

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.0015

    render = True

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            exploration_threshold = np.random.uniform(0, 1)
            if exploration_threshold > exploration_rate:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                    learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state

            rewards_current_episode += reward

            if render:
                env.render()

            if done:
                break

        exploration_rate = min_exploration_rate + \
                            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

        print("Episode: {}, Exploration rate: {:.2f}, Total reward: {:.2f}".format(episode, exploration_rate, rewards_current_episode))

    print(q_table)
    env.close()
