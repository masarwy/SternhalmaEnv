import sternhalma_v0
import random

if __name__ == '__main__':
    env = sternhalma_v0.env(
        render_mode='human',
        num_players=2,
        board_diagonal=5,
        reward_mode='potential_shaped',
        gamma=0.95,
    )
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            break
        available_actions = info['valid_moves']
        action = random.choice(available_actions) if available_actions else None
        env.step(action)
    env.close()
