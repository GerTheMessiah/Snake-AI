from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='src.gym_snake.envs.snake_env:SnakeEnv'
    )
