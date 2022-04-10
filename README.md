# PPO - Reinforcement Learning

## Snake-AI

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)\
\
[![Project](https://img.shields.io/static/v1?label=Game&message=Snake&color=red)]()
[![SonarCloud](https://sonarcloud.io/api/project_badges/measure?project=citrus&metric=alert_status)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.com/IcarusCoding/Speed.svg?token=fchrN5ADWA1xeNzfmo3q&branch=develop)](https://travis-ci.com/IcarusCoding/Speed)
[![Version](https://img.shields.io/static/v1?label=Version&message=0.2&color=green)]()
[![Contributors](https://img.shields.io/static/v1?label=Contributors&message=1&color=yellow)]()

## [Observation](src/gym_snake/envs/snake_env/observation.py)
![obs](src/resources/images/observation.png)
### Visual observation
- Visual observation. The AI is observing a 13x13 space around this head. Six on the left and right site and the Head in the middle.
### Static observation
- Ray tracing along the yellow dashed lines. The AI is able to see himself, walls and the apple.
- Direction (red line) of the snake.
- apple (blue point) and tail compass. Indicates the relativ position according to the apple or the last part of the snake.
- step counter. If the snake doesn't eat an apple in a descried amount of steps the game ends.

## [Evaluation / Reward](src/gym_snake/envs/snake_env/reward.py)
### +100 if the snake reaches the max length. | win
### +2.5 if snake eats an apple.
### -10 if the snake dies. | loss


## [PPO - Implementation](src/snakeAI/agents/ppo_model.py)
- Implementation with ray tune.


## License
#### GPLv3 (General Public License 3)


## Results
<img src="src/resources/images/SnakeAI.gif" width="450" height="450" alt="0">


## Dependencies

Install packages: ```pip install -r requirements.txt```\
Be careful with pytorch dependencies. They often might not work with. If so, install on your own.
