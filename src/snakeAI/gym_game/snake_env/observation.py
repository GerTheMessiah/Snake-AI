import numpy as np


def on_playground(a, b, size):
    return (0 <= a < size[0]) and (0 <= b < size[1])


def dist(ground, p_pos, wanted_hit, fac_0, fac_1):
    dist_, i_0, i_1 = 0, 1, 1
    p_0 = p_pos[0] + fac_0 * i_0
    p_1 = p_pos[1] + fac_1 * i_1
    while on_playground(p_0, p_1, ground.shape) and ground[p_0, p_1] not in wanted_hit:
        i_0 += 1
        i_1 += 1
        dist_ += 1
        p_0 = p_pos[0] + fac_0 * i_0
        p_1 = p_pos[1] + fac_1 * i_1
    if not on_playground(p_0, p_1, ground.shape) and bool(wanted_hit):
        return 0
    # if any((p_pos[i] <= 0 or p_pos[i] >= ground.shape[i]) for i in range(2)):
    #     return 1.2 if dist_ == 0 else dist_
    return 1 / dist_ if dist_ != 0 else 2


# 1x6x13x13
def create_around_view(pos, id, g):
    width = 6
    c_h = id * 2
    c_s = c_h - 1
    tmp_arr = np.zeros((6, width * 2 + 1, width * 2 + 1), dtype=np.float64)

    for row in range(-width, width + 1):
        for column in range(-width, width + 1):
            if on_playground(pos[0] + row, pos[1] + column, g.shape):
                a, b = pos[0] + row, pos[1] + column
                if g[a, b] == c_s:
                    tmp_arr[1, row + width, column + width] = 1
                    continue

                elif g[a, b] == c_h:
                    tmp_arr[2, row + width, column + width] = 1
                    continue

                elif g[a, b] == 0:
                    tmp_arr[3, row + width, column + width] = 1
                    continue

                elif g[a, b] == -1:  # End of snake tail.
                    tmp_arr[4, row + width, column + width] = 1
                    continue

                elif g[a, b] == -2:
                    tmp_arr[5, row + width, column + width] = 1

            else:
                tmp_arr[0, row + width, column + width] = 1

    return np.expand_dims(tmp_arr, axis=0)


# a = 24
def create_distances(pos, ground):
    obs = np.zeros(24, dtype=np.float64)
    a = 0
    for c in [[], list(range(1, 13)), [-2]]:
        obs[a] = dist(ground, pos, c, -1, 0)
        a += 1
        obs[a] = dist(ground, pos, c, -1, 1)
        a += 1
        obs[a] = dist(ground, pos, c, 0, 1)
        a += 1
        obs[a] = dist(ground, pos, c, 1, 1)
        a += 1
        obs[a] = dist(ground, pos, c, 1, 0)
        a += 1
        obs[a] = dist(ground, pos, c, 1, -1)
        a += 1
        obs[a] = dist(ground, pos, c, 0, -1)
        a += 1
        obs[a] = dist(ground, pos, c, -1, -1)
        a += 1
    return obs


# a + 4
def direction_obs(direction):
    obs = np.zeros(4, dtype=np.float64)
    obs[0 + direction] = 1
    return obs


# a + 6
def compass_obs(pos, obj):
    obs = np.zeros(6, dtype=np.float64)
    if obj is None:
        return obs
    obs[0] = 1 if pos[0] < obj[0] else 0
    obs[1] = 1 if pos[1] > obj[1] else 0
    obs[2] = 1 if pos[0] > obj[0] else 0
    obs[3] = 1 if pos[1] < obj[1] else 0
    obs[4] = 1 if pos[0] == obj[0] else 0
    obs[5] = 1 if pos[1] == obj[1] else 0
    return obs


# a + 1
def step_counter_obs(apple_stepCounter, size):
    obs = np.zeros(1, dtype=np.float64)
    obs[0] = 1 / (size - apple_stepCounter) if apple_stepCounter != size else 2
    return obs


def make_obs(p_id, pos, tail_pos, direction, ground, food, iter_apple_counter):
    around_view = create_around_view(pos, p_id, ground)
    distances = create_distances(pos, ground)
    direction = direction_obs(direction)
    apple_dir = compass_obs(pos, food)
    step_obs = step_counter_obs(iter_apple_counter, ground.size)

    tail_obs = compass_obs(pos, tail_pos)
    cat_tensor = np.concatenate((distances, direction, apple_dir, step_obs, tail_obs))
    return around_view, np.expand_dims(cat_tensor, axis=0)
