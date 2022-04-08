import math
import numpy as np


def on_playground(a, b, size):
    return (0 <= a < size[0]) and (0 <= b < size[1])


def dist(ground, p_pos, wanted, fac_0, fac_1):
    dist_, i_0, i_1 = 0, 1, 1
    p_0 = p_pos[0] + fac_0 * i_0
    p_1 = p_pos[1] + fac_1 * i_1
    while on_playground(p_0, p_1, ground.shape) and ground[p_0, p_1] not in wanted:
        i_0 += 1
        i_1 += 1
        dist_ += 1
        p_0 = p_pos[0] + fac_0 * i_0
        p_1 = p_pos[1] + fac_1 * i_1
    if not on_playground(p_0, p_1, ground.shape) and bool(wanted):
        return 0
    return 1 / dist_ if dist_ != 0 else 2


class Observation:
    def __init__(self, id, av_size):
        self.id = id
        self.av_size = (6, *av_size)
        self.new_av_size = (6, 13, 13)
        delta_a = self.new_av_size[-1] - av_size[-1]
        delta_b = self.new_av_size[-2] - av_size[-2]
        self.pad_left = math.ceil(delta_a / 2)
        self.pad_right = math.floor(delta_a / 2)
        self.pad_top = math.ceil(delta_b / 2)
        self.pad_bottom = math.floor(delta_b / 2)
        self.pad = ((self.pad_left, self.pad_right), (self.pad_top, self.pad_bottom))
        self.d_av_r = 8
        self.grad_list = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

    def create_view(self, ground):
        arr = np.zeros(self.new_av_size, dtype=np.float64)
        g = np.pad(ground, self.pad, constant_values=5)
        c_ = [(g == 5), ((g == self.id) | (g == 3)), ((g == self.id * 2) | (g == 3)), (g == 0), (g == -1), (g == -2)]
        for i, condition in enumerate(c_):
            condition_arr = np.where(condition, 1, 0)
            arr[0, i] = condition_arr
        return arr

    def dynamic_av(self, g, pos):
        arr = np.zeros(self.new_av_size, dtype=np.float64)
        ground = np.pad(g, self.d_av_r, constant_values=5)
        new_pos = np.array([pos[0] + self.d_av_r, pos[1] + self.d_av_r])
        x = np.arange(new_pos[0] - 6, new_pos[0] + 7)
        y = np.arange(new_pos[1] - 6, new_pos[1] + 7)
        g_ = ground[np.ix_(x, y)]
        c_ = [(g_ == 5), ((g_ == self.id) | (g_ == 3)), ((g_ == self.id * 2) | (g_ == 3)), (g_ == 0), (g_ == -1), (g_ == -2)]
        for i, condition in enumerate(c_):
            a = np.where(condition, 1, 0)
            arr[i] = a
        return arr

    # a + 6
    @staticmethod
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
    @staticmethod
    def hunger_obs(inter_apple_steps, size):
        obs = np.zeros(1, dtype=np.float64)
        obs[0] = 1 / (size - inter_apple_steps) if inter_apple_steps != size else 2
        return obs

    # a = 24
    def create_distances(self, pos, ground):
        obs = np.zeros(24, dtype=np.float64)
        a = 0
        for wanted in [[], [-1, 1, 2], [-2]]:
            for grad in self.grad_list:
                obs[a] = dist(ground, pos, wanted, *grad)
                a += 1
        return obs

    # a + 4
    @staticmethod
    def direction_obs(direction):
        obs = np.zeros(4, dtype=np.float64)
        obs[0 + direction] = 1
        return obs

    def make_obs(self, pos, tail_pos, direction, ground, food, iter_apple_counter):
        around_view = self.dynamic_av(g=ground, pos=pos)
        distances = self.create_distances(pos, ground)
        direction = self.direction_obs(direction)
        apple_obs = self.compass_obs(pos, food)
        tail_obs = self.compass_obs(pos, tail_pos)
        hunger = self.hunger_obs(iter_apple_counter, ground.size)
        scalar_obs = np.concatenate((distances, direction, apple_obs, hunger, tail_obs))
        return around_view, scalar_obs


# 1x6x13x13
def create_around_view(pos, id, g):
    width = 6
    c_h = id * 2
    c_s = id
    tmp_arr = np.zeros((6, width * 2 + 1, width * 2 + 1), dtype=np.float64)

    for row in range(-width, width + 1):
        for column in range(-width, width + 1):
            if on_playground(pos[0] + row, pos[1] + column, g.shape):
                a, b = pos[0] + row, pos[1] + column
                if g[a, b] == c_s or g[a, b] == 3:
                    tmp_arr[1, row + width, column + width] = 1

                elif g[a, b] == c_h or g[a, b] == 3:
                    tmp_arr[2, row + width, column + width] = 1

                elif g[a, b] == 0:
                    tmp_arr[3, row + width, column + width] = 1

                elif g[a, b] == -1:  # End of snake tail.
                    tmp_arr[4, row + width, column + width] = 1

                elif g[a, b] == -2:
                    tmp_arr[5, row + width, column + width] = 1

            else:
                tmp_arr[0, row + width, column + width] = 1

    return tmp_arr
