import numpy as np


def _on_playground(a, b, shape):
    return (0 <= a < shape[0]) and (0 <= b < shape[1])


class Observation:
    def __init__(self, obs_shape: tuple = (10, 10)):
        self.matrix_shape = obs_shape
        self.previous_obs = None

    def create_view(self, apple: tuple, tail: list, direction: int, hunger: int):
        matrix = np.zeros((5, self.matrix_shape[0], self.matrix_shape[1]), dtype=np.float32)

        if _on_playground(tail[0][0], tail[0][1], self.matrix_shape):
            matrix[0, tail[0][0], tail[0][1]] = 1
            matrix[1, tail[0][0], tail[0][1]] = 1

        for index in tail[1:]:
            matrix[0, index[0], index[1]] = 1

        for i in [-1, 0, 1]:
            tmp_next_pos = [tail[0][0], tail[0][1]]
            tmp_next_pos[((direction + i) % 4) % 2] += -1 if ((direction + i) % 4) % 3 == 0 else 1

            if _on_playground(tmp_next_pos[0], tmp_next_pos[1], self.matrix_shape):
                matrix[2, tmp_next_pos[0], tmp_next_pos[1]] = 1

        if len(tail) > 1:
            matrix[3, tail[-1][0], tail[-1][1]] = 1

        matrix[4, apple[0], apple[1]] = 1

        it = np.ones((1, self.matrix_shape[0], self.matrix_shape[1]), dtype=np.float32) * (hunger / 1000)
        if self.previous_obs is None:
            self.previous_obs = matrix.copy()
            matrix = np.concatenate([matrix, self.previous_obs, it], axis=0)

        else:
            tmp = self.previous_obs
            self.previous_obs = matrix.copy()
            matrix = np.concatenate([matrix, tmp, it], axis=0)

        return np.expand_dims(matrix, axis=0)
