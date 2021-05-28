import torch as T


class Memory:
    def __init__(self, max_mem_size, in_dims_av, in_dims_cat_obs, batch_size=64):
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.around_views = T.zeros((self.mem_size, *in_dims_av), dtype=T.double)
        self.cat_obs = T.zeros((self.mem_size, *in_dims_cat_obs), dtype=T.double)
        self.actions = T.zeros(self.mem_size, dtype=T.long)
        self.rewards = T.zeros(self.mem_size, dtype=T.double)
        self.terminals = T.zeros(self.mem_size, dtype=T.bool)
        self.new_around_views = T.zeros((self.mem_size, *in_dims_av), dtype=T.double)
        self.new_cat_obs = T.zeros((self.mem_size, *in_dims_cat_obs), dtype=T.double)

    def add(self, av, cat_obs, action, reward, done, new_av, new_cat_obs):
        index = self.mem_cntr % self.mem_size
        self.around_views[index, ...] = av if T.is_tensor(av) else T.tensor(av, dtype=T.double)
        self.cat_obs[index, ...] = cat_obs if T.is_tensor(cat_obs) else T.tensor(new_cat_obs, dtype=T.double)
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = done
        self.new_around_views[index, ...] = T.tensor(new_av, dtype=T.double)
        self.new_cat_obs[index, ...] = T.tensor(new_cat_obs, dtype=T.double)
        self.mem_cntr += 1

