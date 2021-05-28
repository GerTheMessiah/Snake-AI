

class Memory:
    def __init__(self):
        self.actions = []
        self.around_view = []
        self.cat_obs = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def __add__(self, other):
        self.actions += other.actions
        self.around_view += other.around_view
        self.cat_obs += other.cat_obs
        self.logprobs += other.logprobs
        self.rewards += other.rewards
        self.dones += other.dones
        return self

    def add(self, around_view, cat_obs, action, log_probs, reward, done):
        self.around_view.append(around_view)
        self.cat_obs.append(cat_obs)
        self.actions.append(action)
        self.logprobs.append(log_probs)
        self.rewards.append(reward)
        self.dones.append(done)

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self.actions)

    def clear_memory(self):
        del self.actions[:]
        del self.cat_obs[:]
        del self.around_view[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
