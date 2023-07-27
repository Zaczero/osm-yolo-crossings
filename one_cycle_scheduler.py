import numpy as np
from keras import backend as K
from keras.callbacks import Callback


class OneCycleScheduler(Callback):
    def __init__(self, max_lr, steps_per_epoch, cycle_epochs, div_factor=25., pct_start=0.3):
        super(OneCycleScheduler, self).__init__()
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.total_iterations = steps_per_epoch * cycle_epochs
        self.mid_cycle_id = int(self.total_iterations * pct_start)
        self.current_iteration = 0
        self.history = {}

    def cosine_annealing(self, start, end, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end + (start - end) / 2 * cos_out

    def on_train_begin(self, logs=None):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr/self.div_factor)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["lr"] = K.get_value(self.model.optimizer.lr)
        self.current_iteration += 1
        lr = self.calc_lr()

        K.set_value(self.model.optimizer.lr, lr)

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.current_iteration)

    def calc_lr(self):
        if self.current_iteration <= self.mid_cycle_id:
            pct = self.current_iteration / self.mid_cycle_id
            return self.cosine_annealing(self.max_lr/self.div_factor, self.max_lr, pct)
        else:
            pct = (self.current_iteration - self.mid_cycle_id) / (self.total_iterations - self.mid_cycle_id)
            return self.cosine_annealing(self.max_lr, self.max_lr/self.div_factor, pct)
