import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class WarmUpCosine(LearningRateSchedule):
    def __init__(self, lrStart, lrMax, warmupSteps, totalSteps):
        super().__init__()
        self.lrStart = lrStart
        self.lrMax = lrMax
        self.warmupSteps = warmupSteps
        self.totalSteps = totalSteps
        self.pi = tf.constant(np.pi)
    def __call__(self, step):
        if self.totalSteps < self.warmupSteps:
                raise ValueError("Total number of steps {} must be larger or equal to warmup steps {}.".format(self.totalSteps, self.warmupSteps))
        # a graph that increases to 1 from the initial step to the
        # warmup step, later decays to -1 at the final step mark
        cosAnnealedLr = tf.cos(self.pi* (tf.cast(step, tf.float32) - self.warmupSteps)/ tf.cast(self.totalSteps - self.warmupSteps, tf.float32))
        learningRate = 0.5 * self.lrMax * (1 + cosAnnealedLr)
        if self.warmupSteps > 0:
            if self.lrMax < self.lrStart:
                    raise ValueError("lr_start {} must be smaller or equal to lr_max {}.".format(self.lrStart, self.lrMax))
            # calculate the slope of the warmup line and build the warmup rate
            slope = (self.lrMax - self.lrStart) / self.warmupSteps
            warmupRate = slope * tf.cast(step, tf.float32) + self.lrStart
            # when the current step is lesser that warmup steps, get
            # the line graph, when the current step is greater than
            # the warmup steps, get the scaled cos graph.
            learning_rate = tf.where(step < self.warmupSteps, warmupRate, learningRate)
        return tf.where(step > self.totalSteps, 0.0, learningRate,name="learning_rate",)
    