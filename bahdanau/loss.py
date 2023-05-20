import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import Loss

class MaskedLoss(Loss):
        def __init__(self):
                self.name = "masked_loss"
                self.loss = SparseCategoricalCrossentropy(from_logits=True,
                        reduction="none")
        def __call__(self, yTrue, yPred):
                loss = self.loss(yTrue, yPred)
                # mask off the losses on padding
                mask = tf.cast(yTrue != 0, tf.float32)
                loss *= mask
                return tf.reduce_sum(loss)