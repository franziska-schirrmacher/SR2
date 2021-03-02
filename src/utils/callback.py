"""
SR²: Super-Resolution With Structure-Aware Reconstruction

sr2/src/utils
@author: Franziska Schirrmacher
"""

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

## Code based on https://github.com/krasserm/super-resolution
# Callback to store checkpoints
class ModelCheckpointAfter(ModelCheckpoint):
    def __init__(self, epoch, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        # epoch: states after how many epochs another checkpoint is saved
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

        self.after_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.after_epoch:
            super().on_epoch_end(epoch, logs)

# Learning rate scheduler
def learning_rate(step_size, decay, verbose=1):
    def schedule(epoch, lr):
        if epoch > 0 and epoch % step_size == 0:
            return lr * decay
        else:
            return lr

    return LearningRateScheduler(schedule, verbose=verbose)

