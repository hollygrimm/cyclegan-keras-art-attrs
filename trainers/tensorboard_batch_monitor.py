from keras.callbacks import TensorBoard
import tensorflow as tf

class TensorBoardBatchMonitor(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super().on_batch_end(batch, logs)
