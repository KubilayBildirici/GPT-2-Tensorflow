import tensorflow as tf


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr, min_lr, warmup_steps, max_steps):
        super().__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        def warmup():
            return self.max_lr * (step + 1) / self.warmup_steps
        
        #cosine decay
        decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        decay_ratio = tf.clip_by_value(decay_ratio, 0.0, 1.0)
        coeff = 0.5 * (1.0 + tf.cos(tf.constant(3.141592653589793) * decay_ratio))
        cosine = self.min_lr + coeff * (self.max_lr - self.min_lr)
        
        lr = tf.where(step < self.warmup_steps,
                     warmup(),
                     tf.where(step > self.max_steps,
                               tf.constant(self.min_lr, dtype=tf.float32),
                               cosine))
        return lr