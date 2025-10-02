import tensorflow as tf

class CustomAdamW(tf.keras.optimizers.AdamW):
    def __init__(self, exclude_from_weight_decay=None, **kwargs):
        super().__init__(**kwargs)
        if exclude_from_weight_decay is None:
            exclude_from_weight_decay = ["bias", "LayerNorm", "layer_norm", "norm"]
        self.exclude_from_weight_decay = exclude_from_weight_decay
        
    def _use_weight_decay(self, variable):
        if not self.weight_decay:
            return False
        var_name = getattr(variable, "name", "")
        for name in self.exclude_from_weight_decay:
            if name in var_name:
                return False
        return True
    
    def _decay_weights_op(self, var):
        if self._use_weight_decay(var.name):
            return super()._decay_weights_op(var)
        return tf.no_op()