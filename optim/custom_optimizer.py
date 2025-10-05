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
        var_name = variable.name
        for name in self.exclude_from_weight_decay:
            if name in var_name:
                return False
        return True
    
    def _decay_weights_op(self, var):
        if self._use_weight_decay(var):
            return super()._decay_weights_op(var)
        return tf.constant(0.0)
    
    def summarize_parameters(self, model):
        decay_vars = [v for v in model.trainable_variables if self._use_weight_decay(v)]
        nodecay_vars = [v for v in model.trainable_variables if not self._use_weight_decay(v)]
        
        num_decay = sum (int(tf.size(v)) for v in decay_vars)
        num_nodecay = sum (int(tf.size(v)) for v in nodecay_vars)
        
        print(f"num decayed parameter tensors: {len(decay_vars)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_vars)}, with {num_nodecay:,} parameters")
