from dataclasses import dataclass
import tensorflow as tf
from transformers import TFGPT2LMHeadModel


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self,config, name="attn"):
        super().__init__(name=name)
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attn_pdrop  = getattr(config, "attn_pdrop", 0.1)
        self.resid_pdrop = getattr(config, "resid_pdrop", 0.1)
        
        self.c_attn = tf.keras.layers.Dense(3 * config.n_embd,name="c_attn",use_bias=True,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                            bias_initializer="zeros") # q,k,v linear projection
        
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj", use_bias=True,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                            bias_initializer="zeros") # output projection
        
        self.attn_drop = tf.keras.layers.Dropout(self.attn_pdrop)
        self.resid_drop = tf.keras.layers.Dropout(self.resid_pdrop)
    
    def call(self, x, training=False):
        # batch,time,channel = batch,seq_len,embd_dim
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        # QKV: (B,T,3C) -> split -> (B,T,C) x3
        qkv = self.c_attn(x) 
        qkv = tf.reshape(qkv,(B,T,self.n_head, 3*self.head_dim))
        qkv = tf.transpose(qkv, [0,2,1,3])
        
        q, k, v = tf.split(qkv,3, axis=-1)
        
        # attention score
        att = tf.matmul(q, k, transpose_b=True) / tf.cast(self.head_dim ** 0.5, q.dtype)

        # causal mask (upper triangle = -inf)
        mask = tf.linalg.band_part(tf.ones((T,T)),-1,0) # lower trianguler
        mask = tf.cast(mask, att.dtype)  
        att = att * mask - 1e9 * (1.0 - mask)
        
        att = tf.nn.softmax(att, axis=-1)
        att = self.attn_drop(att, training=training)
        
        # weighted sum
        out = tf.matmul(att,v)
        out = tf.transpose(out,[0,2,1,3])
        out = tf.reshape(out,(B,T,C))
        
        out = self.c_proj(out)
        out = self.resid_drop(out, training=training)

        return out

class MLP(tf.keras.layers.Layer):
    def __init__(self,config,name="mlp"):
        super().__init__(name=name)
        self.resid_drop = getattr(config, "resid_drop", 0.1)
        
        self.c_fc = tf.keras.layers.Dense(4 * config.n_embd, name="c_fc", use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                          bias_initializer="zeros")
        
        self.gelu = tf.keras.layers.Activation("gelu")
        
        self.c_proj = tf.keras.layers.Dense(config.n_embd, name="c_proj", use_bias=True,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                            bias_initializer="zeros")
        self.drop = tf.keras.layers.Dropout(self.resid_drop)
    
    def call(self,x,training=False):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.drop(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self,config,name):
        super().__init__(name=name)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5,name="ln_1")
        self.attn = CausalSelfAttention(config, name="attn")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5,name="ln_2")
        self.mlp = MLP(config, name="mlp")
    
    def call(self,x,training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


@dataclass
class GPTConfig:
     block_size: int = 1024 # max sequence length
     vocab_size: int = 50257 # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <|endoftext| > token
     n_layer: int = 12 # number of layers
     n_head: int = 12 # number of heads
     n_embd: int = 768 # embedding dimension
     dropout: float = 0.1
     bias: bool = True


class GPT(tf.keras.Model):
    def __init__(self,config,training=False):
        super().__init__(name="transformer")
        self.config = config
        self.wte = tf.keras.layers.Embedding(config.vocab_size, config.n_embd, name="wte",
                                             embeddings_initializer=tf.random_normal_initializer(stddev=0.02))
        
        self.wpe = tf.keras.layers.Embedding(config.block_size, config.n_embd, name="wpe",
                                             embeddings_initializer=tf.random_normal_initializer(stddev=0.02))
        
        self.h = [Block(config, name=f"h_._{_}") for _ in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5,name="ln_f")
        #self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False,name="lm_head")
        
        # weight sharing scheme
        # self._tied = False
    
    def get_input_embeddings(self):
        return self.wte

    def get_output_embeddings(self):
        return self.wte  ## output embeddings == input embeddings
    
    
    def call(self,idx,targets=None):
        # idx is of shape (B, T)
        B, T = tf.shape(idx)[0], tf.shape(idx)[1]
        tf.debugging.assert_less_equal(
            T, self.config.block_size,
            message=f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        )
        
        #assert T <= self.config.block_size,f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        
        pos = tf.range(0,T) # (T,)
        pos_emb = self.wpe(pos) # (T,n_embd)
        tok_emb = self.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) (broadcast)
        
        for block in self.h:
            x = block(x)
            
        x = self.ln_f(x)  # (B, T, n_embd)
        
        #logits = self.lm_head(x) # (B, T, vocab_size)
        
        W = (self.wte.embeddings
             if hasattr(self.wte,"embeddings")
             else self.wte.weights[0])
        logits = tf.matmul(x,W, transpose_b=True)
        
        loss = None
        if targets is not None:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(targets,logits)
        return logits,loss

    @classmethod
    def from_pretrained(cls,model_type="gpt2",override_args=None):
        ## Loads pretrained GPT-2 model weights
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        
        # HuggingFace defaualt config
        config_map = {
            "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), 
            "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        
        config_map["vocab_size"] = 50257
        config_map["block_size"] = 1024
        config_map["bias"] = True
        
        if "dropout" in override_args:
            config_map["dropout"] = override_args["dropout"]
        
        config = GPTConfig(**config_map)
        
        model = cls(config)
        hf_model = TFGPT2LMHeadModel.from_pretrained(model_type,from_pt=False)
        print(f"Loaded HuggingFace {model_type} with {hf_model.num_parameters()} params")
        
        dummy_inpt = tf.zeros((1,1), dtype=tf.int32)
        model(dummy_inpt)
        
        # weights transfer
        for w, w_hf in zip(model.weights,hf_model.weights):
            if w.shape == w_hf.shape:
                w.assign(w_hf)
            elif (len(w.shape) == 1 and
                len(w_hf.shape) == 2 and
                w.shape[0] == w_hf.shape[1]):  
                # Hugging Face bias (1, dim) → bizim bias (dim,)
                w.assign(tf.squeeze(w_hf, axis=0))
                print(f"Reshaped bias: {w.name}")
            else:
                print(f"⚠️ Shape mismatch: {w.name} vs {w_hf.name}, skipping")
        
        return model

# DataLoader
import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('gpt-2-124M\input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = tf.constant(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")
        
        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = tf.reshape(buf[:-1],(B,T))
        y = tf.reshape(buf[1:],(B,T))
        self.current_position += B*T
        if self.current_position + (B * T +1) > len(self.tokens):
            self.current_position = 0
        return x,y

# attempt to autodetect the device
import time


train_loader = DataLoaderLite(B=4, T=256)
tf.config.experimental.enable_tensor_float_32_execution(True) # matmul/conv tf32 precision 

from gpu_config import setup_strategy
strategy, world_size, _, _, _, device = setup_strategy()

with strategy.scope():
    #tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    model = GPT(GPTConfig(vocab_size=50304))

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

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
lr_schedule = WarmupCosineSchedule(max_lr, min_lr, warmup_steps, max_steps)

# optimizer
optimizer = CustomAdamW(learning_rate=lr_schedule,
                        weight_decay=0.1,
                        beta_1=0.9,
                        beta_2=0.95,
                        epsilon=1e-8,
                        exclude_from_weight_decay=["bias","LayerNorm","layer_norm","norm"])

@tf.function(jit_compile=True) # XLA compilation
def train_step(x,y):
    with tf.GradientTape() as tape:
        logits, loss = model(x, y, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, global_norm = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, global_norm

# training Loop
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()        
    loss, norm = train_step(x,y)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    lr = lr_schedule(step).numpy()
    print(f"step {step}, lr={lr:.6f}, loss={loss.numpy():.4f}, norm:{norm.numpy():.4f} time={dt:.2f}ms, tok/sec={tokens_per_sec:.2f}")

    

tf.random.set_seed(42)
import numpy as np
np.random.seed(42)

#prompt = "Hello, I am a language model"
#tokens = enc.encode(prompt)
#x_gen = tf.convert_to_tensor([tokens], dtype=tf.int32)  # (1, T)

#max_length = 30
#top_k = 50
"""
while tf.shape(x)[1] < max_length:
    # forward the model to get the logits
    logits = model(x_gen, training=True)
    logits_last = logits[:, -1, :]
    import sys; sys.exit(0)
    
    # softmax
    probs = tf.nn.softmax(logits_last, axis=-1)
    # top-k sampling
    topk_probs, topk_indices = tf.math.top_k(probs, k=top_k)
    
    sampled_idx = []
    for b in range(probs.shape[0]):
        sampled = np.random.choice(
            topk_indices[b].numpy(),
            p=(topk_probs[b].numpy() / np.sum(topk_probs[b].numpy()))
        )
        sampled_idx.append(sampled)
    
    sampled_idx = tf.convert_to_tensor(sampled_idx, dtype=tf.int32)
    sampled_idx = tf.expand_dims(sampled_idx, axis=1)
    x = tf.concat([x, sampled_idx], axis=1)

# print the generated text
num_return_sequences = 5
for i in range(num_return_sequences):
    tokens = x_gen[0, :max_length].numpy().tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
"""

