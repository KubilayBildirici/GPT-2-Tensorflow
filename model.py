import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from configs.config import load_config
import keras

# Model Config
model_config, train_config = load_config(
    "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\configs\\gpt2_.json",
    "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\configs\\training.yaml"
)


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self,config, name="attn"):
        super().__init__(name=name)
        assert config["n_embd"] % config["n_head"] == 0
        self.config = config
        self.n_head = config["n_head"]
        self.head_dim = config["n_embd"] // config["n_head"]
        self.attn_pdrop  = config.get("attn_pdrop", 0.1)
        self.resid_pdrop = config.get("resid_pdrop", 0.1)
        
        self.c_attn = tf.keras.layers.Dense(3 * config["n_embd"],name="c_attn",use_bias=True,
                                            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                            bias_initializer="zeros") # q,k,v linear projection
        
        self.c_proj = tf.keras.layers.Dense(config["n_embd"], name="c_proj", use_bias=True,
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
        self.resid_drop = config.get("resid_pdrop", 0.1)
        
        self.c_fc = tf.keras.layers.Dense(4 * config["n_embd"], name="c_fc", use_bias=True,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                          bias_initializer="zeros")
        
        self.gelu = tf.keras.layers.Activation("gelu")
        
        self.c_proj = tf.keras.layers.Dense(config["n_embd"], name="c_proj", use_bias=True,
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

@keras.saving.register_keras_serializable(package="model")
class GPT(tf.keras.Model):
    def __init__(self,config,training=False):
        super().__init__(name="transformer")
        self.config = config
        self.wte = tf.keras.layers.Embedding(config["vocab_size"], config["n_embd"], name="wte",
                                             embeddings_initializer=tf.random_normal_initializer(stddev=0.02))
        
        self.wpe = tf.keras.layers.Embedding(config["block_size"], config["n_embd"], name="wpe",
                                             embeddings_initializer=tf.random_normal_initializer(stddev=0.02))
        
        self.h = [Block(config, name=f"h_._{_}") for _ in range(config["n_layer"])]
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
            T, self.config["block_size"],
            message=f"Cannot forward sequence of length {T}, block size is {self.config['block_size']}"
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
        
        config = model_config(**config_map)
        
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