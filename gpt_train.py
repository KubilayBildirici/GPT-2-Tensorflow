from dataclasses import dataclass
import tensorflow as tf
from transformers import TFGPT2LMHeadModel

from optim.gpu_config import setup_strategy
from optim.custom_optimizer import CustomAdamW
from optim.custom_LR import WarmupCosineSchedule
from configs.config import load_config
import numpy as np
from data.fineweb_tf import tf_dataset
from model import GPT
from datetime import datetime
import os

tf.random.set_seed(1337)
np.random.seed(1337)


# Model Config
model_config, train_config = load_config(
    "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\configs\\gpt2_.json",
    "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\configs\\training.yaml"
)

#print("Model Config:", model_config)
#print("Training Config:", train_config)


strategy, num_processes, ddp_rank, ddp_local_rank, is_chief, device = setup_strategy()

# attempt to autodetect the device
import time

lr_schedule = WarmupCosineSchedule(train_config["max_lr"], train_config["min_lr"], train_config["warmup_steps"], train_config["max_steps"])


with strategy.scope():
    train_ds = tf_dataset("C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\data\\edu_fineweb10B",batch_size=train_config["batch_size"],
                          seq_len=train_config["sequence_length"], split="train")
    
    val_ds = tf_dataset("C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\data\\edu_fineweb10B",batch_size=train_config["batch_size"],
                        seq_len=train_config["sequence_length"], split="val")
    
    model = GPT(model_config)
    
    optimizer = CustomAdamW(learning_rate=lr_schedule,
                        weight_decay=train_config["weight_decay"],
                        beta_1=0.9,
                        beta_2=0.95,
                        epsilon=1e-8,
                        exclude_from_weight_decay=["bias","LayerNorm","layer_norm","norm"])

    
tf.config.experimental.enable_tensor_float_32_execution(True) # matmul/conv tf32 precision
 
x_dummy = tf.ones((1, train_config["sequence_length"]), dtype=tf.int32)
y_dummy = tf.ones((1, train_config["sequence_length"]), dtype=tf.int32)
_ = model(x_dummy, y_dummy)

optimizer.summarize_parameters(model)

# checkpoint system
ckpt_dir = "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
print("Checkpoint system ready at", ckpt_dir)


@tf.function(jit_compile=True) # XLA compilation
def train_step(x,y):
    with tf.GradientTape() as tape:
        logits, loss = model(x, y, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    grads, global_norm = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #optimizer.summarize_parameters(model)
    return loss, global_norm

train_ds = train_ds.repeat()
val_ds = val_ds.repeat()

train_iter = iter(train_ds)
val_iter = iter(val_ds)

# training Loop
for step in range(train_config["max_steps"]):
    #validation
    if step % 100 == 0:
        val_losses = []
        for _ in range(10): # 10 validation steps
            x_val, y_val = next(val_iter)
            _, val_loss = model(x_val, y_val, training=False)
            val_losses.append(val_loss.numpy())
        print(f"step {step}, val loss: {np.mean(val_losses):.4f}")

    t0 = time.time()
    x, y = next(train_iter)        
    loss, norm = train_step(x,y)
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_config["batch_size"] * train_config["sequence_length"] * strategy.num_replicas_in_sync) / (t1 - t0)
    lr = lr_schedule(step).numpy()
    print(f"step {step}, lr={lr:.6f}, loss={loss.numpy():.4f}, norm:{norm.numpy():.4f} time={dt:.2f}ms, tok/sec={tokens_per_sec:.2f}")
    
    # checkpointing
    if step % 10000 == 0 and step > 0:
        ckpt_path = ckpt_manager.save()
        print(f"Saved checkpoint to {ckpt_path}")
    
    # Generate text sample every 1000 steps
    if step % 10000 == 0 and step > 0:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        
        prompt = "Hello, I am a language model"
        tokens = enc.encode(prompt)
        x_gen = tf.convert_to_tensor([tokens], dtype=tf.int32)  # (1, T)

        max_length = 30
        top_k = 50
        while tf.shape(x_gen)[1] < max_length:
            # forward the model to get the logits
            logits, _ = model(x_gen, training=False)
            logits_last = logits[:, -1, :]

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
            x_gen = tf.concat([x_gen, sampled_idx], axis=1)

        # print the generated text
        base_dir = "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\generated_text"
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"generated_step{step}.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            num_return_sequences = 5
            for i in range(num_return_sequences):
                x_gen = tf.convert_to_tensor([tokens], dtype=tf.int32)  # reset to prompt
                while tf.shape(x_gen)[1] < max_length:
                    logits, _ = model(x_gen, training=False)
                    logits_last = logits[:, -1, :]

                    probs = tf.nn.softmax(logits_last, axis=-1)
                    topk_probs, topk_indices = tf.math.top_k(probs, k=top_k)

                    sampled = np.random.choice(
                        topk_indices[0].numpy(),
                        p=(topk_probs[0].numpy() / np.sum(topk_probs[0].numpy()))
                    )
                    x_gen = tf.concat([x_gen, [[sampled]]], axis=1)
                
                decoded = enc.decode(x_gen[0].numpy().tolist())
                f.write(f"\n=== Generated Sequence {i+1} ===\n")
                f.write(decoded + "\n")
                
        print(f"Generated text samples saved to {output_path}")

## Saved Fianl Model
final_model_save_dir = "C:\\Users\\kubilay\\PycharmProjects\\gpt-2-124M\\GPT2_Model.keras"
model.save(final_model_save_dir)
print(f"Final model saved to {final_model_save_dir}")
            

                                   

