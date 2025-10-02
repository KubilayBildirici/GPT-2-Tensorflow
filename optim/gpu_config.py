import tensorflow as tf

def setup_strategy():
    # setup memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Physical GPUs: {gpus}")
    for gpu in gpus:
        try:
            print(f"available GPU: {len(gpu)}")
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            print(f'Cannot set memory growth for {gpu}')
    
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        device = "/GPU:0"
        ddp = True
        ddp_rank = 0
        ddp_local_rank = 0
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        device = "/GPU:0"
        ddp = False
        ddp_rank = 0
        ddp_local_rank = 0
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        device = "/CPU:0"
        ddp = False
        ddp_rank = 0
        ddp_local_rank = 0
    
    num_processes  = strategy.num_replicas_in_sync
    is_chief = True # tek node
    print(f"[TF-DDP] world_size={num_processes} | device={device}")
    return strategy, num_processes , ddp_rank, ddp_local_rank, is_chief, device