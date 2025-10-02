import json, yaml

def load_config(model_path: str, train_path: str):
    with open(model_path, 'r') as f:
        model_config = json.load(f)
        
    with open(train_path, 'r') as f:
        train_config = yaml.safe_load(f)
    
    for key in ["max_lr", "min_lr", "weight_decay"]:
        if key in train_config:
            train_config[key] = float(train_config[key])
        for key in ["warmup_steps", "max_steps", "batch_size", "sequence_length"]:
            if key in train_config:
                train_config[key] = int(train_config[key])

    return model_config, train_config