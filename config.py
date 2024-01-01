from pathlib import Path

def get_config():
    return {
        "batch_size": 8, 
        "num_epochs":20,
        "lr": 0.0001,
        "seq_len": 500,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr", # change to german later
        "train_data_ratio": 0.9,
        "model_folder": "weights",
        "model_filename": "transfomer_model_", 
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/transformer_model",
        "datasource": 'opus_books',
    }

def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = model_basename + str(epoch) + ".pt"
    return str(Path('.') / model_folder / model_filename)
