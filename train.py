import torch
import torch.nn as nn
import torch.optim as optim

import os

import tensorboard
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm

from model import Transformer

from config import get_config, get_weights_file_path

from pathlib import Path

from dataset import TranslationDataset, causal_mask
from model import build_transformer

def greedy_inference_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")
    
    # precompute encoder output
    
    encoder_output = model.encode(source, source_mask)
    
    decoder_input = torch.tensor([[sos_idx]]).type_as(source).to(device) # [1, 1], initialize decoder input with sos token
    while True:
        if decoder_input.shape[1] >= max_len:
            break
        
        # create mask for target
        decoder_mask = causal_mask(decoder_input.shape[1]).type_as(source_mask).to(device) 
        
        # calc output
        output = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
        
        # get next token
        probabilities = model.project(output[:, -1]) # only want projection for last token
        
        _, next_word = torch.max(probabilities, dim=1) # greedy sampling of next token
        
        # append new word onto decoder input for next iteration through decoder
        decoder_input = torch.cat(
            [decoder_input, 
             torch.tensor([[next_word.item()]]).type_as(source).to(device)
             ], dim=1)
        
        if next_word == eos_idx:
            break
    
    return decoder_input.squeeze(0) # remove batch dimension    

def run_validation(model, validation_dataset, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2):
    model.eval() # set model to evaluation mode
    
    # inference two sentences to see what output of model would be
    
    source_texts = []
    target_texts = []
    predicted_text = []
    
    console_width = 80
    
    count = 1 
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            
            # validation dataset only has batch size of 1, but lets check anyhow
            
            encoder_input = batch['encoder_input'].to(device) # [batch_size, seq_len]
            encoder_mask = batch['encoder_mask'].to(device) # [batch_size, 1, 1, seq_len]
            
            assert encoder_input.shape[0] == 1, "Validation dataset must have batch size of 1"
            
            model_output = greedy_inference_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy()) # convert tensor to list, then decode using tokenizer
            
            source_texts.append(source_text)
            target_texts.append(target_text)
            predicted_text.append(model_output_text)
            
            # print on console
            
            print_msg('-' * console_width)
            print_msg(f'Source: {source_text}')
            print_msg(f'Target: {target_text}')
            print_msg(f'Predicted: {model_output_text}')
            
            if count >= num_examples:
                print_msg('-' * console_width)
                break
            
    if writer:
        # TorchMetrics for CharrErrorRate, BLEU, WordErrorRate, look into these later
        pass      

def get_sentences(dataset, language):
    for sentence_pair in dataset:
        yield sentence_pair['translation'][language]
        
def get_or_build_tokenizer(dataset, config, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language)) # Path to save tokenizer
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace() # split by whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) # words must appear at least twice in order to be added to the vocabulary
        tokenizer.train_from_iterator(get_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_datset(config):
    raw_dataset = load_dataset(f"{config['datasource']}", f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
    
    tokenizer_src = get_or_build_tokenizer(raw_dataset, config, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(raw_dataset, config, config['lang_tgt'])
    
    # split into training and validation sets
    train_data_size = int(len(raw_dataset) * config['train_data_ratio'])
    val_data_size = len(raw_dataset) - train_data_size
    train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(raw_dataset, [train_data_size, val_data_size])
    
    train_dataset = TranslationDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = TranslationDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # want to check maximum sequence length of both source and target to choose correct seq_len
    
    max_len_src = 0
    max_len_tgt = 0
    
    for pair in raw_dataset:
        src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(pair['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Maximum sequence length for source language {config['lang_src']}: {max_len_src}")
    print(f"Maximum sequence length for target language {config['lang_tgt']}: {max_len_tgt}")
    
    # create dataloaders 
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True) # want batch size of 1 for validation set
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    if config['transformer_type'] == 'vanilla':
        model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    if config['transformer_type'] == 'bitnet':
        model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # device definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # create weights folder if it doesn't exist
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
    # get dataloaders
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_datset(config)
    
    # get model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # set up tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        
    # loss function is cross entropy loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device) # don't want padding tokens to contribute to loss, using label smoothing to prevent overfitting
    
    # training loop
    
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch}")
        
        for i, batch in enumerate(batch_iterator):
            encoder_input = batch['encoder_input'].to(device) # [batch_size, seq_len]
            decoder_input = batch['decoder_input'].to(device) # [batch_size, seq_len]
            encoder_mask = batch['encoder_mask'].to(device) # [batch_size, 1, 1, seq_len]
            decoder_mask = batch['decoder_mask'].to(device) # [batch_size, 1, seq_len, seq_len], different because in one case hiding paddings, in other case hiding future tokens and paddings
            
            # pass tensors through model
            encoder_output = model.encode(encoder_input, encoder_mask) # [batch_size, seq_len, d_model]
            # print(f"encoder output type in train.py: {encoder_output.type()}")
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # [batch_size, seq_len, d_model]
            output = model.project(decoder_output) # [batch_size, seq_len, vocab_tgt_len]
            
            label = batch['translation_label'].to(device) # [batch_size, seq_len]
            
            loss = loss_fn(output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # logging
            batch_iterator.set_postfix({"loss": loss.item()})
            writer.add_scalar("loss", loss.item(), global_step)
            writer.flush()
            
            # backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
            if i % 100 == 0:
                with open(f"{config['datasource']}_{config['model_folder']}/loss.csv", 'a') as f:
                    f.write(f"{epoch},{i},{loss.item()}\n")
              
        # save loss to csv file
        with open(f"{config['datasource']}_{config['model_folder']}/loss.csv", 'a') as f:
            f.write(f"{epoch},{i}{loss.item()}\n")
        
        # validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # save model
        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    config = get_config()
    train_model(config)

            
            
            
            
            
            
            
    
    
        

    
        
    