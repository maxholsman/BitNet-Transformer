import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        # convert text into tokens and then into ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # gives ids of each token in the source text
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids # gives ids of each token in the target text
        
        # pad to seq_len
        
        enc_num_pad_tokens = self.seq_len - len(enc_input_tokens) - 2 # subtract 2 for sos and eos tokens
        dec_num_pad_tokens = self.seq_len - len(dec_input_tokens) - 1 # only add eos in training
        
        if enc_num_pad_tokens < 0 or dec_num_pad_tokens < 0:
            raise Exception("seq_len is too small or sequence too long")
        
        encoder_input = torch.cat(
            [
                self.sos_token, # start of sentence token
                torch.tensor(enc_input_tokens, dtype=torch.int64), # source text
                self.eos_token, # end of sentence token
                self.pad_token.repeat(enc_num_pad_tokens) # pad tokens
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token, # start of sentence token
                torch.tensor(dec_input_tokens, dtype=torch.int64), # target text
                self.pad_token.repeat(dec_num_pad_tokens) # pad tokens, no eos token
            ]
        )
        
        translation_label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), # target text
                self.eos_token, # end of sentence token, no sos token
                self.pad_token.repeat(dec_num_pad_tokens) # pad tokens
            ]
        )
        # ensure that the tensors are of the correct shape
        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert translation_label.shape[0] == self.seq_len
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() # [1, 1, seq_len]
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask((decoder_input.size(0))) # [1, seq_len] & [1, seq_len, seq_len] -> can be broadcasted
        
        
        
        
        return {
            "encoder_input": encoder_input, # [seq_len]
            "decoder_input": decoder_input, # [seq_len]
            "encoder_mask": encoder_mask, # [1, 1, seq_len]
            "decoder_mask": decoder_mask, # [1, seq_len, seq_len]
            "translation_label": translation_label, # [seq_len]
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.ones(1, size, size, dtype=torch.int)
    mask = torch.triu(mask, diagonal=1)
    return mask == 0 # flip


        
        
        
        
        
        