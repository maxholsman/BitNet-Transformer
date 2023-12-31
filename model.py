import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #simply a mapping between a number and a vector, number represents a word in the vocabulary and the vector represents the word in the embedding space
        
    def forward(self, x):
        # print(f"x type: {x.type()}")
        x = self.embedding(x)
        # print(f"x embedded type: {x.type()}")
        return x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) # by paper

class PositionalEncoding(nn.Module):
    # Positional Encoding is a way of encoding the position of the word in the sentence. They will be added to the embedding vector to give the model a sense of the word's position in the sentence.
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        
        #first will build a matrix of shape (seq_len, d_model) because we need seq_len number of positional encodings for each word in the sentence
        pe = torch.zeros(seq_len, d_model)
        
        # apply one formula to all the even indices of the matrix, and another formula to all the odd indices of the matrix
        # will therefore create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # unsqueeze to get shape (seq_len, 1)
        divterm = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)) # shape (d_model/2)
        # sin for even, cosine for odd
        # self.pe[:, 0::2] = torch.sin(position * divterm)
        # self.pe[:, 1::2] = torch.cos(position * divterm)
        # for i in range(0, d_model, 2):
        #     self.pe[:, i] = torch.sin(position * divterm[i])
        #     self.pe[:, i+1] = torch.cos(position * divterm[i])
        
        pe[:, 0::2] = torch.sin(position * divterm)
        pe[:, 1::2] = torch.cos(position * divterm)
                
        # add a batch dimension to the positional encoding matrix so that it can be added to the embedding vector
        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model) which now has a batch dimension
        
        # now we will save this positional encoding matrix as a buffer so that it can be used later
        self.register_buffer('pe', pe) # keeping this tensor NOT as a parameter but still as a part of the model, tensor will now be saved in the file with the model
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # add positional encoding to the embedding vector. sp
        x = self.dropout(x)
        return x

# for each item in batch, calculate mean and variance indpenendent of other items in the batch, then find new values using their own mean and variance
# also add and multiply two additional parameters, to give the model the ability to amplify or reduce the importance of a given feature

class LayerNorm(nn.Module): 
    def __init__(self, d_model, eps=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # alpha is a learnable parameter and will be multiplied
        self.beta = nn.Parameter(torch.zeros(d_model)) # beta is a learnable parameter and will be added
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # mean of each item in the batch, not the whole batch
        std = x.std(dim=-1, keepdim=True) # same thing here
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FFBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        self.w1 = nn.Linear(d_model, d_ff) #includes bias
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # input: [batch, seq_len, d_model]
        x = self.relu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % num_heads == 0, "d_model not divisible by num_heads" # d_model must be divisible by num_heads
        
        self.d_k = d_model // num_heads # d_k is the dimension of each head
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        
        self.w0 = nn.Linear(d_model, d_model, bias=False)
    
    @staticmethod
    def attention(q, k, v, mask=None, dropout=None):
        d_k = q.shape[-1]
        
        att = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32)) # [batch, num_heads, seq_len, seq_len], only transpose last two dimensions beause want each head dimension to stay the same
        
        if mask is not None:
            att.masked_fill_(mask == 0, -1e9)
        
        # att = torch.softmax(att, dim=-1)
        att = att.softmax(dim=-1)
        
        if dropout is not None:
            att = dropout(att)
        
        att_values = torch.matmul(att, v)
        
        return att_values, att
        

    def forward(self, q, k, v, mask=None):
        
        # print datatypes of all inputs
        # print(f"q type: {q.type()}")
        # print(f"k type: {k.type()}")
        # print(f"v type: {v.type()}")
        # print(f"mask type: {mask.type()}")
        
        qprime = self.wq(q) # [batch, seq_len, d_model]
        kprime = self.wk(k)
        vprime = self.wv(v)
        
        # split qprime, kprime, vprime into num_heads
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        qprime = qprime.view(qprime.shape[0], qprime.shape[1], self.num_heads, self.d_k)
        kprime = kprime.view(kprime.shape[0], kprime.shape[1], self.num_heads, self.d_k)
        vprime = vprime.view(vprime.shape[0], vprime.shape[1], self.num_heads, self.d_k)
        
        # transpose qprime, kprime, vprime to [batch, num_heads, seq_len, d_k] so that each head has full access to the sequence
        qprime = qprime.transpose(1, 2)
        kprime = kprime.transpose(1, 2)
        vprime = vprime.transpose(1, 2)
        
        # calculate attention
        x, self.attention = MultiHeadAttention.attention(qprime, kprime, vprime, mask, self.dropout)  
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # [batch, seq_len, d_model]
        
        out = self.w0(x)
        
        return out
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = LayerNorm(d_model) # initialized from class defined above
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.ln(x))) # takes care of both the residual connection and the layer normalization, sublayer will be either the multihead attention or the feed forward block

#now we must construct the entire encoder block

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        self.residual_connection = ResidualConnection(d_model, dropout)
        self.multiheadattention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffblock = FFBlock(d_model, d_ff, dropout)
        
    def forward(self, x, src_mask):
        x = self.residual_connection(x, lambda x: self.multiheadattention(x, x, x, src_mask))
        x = self.residual_connection(x, self.ffblock)
        # print(f"output type inside encoder block: {x.type()}")
        return x
    
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        for i, encoder_layer in enumerate(self.encoder):
            x = encoder_layer(x, src_mask)
            # print(f"output type inside encoder after block {i+1}: {x.type()}")
        # print(f"output type inside encoder: {x.type()}")
        return x
    
# class Encoder(nn.Module):
#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers
#         self.norm = LayerNorm()
        
#     def forward(self, x, src_mask):
#         for layer in self.layers:
#             x = layer(x, src_mask)
#         return self.norm(x)

class OutputEmbedding(nn.Module): # same as input embedding
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #simply a mapping between a number and a vector, number represents a word in the vocabulary and the vector represents the word in the embedding space
        
    def forward(self, x):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32)) # by paper


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        self.residual_connection = ResidualConnection(d_model, dropout)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffblock = FFBlock(d_model, d_ff, dropout)
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        #src_mask is the mask for the encoder output, target_mask is the mask for the decoder output
        x = self.residual_connection(x, lambda x: self.self_attention(x, x, x, target_mask)) # self attention with decoder output
        
        # print(f"x type: {x.type()}")
        # print(f"encoder output type: {encoder_output.type()}")
        
        x = self.residual_connection(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection(x, self.ffblock)
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        self.decoder = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, encoder_output, src_mask, target_mask):
        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x, encoder_output, src_mask, target_mask)
            # print(f"output type inside decoder after block {i+1}: {x.type()}")
        return x

#need to map output back into the vocabulary/dictionary

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        #applying log softmax for numerical stability
        return self.projection(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, input_embedding, input_positional_embedding, output_embedding, output_positional_embedding, projection_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.input_positional_embedding = input_positional_embedding
        self.output_positional_embedding = output_positional_embedding
        self.projection_layer = projection_layer
    
    def encode(self, x, src_mask):
        x = self.input_embedding(x)
        x = self.input_positional_embedding(x)
        x = self.encoder(x, src_mask)
        return x

    def decode(self, x, encoder_output, src_mask, target_mask):
        x = self.output_embedding(x)
        x = self.output_positional_embedding(x)
        # print(f"encoder output inside decode: {encoder_output.type()}")
        x = self.decoder(x, encoder_output, src_mask, target_mask)
        return x

    def project(self, x):
        x = self.projection_layer(x)
        return x


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, num_heads=8, dropout=0.1, d_ff=2048):
    # first create embedding and positional encoding layers
    input_embedding = InputEmbedding(d_model, src_vocab_size)
    input_positional_embedding = PositionalEncoding(d_model, src_seq_len, dropout)
    output_embedding = OutputEmbedding(d_model, tgt_vocab_size)
    output_positional_embedding = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # next create encoder and decoder
    encoder = Encoder(d_model, num_heads, d_ff, N, dropout)
    decoder = Decoder(d_model, num_heads, d_ff, N, dropout)
    
    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) #bc projecting into target vocab
    
    # Create transformer
    transformer = Transformer(encoder, decoder, input_embedding, input_positional_embedding, output_embedding, output_positional_embedding, projection_layer)
    
    #initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # initialize weights with xavier uniform distribution to enable faster training that using random weights
    
    return transformer