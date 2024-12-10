import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    #one head of self-attention

    def __init__(self, head_size, n_embd, block_size, dropout, vocab_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.end_token = vocab_size #id of the end token

    def forward(self,x, idx):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #consider end token
        end_mask = torch.ones(B,T,T).to(device)

        for batch in range(B):
            end_positions = (idx[batch] == self.end_token).nonzero(as_tuple=True)[0] #returns position of the first occurence of an end token in each batch
            if len(end_positions)>0: #if there's an end token in the sequence
                end_mask[batch, end_positions:, :] = 0 #set all values after first end token to 0 


        #compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] * end_mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # prevents some nodes to communicates to prevent overfitting
        #weighted aggregation
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    #multiple attention heads in parallel

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size, vocab_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        out = torch.cat([h(x, idx) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # projection back into residual pathway
        return out

class FeedForward(nn.Module):
    #Linear layer + Non-linearity
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), #multiply by 4 as in the paper
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), #projection layer into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    #Transformer block: communication(attention) followed by computation (ffwd)

    def __init__(self, n_embd, n_head, dropout, block_size, vocab_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, block_size=block_size, dropout=dropout, vocab_size=vocab_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, idx):
        x = x + self.sa(self.ln1(x), idx) #normalizing and implementing residual connections 
        x = x + self.ffwd(self.ln2(x)) #normalizing and implementing residual connections
        #in the paper normalization comes after the layer but we implemented this reversed as it became more common
        return x
        



class GPTLanguageModel(nn.Module):
    
    def __init__(self, parameters):
        super().__init__()
        #load parameters
        self.block_size = parameters['block_size'] # what is the maximum context length for predictions?
        self.n_embd = parameters['n_embd'] #number of embedding dimensions
        self.n_head = parameters['n_head']
        self.n_layer = parameters['n_layer']
        self.dropout = parameters['dropout']
        self.vocab_size = parameters['vocab_size']
        #each token reads logits for next tkn from lookup table
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.ModuleList([Block(self.n_embd, n_head=self.n_head, dropout=self.dropout, block_size=self.block_size, vocab_size=self.vocab_size) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)


    def forward(self, idx, targets = None):
        B, T = idx.shape
        # idx and targets are (B, T) tensor of ints (Batch, Time, Channel)
        #in this context time represents the sequential nature of the data (block_size). Channel is representative of the logits for each token in the embedding table(vocab_size)
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C=n_embd 
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb #(B,T,C) x holds the token and positional identity
        for block in self.blocks:
            x = block(x,idx) #(B,T,C)
        x = self.ln_f(x) #final layer of normalization (B,T,C)
        logits = self.lm_head(x) #(B,T, vocab_size)

        if targets is None:
            loss = None
        else:
        #we reshape the logits and targets in order to use the cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is (B,T) array of indices in current context
        for _ in range(max_new_tokens):
            #crop idx to block_size
            idx_cond = idx[:, -self.block_size:]
            #get predictions
            logits, loss = self(idx_cond)
            #focus on last time step (last element)
            logits = logits[:,-1,:] #becomes (B,C)
            #apply softmax
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from prob distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append sampled idx to the sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    