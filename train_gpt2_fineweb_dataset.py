from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import tiktoken
import inspect
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from hellaswag import render_example, iterate_examples

# ----------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length , embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "headsize", and C (number of channels) = nh * ns
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*ns=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        
        
        # att = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = (att@v).transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # we are using flash attention: Flashattention is a way to quickly calculate attention operations
        # instead of the above line of code, flash attention runs the above four lines really quick
        # Flash attention is a KERNEL FUSION operation, it is very mindful of the memory hierarchy
        # Flash attention does not read and write the large NxN attention matrix to HBM, resulting
        # 7.6x faster on attention computation
        # Trick: Online softmax calculation:
        # Flashattention-2 is better (based on a paper called `Online Normalizer Calculation for Softmax` from Nvidia)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash attention is does by pytorch
        
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re assemble all outputs side by side
        # output projection 
        y = self.c_proj(y)
        return y

# class TanhGELU(nn.Module):
#     """
#     just algorithmically typed out gelu. Why this?
#     If you run this with torch.compile then;
#     first there will be kernel that will raise to the third power `torch.pow(input, 3.0)`
#     and when this kernel runs after that the result is stored back in the memory of the GPU
#     So the travel time (memory access time) causes lot of issues.
#     So then we dispatch another kernel to do 0.044715 * torch.pow(input, 3.0) multiplication
#     so it takes the constant travels to GPU, performs addition and then stores back to global memory
#     So we make a lot of round trips, and PyTorch without `.compile()` doesnot know how to optimize
#     this.
#     So with pytorch compile it will take all the inputs at one time does all the computation 
#     and save the values to global memory.
#     So we are not using it
#     """
#     def forward(self, input):
#         return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) *  (input + 0.044715 * torch.pow(input, 3.0))))



# There are operations that `torch.compile` wont understand.
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # Why approximate version of Gelu
        # At the time when they developed this non linearity the erf function which you need to 
        # evaluate the exact GELU was very slow in tensorflow, so this approximation that ended
        # up being used by Bert2 and other. approximate version is as good as exact version
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        In GPT the layer normalizations are after the application 
        of attention or feed forward and in addition to that note
        the normalizations are inside the residual streams. That means
        your residual pathway has normalization inside them. This is 
        not very good, you would want to have clean residual stream 
        supervision all the way down to the inputs the tokens because the
        gradients that flow from the top distributes gradients during
        the backwards state to both of its branches equally. So addition
        is a brach in the gradients, So the gradients from the top flow 
        straight to the inputs tokens through the pathways unchanged, 
        but in addition to the that the gradient also flows through
        the blocks. And the blocks contribute theour own contribution
        overtime and kick in and change the optimization over time.
        So basically clean residual pathway is desirable from an optimi-
        -zation standpoint, hence the pre-normalization version
        """
        x = x + self.attn(self.ln_1(x)) # This shit where they communicate
        x = x + self.mlp(self.ln_2(x)) # This is where they think individually about the information they gathered
        # and everyone of the  
        return x
    
    """
    Attention is a communication operation, it is where all
    the tokens and there's 1024. So attention is an aggregation
    function/pooling function, ir is a weighted sum function, whereas 
    MLP, happens at every single token individually, there is no
    information being exchanged or collected between the tokens
    So the attention is the reduce and MLP is the map. Hence transformer
    is just and repeated application of MapReduce
    """

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens
    n_layer: int = 12 # number of layers 
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension 

# we want to match up the config of architecture from hugging face so that 
# we can load the weights from the same state dict 

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # transformer is the main container that has all the modules 
        # so we are reflecting that with `nn.Module` dict 
        # basically module that allows you to index into the sub modules 
        # using keys just like dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # weights of the token embedding
                                                                # embedding is just a wrapper module around a single array of numbers or Tensor
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of the position embedding
            # we have something called .h which is index using numbers instead of using strinds
            # somethings like this `transformer.h.0.ln_1.weight torch.Size([768])`.
            # from `.h.0` all the way upto `.h.11` thats because there are 12 layers
            # instead of module dict we have a `ModuleList` so that we can index it using 
            # integers, and the module list has n layer blocks and the block are yet to 
            # defined in another module
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            # In gpt-2 paper we have an additional final layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # then we had the final classifier, which projects from 768, the number of embedding
        # dimensions in this GPT all the way to the vocab size which is 50257
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme because input and output embeddings should be same
        # So this tensor and its going to be used twice in the forward pass
        # Also this is a ton of params, 768x50257=38597376, and this is 124M Params model
        # out of which 40M is this embedding, so now as this 40M is shared, you donot have to train
        # as long. Becomes more efficient
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        We will iterate through all the modules, if the module is nn.Linear
        than we will intialize the weight with the standard deviation of 0.02
        if there're a bias in this layer we will make sure to intialize that to
        zero note that zero initilization for the bias is not actually the pytorch
        default the bias here is initialized with a uniform.

        Typically the standard deviation on this initialization is xavier intialization
        But here you will see 0.02 is a bit consistent because transformer for gpt2
        are roughly 768 to 1600 (the d_model), 1/sqrt(768)=0.03 and 1/sqrt(1600) is 0.025
        and 1/sqrt(3*1600) = 0.0144 

        From the GPT-2 Paper: A modified initialization which accounts for the
        accumulation on the residual path with the model depth is used. We scale the
        weights of the residual layers at initialization by a factor of 1/sqrt(N)\
        where N is the number of residual layers
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std += (2*self.config.n_layer)**-0.5 # I am sclaing down the standard deviation
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shapr (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
        # forward the token and position embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the block of transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

    # load parameters from hugging face
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from hugging face"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768), # 124M Params
            'gpt2-medium':  dict(n_layer=12, n_head=12, n_embd=1024), # 350M Params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M Params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M Params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50527 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized miniGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/ buffer

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight' ,'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use vanila network
        # this means that we have to transpose these weights when we import them
        for key in sd_keys_hf:
            if key not in sd_keys:
                print(key)
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}" 

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params':nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda'  in device # using fused kernels now
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# --------------------------------------------------------------------------------
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb100B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # target
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# ---------------------------------------------------------------------------------
# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps' # apple silicon GPU
print(f"using device: {device}")

# run the training loop
from torch.distributed import init_process_group, destroy_process_group

#-------------------------------- Runnign DDP ----------------------
# earlier runnin was done by
# python3 train_gpt2.py
#  for DDP  for e.g. 4 GPUs
# torchrun --standalone --nproc_per_node=4 train_gpt2.py



# set up DDP (distributed data parallel)
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # Check if this is a ddp run or not



if ddp:
    # use of DDP atm demands CUDA, we set device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl') # if nccl backend is not working then use gloo
    # using `gloo` incase nccl is not working
    # init_process_group(backend='gloo')
    ddp_rank = int(os.environ['RANK']) # each process will run exact same code at same time will have ddp rank different
    # gpu 0 -> rank 0, gpu 1 -> rank 1 etc.
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # generally used in multi node setting, local rank is rank of the gpu on single node
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of processes running, num of GPU
    device = f'cuda:{ddp_local_rank}' # : colon indicates which gpu to use i.e. which rank (0, 1, 2, etc.)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodectect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"using device: {device}")


# to have some reproducability
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# now since we're doing ddp we need to change these params accordingly
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 32 # micro batch size -> 
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # chnaged from B*T to B*T*ddp_world_size
#print(f"total desired batch size: {total_batch_size}") # now we will have 8 compies of the print statement
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
# print(f"=> calculated gradient accumulation steps: {grad_accum_steps}") # we will have 8 copies of the print statement



# earlier we were running `python train_gpt2.py`
# now for DDP (e.g. 8 gpus), we will run 
# torchrun --standalone --nproc_per_node==8 train_gpt2.py


# since we have a small machine and the batch size is 0.5M we would like to accumulate the gradient
# so we will do forward and backward propagation alot of times and accumulate the gradient
# we will update the accumulated gradients after we have reached the given batch size
# train_loader = DataLoaderLite(B=B, T=T) # now we want every processes for ddp to not load exact same data
# we pass on the rank and the world size to the data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
# sets the internal precision of float32 matrix multiplication
# Running float32 matrix multiplications in lower precision may significantly
# increase performance and in some programs the loss of precision has a negligible 
# impact
torch.set_float32_matmul_precision('high') 

# create model
model = GPT(GPTConfig(vocab_size=50304)) # because 50304 is divisible by 2, 4, 8, 16, 32, 64
# so basically its a power of 2
# the extra tokens added are will be unsused and will be driven two zero
model.to(device)

# torch.compile is the latest method to speed up PyTorch code. makes PyTorch code
# run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring
# minimal code changes
# This doesnot work here in my system (windows), It needs linux (I am running it in my lab's server)
use_compile = True # False # torch.compile interferes with hellaswag eval and generations
if use_compile:
    model = torch.compile(model)

#model = torch.compile(model)
if ddp:
    # in a forward pass it behaves identically
    # after you are doing the backwardpass, it will call alrreduce
    model = DDP(model, device_ids=[ddp_local_rank])

# keep the raw model not the ddp model
raw_model = model.module if ddp else model # contains the raw models

max_lr = 6e-4 # for GPT-3 Small max lr is 6e-4 
min_lr = max_lr * 0.1
warmup_steps = 715
# Currently total batch size is 524288, B = 32, T = 1024, so per step I am computing BxT which is 32768 tokens
# I have 10 Billion tokens, so to go over them, I would need 10,000,000,000/B*T*(ddp_rank)
# 10,000,000,000/(batch_size=524288)
max_steps = 190735 # 19073 for 10B, 190735 for 100B dataset
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine deacay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)



# optimizations
# the gpt-2 paper is very vague, and doesnot tell about hyperparameters, also github
# in gpt-3 paper is more clearer, but gpt-3 models were never released
# but gpt-2, gpt-3 architectures are very very similar i.e. context lenght 1024 to 1025
# gpt-3 was trained for lot longer on a bigger dataset, and its 175billion to 1.7billion

# in the paper [gpt-3] in seciton B. Details of Model Training
# Adam with B1 = 0.9, B2 = 0.95 and epsilon=10^-8
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

# uses regularization
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# This one is for future use
# we will create a log directory
# here we will write logs and checkpoints
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: 
    # to write in the file
    pass

# here we have used a fixed learning rate but actually in the paper 
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps -1)

    # once in a while evaluate our validation loss
    if step % 100 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % 5000 == 0 or last_step):
                    # optionally write the model checkpoints
                    checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                    checkpoint = {
                            "model": raw_model.state_dict(),
                            "config": raw_model.config,
                            "step": step,
                            "val_loss": val_loss_accum.item(),
                            }
                    torch.save(checkpoint, checkpoint_path)


    # once in a while evaluate hellaswag
    if (step % 100 == 0 or last_step): # and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the examples into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            #get the logits 
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes running (basically in your case is 4)
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"Hellaswag acc result: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 100 == 0) or last_step): # and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # talk the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the possibilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 10)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # training loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        #For using automatic and mixed precision
        with torch.autocast(device_type=device, dtype=torch.bfloat16):        
            logits, loss = model(x, y)

        # we have to scale the loss into account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() 
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        #import code; code.interact(local=locals())
        loss.backward()
    if ddp: # we need to loss from all the processes and average them
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # and they say in gpt-3 paper "we clip the global norm of the gradient at 1.0"
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # to prevent the model 
    
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups: # way to set learning rate in pytorch
        param_group['lr'] = lr

    
    # from getting too big shocks from big gradients
    # cosine decay learning rate schedule, with linear warmup
    # in paper they say `use cosine decay from learning rate down to 10% of its value`
    optimizer.step()
    torch.cuda.synchronize()
    # print(f"[RANK {ddp_rank}] step {step} done") # for debugging
    t1 = time.time()
    dt = (t1-t0) # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed/dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000: .2f}ms, tok/sec: {tokens_per_sec}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
#print(f"RANK {ddp_rank} finished training loop")


if ddp:
    # print("Destroying Process Groups") # for debugging purpose
    destroy_process_group()
    # print("Process Group Destroyed") # for debugging purpose

