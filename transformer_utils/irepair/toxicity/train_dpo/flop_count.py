from transformers import AutoModelForCausalLM, AutoTokenizer

from constants import DEVICE


def openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab, ff_ratio=4, include_embeddings=True):

    d_attn = d_model // n_heads
    d_ff = d_model * ff_ratio

    # Compute the FLOPs for each component
    embeddings = 4 * d_model if include_embeddings else 0
    attn_qkv = 2 * n_layers * d_model * 3 * (d_attn * n_heads)
    attn_mask = 2 * n_layers * n_ctx * (d_attn * n_heads)
    attn_project = 2 * n_layers * (d_attn * n_heads) * d_model
    ff = 2 * n_layers * 2 * d_model * d_ff
    logits = 2 * d_model * n_vocab

    # Sum up the FLOPs
    total_flops = embeddings + attn_qkv + attn_mask + attn_project + ff + logits

    return total_flops/1e9


def count_flop_sr(n_layers,n_heads,d_model,n_ctx,n_vocab):
    total_forward = 4*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    backward_1 = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab, include_embeddings=False)
    backward_2 = 2 * openai_flops_per_token(1, n_heads, d_model, n_ctx, n_vocab, include_embeddings=False)
    flops=total_forward+backward_1+backward_2
    print(f"Estimated FLOPs per token (SR): {flops}")

def count_flop_sr_free(n_layers,n_heads,d_model,n_ctx,n_vocab):
    total_forward = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    backward_1 = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab, include_embeddings=False)
    backward_2 = 2 * openai_flops_per_token(1, n_heads, d_model, n_ctx, n_vocab, include_embeddings=False)
    flops=total_forward+backward_1+backward_2
    print(f"Estimated FLOPs per token (SR Free): {flops}")

def count_flop_dpo(n_layers,n_heads,d_model,n_ctx,n_vocab):
    total_forward = 4*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    backward = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    flops=total_forward+backward
    print(f"Estimated FLOPs per token: {flops}")

def count_flop_sft(n_layers,n_heads,d_model,n_ctx,n_vocab):
    total_forward = openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    backward = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    flops=total_forward+backward
    print(f"Estimated FLOPs per token: {flops}")

def count_flop_sftkl(n_layers,n_heads,d_model,n_ctx,n_vocab):
    total_forward = 3*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    backward = 2*openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab)
    flops=total_forward+backward
    print(f"Estimated FLOPs per token: {flops}")


model_name='EleutherAI/gpt-neo-1.3b'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Retrieve model configuration
config = model.config

# Parameters
n_layers = config.num_hidden_layers
n_heads = config.num_attention_heads
d_model = config.hidden_size
if 'neo' in model_name:
    n_ctx=config.max_position_embeddings
else:
    n_ctx = config.n_positions
n_vocab = len(tokenizer.get_vocab())

count_flop_sr_free(n_layers,n_heads,d_model,n_ctx,n_vocab)
# count_flop_sr(n_layers,n_heads,d_model,n_ctx,n_vocab)
# count_flop_dpo(n_layers,n_heads,d_model,n_ctx,n_vocab)
# count_flop_sft(n_layers,n_heads,d_model,n_ctx,n_vocab)
# count_flop_sftkl(n_layers,n_heads,d_model,n_ctx,n_vocab)