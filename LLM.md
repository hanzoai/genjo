# Genjo Project Documentation

## Project Overview

Genjo is an implementation of the Large Language Diffusion model (Genjo), which represents a novel approach to language modeling using diffusion models rather than traditional autoregressive methods. The project offers two model variants:

- **Genjo-8B-Base**: The foundation model
- **Genjo-8B-Instruct**: Fine-tuned for instruction following and chat applications

The core innovation of Genjo is implementing a masked diffusion approach to language modeling, which demonstrates that diffusion models can achieve competitive performance compared to autoregressive models like LLaMA.

## Architecture

### Key Components

1. **Mask Predictor**: Uses a Transformer Encoder (not Decoder) as the backbone
   - Differs from autoregressive models by removing the causal mask from self-attention
   - Uses bidirectional attention instead of unidirectional attention

2. **Masking Approach**: 
   - Employs a masking ratio that varies randomly between 0 and 1 (unlike BERT's fixed ratio)
   - The training objective serves as an upper bound on the negative log-likelihood
   - Uses a reserved token (ID: 126336) as the mask token

3. **Generation Process**:
   - Implements a denoising diffusion process rather than autoregressive generation
   - Supports both fixed-length and semi-autoregressive generation strategies
   - Uses remasking techniques like "low_confidence" or "random" during generation

### Comparison with Traditional Models

| Feature | Genjo (Genjo) | Autoregressive Models (e.g., LLaMA) | BERT-like Models |
|---------|---------------|-------------------------------------|------------------|
| Network Architecture | Transformer Encoder | Transformer Decoder | Transformer Encoder |
| Attention Mechanism | Bidirectional | Unidirectional (Causal) | Bidirectional |
| Training Objective | Masked diffusion with varying mask ratio | Next-token prediction | Masked language modeling with fixed mask ratio |
| Generation Process | Parallel prediction and iterative denoising | Sequential token-by-token generation | Not designed for generation |
| In-context Learning | Yes | Yes | Limited |
| Sampling Speed | Currently slower, requires fixed context length | Faster, leverages KV-cache | N/A |
| Theoretical Foundation | Upper bound on negative log-likelihood | Direct log-likelihood | Pretraining only, not generative |

Unlike autoregressive models (which generate one token at a time) or BERT-like models (which use a fixed masking ratio), Genjo combines aspects of both approaches while using diffusion techniques for the generative process. This gives it unique capabilities while creating different performance characteristics.

## Project Structure

```
genjo/
├── imgs/                   # Images for documentation
├── visualization/          # Visualization utilities
├── EVAL.md                 # Evaluation documentation
├── GUIDELINES.md           # Model architecture and usage guidelines
├── README.md               # Project overview
├── app.py                  # Gradio web interface
├── chat.py                 # Chat interface for Genjo-Instruct
├── eval_genjo.py           # Evaluation script
├── eval_genjo.sh           # Evaluation shell script
├── generate.py             # Text generation implementation
└── get_log_likelihood.py   # Conditional likelihood calculation
```

## Core Functionality

### Text Generation Process

The text generation process in Genjo differs fundamentally from autoregressive models:

1. **Initialization**: Start with a sequence where the prompt is fixed and the response is fully masked
2. **Iterative Denoising**: Gradually unmask tokens through a diffusion process
   - The mask predictor predicts all masked tokens simultaneously
   - Tokens are then selectively unmasked based on confidence or randomly
3. **Sampling Approaches**:
   - Fixed-length: Generate with a fixed context length
   - Semi-autoregressive-padding: Use block-based generation with padding
   - Semi-autoregressive-origin: An experimental approach (not used in final models)

Key functions in the generation process:
- `add_gumbel_noise()`: Adds noise for sampling from categorical distributions
- `get_num_transfer_tokens()`: Calculates how many tokens to unmask at each step
- `generate()`: The main generation function implementing the diffusion process

#### Detailed Diffusion Algorithm

The core diffusion process in Genjo works as follows:

1. **Forward Process (Training)**:
   ```python
   def forward_process(input_ids, eps=1e-3):
       b, l = input_ids.shape
       t = torch.rand(b, device=input_ids.device)  # Random masking ratio between 0-1
       p_mask = (1 - eps) * t + eps  # Ensure minimum masking probability
       p_mask = p_mask[:, None].repeat(1, l)  # Expand to sequence length
       
       # Apply masks randomly based on p_mask probability
       masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
       # Replace masked tokens with mask token ID
       noisy_batch = torch.where(masked_indices, 126336, input_ids)
       return noisy_batch, masked_indices, p_mask
   ```

2. **Reverse Process (Generation)**:
   ```python
   for num_block in range(num_blocks):  # Block-based generation
       # Calculate mask indices for current block
       block_mask_index = (x == mask_id)  # Locations of mask tokens
       # Calculate tokens to unmask at each step
       num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
       
       for i in range(steps):  # Progressive denoising
           # Get predictions for all masked tokens
           logits = model(x).logits
           # Apply classifier-free guidance if enabled
           if cfg_scale > 0:
               # [CFG implementation]
           
           # Apply gumbel noise for sampling
           logits_with_noise = add_gumbel_noise(logits, temperature)
           x0 = torch.argmax(logits_with_noise, dim=-1)  # Token predictions
           
           # Calculate confidence scores for each prediction
           if remasking == 'low_confidence':
               p = F.softmax(logits, dim=-1)
               x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
           elif remasking == 'random':
               x0_p = torch.rand(x0.shape, device=x0.device)
           
           # Select tokens to unmask based on confidence
           transfer_index = torch.zeros_like(x0, dtype=torch.bool)
           for j in range(x0_p.shape[0]):
               _, indices = torch.topk(x0_p[j], k=num_transfer_tokens[j, i])
               transfer_index[j, indices] = True
           
           # Update sequence with new tokens
           x[transfer_index] = x0[transfer_index]
   ```

This approach allows Genjo to generate text by iteratively denoising a masked sequence, using the model to predict all masked tokens simultaneously and then selectively revealing the most confident predictions at each step. The number of tokens revealed at each step is calculated to ensure a consistent unmasking rate throughout the generation process.

### Log Likelihood Calculation

For evaluation, the model calculates conditional log-likelihood using:

- Monte Carlo estimation to estimate the likelihood of a response given a prompt
- Classifier-free guidance to improve generation quality
- Forward process that randomly masks tokens based on a probability mask

## Usage

### Inference

To load and use the Genjo model:

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/Genjo-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/Genjo-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)

# For the base model
from generate import generate
prompt = "Your prompt text here"
input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
output = generate(model, input_ids, steps=128, gen_length=128, block_length=32)
result = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

# For the instruct model (chat)
messages = [{"role": "user", "content": "Your question here"}]
chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
input_ids = tokenizer(chat_input)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
output = generate(model, input_ids, steps=128, gen_length=128, block_length=32)
result = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
```

### Chat Interface

The project provides a simple chat interface via `chat.py` and a more advanced Gradio web interface via `app.py`:

- `chat.py`: Terminal-based chat interface for Genjo-Instruct
- `app.py`: Gradio web interface with visualization of the denoising process

### Evaluation

Evaluation is done using:

1. **Conditional likelihood estimation**: For specific metrics using the log-likelihood calculation
2. **Conditional generation**: For evaluating the model's generation capabilities

The evaluation script `eval_genjo.py` integrates with the `lm-evaluation-harness` library to evaluate on benchmarks like:
- BBH (Big-Bench Hard)
- GSM8K (Grade School Math 8K)
- Math
- HumanEval
- MBPP (Mostly Basic Python Programming)

#### Evaluation Implementation

The `GenjoEvalHarness` class in `eval_genjo.py` implements the interface required by the `lm-evaluation-harness` library with the following key methods:

- `get_loglikelihood()`: Calculates log-likelihood using Monte Carlo estimation
- `suffix_greedy_prediction()`: Determines if a target continuation would be greedily predicted
- `loglikelihood()`: Handles loglikelihood requests for evaluation
- `generate_until()`: Generates text until reaching specified stop tokens

Evaluation commands are provided in `eval_genjo.sh`, which runs various benchmarks using different configurations:

```bash
# Conditional likelihood estimation benchmarks
accelerates launch eval_genjo.py --tasks gpqa_main_n_shot --num_fewshot 5 --model genjo_dist

# Conditional generation benchmarks
accelerates launch eval_genjo.py --tasks bbh --model genjo_dist --model_args gen_length=1024,steps=1024,block_length=1024
```

Key evaluation parameters:
- `mc_num`: Number of Monte Carlo samples for likelihood estimation (default: 128)
- `batch_size`: Batch size for evaluation
- `cfg`: Classifier-free guidance scale
- `steps`: Number of denoising steps
- `gen_length`: Maximum generation length
- `block_length`: Block length for semi-autoregressive generation

## Limitations and Future Work

1. **Sampling Efficiency**:
   - Currently slower than autoregressive models
   - Cannot leverage techniques like KV-Cache yet
   - Optimal performance requires sampling steps equal to response length

2. **Future Optimizations**:
   - Semi-autoregressive sampling to mitigate fixed context length issues
   - Consistency distillation to reduce the number of sampling steps
   - Further research into improving sampling efficiency

## Key Innovations and Research Findings

1. The training objective of Genjo serves as an upper bound on the negative log-likelihood
2. Masked diffusion models do not require time t as an input to Transformer
3. The training objective is equivalent to that of any-order autoregressive models
4. The model demonstrates that diffusion-based approaches can achieve competitive performance with autoregressive models
5. The unsupervised classifier-free guidance method significantly improves benchmark performance

## References

- Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)
- Model: [Hugging Face - Genjo-8B-Base](https://huggingface.co/GSAI-ML/Genjo-8B-Base)
- Model: [Hugging Face - Genjo-8B-Instruct](https://huggingface.co/GSAI-ML/Genjo-8B-Instruct)
- Demo: [Hugging Face Spaces](https://huggingface.co/spaces/multimodalart/Genjo)
