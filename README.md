# Genjo: Large Language Diffusion Model

[![arXiv](https://img.shields.io/badge/arXiv-2502.09992-red.svg)](https://arxiv.org/abs/2502.09992)
[![deploy](https://img.shields.io/badge/Hugging%20Face%20-Genjo_Base%20-FFEB3B)](https://huggingface.co/GSAI-ML/Genjo-8B-Base)
[![deploy](https://img.shields.io/badge/Hugging%20Face%20-Genjo_Instruct%20-FFEB3B)](https://huggingface.co/GSAI-ML/Genjo-8B-Instruct)
[![deploy](https://img.shields.io/badge/🤗%20Hugging%20Face%20-Spaces%20demo%20-blue)](https://huggingface.co/spaces/multimodalart/Genjo)
[![deploy](https://img.shields.io/badge/Zhihu-知乎-blue)](https://zhuanlan.zhihu.com/p/24214732238)

We introduce Genjo, a **L**arge **La**nguage **D**iffusion model with unprecedented performance and scale. Genjo serves as the core foundation for our advanced diffusion-based language models, which will integrate with our Enso diffusion Mixture of Experts (MoE) architecture.

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/Genjo_vs_LLaMA.svg" style="width: 45%" />
    <img src="./imgs/Genjo_vs_LLaMA_chat.svg" style="width: 46%" />
</div>

## Overview

Genjo represents a novel approach to language modeling using diffusion models rather than traditional autoregressive approaches. The model employs a masked diffusion technique that:

- Uses a Transformer Encoder with bidirectional attention instead of a Decoder with causal attention
- Employs a varying masking ratio that provides an upper bound on the negative log-likelihood
- Achieves competitive performance with autoregressive models in numerous benchmarks

## Integration with Enso

Genjo will serve as the foundation for integration with our [Enso](https://github.com/feizc/DiT-MoE) diffusion Mixture of Experts (MoE) architecture. This integration will enable:

1. **Scaling to Larger Models**: Using MoE architecture to scale to 16B+ parameters while maintaining efficiency
2. **Expert Specialization**: Leveraging specialized experts for different linguistic tasks and patterns
3. **Efficient Inference**: Optimizing sampling through expert routing and batched computation
4. **Improved Performance**: Combining the strengths of diffusion models with the parameter efficiency of MoE

## Inference
The [Genjo-8B-Base](https://huggingface.co/GSAI-ML/Genjo-8B-Base) and [Genjo-8B-Instruct](https://huggingface.co/GSAI-ML/Genjo-8B-Instruct) models are available on Hugging Face. Please first install `transformers==4.38.2` and use the [transformers](https://huggingface.co/docs/transformers/index) library to load them.

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/Genjo-8B-Base', trust_remote_code=True)
model = AutoModel.from_pretrained('GSAI-ML/Genjo-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16)
```

We provide `get_log_likelihood()` and `generate()` functions in `get_log_likelihood.py`
and `generate.py` respectively, for conditional likelihood evaluation and conditional generation.

You can directly run `python chat.py` to have multi-round conversations with Genjo-8B-Instruct.

In addition, please refer to our paper and [GUIDELINES.md](GUIDELINES.md) for more details about the inference methods.

## Gradio Demo
Thank you very much to [apolinário](https://github.com/apolinario) for helping us create this amazing demo!

First, install [Gradio](https://www.gradio.app) `pip install gradio`, and then you can directly run `python app.py`

<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Model Architecture

Genjo uses a Transformer Encoder as the backbone for its mask predictor. The key differences from autoregressive models (like LLaMA) and other masked language models (like BERT) are:

1. **Bidirectional Attention**: Removes causal masking in self-attention, allowing the model to attend to all tokens.
2. **Dynamic Masking Ratio**: Employs a masking ratio that varies randomly between 0 and 1 during training.
3. **Diffusion Process**: Uses an iterative denoising process for text generation rather than autoregressive token prediction.

The model performs generation through an iterative masked diffusion process:
1. Start with a prompt and fully masked response
2. Predict all masked tokens simultaneously
3. Selectively unmask tokens based on confidence or random selection
4. Repeat until all tokens are unmasked

## Evaluation

We use two evaluation methods:
1. **Conditional likelihood estimation** for specific metrics
2. **Conditional generation** for other benchmarks

The model achieves competitive performance on a wide range of benchmarks including:
- BBH (Big-Bench Hard)
- GSM8K (Grade School Math 8K)
- Math
- HumanEval
- MBPP (Mostly Basic Python Programming)

## Future Directions

Our roadmap for Genjo + Enso integration includes:

1. **Sampling Efficiency Optimizations**:
   - Implementing semi-autoregressive sampling approaches
   - Applying consistency distillation to reduce sampling steps
   - Leveraging MoE architecture for more efficient inference

2. **Model Scaling**:
   - Integrating Enso's MoE architecture to scale to 16B+ parameters
   - Implementing expert specialization for different linguistic tasks
   - Optimizing expert routing for language processing

3. **Training Improvements**:
   - Applying rectified flow training for better performance and faster convergence
   - Implementing DeepSpeed for efficient training of large-scale models
   - Exploring hybrid MoE architectures combining dense and sparse layers

## Citation

If you find our work useful, please cite:

```bibtex
@article{nie2025large,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Zebin and Zhang, Xiaolu and Ou, Jingyang and Hu, Jun and Zhou, Jun and Lin, Yankai and Wen, Ji-Rong and Li, Chongxuan},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

For the MoE architecture integration:

```bibtex
@article{FeiDiTMoE2024,
  title={Scaling Diffusion Transformers to 16 Billion Parameters},
  author={Zhengcong Fei, Mingyuan Fan, Changqian Yu, Debang Li, Jusnshi Huang},
  year={2024},
  journal={arXiv preprint},
}
```