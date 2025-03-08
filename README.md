# Llama 3.3 8B Fine-Tuning Project

This repository contains code and resources for fine-tuning the Llama 3.3 8B model quantized to 4-bit precision using the Unsloth library. The model is fine-tuned on a dataset of Amazon product information, focusing on product IDs, titles, and content.

## Project Overview

This project demonstrates how to efficiently fine-tune a large language model (LLM) with limited computational resources by using quantization techniques. The Llama 3.3 8B model is quantized to 4-bit precision to reduce memory requirements while maintaining performance.

## Dataset

The dataset consists of Amazon product information with the following structure:
- Product ID
- Product Title
- Product Content

All other columns from the original dataset are stripped out to focus on these key attributes.

## Key Features

- 4-bit quantization of Llama 3.3 8B model
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Optimized training using Unsloth library
- Inference examples and evaluation

## Requirements

The project requires the following dependencies:
- bitsandbytes
- accelerate
- xformers
- peft
- trl
- triton
- cut_cross_entropy
- unsloth_zoo
- unsloth
- transformers
- datasets

## Setup and Installation

To set up the environment, run:

```bash
pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
pip install --no-deps cut_cross_entropy unsloth_zoo
pip install datasets
pip install unsloth
```

## Training Process

The training process involves:

1. Loading the Llama 3.3 8B model with 4-bit quantization
2. Applying LoRA adapters for parameter-efficient fine-tuning
3. Preparing the Amazon product dataset
4. Training with the SFTTrainer from TRL
5. Saving the fine-tuned model

## Inference

After training, the model can be used for inference by:

1. Loading the fine-tuned model
2. Converting it to inference mode
3. Generating responses based on prompts

## Usage

The main workflow is implemented in the Jupyter notebook `llama3-finetunning.ipynb`. Run the notebook cells sequentially to:
1. Install dependencies
2. Load and prepare the model
3. Load and preprocess the dataset
4. Train the model
5. Run inference with the fine-tuned model

## License

Please refer to the Llama 3 license for usage restrictions and permissions.

## Acknowledgements

- Meta AI for the Llama 3.3 model
- Unsloth team for the optimization library
- Hugging Face for the transformers library and datasets tools
