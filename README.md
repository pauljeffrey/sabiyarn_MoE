# SabiYarn: Pretraining a Small Language Model on African Languages

[![CI/CD](https://github.com/NaijaAI/sabiyarn/actions/workflows/test.yml/badge.svg)](https://github.com/NaijaAI/sabiyarn/actions/workflows/test.yml)

SabiYarn is a research codebase for pretraining, fine-tuning, and running inference on small Large Language Models (LLMs) on African Languages. It includes modular implementations of transformer blocks, differential attention mechanisms, and advanced features such as Mixture-of-Experts (MoE) and Multi-Head Latent Attention (MLA).

[Read the Paper here](https://openreview.net/forum?id=3U1LCDdYwy)

## Features
- Modular transformer architecture (see `sabiyarn/model.py`)
- Support for rotary embeddings, LoRA, and MoE
- Multi-Head Latent Attention (MLA) module (`sabiyarn/MLA.py`)
- Differential Attention module (`sabiyarn/differential_attention.py`)
- Utilities for pretraining, fine-tuning, and inference
- Designed for use with [Modal](https://modal.com/) for scalable training and inference
- Test suite for model forward pass and attention modules

## Quickstart

**The triton kernels for distributed training require that the code be run on a GPU compatible machine. It is highly recommended to run the code on a GPU machine**

### Create a virtual env
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh #install uv
surce ~/.bashrc #restart your shell
uv venv #create venv
source .venv/bin/activate # activate the venv
```
### 2. Install Requirements
```bash
uv pip install -r requirements.txt
```

### 3. Run Model Tests
Run the included tests:
```bash
python tests/test_diff_attn.py
python tests/test_mha.py
python tests/test_mla.py
python tests/test_model_initialization.py 
```

Add more tests in `sabiyarn/test.py` as needed.

### 3. Training & Inference on Modal
- All training and inference jobs are designed to run on [Modal](https://modal.com/). See `scripts/` for example Modal entrypoints.
```

## Project Structure
```
├───config
├───cut_cross_entropy
│   └───transformers
├───data
├───eval
├───finetuning
├───inference
├───Notebooks
├───sabiyarn
└───training

## Testing & CI/CD
- Use [pytest](https://docs.pytest.org/) for unit and integration tests.
- Example test: ensure model can instantiate and run a forward pass.
- Recommended: set up GitHub Actions to run tests on every PR and push.

## Modal Integration
- All heavy compute (training, inference) should be run on Modal.
- Use Modal Volumes or cloud storage for datasets and checkpoints.
- See [Modal documentation](https://modal.com/docs/) for more.

## MLOps Best Practices
- Version control all code and configuration
- Use Docker for reproducible environments (Modal supports custom Dockerfiles)
- Log all hyperparameters and environment details
- Track experiments with [Weights & Biases](https://wandb.ai/) or similar
- Save model checkpoints with metadata
- Write and automate tests for all modules

## License
[MIT](LICENSE)