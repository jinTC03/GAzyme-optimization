# GAzyme

**GAzyme** is a genetic algorithm framework based on the **[Enzeptional](https://github.com/GT4SD/enzeptional.git)** framework for protein sequence optimization using ESM2 protein large language model and fitness Kcat predictor.  
It is designed for mutating enzyme sequences in a biologically meaningful way, combining guided crossover, selection, and mutation strategies.

---

## Purpose

This repository supports the implementation and experimentation of:
- Genetic algorithm-based protein sequence optimization
- LLM-guided mutation strategies
- Sequence scoring using pretrained Kcat model
- Evaluation of crossover, mutation, and selection techniques
---

## Getting Started
### 1. Clone the repository

```bash
git clone https://github.com/ashi198/GAzyme.git
```

### 2. Clone the Enzeptional repository within GAzyme to use its functionality
```bash
cd GAzyme
git clone https://github.com/GT4SD/enzeptional.git
```
### 3. Create conda environment 
```bash
conda create -n gt4sd python=3.9
cd enzeption
pip install -e ".[dev]"
pip install torch --upgrade #to avoid No module named 'torch.distributed._tensor error
```
---
## Running an Optimization
To run the optimization pipeline with default configuration:
```bash
python main.py
```
---
## References 
Thanks to the following repository:
**[Enzeptional](https://github.com/GT4SD/enzeptional.git)**
```bash
@inproceedings{teukam2023enzyme,
  title={Enzyme optimization via a generative language modeling-based evolutionary algorithm},
  author={Teukam, Yves Gaetan Nana and Grisoni, Francesca and Manica, Matteo and Zipoli, Federico and Laino, Teodoro},
  booktitle={American Chemical Society (ACS) Spring Meeting},
  year={2023}
}
```

