# Multi-Scale Collision Counting for Rényi Entropy Rate Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Estimate Rényi 2-entropy rate from raw byte streams using multi-scale collision counting. No model fitting required.

**Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Key Result

On 124 DNS tunnel pcaps, Rényi entropy rate achieves ARI = 0.720 vs Shannon's 0.204 for tool classification (Δ = 0.516), validated through a controlled 9-method ablation with debiasing controls.

| Method                      | α      | Dim    | ARI       |
| --------------------------- | ------ | ------ | --------- |
| Shannon H₁ (byte)           | 256    | 1D     | 0.246     |
| Shannon rate (nibble slope) | 16     | 1D     | 0.204     |
| Shannon rate (MM debiased)  | 16     | 1D     | 0.204     |
| **Rényi h₂ (nibble slope)** | **16** | **1D** | **0.720** |
| **Rényi (h₂,h₃,h₄)**        | **16** | **3D** | **0.747** |
| Shannon 3D (nibble)         | 16     | 3D     | 0.460     |

## Quick Start

```python
from fq_core import multiscale_fq, fingerprint

# Estimate entropy rate from bytes
with open('somefile.bin', 'rb') as f:
    data = f.read()

result = multiscale_fq(data, alpha=16, q_list=(2, 3, 4))
print(f"h₂ = {result['h_q'][2]:.4f}")
print(f"R² = {result['R2']:.6f}")

# Classification fingerprint
fp = fingerprint(data, alpha=16, q_list=(2, 3, 4), mode='slope')
print(f"(h₂, h₃, h₄) = ({fp[0]:.3f}, {fp[1]:.3f}, {fp[2]:.3f})")
```

## Installation

```bash
git clone https://github.com/0x1Adi/renyi-entropy-rate-estimation.git
cd renyi-entropy-rate-estimation
pip install numpy scipy  # only dependencies
```

Optional for DNS experiments:

```bash
pip install dpkt matplotlib scikit-learn
```

## Files

| File                   | Description                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| `fq_core.py`           | Core library: debiased F_q, multi-scale slope, fingerprint, classification |
| `run_experiment.py`    | Full experiment suite (7 experiments, 8 figures)                           |
| `mega_validation.py`   | 9-method ablation + all paper claims validation                            |
| `standalone_verify.py` | Independent verifier (zero imports from fq_core)                           |
| `test_fq_core.py`      | 33 unit + integration tests                                                |
| `paper/paper.tex`      | LaTeX source                                                               |
| `paper/references.bib` | Bibliography                                                               |

## Reproducing Paper Results

### Synthetic experiments (no data needed)

```bash
python run_experiment.py --skip-dns --skip-evm
# Generates: exp1_ground_truth.json, exp2_sqrt_n.json, exp3_r2_order.json, exp6_crypto_hierarchy.json
```

### DNS tunnel classification

Requires the [GraphTunnel dataset](https://github.com/ggyggy666/DNS-Tunnel-Datasets):

```bash
python mega_validation.py \
  --dns-path /path/to/DNS-Tunnel-Datasets/ \
  --evm-path /path/to/evm_bytecodes/
```

### Independent verification

```bash
python standalone_verify.py --dns-path /path/to/DNS-Tunnel-Datasets/
```

This reimplements everything from scratch with zero imports from `fq_core.py` to cross-check results.

## Running Tests

```bash
python test_fq_core.py
# Expected: 33/33 passed
```

## Method

1. **Expand** bytes to nibbles (α=16)
2. **Count** k-grams for k = 1, ..., K (adaptive K)
3. **Debias** collision probability via falling factorial: F̂_q = Σ c_i^{(q)} / n^{(q)}
4. **Fit** OLS slope of log F̂₂(k) vs k
5. **Extract** h₂ = −slope / ln(2)

For a stationary Markov(1) chain with transition matrix P, define M₂ by (M₂)ᵢⱼ = Pᵢⱼ². Then F₂(k) = v⊤ M₂^{k−1} 𝟏 where vᵢ = πᵢ². By Perron–Frobenius, log F₂(k) is asymptotically linear in k with slope log ρ(M₂), giving h₂ = −log₂ ρ(M₂).

## Citation

```bibtex
@article{tiwari2026multiscale,
  author  = {Tiwari, Aditya},
  title   = {Multi-Scale Collision Counting for {R}\'enyi Entropy Rate Estimation},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT
