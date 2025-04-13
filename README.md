 ## Overview

This repository compares Fourier Neural Operators (FNOs) and U-Nets as deep learning surrogates for accelerating simulations of 2D reactive transport systems involving radioactive decay chains. The models are conditioned on physical parameters (vₓ, dispersion D, decay rates k_P, k_D, inlet concentration P_in) to predict spatiotemporal evolution of parent (P) and daughter (D) radionuclides.

## Key Features

- GPU-accelerated ground truth generation using Finite Difference Method (CuPy).

- Parameter-aware architectures: FNO and U-Net conditioned on 5 physical parameters.

- Benchmarking: Single-step accuracy, long-term rollouts, and mass conservation.

- Generalization tests: Interpolation to unseen parameter combinations.

## Requirements

1. Python 3.8+

2. PyTorch (>=1.10)

3. CuPy (matching your CUDA version)

4. NumPy

5. Matplotlib

6. SciPy (for LHS)

7. neuraloperator (or neuralop)

8. scikit-learn (for train_test_split)

## Problem Statement

Reactive transport models (RTMs) for radionuclide decay chains are computationally expensive. This project explores neural surrogates to replace traditional solvers, enabling:

- Fast uncertainty quantification
- Inverse parameter estimation
- Real-time scenario analysis


### Governing PDEs:

    ∂P/∂t = D∇²P − vₓ∂P/∂x − k_P P  
    ∂D/∂t = D∇²D − vₓ∂D/∂x + k_P P − k_D D  

## Methodology

### 1. Data Generation

   Numerical solver: Explicit Euler FDM on GPU (CuPy).

   Parameters sampled: Latin Hypercube Sampling (LHS) over ranges:
   
   | Parameter       | Range         | Units  |
   |-----------------|---------------|--------|
   | Velocity (`vₓ`) | 0.1 – 1.0     | L/T    |
   | Dispersion (`D`) | 0.001 – 0.01  | L²/T   |
   | Decay rate (`k_P`)| 0.05 – 0.5   | 1/T    |
   | Decay rate (`k_D`)| 0.01 – 0.1   | 1/T    |
   | Inlet conc. (`P_in`)| 0.5 – 2.0  | M/L³   |

### 3. Model Architectures

| Model | Architecture      | Inputs                                | Outputs               |
|-------|-------------------|----------------------------------------|------------------------|
| FNO   | 4-layer, 32 modes | [P(t), D(t), vₓ, D, k_P, k_D, P_in]   | [P(t+Δt), D(t+Δt)]     |
| U-Net | 4 down/up blocks  | Same as FNO                            | Same as FNO            |

   
### 5. Training (Adjust as required)

- **Loss Function**: Relative L₂ error  
- **Optimizer**: AdamW  
  - Learning Rate: `1e-3`  
  - Weight Decay: `1e-4`  
- **Learning Rate Scheduler**: StepLR  
  - Decay Factor (γ): `0.5`  
  - Step Size: every `5` epochs



## Key Observations

FNO outperforms U-Net in long-term stability and parameter generalization.




9. tqdm (for progress bars)

    
