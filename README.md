📌 This repository compares Fourier Neural Operators (FNOs) and U-Nets as deep learning surrogates for accelerating simulations of 2D reactive transport systems involving radioactive decay chains. The models are conditioned on physical parameters (vₓ, dispersion D, decay rates k_P, k_D, inlet concentration P_in) to predict spatiotemporal evolution of parent (P) and daughter (D) radionuclides.

Key Features:

     GPU-accelerated ground truth generation using Finite Difference Method (CuPy).

     Parameter-aware architectures: FNO and U-Net conditioned on 5 physical parameters.

     Benchmarking: Single-step accuracy, long-term rollouts, and mass conservation.

     Generalization tests: Interpolation to unseen parameter combinations.

🔍 Problem Statement

Reactive transport models (RTMs) for radionuclide decay chains are computationally expensive. This project explores neural surrogates to replace traditional solvers, enabling:

    Fast uncertainty quantification

    Inverse parameter estimation

    Real-time scenario analysis

    Governing PDEs:

    ∂P/∂t = D∇²P − vₓ∂P/∂x − k_P P  
    ∂D/∂t = D∇²D − vₓ∂D/∂x + k_P P − k_D D  

🚀 Methodology

1. Data Generation

    Numerical solver: Explicit Euler FDM on GPU (CuPy).

    Parameters sampled: Latin Hypercube Sampling (LHS) over ranges:
    Parameter	Range
    vₓ	[0.1, 1.0]
    D	[0.001, 0.01]
    k_P	[0.05, 0.5]
    k_D	[0.01, 0.1]
    P_in	[0.5, 2.0]

2. Model Architectures
      Model	Architecture	Inputs	Outputs
      FNO	4-layer, 32 modes	[P(t), D(t), vₓ, D, k_P, k_D, P_in]	[P(t+Δt), D(t+Δt)]
      U-Net	4 down/up blocks	Same as FNO	Same as FNO
   
4. Training

    Loss: Relative L₂ error
    Optimizer: AdamW (lr=1e-3, weight decay=1e-4)
    Scheduler: StepLR (γ=0.5 every 5 epochs)

📊 Results

    (Example findings – replace with your actual results)
    Metric	FNO	U-Net
    Single-step test error	0.8%	1.2%
    100-step rollout error	3.5%	12.7%
    Mass conservation error	<1%	~5%
    Training time (30 epochs)	2.1 hrs	3.4 hrs

Key Observations:

    FNO outperforms U-Net in long-term stability and parameter generalization.

    
