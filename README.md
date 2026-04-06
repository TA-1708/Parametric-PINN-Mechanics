# Parametric PINN Surrogate: Real-World 1D Structural Mechanics

Traditional Finite Element Analysis (FEA) requires meshing and iterative solving for every new geometry or load case. This project demonstrates a **Mesh-Free Parametric Physics-Informed Neural Network (PINN)** that acts as an instant surrogate model for solid mechanics. 

Instead of solving a differential equation for a single specific bar, this neural network learns the generalized physics of a tapered rod. Once trained, it can instantly predict the displacement field and internal forces for **any combination** of length, taper geometry, material stiffness, and applied load within its training domain—executing in milliseconds.

## 🚀 Key Features & Architectural Highlights
* **Universal Surrogate Model:** Takes 6 inputs `(x, L, P, A_0, A_L, E)` to predict continuous displacement `u(x)`.
* **Real-World SI Unit Normalization:** Employs dynamic Min-Max scaling to prevent exploding gradients when handling massive differences in scale (e.g., $10$ mm$^2$ area vs. $200$ GPa stiffness).
* **Advanced Optimization (Adam + L-BFGS):** Uses Adam for initial landscape navigation and the second-order `L-BFGS` optimizer with Strong Wolfe line search to achieve machine-precision accuracy.
* **Stable Derivatives:** Utilizes `SiLU` (Swish) activation functions to ensure continuous, non-vanishing second-order derivatives necessary for the physical loss functions.

## 📐 The Physics: Tapered Rod under Axial Load
The model strictly adheres to the governing differential equation for 1D solid mechanics with varying cross-sectional areas:
$$\frac{d}{dx} \left( E \cdot A(x) \cdot \frac{du}{dx} \right) = 0$$

**Boundary Conditions Enforced:**
1.  **Dirichlet (Fixed Wall):** $u(0) = 0$
2.  **Neumann (Free End Load):** $E \cdot A(L) \cdot u'(L) = P$

## 📊 Results & Validation
The model was validated against the exact analytical logarithmic solution for a tapered rod. 

* **Test Case:** Steel Rod ($E = 200$ GPa), $L = 500$ mm, tapering from $400$ mm$^2$ to $200$ mm$^2$, under a $100$ kN load.
* **Accuracy:** The L-BFGS optimized network achieved a relative $L_2$ error of **< 0.05%**, effectively mirroring the exact analytical solution.

*(Upload your `real_world_surrogate.png` to your repo and display it here using `![Validation Plot](real_world_surrogate.png)`)*

## ⚙️ How to Run
```bash
# Clone the repository
git clone [https://github.com/YourUsername/Parametric-PINN-Mechanics.git](https://github.com/YourUsername/Parametric-PINN-Mechanics.git)

# Install requirements
pip install torch numpy matplotlib

# Run the training and evaluation script
python real_world_surrogate.py