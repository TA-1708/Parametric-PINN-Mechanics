# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 1. Real-World Parameter Bounds (SI Units) ────────────────────────────────
bounds = {
    'L_min': 0.01,  'L_max': 1.0,         # 10mm to 1000mm
    'P_min': 1e3,   'P_max': 1e6,         # 1kN to 1000kN
    'A_min': 1e-5,  'A_max': 1e-3,        # 10mm^2 to 1000mm^2
    'E_min': 1e10,  'E_max': 1e12         # 10GPa to 1000GPa
}

# ─── 2. Normalized Parametric Neural Network ──────────────────────────────────
class RealWorldSurrogatePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 6 Inputs: x, L, P, A0, AL, E
        self.network = nn.Sequential(
            nn.Linear(6, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
        
    def normalize(self, tensor, min_val, max_val):
        """ Squashes any real-world tensor into a safe [0, 1] range """
        return (tensor - min_val) / (max_val - min_val)

    def forward(self, x, L, P, A0, AL, E):
        # 1. Normalize every single input
        x_norm  = x / L 
        L_norm  = self.normalize(L, bounds['L_min'], bounds['L_max'])
        P_norm  = self.normalize(P, bounds['P_min'], bounds['P_max'])
        A0_norm = self.normalize(A0, bounds['A_min'], bounds['A_max'])
        AL_norm = self.normalize(AL, bounds['A_min'], bounds['A_max'])
        E_norm  = self.normalize(E, bounds['E_min'], bounds['E_max'])
        
        inputs = torch.cat([x_norm, L_norm, P_norm, A0_norm, AL_norm, E_norm], dim=1)
        
        # 2. Network predicts in safe [0, 1] space
        u_pred_normalized = self.network(inputs)
        
        # 3. Scale back to real-world magnitude
        scale_factor = (P * L) / (E * A0) 
        return u_pred_normalized * scale_factor

model = RealWorldSurrogatePINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ─── 3. Physics Loss with Parameter Sampling ──────────────────────────────────
def compute_parametric_loss(batch_size=128):
    # Sample random real-world scenarios
    L_samples  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['L_max'] - bounds['L_min']) + bounds['L_min']
    P_samples  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['P_max'] - bounds['P_min']) + bounds['P_min']
    A0_samples = torch.rand(batch_size, 1, device=DEVICE) * (bounds['A_max'] - bounds['A_min']) + bounds['A_min']
    AL_samples = torch.rand(batch_size, 1, device=DEVICE) * (bounds['A_max'] - bounds['A_min']) + bounds['A_min']
    E_samples  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['E_max'] - bounds['E_min']) + bounds['E_min']
    
    # Sample random x points (0 to L)
    x_real = torch.rand(batch_size, 1, device=DEVICE) * L_samples
    x_real.requires_grad_(True)
    
    # Predict Displacement
    u = model(x_real, L_samples, P_samples, A0_samples, AL_samples, E_samples)
    
    # Calculate Strain and Force
    strain = torch.autograd.grad(u, x_real, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    A_x = A0_samples - ((A0_samples - AL_samples) / L_samples) * x_real
    internal_force = E_samples * A_x * strain
    
    # PDE Loss (Force gradient should be 0). Normalized to prevent explosion.
    force_grad = torch.autograd.grad(internal_force, x_real, grad_outputs=torch.ones_like(internal_force), create_graph=True)[0]
    loss_pde = torch.mean((force_grad / bounds['P_max'])**2)
    
    # Wall Boundary Condition (u = 0)
    x_wall = torch.zeros_like(x_real)
    u_wall = model(x_wall, L_samples, P_samples, A0_samples, AL_samples, E_samples)
    scale_factor = (P_samples * L_samples) / (E_samples * A0_samples)
    loss_wall = torch.mean((u_wall / scale_factor)**2)
    
    # Free End Boundary Condition (Force = P)
    x_tip = L_samples.clone().requires_grad_(True)
    u_tip = model(x_tip, L_samples, P_samples, A0_samples, AL_samples, E_samples)
    strain_tip = torch.autograd.grad(u_tip, x_tip, grad_outputs=torch.ones_like(u_tip), create_graph=True)[0]
    
    force_tip = E_samples * AL_samples * strain_tip
    loss_free = torch.mean(((force_tip - P_samples) / bounds['P_max'])**2)
    
    return loss_pde, loss_wall, loss_free

# ─── 4. Training ──────────────────────────────────────────────────────────────
def train_surrogate(epochs=30000):
    print("Training Real-World Tapered Rod Surrogate Model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde, loss_wall, loss_free = compute_parametric_loss()
        
        # Heavy boundary enforcement
        loss = 100.0 * loss_pde + 5000.0 * loss_wall + 1000.0 * loss_free
        loss.backward()
        optimizer.step()
        
        if epoch % 2000 == 0:
            print(f"Epoch {epoch:5d} | PDE: {loss_pde.item():.6e} | Wall: {loss_wall.item():.6e} | Free: {loss_free.item():.6e}")

# ─── 4.5. The "Last Mile" L-BFGS Optimizer ────────────────────────────────────
def train_lbfgs(max_iterations=5000):
    print("\nStarting L-BFGS Optimization for extreme precision...")
    
    # 1. Generate a massive, FIXED batch of scenarios. No randomness inside the loop!
    batch_size = 2000
    L_f  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['L_max'] - bounds['L_min']) + bounds['L_min']
    P_f  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['P_max'] - bounds['P_min']) + bounds['P_min']
    A0_f = torch.rand(batch_size, 1, device=DEVICE) * (bounds['A_max'] - bounds['A_min']) + bounds['A_min']
    AL_f = torch.rand(batch_size, 1, device=DEVICE) * (bounds['A_max'] - bounds['A_min']) + bounds['A_min']
    E_f  = torch.rand(batch_size, 1, device=DEVICE) * (bounds['E_max'] - bounds['E_min']) + bounds['E_min']
    
    x_f = torch.rand(batch_size, 1, device=DEVICE) * L_f
    x_f.requires_grad_(True)
    
    # Wall and Tip tensors (Fixed)
    x_wall = torch.zeros_like(x_f)
    x_tip  = L_f.clone().requires_grad_(True)
    
    # 2. Define the L-BFGS Optimizer
    # 'strong_wolfe' line search is the secret ingredient that prevents PINNs from diverging
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=max_iterations, 
        tolerance_change=1e-9, 
        history_size=50,
        line_search_fn="strong_wolfe" 
    )
    
    iteration = 0
    
    # 3. Define the Closure Function
    def closure():
        nonlocal iteration
        lbfgs_optimizer.zero_grad()
        
        # --- A. PDE Loss (Using fixed tensors) ---
        u = model(x_f, L_f, P_f, A0_f, AL_f, E_f)
        strain = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        A_x = A0_f - ((A0_f - AL_f) / L_f) * x_f
        internal_force = E_f * A_x * strain
        
        force_grad = torch.autograd.grad(internal_force, x_f, grad_outputs=torch.ones_like(internal_force), create_graph=True)[0]
        loss_pde = torch.mean((force_grad / bounds['P_max'])**2)
        
        # --- B. Wall Loss ---
        u_wall = model(x_wall, L_f, P_f, A0_f, AL_f, E_f)
        scale_factor = (P_f * L_f) / (E_f * A0_f)
        loss_wall = torch.mean((u_wall / scale_factor)**2)
        
        # --- C. Free End Loss ---
        u_tip = model(x_tip, L_f, P_f, A0_f, AL_f, E_f)
        strain_tip = torch.autograd.grad(u_tip, x_tip, grad_outputs=torch.ones_like(u_tip), create_graph=True)[0]
        force_tip = E_f * AL_f * strain_tip
        loss_free = torch.mean(((force_tip - P_f) / bounds['P_max'])**2)
        
        # Combine and compute gradients
        total_loss = 100.0 * loss_pde + 5000.0 * loss_wall + 1000.0 * loss_free
        total_loss.backward()
        
        if iteration % 100 == 0:
            print(f"L-BFGS Step {iteration:4d} | Total Loss: {total_loss.item():.6e}")
        iteration += 1
        
        return total_loss

    # 4. Execute the optimizer
    lbfgs_optimizer.step(closure)
    print("L-BFGS Optimization Complete!")

# ─── 5. Evaluation for an "Unseen" Real-World Case ────────────────────────────
def test_surrogate():
    # Test Case: 500mm Steel Rod, 100kN load, tapering from 400mm^2 to 200mm^2
    test_L  = 0.5 
    test_P  = 100000.0 
    test_A0 = 0.0004 
    test_AL = 0.0002 
    test_E  = 200e9 
    
    x_vals = np.linspace(0, test_L, 100)
    x_t = torch.tensor(x_vals, dtype=torch.float32, device=DEVICE).view(-1, 1)
    
    L_t  = torch.full_like(x_t, test_L); P_t  = torch.full_like(x_t, test_P)
    A0_t = torch.full_like(x_t, test_A0); AL_t = torch.full_like(x_t, test_AL)
    E_t  = torch.full_like(x_t, test_E)
    
    with torch.no_grad():
        u_pred = model(x_t, L_t, P_t, A0_t, AL_t, E_t).cpu().numpy().flatten()
    
    # Analytical Verification
    k = (test_A0 - test_AL) / test_L
    u_exact = (test_P / (test_E * k)) * np.log(test_A0 / (test_A0 - k * x_vals))
    
    plt.figure(figsize=(8, 5))
    # Plot in millimeters for readability
    plt.plot(x_vals * 1000, u_exact * 1000, 'k--', lw=2, label='Exact Analytical')
    plt.plot(x_vals * 1000, u_pred * 1000, 'r-', lw=1.5, label='Real-World PINN Prediction')
    plt.title(f'Real-World Surrogate: Steel Rod (L={test_L*1000}mm, P={test_P/1000}kN)')
    plt.xlabel('Position (mm)')
    plt.ylabel('Displacement (mm)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig('real_world_surrogate.png', dpi=150)
    print("\nInference complete! Saved real_world_surrogate.png")

if __name__ == "__main__":
    train_surrogate(epochs=20000)
    test_surrogate()