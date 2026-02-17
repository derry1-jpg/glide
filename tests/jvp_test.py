import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from glide import IcePhysics


# =============================================================================
# Configuration - modify these paths and parameters
# =============================================================================

L = 20000
EXP = 'C'

# Physical constants
RHO_ICE = 917.0
G = 9.81
N_GLEN = 3.0

# =============================================================================
# Configure Domain
# =============================================================================

base_res = 128

y_factr = 7
x_factr = 7

ny = base_res*y_factr
nx = base_res*x_factr

y_slice = int((y_factr//2  +  1./4) * base_res)
x_slice = slice(x_factr//2*base_res,(x_factr//2 + 1)*base_res,1)

x = cp.linspace(0,x_factr*L,nx,dtype=cp.float32)
y = cp.linspace(0,y_factr*L,ny,dtype=cp.float32)
dx = (x[1] - x[0]).item()

X,Y = cp.meshgrid(x,y)

srf = 1000.0 * cp.ones((ny,nx),dtype=cp.float32) - cp.tan(cp.deg2rad(0.1))*Y + 10000
bed = srf - 1000 
thk = srf - bed

if EXP == 'C':
    beta = (1000*cp.sin(2*cp.pi*X/L)*cp.sin(2*cp.pi*Y/L) + 1000)/(RHO_ICE*G)
elif EXP == 'D':
    beta = (1000*cp.sin(2*cp.pi*X/L) + 1000)/(RHO_ICE*G)
else:
    raise NotImplementedError('Only support ISMIP-HOM C and D for now')

#beta.fill(beta.mean())
#beta*=0.1

smb = cp.zeros_like(thk)

# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B_scalar = cp.float32(1e-16 ** (-1.0 / N_GLEN) / (RHO_ICE * G))
B = B_scalar * cp.ones((ny, nx), dtype=cp.float32)
#B.fill(0)

# =============================================================================
# Initialize physics
# =============================================================================

print("Initializing physics...")
physics = IcePhysics(ny, nx, dx, n_levels=1,m=1./3)
physics.set_geometry(bed, thk)
physics.set_parameters(B=B, beta=beta, smb=smb)

#cp.random.seed(0)

# Access the grid hierarchy
grid = physics.grid
grid.set_rhs()
grid.vanka_sweep(2000)

grid.mask[:,:] = cp.random.randint(0,2,size=grid.mask.shape).astype(cp.float32)
grid.mask.fill(0)

grid.d_u[:,:] = cp.random.randn(*grid.u.shape,dtype=cp.float32)
grid.d_u[:,0].fill(0)
grid.d_u[:,-1].fill(0)

grid.d_v[:,:] = cp.random.randn(*grid.v.shape,dtype=cp.float32)
grid.d_v[0].fill(0)
grid.d_v[-1].fill(0)

grid.d_H[:,:] = cp.random.randn(*grid.H.shape,dtype=cp.float32)
grid.d_H[grid.mask>0.5] = 0
#grid.d_H.fill(0)

grid.compute_jvp()

u_0 = cp.array(grid.u)
v_0 = cp.array(grid.v)
H_0 = cp.array(grid.H)

eps = cp.float32(1e-2)

grid.u[:,:] = u_0 + eps * grid.d_u
grid.v[:,:] = v_0 + eps * grid.d_v
grid.H[:,:] = H_0 + eps * grid.d_H
grid.compute_residual()
r1_u = cp.array(grid.r_u)
r1_v = cp.array(grid.r_v)
r1_H = cp.array(grid.r_H)


grid.u[:,:] = u_0 - eps * grid.d_u
grid.v[:,:] = v_0 - eps * grid.d_v
grid.H[:,:] = H_0 - eps * grid.d_H

grid.compute_residual()
r0_u = cp.array(grid.r_u)
r0_v = cp.array(grid.r_v)
r0_H = cp.array(grid.r_H)

j_u_fd = (r1_u - r0_u)/(2*eps)
j_v_fd = (r1_v - r0_v)/(2*eps)
j_H_fd = (r1_H - r0_H)/(2*eps)

abs_err_u = cp.linalg.norm(j_u_fd - grid.j_u)
abs_err_v = cp.linalg.norm(j_v_fd - grid.j_v)
abs_err_H = cp.linalg.norm(j_H_fd - grid.j_H)

rel_err_u = abs_err_u / cp.linalg.norm(grid.j_u)
rel_err_v = abs_err_v / cp.linalg.norm(grid.j_v)
rel_err_H = abs_err_H / cp.linalg.norm(grid.j_H)

print(f"Relative norm of jvp versus finite difference: {rel_err_u}, {rel_err_v}, {rel_err_H}")




