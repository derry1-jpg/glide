"""
Mountain glacier forward simulation example, in
which we build a glacier system over the Bitterroot
Mountains in western Montana

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import torch
import numpy as np
import pyproj

from torch.nn.functional import avg_pool2d, interpolate
from torch.utils.checkpoint import checkpoint

from glide.model import IceDynamics
from glide.io import VTIWriter
from glide.torch import GlideStep
from glide.field import Field,GridEntity

from glare.model import ImprovedTemperatureIndex
from glare.torch import GlareStep

import xarray as xr

# =============================================================================
# Load data
# =============================================================================

print("Loading geometry...")

N_LEVELS = 6

dem = xr.load_dataset('data/gridded_dem.nc')
vel = xr.load_dataset('data/gridded_velocity.nc')
clm = xr.load_dataset('~/Source/glare/examples/wrangell/data/gridded_climate.nc')
ins = xr.load_dataset('~/Source/glare/examples/wrangell/data/gridded_insolation.nc')
ano = 2.5*xr.load_dataset('data/temp_anomaly/merged_temperature.nc')

crs = pyproj.CRS(dem.spatial_ref.crs_wkt)

x           = cp.array(dem.x)
y           = cp.array(dem.y)

factor = 2**N_LEVELS
nx_target = (len(x) // factor) * factor
ny_target = (len(y) // factor) * factor

# Center the subregion
x_start = (len(x) - nx_target) // 2
y_start = (len(y) - ny_target) // 2

x_slice = slice(x_start, x_start + nx_target)
y_slice = slice(y_start, y_start + ny_target)

x,y = x[x_slice],y[y_slice]

srf         = cp.array(dem.elevation.values[y_slice,x_slice],dtype=cp.float32)
domain_mask = cp.array(dem.domain_mask.values[y_slice,x_slice])
rgi_mask    = cp.array(dem.rgi_mask.values[y_slice,x_slice])

t2m         = cp.array(clm.monthly_t2m.values[:,y_slice,x_slice],dtype=cp.float32)
precip      = cp.array(clm.monthly_precip.values[:,y_slice,x_slice],dtype=cp.float32)
I_pot_mean  = cp.array(ins.monthly_solar_potential_mean[:,y_slice,x_slice],dtype=cp.float32)
I_pot_cos   = cp.array(ins.monthly_solar_potential_cos[:,y_slice,x_slice],dtype=cp.float32)
I_pot_sin   = cp.array(ins.monthly_solar_potential_sin[:,y_slice,x_slice],dtype=cp.float32)

vx          = cp.array(vel.vx.values[y_slice,x_slice],dtype=cp.float32)
vy          = cp.array(vel.vy.values[y_slice,x_slice],dtype=cp.float32)

#smb[~domain_mask] = -10
vx[cp.isnan(vx)]  = 0.0
vy[cp.isnan(vy)]  = 0.0
vx[~domain_mask]  = 0.0
vy[~domain_mask]  = 0.0

ny,nx = srf.shape
dx = (x[1]-x[0]).item()

### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
model = IceDynamics(n_levels=N_LEVELS,ny=ny,nx=nx,dx=dx,
        x0=x[0].item(),y0=y[0].item(),
        crs=crs)
mg = model.mg

### Initialize state
mg.state.H.set(cp.ones((ny,nx),dtype=cp.float32)*0.1)
mg.state.H_prev.set(cp.ones((ny,nx),dtype=cp.float32)*0.1)

### Initialize geometry
mg.geometry.bed.set(srf)
mg.geometry.sigmoid_c.set(1.0)
mg.geometry.sigmoid_k.set(5.0)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-16 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-5)
mg.rheology.n.set(3.0)

beta = cp.zeros((ny,nx), dtype=cp.float32)
beta.fill(2.0)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)

smb_model = ImprovedTemperatureIndex(ny=ny,nx=nx,nt=12,
        dx=dx,dt=1./12,
        x0=x[0].item(),y0=y[0].item(),
        crs=crs)

smb_model.grid.insolation.insol_mean.set(I_pot_mean)
smb_model.grid.insolation.insol_cos.set(I_pot_cos)
smb_model.grid.insolation.insol_sin.set(I_pot_sin)

smb_model.grid.temperature.t2m.set(t2m)

smb_model.grid.precipitation.precip.set(precip)

smb_model.forward()

### Initialize forcing
mg.forcing.smb.set(smb_model.grid.state.smb.data.mean(axis=0))

# Build hierarchy of observations
observation_levels = [(vx,vy,srf,rgi_mask,domain_mask)]#,srf,rgi_mask,domain_mask)]
for j in range(1,N_LEVELS):
    u_obs_coarse = mg.restrict_cell(observation_levels[-1][0])
    v_obs_coarse = mg.restrict_cell(observation_levels[-1][1])
    S_obs_coarse = mg.restrict_cell(observation_levels[-1][2])
    rgi_mask_coarse = mg.restrict_cell(observation_levels[-1][3].astype(cp.float32)).astype(cp.bool_)
    domain_mask_coarse = mg.restrict_cell(observation_levels[-1][4].astype(cp.float32)).astype(cp.bool_)
    observation_levels.append((u_obs_coarse,v_obs_coarse,S_obs_coarse,rgi_mask_coarse,domain_mask_coarse))

### Set multigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

model.adjoint_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10,
        post_steps=50, finest_steps=50,
        relative_tolerance=1e-2, absolute_tolerance=1e-6, # Note that adjoint var
        report_norms=True)                               # adjoint var is small 
                                                          # in magnitude

model.adjoint_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(1.0))

log_beta = torch.tensor(cp.log(mg[0].sliding.beta.data),
        device='cuda',requires_grad=True)
H_prev = torch.tensor(mg[0].state.H_prev.data,
        device='cuda',requires_grad=False)
bed = torch.tensor(mg[0].geometry.bed.data,
        device='cuda',requires_grad=True)

t2m = torch.tensor(smb_model.grid.temperature.t2m.data,
        device='cuda',requires_grad=False)

precip = torch.tensor(smb_model.grid.precipitation.precip.data,
        device='cuda',requires_grad=False)

def V(x,l,root_a,eig_tol=1e-3):
    K =  root_a * cp.exp(-((x - x[:,None])/l)**2)
    l,v = cp.linalg.eigh(K)
    mask = l > 1e-3*l.max()
    return v[:,mask]*cp.sqrt(l[mask])

Vx = 2*torch.as_tensor(V(mg[0].x_cell,20000,1.0)).to(torch.float32)
Vy = 2*torch.as_tensor(V(mg[0].y_cell,20000,1.0)).to(torch.float32)

precipitation_bias = torch.zeros((Vy.shape[1],Vx.shape[1]),
        device='cuda',dtype=torch.float32,
        requires_grad=True)

log_mf = torch.tensor(cp.log(smb_model.grid.temperature.mf.value),
        device='cuda',requires_grad=True)
log_rf = torch.tensor(cp.log(smb_model.grid.insolation.rf.value),
        device='cuda',requires_grad=True)

d = torch.load('inverse_with_glare_refined/level_1/torch_vars.p')
log_beta = d['log_beta']
bed = d['bed']
precipitation_bias = d['precipitation_bias']
log_mf = d['log_mf']
log_rf = d['log_rf']

# Thin Pytorch wrapper of a single glide time step
glide_step = GlideStep.apply
glare_step = GlareStep.apply

rgi_mask_ = torch.tensor(observation_levels[0][3])
domain_mask_ = torch.tensor(observation_levels[0][4])

u_obs,v_obs,S_obs,rgi_mask,domain_mask = (torch.tensor(var) for var in observation_levels[0])

base_anomaly = ano.sel(time=2012).temp_anomaly.item()

for level in range(0,-1,-1):
    model.set_top_level(level)
    delta = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    srf = Field(cp.zeros((mg[level].ny,mg[level].nx),dtype=cp.float32),
            grid_entity=GridEntity.CELL,
            dx=mg[level].dx,
            grid=mg[level])

    uz = cp.zeros(mg[level].state.u.data.shape,dtype=cp.float32)
    vz = cp.zeros(mg[level].state.v.data.shape,dtype=cp.float32)
    Hz = cp.zeros(mg[level].state.H.data.shape,dtype=cp.float32)
    H0 = cp.array(mg[level].state.H.data,dtype=cp.float32)

    mask = torch.tensor(rgi_mask*domain_mask,
            dtype=torch.float32,device='cuda')


    optimizer = torch.optim.Adam([{'params':log_beta,'lr':0.05},
                                     {'params':bed,'lr':10.0},
                                     {'params':precipitation_bias,'lr':0.005},
                                     {'params':log_mf,'lr':0.01},
                                     {'params':log_rf,'lr':0.01}
                                     ],lr=1e-3,betas=(0.5,0.99))
    

    S_obs_0 = observation_levels[level][2]
    vti_writer = VTIWriter(f'inverse_with_glare_refined/level_{level}/vti', base='wrangell', dx=mg[level].dx,
            static_fields={'U_obs':[u_obs,v_obs],
                           'S_obs':torch.as_tensor(S_obs_0)},
    #                       'mask':mask},
            dynamic_fields={'bed':mg[level].geometry.bed,
                            'beta':mg[level].sliding.beta,
                            'thk':mg[level].state.H,
                            'U':[mg[level].state.u,mg[level].state.v],
                            'srf':srf,
                            'delta':delta,
                            'smb':mg[level].forcing.smb}
        )

    vti_writer.initialize(mg[level])


    def compute_smb(smb_model,t2m,t_anomaly,base_anomaly,precip_,mf,rf,domain_mask):
        smb = glare_step(smb_model,t2m + (t_anomaly - base_anomaly),precip_,mf,rf).mean(axis=0)
        smb_ = smb.masked_fill(~domain_mask,-10)
        return smb_
    
    def evaluate_loss(i,compute_gradient=True,write_vti=True):
        optimizer.zero_grad()
        
        mg.state.u.set(uz,start_level=level)
        mg.state.v.set(vz,start_level=level)
        mg.state.H.set(H0,start_level=level)
        mg.state.H_prev.set(H0,start_level=level)
        mg.state.mask.set(Hz,start_level=level)

        mg.adjoint.lambda_u.set(uz,start_level=level)
        mg.adjoint.lambda_v.set(vz,start_level=level)
        mg.adjoint.lambda_H.set(Hz,start_level=level)


        precip_ = precip + (Vy @ precipitation_bias @ Vx.T)[None,:,:]
        mf = torch.exp(log_mf)
        rf = torch.exp(log_rf)

        bed_ = bed
        H_prev_ = H_prev
        log_beta_ = log_beta
        for _ in range(level):
            bed_ = avg_pool2d(bed_[None,:,:],(2,2))[0]
            H_prev_ = avg_pool2d(H_prev_[None,:,:],(2,2))[0]
            #smb_ = avg_pool2d(smb_[None,:,:],(2,2))[0]
            log_beta_ = avg_pool2d(log_beta_[None,:,:],(2,2))[0]

        beta_ = torch.exp(log_beta_)

        t = cp.float32(2012-1000)
        t_end = cp.float32(2012)
        dt = cp.float32(20.0)

        if i % 10 == 0:
            time_writer = VTIWriter(f'inverse_with_glare_refined/level_{level}/vti', base='time', dx=mg[level].dx,
                dynamic_fields={'thk':mg[level].state.H,
                            'U':[mg[level].state.u,mg[level].state.v],
                            'smb':mg[level].forcing.smb}
            )


        while t < t_end:
            t_anomaly_0 = ano.sel(time=int(t)).temp_anomaly.item()
            t_anomaly_1 = ano.sel(time=int(t+dt)).temp_anomaly.item()
            t_anomaly = 0.5*(t_anomaly_0 + t_anomaly_1)
            smb_ = checkpoint(compute_smb,smb_model,t2m,t_anomaly,base_anomaly,precip_,mf,rf,domain_mask,use_reentrant=False)
            for _ in range(level):
                smb_ = avg_pool2d(smb_[None,:,:],(2,2))[0]

            u,v,H = glide_step(t,dt,model,level,H_prev_,bed_,beta_,smb_)

            t += dt
            H_prev_ = H
            
            if i % 20 == 0:
                time_writer.append(mg[level],time=t)
                time_writer.write_pvd()

        # Surface elevation loss
        S = bed_ + H
        S_0 = S

        # Velocity loss
        for _ in range(level):
            u = interpolate(u[None,None,:,:],(2*u.shape[0],2*(u.shape[1] - 1) + 1),mode='bilinear').squeeze()
            v = interpolate(v[None,None,:,:],(2*(v.shape[0] - 1) + 1,2*v.shape[1]),mode='bilinear').squeeze()
            H = interpolate(H[None,None,:,:],(2*H.shape[0],2*H.shape[1]),mode='bilinear').squeeze()
            S = interpolate(S[None,None,:,:],(2*S.shape[0],2*S.shape[1]),mode='bilinear').squeeze()

        J_srf = 2.0*(torch.sqrt((S - S_obs)**2 + 100.0).mean() - 10)
        #J_srf = 10.0*(torch.sqrt((S - S_obs)**2 + 10000.0).mean() - 100)
        #J_srf = 0.1*((S - S_obs)**2).mean()
        
        u_pred = (u[:,1:] + u[:,:-1])/2.
        v_pred = (v[1:,:] + v[:-1,:])/2.        
        J_vel = 3.0*(torch.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2 + 100.0).mean() - 10.0)
        #J_vel = 10.0*(torch.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2 + 10000.0).mean() - 100.0)

        # Extent loss
        p_extent = (2./(1+torch.exp(-H/100.0))-1).clip(min=0.001,max=0.999)
        J_extent = -100.0*(mask*torch.log(p_extent)).mean()
         
        # Tikhonov Regularization - bed
        gy = torch.diff(bed,dim=0)/mg[0].dx
        gx = torch.diff(bed,dim=1)/mg[0].dx
        J_tikh_bed = 1e-8*((gy**2).sum() + (gx**2).sum())*mg[0].dx**2

        # Tikhonov Regularization - log_beta
        gy = torch.diff(log_beta,dim=0)/mg[0].dx
        gx = torch.diff(log_beta,dim=1)/mg[0].dx
        J_tikh_beta = 1e-5*((gy**2).sum() + (gx**2).sum())*mg[0].dx**2
                                          

        # L2 Regularization - log_beta
        J_L2_beta = 10*((log_beta - .69)**2).mean()

        # L2 Regularization - smb offset
        J_smb = (precipitation_bias**2).sum() + (log_rf - 2.9)**2 + (log_mf - .6016)**2

        J = J_srf + J_vel + J_extent + J_tikh_bed + J_tikh_beta + J_L2_beta + J_smb

        if write_vti:
            delta.data[:,:] = cp.asarray(S_0.detach()) - S_obs_0
            srf.data[:,:] = cp.asarray(S_0.detach())
            vti_writer.append(mg[level],time=i)
            vti_writer.write_pvd()

        print(f"{i}, {J.item():.2f}, {J_srf.item():.2f}, {J_vel.item():.2f}, {J_extent.item():.2f}, {J_tikh_bed.item():.2f}, {J_tikh_beta.item():.2f}, {J_L2_beta:.2f}")
        if compute_gradient:
            J.backward()
            bed.grad[~(rgi_mask_)] = 0.0
        return J


    max_iters = 250
    for i in range(0,max_iters):
        evaluate_loss(i)
        optimizer.step()
    evaluate_loss(0,write_vti=False,compute_gradient=False)


    u_xr = mg[level].state.u.to_dataarray()
    v_xr = mg[level].state.v.to_dataarray()
    H_xr = mg[level].state.H.to_dataarray()
    bed_xr = mg[level].geometry.bed.to_dataarray()
    beta_xr = mg[level].sliding.beta.to_dataarray()
    smb_xr = mg[level].forcing.smb.to_dataarray()
    ds = xr.merge([u_xr,v_xr,H_xr,bed_xr,beta_xr,smb_xr])
    ds.to_netcdf(f'inverse_with_glare_refined/level_{level}/inverse_soln.nc')
    torch.save({'log_beta':log_beta,'bed':bed,'precipitation_bias':precipitation_bias,'log_rf':log_rf,'log_mf':log_mf},f'inverse_with_glare_refined/level_{level}/torch_vars.p')



