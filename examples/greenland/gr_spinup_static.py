"""
Greenland spinup and forward simulation using SSP 5-8.5 SMB values from ISMIP7
Run from 2015-2300  
Spinup with average SMB from 1979-2019. Only ran through 2014 in script

Run interactively or execute as a script. Modify the paths and parameters
below to match your setup.
"""
import cupy as cp
import numpy as np
import pyproj
import xarray as xr
from scipy.ndimage import gaussian_filter

from glide.model import IceDynamics
from glide.data import load_greenland_preprocessed
from glide.io import ZarrWriter, VTIWriter

### Load a dataset (here a preprocessed greenland dataset)
dataset = load_greenland_preprocessed()


### Initialize grid
# ny and nx must both divide by 2^(n_levels - 1) cleanly!
ny,nx,dx = dataset.ny,dataset.nx,dataset.dx
model = IceDynamics(n_levels=6,ny=ny,nx=nx,dx=dx,
        x0=dataset.x[0].item(),y0=dataset.y[0].item(),
        crs=pyproj.CRS("EPSG:3413"))
mg = model.mg

### Initialize state
thk = gaussian_filter(dataset.thickness.values,1)
mg.state.H.set(thk)
mg.state.H_prev.set(thk)

### Initialize geometry
bed = gaussian_filter(dataset.bed.values,1)
mg.geometry.bed.set(bed)
mg.geometry.depth.set(np.maximum(-bed,0))
mg.geometry.sigmoid_c.set(0.1)
mg.geometry.sigmoid_k.set(3.0)

### Initialize rheology
# Compute B (rate factor - we measure driving stress in units of head, so the rho g factor gets subsumed into definitions of beta and B!)
B = cp.zeros((ny,nx), dtype=cp.float32)
B.fill(1e-17 ** (-1.0 / 3.0) / (917 * 9.81)) 
mg.rheology.B.set(B)
mg.rheology.eps_reg.set(1e-6)
mg.rheology.n.set(3.0)

### Initialize sliding
#BETA_PATH = None
BETA_PATH ="inverse/level_0/beta_opt.nc"
if BETA_PATH:
    import xarray as xr
    beta = cp.array(xr.load_dataarray(BETA_PATH))
else:
    beta = cp.zeros((ny,nx), dtype=cp.float32)
    beta.fill(2.5)

mg.sliding.beta.set(beta)
mg.sliding.m.set(1./3.)
mg.sliding.water_drag.set(1e-4)

### Initialize calving
# Specifies calving velocity for a non-conservative
# calving flux over facets between adjacent floating cells
mg.calving.calving_rate.set(2000.0) 

target_grid = xr.DataArray(
    np.zeros((dataset.ny, dataset.nx)),
    coords={"y": dataset.y, "x": dataset.x},
    dims=("y","x")
)


### Initialize forcing (ISMIP7 SMB - GLIDE SAFE VERSION) **ChatGPT Assisted**

from pathlib import Path

# -------------------------------------------------------
# Paths (fully explicit to avoid runtime confusion)
# -------------------------------------------------------
SMB_DIR = Path("/home/wilson/Sandbox/glide/examples/greenland")

def load_smb_year(year, target_grid):

    file = SMB_DIR / f"acabf_GrIS_CESM2-WACCM_ssp585_dEBM2-1000m_v1_{year}.nc"

    if not file.exists():
        raise FileNotFoundError(f"Missing SMB file: {file}")

    with xr.open_dataset(file) as ds:

        # Mean over monthly cycle → annual climatology
        smb = ds["acabf"].mean(dim="time")


        # -------------------------------------------------------
        # UNIT CONVERSION (kg m^-2 s^-1 → m yr^-1 ice equivalent)
        # -------------------------------------------------------
        SECONDS_PER_YEAR = 365.25 * 24 * 3600
        RHO_ICE = 917.0

        smb = smb * SECONDS_PER_YEAR / RHO_ICE

    
        # Regrid SMB → GLIDE grid
   
        smb_on_glide = smb.interp_like(target_grid, method="linear").fillna(0.0)

    return smb_on_glide




### Setmultigrid solver parameters ###
model.forward_solver.fas_options.set(
        coarsest_steps=200, pre_steps=10, 
        post_steps=150, finest_steps=0,
        relative_tolerance=1e-2, absolute_tolerance=10.0,
        report_norms=True)

#model.forward_solver.vanka_options.relax_phi.set(cp.float32(0.5))
#model.forward_solver.vanka_options.newton_options.ssa_damping.set(cp.float32(0.1))


# Examples of different writing utilities - First writes to vti/pvd
vti_writer = VTIWriter('forward/vti/', base='greenland', dx=mg[0].dx,
        static_fields={'bed':mg[0].geometry.bed,
                       'beta':mg[0].sliding.beta,},
        dynamic_fields={'H':mg[0].state.H,
                        'mask':mg[0].state.mask,
			'SMB':mg[0].forcing.smb}
        )
vti_writer.initialize(mg[0])

# Second writes to zarr archive, which can be converted to netcdf via xarray
zarr_writer = ZarrWriter('forward/example_run.zarr',
        static_fields={'bed':mg[0].geometry.bed,
                       'beta':mg[0].sliding.beta,},
        dynamic_fields={'H':mg[0].state.H,
                        'u':mg[0].state.u,
                        'v':mg[0].state.v,
                        'mask':mg[0].state.mask,}
        )

zarr_writer.initialize(mg[0],overwrite=True)

# Loop Parameters:

start_year = 1979

t = cp.float32(start_year)
t_end = cp.float32(2301.0)
dt = cp.float32(1.0)

### SMB Spinup
smb_spinup = dataset.smb.values

smb_spinup_gpu=cp.asarray(smb_spinup)

while t < t_end:
    print(f"Solving forward problem at t={t} with dt={dt:.2f}")

    year = int(t)
    
    #Switch SMB at 2015
    if year < 2015:
        smb_gpu =smb_spinup_gpu
    else:
        smb_on_glide = load_smb_year(year, target_grid)
        smb_gpu = cp.asarray(smb_on_glide.values)

        print("Year:", year, "SMB mean:", float(smb_on_glide.mean()))


   
    mg.forcing.smb.set(smb_gpu)


    model.forward(t,dt)
   

    # Write
    vti_writer.append(mg[0],time=t)
    vti_writer.write_pvd()
    zarr_writer.append(mg[0],time=t)
    t+= dt

# Finalize zarr for fast xarray reading
zarr_writer.consolidate_metadata()

# If you want a netcdf of the simulation, uncomment:
#import xarray as xr
#sim_ds = xr.load_dataset('forward/example_run.zarr')
#sim_ds.to_netcdf('forward/example_run.nc')
