import xarray as xr
import numpy as np
from glide.data import load_bedmachine,load_smb_racmo,prepare_grid,load_antarctic_velocity,interpolate_to_grid

SKIP=4
N_LEVELS=6

GEOMETRY_PATH = "./NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
SMB_PATH = "./smbgl_monthlyS_ANT11_RACMO2.4p1_ERA5_197901_202312.nc"
BETA_PATH = "../inverse/level_0/beta_opt.nc"
U_OBS_PATH = "./antarctica_ice_velocity_450m_v2.nc"

geometry = load_bedmachine(GEOMETRY_PATH, skip=SKIP, thklim=0.1,bbox_pad=[1100,1000,1600,1600])
geometry = prepare_grid(geometry, n_levels=N_LEVELS-1)
bed = geometry['bed']
thickness = geometry['thickness']
surface = geometry['surface']
ny, nx = geometry['ny'], geometry['nx']

dx = geometry['dx']
x, y = geometry['x'], geometry['y']
smb = load_smb_racmo(SMB_PATH,x,y)
x_vel,y_vel,vx,vy = load_antarctic_velocity(U_OBS_PATH)
u_obs_cell = interpolate_to_grid(vx, x_vel, y_vel, x, y)
v_obs_cell = interpolate_to_grid(vy, x_vel, y_vel, x, y)
beta = np.array(xr.load_dataarray(BETA_PATH))

ds = xr.Dataset(
    data_vars=dict(
        bed=(["y","x"],bed),
        surface=(["y","x"],surface),
        thickness=(["y","x"],thickness),
        smb=(["y","x"],smb),
        beta=(["y","x"],beta),
        vx=(["y","x"],u_obs_cell.get()),
        vy=(["y","x"],v_obs_cell.get()),
        ),

    coords=dict(
        x=("x",x),
        y=("y",y)
        ),

    attrs={
        'title':'Input data for GLIDE Antarctica',
        'author':'Doug Brinkerhoff',
        'spatial_ref':"EPSG:3031",
        'ny':ny,
        'nx':nx,
        'dx':dx
        }
)
ds.to_netcdf('GLIDE_antarctica_inputs_v2.nc')

