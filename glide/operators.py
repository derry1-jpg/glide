import cupy as cp
from pathlib import Path

class ForwardOperators:
    def __init__(self,grid,use_fast_math=True):
        self.grid = grid

        cuda_dir = Path(__file__).parent / "cuda"

        # Concatenate ice kernel files in dependency order
        cuda_files = ['common.cu', 'viscosity.cu', 'stress.cu', 'flux.cu',
                          'residuals.cu', 'vanka.cu', 'grad.cu']
        cuda_source = '\n'.join((cuda_dir / f).read_text() for f in cuda_files)
        
        if use_fast_math:
            options=("--use_fast_math",)
        else:
            options=()

        self.kernels = cp.RawModule(code=cuda_source, options=options)

        self.Z_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.r_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.r_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.r_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        
        self.f_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.f_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.f_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.delta_u = cp.zeros((grid.ny,grid.nx+1),dtype=cp.float32)
        self.delta_v = cp.zeros((grid.ny+1,grid.nx),dtype=cp.float32)
        self.delta_H = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)

        self.gamma = cp.zeros((grid.ny,grid.nx),dtype=cp.float32)
        self.gamma.fill(grid.geometry.thklim.value)

    @property
    def _kernel_config(self):
        block_size = (16, 16)
        stride = 14
        halo = 1
        grid_size = (self.grid.nx // stride + 1, self.grid.ny // stride + 1)
        return grid_size, block_size, stride, halo

    def compute_residual(self, dt, use_mask=True, enable_calving=True, recompute_grounded=True):
        kernel = self.kernels.get_function('compute_residual')
        grid_size, block_size, stride, halo = self._kernel_config
  
        grid = self.grid

        if use_mask:
            mask = grid.state.mask.data
        else:
            mask = self.Z_H     

        if enable_calving:
            calving_rate = grid.calving.calving_rate.value
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded()

        kernel(grid_size, block_size,
               (self.r_u, self.r_v, self.r_H,
                grid.state.u.data, 
                grid.state.v.data,
                grid.state.H.data, 
                grid.state.grounded.data,
                self.f_u, self.f_v, self.f_H,
                grid.geometry.bed.data, 
                grid.rheology.B.data, 
                grid.sliding.beta.data,
                mask, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, 
                grid.sliding.m.value, grid.sliding.u_reg.value,
                grid.sliding.water_drag.value,
                calving_rate, grid.geometry.sigmoid_c.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo)) 

    def compute_grounded(self, relaxation=cp.float32(0.0)):
        kernel = self.kernels.get_function('compute_grounded')
        grid_size, block_size, stride, halo = self._kernel_config
        
        grid = self.grid
        kernel(grid_size, block_size,
               (grid.state.grounded.data,
                grid.state.H.data, grid.geometry.bed.data, 
                grid.geometry.sigmoid_c.value,
                relaxation,
                grid.ny, grid.nx, 
                stride, halo))

    def vanka_smooth(self, dt, n_inner=1, enable_calving=True, recompute_grounded=True, relax_grounded=cp.float32(0.5)):
        kernel = self.kernels.get_function('vanka_smooth')
        grid_size, block_size, stride, halo = self._kernel_config

        grid = self.grid

        if enable_calving:
            calving_rate = grid.calving.calving_rate.value
        else:
            calving_rate = cp.float32(0.0)

        if recompute_grounded:
            self.compute_grounded(relaxation=relax_grounded)

        self.delta_u.fill(0.0)
        self.delta_v.fill(0.0)
        self.delta_H.fill(0.0)
        kernel(grid_size, block_size,
               (self.delta_u, self.delta_v, self.delta_H, 
                grid.state.mask.data,
                grid.state.u.data, grid.state.v.data, grid.state.H.data, grid.state.grounded.data,
                self.f_u, self.f_v, self.f_H,
                grid.geometry.bed.data, grid.rheology.B.data, grid.sliding.beta.data, self.gamma,
                grid.rheology.n.value, grid.rheology.eps_reg.value, 
                grid.sliding.m.value, grid.sliding.u_reg.value,
                grid.sliding.water_drag.value,
                calving_rate, grid.geometry.sigmoid_c.value,
                grid.dx, dt,
                grid.ny, grid.nx, stride, halo,
                n_inner))

    def vanka_sweep(self, dt, n_iter, verbose=True, n_inner=30, omega=cp.float32(0.5),enable_calving=True,recompute_grounded=True):

        for _ in range(n_iter):
            self.vanka_smooth(dt,n_inner=n_inner,enable_calving=enable_calving,recompute_grounded=recompute_grounded)
            self.grid.state.u.data[:] += omega * self.delta_u
            self.grid.state.v.data[:] += omega * self.delta_v
            self.grid.state.H.data[:] += omega * self.delta_H
        if verbose:
            self.compute_residual(dt,use_mask=True,recompute_grounded=False)
            print(self.grid.dx,cp.linalg.norm(self.r_u),
                  cp.linalg.norm(self.r_v),
                  cp.linalg.norm(self.r_H))

    def set_rhs(self,dt):
        self.f_H[:,:] = self.grid.state.H_prev.data/dt + self.grid.forcing.smb.data
