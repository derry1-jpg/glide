import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
print("Opening dataset...")

ds = xr.open_zarr("forward/example_run.zarr")

print(ds)
print(ds.attrs)

dx = ds.attrs["dx"]
dy = dx

H = ds["H"]

cell_area = dx * dy

volume = H.sum(dim=["y_cell", "x_cell"])*(dx*dx)

volume_km3 = volume / 1e9

years = ds["time"].values

plt.figure(figsize=(8,5))

plt.xlim(years[0], years[-1])

plt.plot(years, volume_km3.values)

dynamic_ticks = np.linspace(years[0],years[-1],num=6, dtype=int)

#if 2015 not in dynamic_ticks:
 #   dynamic_ticks =np.sort(np.append(dynamic_ticks,2015))
#plt.xticks(dynamic_ticks)

plt.axvline(2015, color='red', linestyle='--', alpha=0.5, label='Projection starts (2015)')


plt.xlabel("Simulation Year")
plt.ylabel("Ice Volume (km^3)")
plt.title("Projected Greenland Ice Sheet Volume Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
