### Code to derive ice volume and plot it over time.

###     Written Primarily by ChatGPT 



import xarray as xr
import matplotlib.pyplot as plt

print("Opening dataset...")

# 1. LOAD FIRST
ds = xr.open_zarr("forward/example_run.zarr")

# 2. INSPECT
print(ds)
print(ds.attrs)

# 3. NOW you can use ds safely
dx = ds.attrs["dx"]
dy = dx

H = ds["H"]

cell_area = dx * dy

volume = (H * cell_area).sum(dim=["y_cell", "x_cell"])

# 4. Unit Conversion m to km
volume_km3=volume/1e9
years=ds["time"]
# 5. PLOT

plt.figure(figsize=(8,5))

plt.xlim(2015, 2300)

plt.plot(years, volume_km3)
plt.xticks([2015,2050,2100,2150,2200,2250,2300])

plt.xlabel("Simulation Year")
plt.ylabel("Ice Volume (km^3)")
plt.title("Projected Greenland Ice Sheet Volume Over TIme")
plt.grid(True, alpha=0.3)
plt.show()
