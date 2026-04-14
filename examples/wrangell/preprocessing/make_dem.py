import xarray as xr
import rioxarray
import numpy as np
import shapely
import geopandas
from projection_dictionary import crs

def largest_valid_rectangle(valid):
    rows, cols = valid.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, 0, 0, 0, 0)  # area, y_start, x_start, y_end, x_end

    for i in range(rows):
        heights = np.where(valid[i], heights + 1, 0)

        # Largest rectangle in this histogram row
        stack = []
        for j in range(cols + 1):
            h = heights[j] if j < cols else 0
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                x_start = 0 if not stack else stack[-1] + 1
                width = j - x_start
                area = height * width
                if area > best[0]:
                    best = (area, i - height + 1, x_start, i, x_start + width - 1)
            stack.append(j)

    _, y_start, x_start, y_end, x_end = best
    return y_start, y_end, x_start, x_end

DEM_PATH = '../data/dem/cop90/output_hh.tif'
DOMAIN_MASK_PATH = '../data/area/domain/outline.csv'
GLACIER_MASK_PATH = '../data/area/rgi/rgi_ak/RGI2000-v7.0-C-01_alaska.shp'

OUTPUT_PATH = 'model_inputs/gridded_dem.nc'

# Load the COP90 DEM
ds = xr.open_dataset(DEM_PATH)
ds = ds.rename({'band_data':'elevation'})
ds = ds.isel(band=0)
ds = ds.drop_vars('band')

if ds.rio.crs is None:
    ds.rio.write_crs("EPSG:4326", inplace=True)

# Reproject to project CRS
ds = ds.rio.reproject(crs,resolution=90)

# Get the elevation part of projected DEM
da = ds["elevation"]  
valid = da.notnull().values

# Get the largest bounding box that contains all valid pixels
y_start, y_end, x_start, x_end = largest_valid_rectangle(valid)
da_trimmed = da.isel(y=slice(y_start, y_end + 1), x=slice(x_start, x_end + 1))
ds_trimmed = da_trimmed.to_dataset()

# Load a csv of domain to model (here we exclude the chunk
# of the St. Elias that lives in the bounding box
outline = np.loadtxt(DOMAIN_MASK_PATH,delimiter=',')
poly = shapely.Polygon(outline[:,:2])

# Convert to a mask
mask = da_trimmed.rio.clip([poly],crs="EPSG:4326",invert=False,drop=False).notnull()
ds_trimmed['domain_mask'] = mask

# Load RGI glaciers and convert to a raster mask
geodf = geopandas.read_file(GLACIER_MASK_PATH)
glacier_mask = ds_trimmed.elevation.rio.clip(geodf.geometry.values,geodf.crs,drop=False).notnull()
ds_trimmed['rgi_mask'] = glacier_mask


# Write combined DEM and masks to nc
ds_trimmed.to_netcdf(OUTPUT_PATH)





