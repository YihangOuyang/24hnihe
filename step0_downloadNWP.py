import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import timedelta
import pvlib

# ================= Configuration =================
SOURCE_DIR = r"G:/data/NWP/HRES/18Z"
OUTPUT_DIR = r"./Stanford_NWP_Processed"
OUTPUT_FILENAME = "NWP_Stanford_2018_2019_Full.parquet"  # <--- Note the extension

START_DATE = "2018-01-01"
END_DATE = "2019-11-30"

# Stanford Location
TARGET_LAT = 37.427963
TARGET_LON = -122.154785
TARGET_TZ = 'America/Los_Angeles'
TARGET_ALTITUDE = 30

def calculate_rh(t2m_k, d2m_k):
    # (Same function as before, omitted for brevity)
    t_c = t2m_k - 273.15
    d_c = d2m_k - 273.15
    return 100.0 * (np.exp((17.625 * d_c) / (243.04 + d_c)) / 
                    np.exp((17.625 * t_c) / (243.04 + t_c))).clip(0, 100)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Setup Time and Location
    target_dates = pd.date_range(start=START_DATE, end=END_DATE, tz=TARGET_TZ)
    site_location = pvlib.location.Location(TARGET_LAT, TARGET_LON, tz=TARGET_TZ, 
                                            altitude=TARGET_ALTITUDE, name='Stanford')

    # 2. Container for all processed days
    # List accumulation is much faster than repeatedly calling pd.concat
    daily_dataframes = [] 

    print(f"=== Processing Pipeline Started: {START_DATE} to {END_DATE} ===")

    for local_date in target_dates:
        # --- Logic to find the correct NWP file (D-1 18Z) ---
        run_date_local = local_date - timedelta(days=1)
        run_str = run_date_local.strftime("%Y%m%d")
        filename = f"ECMWF_HRES_18Z_{run_str}.nc"
        file_path = os.path.join(SOURCE_DIR, filename)
        
        if not os.path.exists(file_path):
            # Optional: Log missing dates to a list to analyze later
            continue

        try:
            with xr.open_dataset(file_path) as ds:
                # Spatial Interpolation
                ds_point = ds.sel(latitude=TARGET_LAT, longitude=TARGET_LON, method='nearest')
                
                # --- Variable Extraction (De-accumulation) ---
                # Load data into memory (.load()) to close file handle appropriately
                ghi = (ds_point['ssrd'].diff(dim='time') / 3600.0).load()
                bhi = (ds_point['fdir'].diff(dim='time') / 3600.0).load()
                precip = (ds_point['tp'].diff(dim='time') * 1000.0).load()
                
                valid_times = ghi.time
                
                # Instantaneous variables
                t2m = (ds_point['t2m'].sel(time=valid_times) - 273.15).load()
                d2m = ds_point['d2m'].sel(time=valid_times).load()
                u10 = ds_point['u10'].sel(time=valid_times).load()
                v10 = ds_point['v10'].sel(time=valid_times).load()
                lcc = ds_point['lcc'].sel(time=valid_times).load()
                sp = ds_point['sp'].sel(time=valid_times).load()

            # Derived Physics
            rh = calculate_rh(ds_point['t2m'].sel(time=valid_times), d2m)
            ws = np.sqrt(u10**2 + v10**2)
            dhi = (ghi - bhi).clip(min=0)

            # --- DataFrame Construction ---
            df = pd.DataFrame(index=valid_times.values)
            df.index.name = 'time_utc'
            
            # Vectorized assignment
            df = df.assign(
                NWP_GHI=ghi.values,
                NWP_BHI=bhi.values,
                NWP_DHI=dhi.values,
                NWP_T2m=t2m.values,
                NWP_WS10m=ws.values,
                NWP_RH=rh.values,
                NWP_LCC=lcc.values * 100.0,
                NWP_Precip=precip.values,
                NWP_Press=sp.values
            )

            # --- Timezone & Filtering ---
            df.index = df.index.tz_localize('UTC')
            df_local = df.tz_convert(TARGET_TZ)
            
            # Mask: Keep only the target local day
            mask = (df_local.index.date == local_date.date())
            df_target = df_local[mask].copy()

            if df_target.empty:
                continue

            # --- PVLib Physics Enhancement ---
            # Do this on the small slice; it's fast
            solar_pos = site_location.get_solarposition(df_target.index)
            cs = site_location.get_clearsky(df_target.index, model='ineichen')
            dni_extra = pvlib.irradiance.get_extra_radiation(df_target.index)
            cos_zenith = np.cos(np.deg2rad(solar_pos['zenith']))

            df_target['Solar_Zenith'] = solar_pos['zenith'].values
            df_target['Solar_Azimuth'] = solar_pos['azimuth'].values
            df_target['Solar_Elevation'] = solar_pos['elevation'].values
            df_target['CS_GHI'] = cs['ghi'].values
            df_target['CS_DNI'] = cs['dni'].values
            df_target['TOA_GHI'] = (dni_extra * cos_zenith).clip(lower=0).values

            # *** CRITICAL CHANGE: Append to list instead of saving ***
            daily_dataframes.append(df_target)

            if len(daily_dataframes) % 30 == 0:
                print(f"-> Processed {len(daily_dataframes)} days...")

        except Exception as e:
            print(f"[Error] Processing {run_str} for {local_date.date()}: {e}")

    # ================= Final Aggregation =================
    print("=== Aggregating and Saving... ===")
    
    if daily_dataframes:
        # Concatenate all days at once
        full_df = pd.concat(daily_dataframes)
        
        # Sort index just in case
        full_df.sort_index(inplace=True)
        
        # Save as Parquet (Requires: pip install pyarrow or fastparquet)
        save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        full_df.to_parquet(save_path, index=True)
        
        print(f"SUCCESS: Saved {len(full_df)} rows to {save_path}")
        print("Columns:", full_df.columns.tolist())
    else:
        print("WARNING: No data was processed.")

if __name__ == "__main__":
    main()