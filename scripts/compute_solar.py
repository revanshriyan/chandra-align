import numpy as np
from datetime import datetime

def compute_solar_azimuth_jd(jd, latitude, longitude):
    """Compute solar altitude and azimuth using approximate algorithm.
    Returns (altitude in degrees, azimuth in degrees from North clockwise)."""
    M = 357.5291 + 0.98560028 * (jd - 2451545.0)
    M = M % 360
    L = 280.4665 + 0.98564736 * (jd - 2451545.0)
    L = L % 360
    C = 1.9148 * np.sin(np.radians(M)) + 0.0200 * np.sin(np.radians(2*M))
    lon = L + C
    ecl = 23.4393 * np.sin(np.radians(lon))
    ra = np.degrees(np.arctan2(np.cos(np.radians(lon)) * np.cos(np.radians(ecl)), np.cos(np.radians(lon))))
    dec = np.degrees(np.arcsin(np.sin(np.radians(ecl)) * np.sin(np.radians(lon))))
    GST = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
    GST = GST % 360
    LST = GST + longitude/15.0
    LST = LST % 360
    HA = LST - ra
    HA = (HA + 540) % 360 - 180
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(dec)
    ha_rad = np.radians(HA)
    sin_alt = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    alt = np.degrees(np.arcsin(sin_alt))
    cos_az = (np.sin(dec_rad) - np.sin(lat_rad) * sin_alt) / np.maximum(np.cos(lat_rad) * np.maximum(np.cos(np.arcsin(sin_alt)), 1e-10), 1e-10)
    sin_az = (np.cos(dec_rad) * np.sin(ha_rad)) / np.maximum(np.cos(np.arcsin(sin_alt)), 1e-10)
    az = np.degrees(np.arctan2(sin_az, cos_az))
    az = (az + 360) % 360
    return alt, az

def jd_from_date(y, m, d):
    epoch = datetime(2000, 1, 1)
    date = datetime(y, m, d)
    delta = date - epoch
    days = delta.days
    jd = 2451545.0 + days + 0.5
    return jd

# Compute solar azimuths for the NAC products
# M113679075LC: 2009-11-24, assumption: latitude=-89 (south pole), longitude=0
jd1 = jd_from_date(2009, 11, 24)
alt1, az1 = compute_solar_azimuth_jd(jd1, -89, 0)
print(f"M113679075LC: 2009-11-24, lat=-89, lon=0")
print(f"  Computed alt={alt1:.2f}, az={az1:.2f}")
print(f"  From download_summary: solar_elevation=1.77, solar_incidence=88.23")

# M1315066257LE: 2019-06-13, assumption: latitude=-89, longitude=0
jd2 = jd_from_date(2019, 6, 13)
alt2, az2 = compute_solar_azimuth_jd(jd2, -89, 0)
print(f"\nM1315066257LE: 2019-06-13, lat=-89, lon=0")
print(f"  Computed alt={alt2:.2f}, az={az2:.2f}")
print(f"  From download_summary: solar_elevation=0.71, solar_incidence=89.29")

# OHRC reference: 2024-11-15, sun_azimuth=242.98 from PDS label
print(f"\nOHRC reference: sun_azimuth=242.98 (from PDS label)")

# Azimuth differences
print(f"\nAzimuth differences vs CH-2 value of 242.98:")
print(f"  M113679075LC: {az1:.2f} - 242.98 = {(az1 - 242.98):.2f} degrees")
print(f"  M1315066257LE: {az2:.2f} - 242.98 = {(az2 - 242.98):.2f} degrees")

# Also try with different longitudes
print("\n\nWith adjusted longitudes (to match solar elevation):")
# The computed altitude should match the solar elevation from download_summary
# If not, adjust longitude
for lon in [0, 45, 90, 135, 180, 225, 270, 315]:
    alt1_, az1_ = compute_solar_azimuth_jd(jd1, -89, lon)
    alt2_, az2_ = compute_solar_azimuth_jd(jd2, -89, lon)
    match1 = abs(alt1_ - 1.77)
    match2 = abs(alt2_ - 0.71)
    print(f"  lon={lon:3d}: M113679075LC alt={alt1_:.2f} (err={match1:.2f}), M1315066257LE alt={alt2_:.2f} (err={match2:.2f})")