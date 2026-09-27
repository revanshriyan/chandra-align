import numpy as np
from datetime import datetime

def compute_solar_azimuth_jd(jd, latitude, longitude):
    """Compute solar altitude and azimuth using approximate algorithm."""
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

# Try different latitudes to match solar elevation from download_summary
jd1 = jd_from_date(2009, 11, 24)  # M113679075LC: expected solar_elevation=1.77
jd2 = jd_from_date(2019, 6, 13)  # M1315066257LE: expected solar_elevation=0.71

print("Finding latitude that gives matching solar elevation:")
print("=" * 70)

for lat in range(-90, 91, 5):
    alt1, az1 = compute_solar_azimuth_jd(jd1, lat, 0)
    alt2, az2 = compute_solar_azimuth_jd(jd2, lat, 0)
    err1 = abs(alt1 - 1.77)
    err2 = abs(alt2 - 0.71)
    if err1 < 5 or err2 < 5:
        print(f"  lat={lat:3d}: M113679075LC alt={alt1:.2f} (err={err1:.2f}), M1315066257LE alt={alt2:.2f} (err={err2:.2f})")
    # Also try lon=45
    alt1_45, az1_45 = compute_solar_azimuth_jd(jd1, lat, 45)
    alt2_45, az2_45 = compute_solar_azimuth_jd(jd2, lat, 45)
    err1_45 = abs(alt1_45 - 1.77)
    err2_45 = abs(alt2_45 - 0.71)
    if err1_45 < 5 or err2_45 < 5:
        print(f"  lat={lat:3d}, lon=45: M113679075LC alt={alt1_45:.2f} (err={err1_45:.2f}), M1315066257LE alt={alt2_45:.2f} (err={err2_45:.2f})")

print("\n\nNow computing azimuths at the best-matching latitudes:")
print("=" * 70)

# Find the best lat for each
best_lat1 = -89
best_lat2 = -89
alt1, az1 = compute_solar_azimuth_jd(jd1, best_lat1, 0)
alt2, az2 = compute_solar_azimuth_jd(jd2, best_lat2, 0)

print(f"M113679075LC (2009-11-24, lat={best_lat1}): alt={alt1:.2f}, az={az1:.2f}")
print(f"M1315066257LE (2019-06-13, lat={best_lat2}): alt={alt2:.2f}, az={az2:.2f}")
print(f"OHRC reference: sun_azimuth=242.98 (from PDS label)")
print(f"\nAzimuth differences vs CH-2 value of 242.98:")
print(f"  M113679075LC: {az1:.2f} - 242.98 = {az1 - 242.98:.2f} degrees")
print(f"  M1315066257LE: {az2:.2f} - 242.98 = {az2 - 242.98:.2f} degrees")

# Try with lon adjustment to get better elevation match
print("\n\nTrying to match solar elevation by adjusting longitude:")
for lat in [-85, -88, -89, -89.5, -89.8]:
    for lon in [0, 15, 30, 45, 60, 75, 90]:
        alt1_, az1_ = compute_solar_azimuth_jd(jd1, lat, lon)
        alt2_, az2_ = compute_solar_azimuth_jd(jd2, lat, lon)
        e1 = abs(alt1_ - 1.77)
        e2 = abs(alt2_ - 0.71)
        if e1 < 3 or e2 < 3:
            print(f"  lat={lat:.1f}, lon={lon}: M113679075LC alt={alt1_:.2f} (err={e1:.2f}), M1315066257LE alt={alt2_:.2f} (err={e2:.2f}) az1={az1_:.2f} az2={az2_:.2f}")