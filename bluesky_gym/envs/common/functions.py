import numpy as np
def bound_angle_positive_negative_180(angles_deg: np.ndarray) -> np.ndarray:
    
    # Convert input to a numpy array if it isn't already
    angles_deg = np.asarray(angles_deg)
    
    # Apply the mapping using numpy's vectorized operations
    mapped_angles = np.where(angles_deg > 180, -(360 - angles_deg), angles_deg)
    mapped_angles = np.where(mapped_angles < -180, 360 + mapped_angles, mapped_angles)
    
    return mapped_angles

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    a = np.radians(bearing)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d/R) + np.cos(lat1) * np.sin(d/R) * np.cos(a))
    lon2 = lon1 + np.arctan2(
        np.sin(a) * np.sin(d/R) * np.cos(lat1),
        np.cos(d/R) - np.sin(lat1) * np.sin(lat2)
    )
    return np.degrees(lat2), np.degrees(lon2)