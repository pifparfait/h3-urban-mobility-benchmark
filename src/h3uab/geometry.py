import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lng1, lat2, lng2):
    lat1 = np.asarray(lat1, dtype=float)
    lng1 = np.asarray(lng1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lng2 = np.asarray(lng2, dtype=float)

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lng2 - lng1)

    a = (
        np.sin(dp / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    )
    return (
        2.0
        * EARTH_RADIUS_M
        * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    )
