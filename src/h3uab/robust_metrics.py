import numpy as np


def absolute_error(reference, approximation):
    r = np.asarray(reference, dtype=float)
    a = np.asarray(approximation, dtype=float)
    return np.abs(a - r)


def data_driven_relative_floor(
    reference,
    absolute_floor=1e-6,
    fraction_of_positive_median=0.01,
):
    r = np.asarray(reference, dtype=float)
    positive = np.abs(
        r[np.isfinite(r) & (np.abs(r) > absolute_floor)]
    )

    if positive.size == 0:
        return float(absolute_floor)

    return float(
        max(
            absolute_floor,
            fraction_of_positive_median * np.median(positive),
        )
    )


def conditional_relative_error(reference, approximation, floor):
    r = np.asarray(reference, dtype=float)
    a = np.asarray(approximation, dtype=float)

    out = np.full(r.shape, np.nan, dtype=float)
    mask = np.isfinite(r) & np.isfinite(a) & (np.abs(r) >= floor)

    out[mask] = (
        np.abs(a[mask] - r[mask])
        / np.abs(r[mask])
    )
    return out


def smape(reference, approximation):
    r = np.asarray(reference, dtype=float)
    a = np.asarray(approximation, dtype=float)

    num = 2.0 * np.abs(a - r)
    den = np.abs(r) + np.abs(a)

    out = np.zeros(r.shape, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def nmae(reference, approximation):
    r = np.asarray(reference, dtype=float)
    a = np.asarray(approximation, dtype=float)

    scale = float(np.mean(np.abs(r)))
    if scale <= 0:
        return np.nan

    return float(
        np.mean(np.abs(a - r))
        / scale
    )
