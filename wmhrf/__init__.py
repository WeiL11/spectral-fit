"""Shot-noise model of the white-matter resting-state BOLD spectrum filtered by the HRF."""
from . import model, fitting, data  # noqa: F401
import json, os


def save_result(name, obj):
    os.makedirs(data.RESULTS, exist_ok=True)
    with open(os.path.join(data.RESULTS, name), 'w') as fh:
        json.dump(obj, fh, indent=2, default=float)


def load_result(name):
    with open(os.path.join(data.RESULTS, name)) as fh:
        return json.load(fh)


def report(label, value, expected=None, fmt='{:.4g}'):
    """Print a computed value beside the value reported in the manuscript."""
    s = f'  {label:<48} {fmt.format(value)}'
    if expected is not None:
        s += f'   (manuscript: {expected})'
    print(s)


def starts(n):
    """Number of random starts; set WMHRF_QUICK=1 for a fast smoke test (results will be less reliable)."""
    return max(3, n // 20) if os.environ.get('WMHRF_QUICK') else n
