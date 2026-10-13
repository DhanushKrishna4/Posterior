"""Regenerate the checked-in oracle with the pinned development environment.

Run .venv/bin/python tests/generate_diagnostic_fixtures.py from the repository.
Production Posterior diagnostics are deliberately not called by this script.
"""

import json
import os
from pathlib import Path
import platform
import runpy
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/posterior-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/posterior-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import arviz as az
import numpy as np
import scipy


def main():
    if az.__version__ != "0.21.0":
        raise RuntimeError("Fixture generation requires ArviZ 0.21.0")
    source = Path(__file__).with_name("test_diagnostics.py")
    values = runpy.run_path(str(source))["fixtures"]()
    records = {}
    for name, array in values.items():
        record = {"rank_rhat": az.rhat(array, method="z_scale"),
                  "folded_rhat": az.rhat(array, method="folded"),
                  "rhat": az.rhat(array, method="rank"),
                  "ess_bulk": az.ess(array, method="bulk"),
                  "ess_tail": az.ess(array, method="tail"),
                  "ess_mean": az.ess(array, method="mean"),
                  "mcse_mean": az.mcse(array, method="mean")}
        records[name] = {key: float(value) for key, value in record.items()}
    report = {"reference": "ArviZ public diagnostic API", "reference_version": az.__version__,
              "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
              "input_recipe": "tests/test_diagnostics.py:fixtures; PCG64 seed 20261006; float64",
              "tolerance": {"relative": 2e-10, "absolute": 2e-10}, "fixtures": records}
    destination = Path(__file__).parent / "fixtures" / "diagnostics-arviz-0.21.0.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
