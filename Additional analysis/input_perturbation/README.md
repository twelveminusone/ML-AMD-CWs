# Input Perturbation Sensitivity

Evaluate how multiplicative perturbations on **input features** affect model performance **at inference time** (no retraining).

- **Dataset**: `../../data/full_dataset.xlsx` (fully imputed).  
- **Features (default C-set)**: `i_COD, i_pH, i_acidity, i_EC, day, height`  
- **Targets**: `TFe (%), Zn (%), Al (%), Mn (%), Ni (%), Co (%), Cr (%)`  
- **Outputs**: `../../outputs/input_perturbation_sensitivity/` → `results.xlsx` + optional figures

## Quick start (run from this folder)

```bash
python input_perturb_sensitivity.py   --input ../../data/full_dataset.xlsx   --output ../../outputs/input_perturbation_sensitivity/results.xlsx   --test_size 0.2 --seed 42 --grid --scales 0.9,1.1
```

Make figures:

```bash
python make_input_perturb_heatmap.py   --input ../../outputs/input_perturbation_sensitivity/results.xlsx   --outdir ../../outputs/input_perturbation_sensitivity/figures   --metric R2 --fmt png --dpi 200

python make_input_perturb_lineplots.py   --input ../../outputs/input_perturbation_sensitivity/results.xlsx   --outdir ../../outputs/input_perturbation_sensitivity/figures   --metric R2 --mode delta --fmt png --dpi 200
```

## Shared utils import

This folder **reuses** the `utils/` package located in
`../imputation_order_sensitivity/utils/` to avoid duplication.

At the top of each script we add a small shim to extend `sys.path` so Python can find the sibling package:

```python
# --- add sibling 'utils/' (from imputation_order_sensitivity) to PYTHONPATH ---
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "imputation_order_sensitivity"
sys.path.insert(0, str(SIBLING))  # so 'from utils.io import ...' works
# -----------------------------------------------------------------------------
```

> Make sure the `utils/` folder contains an `__init__.py`.

This way we keep **one source of truth** for `io.py`, `meta.py`, `common.py`, etc., and all submodules can import them consistently.
