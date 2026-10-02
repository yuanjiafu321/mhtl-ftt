# MHTL-FTT

This project implements a multi-task FT-Transformer for landslide, collapse, and debris-flow susceptibility modeling. It includes dataset splitting, task-specific negative sampling, preprocessing, model training, evaluation, and probability prediction. 

The code requires Python 3.9 or later, PyTorch 2.1 or later, NumPy, pandas, SciPy, scikit-learn, Matplotlib, joblib, and PyYAML. Dependency versions are specified in `pyproject.toml` and installed automatically. CPU execution is supported; a CUDA-enabled PyTorch installation can be used for GPU execution.

The original research data are confidential and are not included. Synthetic data are provided in `examples/synthetic` to run the program. These files can be replaced with your own positive samples and background pool, using the identifiers and factor columns defined in `configs/default.yaml`. 

From the project directory, install the package in your Python environment and run the default configuration:

```bash
python -m pip install -e .
python -m mhtl_ftt run --config configs/default.yaml --data examples/synthetic --output outputs/full
```

For a shorter CPU test, use:

```bash
python -m mhtl_ftt demo --config configs/smoke.yaml --output outputs/demo
```
