# sky-anisotropy

_Note: this repository is under active development_

Python code for calculating distribution anisotropies vs. sky position.

## Installation

To install directly from GitHub

```
pip install git+https://github.com/jrbourbeau/sky-anisotropy.git
```

or fork the sky-anisotropy GitHub repository and install `sky-anisotropy` on your local machine via

```
git clone https://github.com/<your-username>/sky-anisotropy.git
pip install -e sky-anisotropy
```

### Dependencies

The dependencies for `sky-anisotropy` are:

- NumPy
- SciPy
- pandas
- Healpy
- dask
- xarray

They can be installed using `pip`:

```bash
pip install -r requirements.txt
```

or `conda`:

```bash
conda install --file requirements.txt
```

## Examples

For a (brief) walkthrough of how to use `sky_anisotropy`, see the [example notebook](example.ipynb).
