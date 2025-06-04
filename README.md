# ssp_data_calibration
Calibration pipeline for Sisepuede initial conditions.  The repository
contains a set of utilities and scripts to calibrate SSP input data so
that projected emissions are consistent with reference inventories (for
example EDGAR).  Optimisation is performed via Particle Swarm
Optimisation (PSO).

## Get Started

Create a conda env with python 3.11 (You can use any name)

```sh
conda create -n opt_env python=3.11
```

Activate the env

```sh
conda activate opt_env
```

Install the working version of the sisepuede package

```sh
pip install git+https://github.com/jcsyme/sisepuede.git@working_version
```

Install additional libraries

```sh
pip install -r requirements.txt
```

## Repository structure

```
├── data/                # Input CSV files used for calibration
├── src/
│   ├── pso_opt.py       # Main optimisation script
│   ├── pso_opt_uganda.py
│   ├── config/          # YAML configuration files
│   ├── utilities/       # Helper classes and error functions
│   └── misc/            # Crosswalk tables and example reports
└── requirements.txt     # Python dependencies
```

Output from the optimisation is written to an `output/<region>/<run id>`
directory that is created when a run starts.

## How the optimisation works

1. Parameters are read from a YAML file in `src/config/`.  These specify
   the region, input file, variable bounds and PSO settings.
2. The input dataset is loaded from `data/`.  Groups of variables are
   scaled according to the optimisation vector.  Optional normalisation
   and clipping can be applied.
3. The stressed dataset is passed to the Sisepuede model
   (`utilities.SSPModelForCalibration`) to obtain simulated emissions.
4. `utilities.DiffReportUtils` compares the simulation with EDGAR (or a
   custom inventory) and produces sectoral and subsector reports.
5. An error value is computed using `utilities.ErrorFunctions` (e.g.
   WMSE, MAPE).  This error is minimised by the `pyswarm` PSO
   optimiser.
6. The best scaling factors, input file and diff reports are stored in
   the run output directory.

## Usage

After setting up the Python environment you can start the optimisation
by running one of the scripts in `src/`:

```sh
# Example for Croatia
python src/pso_opt.py

# Example for Uganda
python src/pso_opt_uganda.py
```

Each script reads its corresponding YAML configuration file.  Adjust the
parameters there to change the input file, number of iterations or error
settings.  Results will be written inside `output/<region>/` with a
timestamped run identifier.
