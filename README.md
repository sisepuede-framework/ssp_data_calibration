# ssp_data_calibration
Calibration pipeline for Sisepuede initial conditions.  The repository
contains a set of utilities and scripts to calibrate SSP input data so
that projected emissions are consistent with reference inventories (for
example EDGAR).  Optimisation is performed via Particle Swarm
Optimisation (PSO).

## Instructions: Setting Up the SISEPUEDE Environment

### 1. **Go to the `environment.yml` file**

Obtain the provided `environment.yml` file for SISEPUEDE.

---

### 2. **Create the Environment from the `.yml` File**

In your terminal, navigate to the directory containing your `environment.yml` file, then run:

```bash
conda env create -f environment.yml
```

This will create a new Conda environment.

---

### 4. **Activate the Environment**

After installation, activate your new environment with:

```bash
conda activate <your_env_name>
```

*(Replace `<your_env_name>` with the name specified in the `.yml` file, e.g., `opt_env`)*

---

### 5. **Done!**

Your environment is now ready to use, with all dependencies (including those installed via pip) preconfigured.

---

#### **Tips:**

* If you update the `environment.yml` file later, you can update your environment with:

  ```bash
  conda env update -f environment.yml --prune
  ```
* You can list all your environments with:

  ```bash
  conda env list
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
   and clipping can be applied although this is outdated and it will be removed in a upcoming version.
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

## Additional Comments
- The latest version of the opt routine is the Uganda one. if you wish to create a opt routine for a new country please use Uganda
as the foundation.