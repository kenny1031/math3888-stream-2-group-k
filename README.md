# MATH3888 PPI Project
This repo contains MATH3888 Stream 2 Group K's resources and code.

We mainly use Python (Jupyter Notebook) with:
* NetworkX
* NumPy
* SciPy
* pandas
* Matplotlib

Protein interaction data mainly come from [STRING-DB](https://string-db.org).

## 1. First-time setup
You only need to do this ONCE.

### Step 1: Install Python
Install **Python 3.12** to your local device if you do not already have it. 

Check your Python version by typing in your terminal:
```bash
python3 --version
```

You should see something like:
```text
Python 3.12.x
```

### Step 2: Clone the repository
In your terminal, run:
```bash
git clone https://github.com/kenny1031/math3888-stream-2-group-k
```

Then move into the project folder:
```bash
cd math3888-stream-2-group-k
```

### Step 3: Create a virtual environment
A virtual environment keeps the packages required separate from the rest of your local device.

Run:
```bash
python -m venv .venv
```

Activate it.

macOS / Linux
```bash
source .venv/bin/activate
```

Windows
```
.venv\Scripts\activate
```

When it is activated, you should see something like `(.venv)` at the beginning of your terminal line.

### Step 4: Install the required packages
With the virtual environment activated:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs all packages needed for this project (I already written up `requirements.txt`)

### Step 5: Start Jupyter
Run:
```bash
jupyter lab
```
A browser window should open automatically. Most of our work will be inside the `notebooks/` folder.

## 2. Every time you work on the project
Open Terminal and go to the repository:
```bash
cd math3888-stream-2-group-k
```

Activate environment:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```text
.venv\Scripts\activate
```

Before changing anything, download the latest version from GitHub:
```bash
git pull
```

Then start Jupyter"
```bash
jupyter lab
```

## 3. Repo structure
```text
.
├── data/
│   ├── raw/              # Original data downloaded from STRING
│   └── processed/        # Cleaned / processed data
│
├── notebooks/            # Jupyter notebooks
│
├── src/                  # Reusable Python functions
│
├── figures/              # Generated figures
│
├── outputs/              # Generated results
│
├── requirements.txt      # Required Python packages
├── .gitignore
└── README.md
```

In general:
* use `notebooks/` for analysis, experiments, explanations and plots;
* use `src/` for Python functions that we want to reuse;
* DO NOT manually modify `.venv/`;
* large downloaded datasets should usually stay inside `data/` and should NOT be uploaded to GitHub.

## 4. Basic Git workflow
Before starting work:
```bash
git pull
```

After making changes, check what you changed:
```bash
git status
```

Add your changes:
```bash
git add .
```

Create a commit:
```bash
git commit -m "brief description of your changes"
```
E.g.
```bash
git commit -m "add PPI network construction"
```
Then upload your changes:
```bash
git push
```

A typical workflow is therefore:
```bash
git pull

# do your work

git add .
git commit -m "changes description"
git push
```

## 5. Important GitHub rules
### Always `git pull` before starting work
Someone else may have changed the repo since you last worked on it.

### Do not edit the same notebook at the same time
Jupyter notebooks can be difficult to merge when two people change the same notebook simultaneously.

### Do not upload `.venv`
It contains your local Python environment and should never be committed. It is already excluded by `.gitignore`.

### Do not upload large raw datasets unless we agree to

STRING datasets may be large. Instead, we should record:
* where the data came from;
* how it was downloaded;
* which version / organism was used;
* how it was processed.

### Use meaningful commit messages
Good:
```text
add shortest path function
filter STRING interactions above threshold 750
plot degree distribution
fix essential protein lookup
```

Not very useful:
```text
update
stuff
changes
final
final2
```

## 6. If you install a new Python package
E.g.
```bash
pip install seaborn
```

If the package becomes necessary for the project, also add it to `requirements.txt`. This ensures everyone else can install the same dependencies.

## 7. Common problems
### `command not found: python`
Try `python3` instead of `python`.

### A Python package cannot be found
Make sure the virtual environment is activated:
```bash
source .venv/bin/activate
```
Then run:
```bash
pip install -r requirements.txt
```
again.

### Git says there is a merge conflict
Do not randomly delete files or force-push.

Send the error message to the group first so we can figure it out together.

### Jupyter cannot find the packages
Make sure Jupyter was started after activating `.venv`:
```bash
source .venv/bin/activate
jupyter lab
```