# PPO-HER
Test if we can use Hindsight Experience Replay (HER) with Proximal Policy Optimization (PPO).  Spoiler alert - we can, at least for our predator-prey environments.

This repo provides code for 2 papers:

[PPO-HER](https://arxiv.org/abs/2410.22524) - provides initial evidence for how HER can accelerate PPO

[Maximum Entropy HER (MEHER)](https://arxiv.org/abs/2410.24016) - provides an information theoretic improvement to PPO-HER

# Installation
Linux is the only officially-supported OS.

## Linux

To install via `pip` with a virtual environment (callend `venv`):
1. `git clone <repo_url>`
2. `cd ppo-her`
3. `bash setup.sh`

To run code:
1. `cd ppo-her`
2. `source venv/bin/activate`
3. `cd ppo_her/experiments/ppo-her`
4. `python experiment.py`


## MacOS
This has not been tested and is not supported, but you can try to use the instructions for Linux.

## Windows
This has not been tested and is not supported, but you can try to manually install the libraries listed in `setup.sh` and `requirements.txt`.


Things that may help:
    Using "Future" method
    Changing p(using future goal)
    Maximum entropy HER
    Clipping VF gradients

# Repo Organization

## Images
This folder will only appear after results have been generated.  It contains all plots in sub-folders that are organized by experiment.

## ppo_her
This folder contains all code.  Ideally, this folder will contain no results (although, for now, the slurm logs get saved to sub_folders)

### base
Code for running experiments including:
1. `config.py` - default configuration file
2. `custom_ppo.py` - PPO from stable-baselines3, overwritten to include HER
3. `env.py` - the environment.  The behavior of the enviroment is changed by changing the configs.
4. `experiment.py` - contains a script for training the PPO agent
5. `get_model.py` - gets the RL agent model when provided with the config
6. `run.py` - a script for training RL agents in parallel using ray.tune
7. `utils.py` - various helper functions that mostly help with organizing results

### experiments
Each sub-folder is a specific experiment, which should contain a README with a specific description of the experiment.  To reproduce the results using a slurm cluster, we recommend the following workflow:
1. `cd <experiment_name>`
2. `sbatch optimize.slurm`
3. Wait until the the job completes
4. `sbatch process.slurm`

In general, each sub-folder will contain the following files:
1. `calc_diff.py` - Compare conditions
2. `calc_stats.py` - Calculate RL agent performance and time-to-learn
3. `experiment.py` - Set the configuration and run the experiment
4. `optimize.slurm` - Run the experiment on the slurm cluster
5. `plot.py` - Plot the results of the summary
6. `process.py` - Load the tensorboard log data and summarize it
7. `process.slurm` - Run all of the processing scripts using slurm
8. `README.md` - a description of the experiment

### processing
These are scripts for processing the results:
1. `bar_plot.py` - Create a bar plot
2. `base_x_formatter.py` - Scale and label the X-axis depending on whether we are using time or steps
3. `base.py` - The base processor.  Loads data from tensorboard files
4. `calc_diff.py` - Compare experimental conditions
5. `calc_stats.py` - Calculate performance and time-to-learn stats.  Output a table.
6. `plot_only.py` - Plot pre-processed summary data produced by `base.py`

## ray_results
These are the tensorboard logs produced by running experiments.  Each experiment in `ray_results` should correspond with a folder in `experiments`.

## stats
Each sub-folder contains summary statistics for each experiment, in the form of a csv file.  Fields include time-to-learn and performance.

## summary_data
Each sub-folder contains summary statistics for each experiment, in the form of a json file.  Fields include time-series data (stats contains max/min over these time-series).

## tex
Summary data in the form of tex tables that are mostly acceptable for adding to Latex papers.

## Other files:

The following files are for setup:
1. `requirements.txt`
2. `setup.py`
3. `setup.sh`

`TODO.txt` is a list of experimental ideas

`papers.txt` is a list of high-level tasks required for writing papers

## PIP list
For reproducability, here is the output of PIP list:
Package                   Version     Location
------------------------- ----------- ------------------------------------------
absl-py                   2.3.0
aiosignal                 1.3.2
asttokens                 3.0.0
async-timeout             5.0.1
attrs                     25.3.0
certifi                   2025.6.15
charset-normalizer        3.4.2
click                     8.0.4
cloudpickle               3.1.1
contourpy                 1.3.0
cycler                    0.12.1
decorator                 5.2.1
distlib                   0.3.9
exceptiongroup            1.3.0
executing                 2.2.0
Farama-Notifications      0.0.4
filelock                  3.18.0
flatdict                  4.0.1
flatten-dict              0.4.2
fonttools                 4.58.4
frozenlist                1.7.0
fsspec                    2025.5.1
future                    1.0.0
gitdb                     4.0.12
GitPython                 3.1.44
glfw                      2.9.0
grpcio                    1.73.1
gymnasium                 0.29.1
gymnasium-robotics        1.2.4
hyperopt                  0.2.7
idna                      3.10
imageio                   2.37.0
importlib-metadata        8.7.0
importlib-resources       6.5.2
ipdb                      0.13.13
ipython                   8.18.1
jedi                      0.19.2
jinja2                    3.1.6
jsonschema                4.24.0
jsonschema-specifications 2025.4.1
kiwisolver                1.4.7
markdown                  3.8.2
MarkupSafe                3.0.2
matplotlib                3.9.4
matplotlib-inline         0.1.7
mpmath                    1.3.0
msgpack                   1.1.1
mujoco                    2.3.7
networkx                  3.2.1
numpy                     1.24.4
nvidia-cublas-cu12        12.6.4.1
nvidia-cuda-cupti-cu12    12.6.80
nvidia-cuda-nvrtc-cu12    12.6.77
nvidia-cuda-runtime-cu12  12.6.77
nvidia-cudnn-cu12         9.5.1.17
nvidia-cufft-cu12         11.3.0.4
nvidia-cufile-cu12        1.11.1.6
nvidia-curand-cu12        10.3.7.77
nvidia-cusolver-cu12      11.7.1.2
nvidia-cusparse-cu12      12.5.4.2
nvidia-cusparselt-cu12    0.6.3
nvidia-nccl-cu12          2.26.2
nvidia-nvjitlink-cu12     12.6.85
nvidia-nvtx-cu12          12.6.77
packaging                 25.0
pandas                    2.3.0
parso                     0.8.4
pettingzoo                1.24.3
pexpect                   4.9.0
pillow                    11.2.1
pip                       21.0
platformdirs              4.3.8
ppo-her                   0.0.0
prompt-toolkit            3.0.51
protobuf                  6.31.1
ptyprocess                0.7.0
pure-eval                 0.2.3
py4j                      0.10.9.9
pygments                  2.19.2
PyOpenGL                  3.1.9
pyparsing                 3.2.3
python-dateutil           2.9.0.post0
pytz                      2025.2
PyYAML                    6.0.2
ray                       2.1.0
redis                     6.2.0
referencing               0.36.2
requests                  2.32.4
rpds-py                   0.25.1
scipy                     1.13.1
setuptools                65.5.0
six                       1.17.0
smmap                     5.0.2
stable-baselines3         2.1.0
stack-data                0.6.3
sympy                     1.14.0
tabulate                  0.9.0
tbparse                   0.0.9
tensorboard               2.19.0
tensorboard-data-server   0.7.2
tensorboardx              2.6.4
tomli                     2.2.1
torch                     2.7.1
tqdm                      4.67.1
traitlets                 5.14.3
triton                    3.3.1
typing-extensions         4.14.0
tzdata                    2025.2
urllib3                   2.5.0
virtualenv                20.31.2
wcwidth                   0.2.13
werkzeug                  3.1.3
wheel                     0.38.4
zipp                      3.23.0
