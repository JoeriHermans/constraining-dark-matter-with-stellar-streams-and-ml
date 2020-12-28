[![ArXiv](http://img.shields.io/badge/phys.CO-arXiv%3A2020.2011-14923.svg)](https://arxiv.org/abs/2011.14923)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/master?filepath=notebooks%2F01_overview.ipynb)

We put forward several techniques and guidelines for the application of (amortized) neural simulation-based
inference to scientific problems.
In this work we examine the relation between dark matter subhalo impacts and
the observed stellar density variations in the GD-1 stellar stream to differentiate between [Warm Dark Matter](https://en.wikipedia.org/wiki/Warm_dark_matter) and [Cold Dark Matter](https://en.wikipedia.org/wiki/Cold_dark_matter).

<p align="center">
  <img height=270 alt="WDM 1D posterior GD-1" src="https://github.com/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/blob/master/.github/posterior-gd1-1d.png?raw=true">
  <img height=270 alt="WDM 2D posterior GD-1" src="https://github.com/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/blob/master/.github/posterior-gd1-2d.png?raw=true">
  <img height=282 alt="Posteriors" src="https://github.com/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/blob/master/.github/posteriors.gif?raw=true">
</p>

> **Disclaimer**: Baryonic effects are not accounted for, see paper for details.

This repository contains the code to reproduce this work on a Slurm enabled HPC cluster or on your local machine.

The Slurm arguments you typically use in your batch submission scripts will flawlessly run on your development machine without actually requiring or installing Slurm binaries. Futhermore, our scripts will automatically manage the Anaconda environment related to this work.

## Table of contents

- [Demonstration notebooks](#demonstration-notebooks)
- [Requirements](#requirements)
- [Datasets and models](#datasets-and-models)
- [Usage](#usage)
- [Pipelines](#pipelines)
- [Notebooks](#notebooks)
- [Manuscripts](#manuscripts)
- [Attribution](#attribution)

## Demonstration notebooks

> **Note**. If you are viewing this notebook right after release, it might be possible that the Binder links do no work yet. We are actively solving this!

In addition to the code related to the contents of this paper, we provide several demonstration notebooks
to familiarize yourself with simulation-based inference.

| Short description | Render  | Binder |
| ----------------- | ----- | ------ |
| Overview notebook with presimulated data and pretrained models | [[view]](notebooks/01_overview.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/master?filepath=notebooks%2F01_overview.ipynb)     |
| Toy problem to demonstrate the technique | [[view]](notebooks/02_toy.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/master?filepath=notebooks%2F02_toy.ipynb)     |
| Out-of-distribution or model misspecification detection | [[view]](notebooks/03_out_of_distribution.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/master?filepath=notebooks%2F03_out_of_distribution.ipynb)     |
| Changing the implicit prior of the ratio estimator through MCMC | [[view]](notebooks/04_prior.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JoeriHermans/constraining-dark-matter-with-stellar-streams-and-ml/master?filepath=notebooks%2F04_prior.ipynb)     |

## Requirements

> **Required**. The project assumes you have a working Anaconda installation.

In order to execute this project, you need at least `40 GB` of available storage space. We do not recommend to run the simulations on a single machine, as this would take about *60 years* to complete. On a HPC cluster, the simulations will take about 2-3 weeks. Training all ratio estimators will take 1-2 days depending on the availability of GPU's. Diagnostics another day.

### Installation of the Anaconda environment

```console
you@localhost:~ $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
you@localhost:~ $ sh Miniconda3-latest-Linux-x86_64.sh
```

The corresponding environment can be installed by executing

```console
you@localhost:~ $ sh scripts/install.sh
```

in the root directory of the project. This will install several dependencies in a certain order due to some quirks in Anaconda.

## Datasets and models

The required computational resources mentioned above might not be available to everyone.
As such, the presimulated datasets and pretrained models can be made available on request by e-mailing
[joeri.hermans@doct.uliege.be](mailto:joeri.hermans@doct.uliege.be), or by opening an issue in this GitHub repository.

## Usage

Simply execute `./run.sh -h` to display all available options or`./run.sh` to install the Anaconda environment and dependencies related to this project.

A specific set of experiments can be executed by supplying a comma-seperated list.
```console
you@localhost:~ $ bash run.sh -e simulations,inference
```

If you update the `environment.yml` file by adding or removing dependencies, please run `bash run.sh -i` first. The script will automatically synchronize the changes with the Anaconda environment associated to this project.

## Pipelines

This section gives a quick overview of our results.

A link to a detailed description of every experiment is listed. As described in the [usage](#usage) section, the `identifier` plays an important roll if the developer or end-user wishes to execute a subset of pipelines (experiments).

| Identifier     | Short description                                           | Link                                                      |
| -------------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| *inference*    | Analyses and plots.                                    | [[details]](experiments/experiment-inference/pipeline.sh)             |
| *simulations*  | A pipeline for simulating the datasets and GD-1 mocks.      | [[details]](experiments/experiment-simulations/pipeline.sh)           |


## Notebooks

Overview of a non-exclusive list of interesting notebooks in this repository, not included in the main paper.

| Short description | Render  |
| ----------------- | ----- |
| In this notebook we explore in a ad-hoc fashion how the neural network uses the high-level features in a stellar stream to differentiate between CDM and WDM. | [[view]](experiments/experiment-inference/edge-case.ipynb) |


## Manuscripts

The preprint is available at [`manuscript/preprint/main.pdf`](manuscript/preprint/main.pdf).

Our NeurIPS submission can be found at [`manuscript/neurips/main.pdf`](manuscript/neurips/main.pdf).

## Attribution

If you use our code or methodology, please cite our paper
```
@ARTICLE{2020arXiv201114923H,
       author = {{Hermans}, Joeri and {Banik}, Nilanjan and {Weniger}, Christoph and {Bertone}, Gianfranco and {Louppe}, Gilles},
        title = "{Towards constraining warm dark matter with stellar streams through neural simulation-based inference}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Machine Learning},
         year = 2020,
        month = nov,
          eid = {arXiv:2011.14923},
        pages = {arXiv:2011.14923},
archivePrefix = {arXiv},
       eprint = {2011.14923},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201114923H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
and the original method paper published at ICML2020
```
@ARTICLE{hermansSBI,
       author = {{Hermans}, Joeri and {Begy}, Volodimir and {Louppe}, Gilles},
        title = "{Likelihood-free MCMC with Amortized Approximate Ratio Estimators}",
      journal = {arXiv e-prints},
     keywords = {Statistics - Machine Learning, Computer Science - Machine Learning},
         year = "2019",
        month = "Mar",
          eid = {arXiv:1903.04057},
        pages = {arXiv:1903.04057},
archivePrefix = {arXiv},
       eprint = {1903.04057},
 primaryClass = {stat.ML},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190304057H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
The simulation model uses [`galpy`](https://github.com/jobovy/galpy) extensively, and
was originally conceived by [Jo Bovy](https://github.com/jobovy) and ![Nilanjan Banik](https://github.com/nbanik).
We adapted their [codebase](https://github.com/nbanik/Baryonic-effects-on-Pal5) to fit our purposes.
