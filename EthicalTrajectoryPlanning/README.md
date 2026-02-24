[![DOI](https://zenodo.org/badge/491550412.svg)](https://zenodo.org/badge/latestdoi/491550412)
[![Linux](https://img.shields.io/badge/os-linux-blue.svg)](https://www.linux.org/)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
# Integrating Spatial-Temporal Risk Maps with Candidate Trajectory Trees for Explainable Autonomous Driving Planning

This repository contains the official implementation of the paper "Integrating Spatial-Temporal Risk Maps with Candidate Trajectory Trees for Explainable Autonomous Driving Planning", which has been accepted by "Communications in Transportation Research"

## System Requirements
* Operating System: Linux Ubuntu (tested on 20.04)
* Programming Language: Python >= 3.7 (tested on 3.8)
* [Software Dependencies](/requirements.txt)

## Installation

The installation of this repository takes around 10 min and consists of three steps.
We recommend using an isolated [virtual environment](https://pypi.org/project/virtualenv/) for installation.

1. Clone this repository with:

    `git clone https://github.com/Leon-u/risk_map.git`

2. Navigate to the root folder of the repository (`[..]/risk_map`) and install requirements:

    `pip install -r requirements.txt`

## Quick Start Demo

To run the candidate trjectory tree planner on an exemplary default scenario, execute the following command from the root directory of this repository:
    
* `python planner/Frenet/frenet_planner.py`


You will run a planning algorithm that integrates candidate trajectory trees with risk maps in an open-loop simulation.
Now you can start with your own experiments by selecting another scenario by adding

* `--scenario <path-to-scenario>`

to the command.

