# Quantum State Preparation using Genetic Algorithm


### About this project
This project was built by members of Team #5 at the NTU-Qiskit Hackathon Taiwan 2023. This project focuses on performing Quantum State Preparation by utilizing a modification of a genetic algorithm structure.

### Usage guidelines
The main use file is titled `GA.py`. We recommend cloning this repository and running it with individual IBM Quantum account backends. Users and community members may freely use parts of this project under the applicable open source licenses. We encourage replicating results from this experiment as well as applying it to your own use cases.

### Environment setup
This project was built using Python 3.8.5. The required packages can be installed using the following command:
#### No-GPU
```zsh
pip install -r requirement/requirement.txt
```
#### NVIDIA-GPU
For cuda11.
```zsh
pip install -r requirement/requirement-cu11.txt
```
For cuda12.
```zsh
pip install -r requirement/requirement-cu12.txt
```


### Updates
* Update code.v1 with feedback to improve performance.
* Benchmarking:
  - different algorithms (qGAN, VQE) and try different parameter to optimize the data to fit the distribution.
  - different probability distributions.
* Apply to more practical use cases:
  - Quantum Simulation (e.g. financial applications)
  - Quantum Machine Learning
* Minimize use of classical resources during optimization.
* Invite quantum community members to test solution for various use cases.
