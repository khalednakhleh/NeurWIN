This code is for the paper titled: "NeurWIN: Neural Whittle Index Network for Restless Bandits Via Deep RL" presented at NeurIPS 2021.
The training, inference, and plotting methods used in the paper are contained in this repository. 
---
## Requirements

Run setup.py to get requirements. The code was ran using: 
Python 3.8
PyTorch 1.5.1
graphviz 0.14.1
gym 0.17.2
numpy 1.18.5
pandas 1.0.5
scipy 1.4.1
matplotlib 3.3.0
tensorflow 2.3.0

Note the sizeAware files are the wireless scheduling case considered in the paper.
Also using Gym's *checkenv* function requires tensorflow 1.x, which requires running an Python 3.6 environment. 
---
## Repository structure

```
main
├── **envs:** directory containing all restless arms modelled as Gym environments
├── **plotResults:** directory containing final results for all considered cases (showing the total discounted rewards)
├── **plottingScripts:** directory with .py scripts for showing inference plots
├── **testingScripts:** directory with .py scripts for control policy testing of the trained models' for all cases
├── **testResults:** directory for saving the inference results 
├── **trainResults:** directory with NeurWIN, REINFORCE, and QWIC trained models for all cases
├── **main.py:** starting point for training NeurWIN or REINFORCE for all considered cases (uncomment any code snippet to select a case)
├── **neurwin.py:** NeurWIN's implementation using PyTorch 
├── **reinforce.py:** REINFORCE implementation using PyTorch [implemented from this link](https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0)
```
---
## Running steps
The running procedure is:

1) Uncomment the algorithm's code snippet to train in main.py (trained models are located in trainResults directory).
2) For inference, run the desired case (e.g. deadline scheduling, recovering bandits) found in testingScripts directory.
3) Plot the results using the plottingScripts files (graphs are saved in the plotResults directory). 
---
## Acknowledgment:

Wolpertinger-ddpg original source is (please follow instructions in source repository to run the code):
https://github.com/ChangyWen/wolpertinger_ddpg

reinforce original source is: 
https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
