# Physics-Informed Model-Based Reinforcement Learning for Soft Robot Control

This repo contains the code related to my master thesis at Politecnico di Torino, Italy, in the [VANDAL department](https://vandal.polito.it), 
under the supervision of [Prof. Giuseppe Averta](mailto:giuseppe.averta@polito.it).
The thesis was partially carried out at TU Delft, Netherlands, in the [Cognitive Robotics](https://www.tudelft.nl/en/me/about/departments/cognitive-robotics-cor) department, under the supervision
of [Prof. Cosimo Della Santina](mailto:c.dellasantina@tudelft.nl).

The code is highly inspired, and based, on the work of Adithya Ramesh and Balaraman Ravindran, which can be found [here](https://github.com/adi3e08/Physics_Informed_Model_Based_RL).


<table style="text-align: center">
<tr>
<td><img src="animations/env.gif"/></td>
<td><img src="animations/lnn.gif"/></td>
<td><img src="animations/mlp.gif"/></td>
</tr>
<tr>
<td>reference</td>
<td>LNN-based</td>
<td>MLP-based</td>
</tr>
</table>

## Abstract
Soft robots are gaining popularity in the scientific community, due to the numerous applications for which they outperform their rigid counterparts. The trade-off lies in the difficulty to model and control them. In the last decade, the community has developed various advanced techniques to improve our ability to model them, with the purpose of developing effective controllers. While first-principle solutions require expert knowledge, data-driven techniques, which overcome this limitation, are data inefficient (i.e., they require a significant amount of real-world data) and are time-consuming in training. In addition, off-the-shelf data-driven approaches such as the Multi Layer Perceptron learn solutions that don’t obey physical laws.

This work presents a data efficient strategy to derive a model of the soft system and a controller for it: the learned model regresses the forward dynamics of the system purely from kinematic observational data, and the controller is a Model-Based Deep Reinforcement Learning policy trained on model-generated data. The model regression makes use of Physics-based inductive biases to learn plausible behaviors, following the paradigm of Physics-Informed Neural Networks (PINNs), in particular that of Deep Lagrangian Networks (DeLaNs).

The proposed solution is composed of three stages: i) collection of the data samples from the real (simulated) system using a random policy; ii) training of a set of simple MLPs to learn the coefficients of the Euler-Lagrange equations of the system; iii) training of the RL-based controller using data from the learned model, to obtain a robust policy without interacting with the real system. Ultimately, an iterative reapplication of these three stages – by replacing the random policy with the one under training – is investigated, aiming to obtain a more accurate policy with fewer interactions. Experimental validation has been carried out for a single-segment soft rod modeled with the Piecewise Constant Strain approximation, with the goal of controlling it to make its tip reach an arbitrary point in space.

This approach combines, for the first time on a soft robotics system, the ability of Deep Reinforcement Learning to generate robust controllers with minimal

knowledge of the real system, the power of DeLaNs to learn solutions that comply with fundamental physical laws, and the data efficiency of Model-Based RL, which allows for the training of an effective policy while minimizing the interactions with the real system.

## Requirements
- python 3
- numpy
- scipy
- pytorch
- tensorboard
- pygame
- matplotlib
- torchode
- tqdm


## Usage
To train MBRL LNN on Acrobot task, run,

    python mbrl.py --env acrobot --mode train --episodes 500 --seed 0 

The data from this experiment will be stored in the folder "./log/acrobot/mbrl_lnn/seed_0". This folder will contain two sub folders, (i) models : here model checkpoints will be stored and (ii) tensorboard : here tensorboard plots will be stored.

To evaluate MRBL LNN on Soft Reacher task, run,

    python mbrl.py --env soft_reacher --mode eval --episodes 3 --seed 100 --checkpoint ./export/soft_reacher/seed_0/last.ckpt --render

## Citation
If you find this work helpful, please consider starring this repo and citing our paper using the following Bibtex.
```bibtex
@inproceedings{ramesh2023physics,
  title={Physics-Informed Model-Based Reinforcement Learning},
  author={Ramesh, Adithya and Ravindran, Balaraman},
  booktitle={Learning for Dynamics and Control Conference},
  pages={26--37},
  year={2023},
  organization={PMLR}
}


