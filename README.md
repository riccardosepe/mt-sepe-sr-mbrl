# Physics-Informed Model-Based Reinforcement Learning for Soft Robot Control

This repo contains the code related to my master thesis at Politecnico di Torino, Italy, in the [VANDAL](https://vandal.polito.it) department, 
under the supervision of [Prof. Giuseppe Averta](mailto:giuseppe.averta@polito.it).
The thesis was partially carried out at TU Delft, Netherlands, in the [Cognitive Robotics](https://www.tudelft.nl/en/me/about/departments/cognitive-robotics-cor) department, under the supervision
of [Prof. Cosimo Della Santina](mailto:c.dellasantina@tudelft.nl).

The code is highly inspired, and based, on the work of Adithya Ramesh and Balaraman Ravindran, which can be found [here](https://github.com/adi3e08/Physics_Informed_Model_Based_RL).


<table style="text-align: center">
<tr style="font-style: italic; text-align: center">
<td>reference</td>
<td>LNN-based</td>
<td>MLP-based</td>
</tr>
<tr>
<td><img src="images/env.gif" alt="reference"/></td>
<td><img src="images/lnn.gif" alt="lnn-based"/></td>
<td><img src="images/mlp.gif" alt="mlp-based"/></td>
</tr>
<tr>
<td colspan="3">Here you can see the comparison between the reference (model-free RL) behavior, the RL algorithm trained on the LNN-based model, and the RL algorithm trained on the MLP-based model.
</td>
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
In the final version of this project, a model needs to be pre-trained and then used as a data generator for a Soft Actor-Critic 
algorithm. 

0. Data generation

   1. The data generation can be done using the following command:
        
       `python collect_dataset.py`
    
       A seed can be set for this step. It is currently hardcoded in the `collect_dataset.py` file.
       This code will generate data in the folder `./data/seed_0`. Inside there will be a `data.pt` file.

1. Pre-training the model

   1. The pre-training of the LNN-based model can be done using the following command:
        
       `python train_model.py --seed 0 --lr 3e-4`
    
       This code will generate data in the folder `./log/model/seed_0`. Inside it there will be a `tensorboard` folder and a `.ckpt` with the networks weights.

   2. The pre-training of the MLP-based model can be done using the following command:

      `python train_model_mlp.py --seed 0 --lr 3e-4`
    
      This code will generate data in the folder `./log/model_mlp/seed_0`. Inside it there will be a `tensorboard` folder and a `.ckpt` with the networks weights.

After inspecting the result, the best model weights must be placed in tge `./weights` folder under the name `best_model_lnn.ckpt` and `best_model_mlp.ckpt`.

2. Training the SAC algorithm

   1. The training of the SAC algorithm in a model-free way can be done using the following command:

      `python sac.py --env soft_reacher --mode train --episodes 500 --seed 0`
        
      This code will generate data in the folder `./log/soft_reacher/sac/seed_0`. Inside it there will be a `tensorboard` folder and a `models` folder with the networks weights checkpointed every 50 episodes. 

   2. The training of the SAC algorithm in a model-based way using the pretrained LNN model can be done using the following command:
    
      `python sac_with_model.py --env soft_reacher --mode train --episodes 500 --seed 0 --model-type lnn`

      This code will generate data in the folder `./log/soft_reacher/sac_on_lnn/seed_0`. Inside it there will be a `tensorboard` folder and a `models` folder with the networks weights checkpointed every 50 episodes.

   3. The training of the SAC algorithm in a model-based way using the pretrained MLP model can be done using the following command:

      `python sac_with_model.py --env soft_reacher --mode train --episodes 500 --seed 0 --model-type mlp`

      This code will generate data in the folder `./log/soft_reacher/sac_on_mlp/seed_0`. Inside it there will be a `tensorboard` folder and a `models` folder with the networks weights checkpointed every 50 episodes.
    
3. Evaluating the trained models

    1. To evaluate the SAC algorithm on Soft Reacher task, run,
   
       `python sac.py --env soft_reacher --mode eval --checkpoint path_to_checkpoint`
  

## Results

1. Model performances
<p align="center">
<img src="images/ee.png" alt="end-effector plots" width="50%">
<img src="images/ee_error.png" alt="end-effector error plots" width="50%">
</p>

These plots show the results of the model training step over a fixed evaluation trajectory. The first one shows the Cartesian coordinates of the real (simulated) robot, of the LNN-based model, and of the MLP-based one.
The second one shows instead the absolute errors.</br>
While the maximum errors of both the trained methods are comparable, the LNN-based one tracks the reference behavior more closely over the trajectory.

2. Controller performances

<p align="center">
<img src="images/sac_val_all_returns.png" alt="rl plots" width="50%">
</p>

This plot shows the performance of the SAC algorithm trained in a model-free way (_ref._), with a MLP-based model and with 
a LNN-based one. The LNN-based model reaches performance comparable to the model-free one, using way less real-world interactions.
On the other hand, the MLP-based one fails to learn a good policy, due to the lack of physical constraints in the model.

## Bibliography

This is a short list of the few most relevant papers that have been used in this work:

- [Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning](https://arxiv.org/abs/1907.04490)
- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [Physics-Informed Model-Based Reinforcement Learning](https://arxiv.org/abs/2212.02179)
- [Physics-Informed Neural Networks to Model and Control Robots: A Theoretical and Experimental Investigation](https://arxiv.org/abs/2305.05375)

## Citation
If you find this work helpful, please consider starring this repo and citing it as follows:
```bibtex
@mastersthesis{sepe2024pimbrl,
  title={Physics-Informed Model-Based Reinforcement Learning for Soft Robot Control},
  author={Sepe, Riccardo},
  school={Politecnico di Torino},
  year={2024}
}
```


