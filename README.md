# Quadruped

[TOC]

## Installation using Conda

```bash
conda create -n quadruped python==3.8
conda activate quadruped
```

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:

   ```bash
   tar -xf IsaacGym_Preview_4_Package.tar.gz
   ```

3. now install the python package
   ```bash
   cd isaacgym/python && pip install -e .
   ```
4. Verify the installation by try running an example

   ```bash
   python examples/1080_balls_of_solitude.py
   ```

5. For troubleshooting check docs `isaacgym/docs/index.html`

### Install the repo itself

In this repository, run `pip install -e .`

## Environment and Model Configuration

**CODE STRUCTURE** The main environment for simulating a legged robot is
in [legged_robot.py](dribblebot/envs/base/legged_robot.py). The default configuration parameters including reward
weightings are defined in [legged_robot_config.py::Cfg](dribblebot/envs/base/legged_robot_config.py).

There are three scripts in the [scripts](scripts/) directory:

```bash
scripts
├── __init__.py
├── play_dribbling_custom.py
├── play_dribbling_pretrained.py
└── train_dribbling.py
```

## Training, Logging and evaluation

To train the Go1 controller, run:

```bash
python scripts/train_dribbling.py
```

After initializing the simulator, the script will print out a list of metrics every ten training iterations.

Training with the default configuration requires about 12GB of GPU memory. If you have less memory available, you can
still train by reducing the number of parallel environments used in simulation (the default is `Cfg.env.num_envs = 1000`).

To visualize training progress, first set up weights and bias (wandb):

### Set Up Weights and Bias (wandb):

Weights and Biases is the service that will provide you a dashboard where you can see the progress log of your training runs, including statistics and videos.

First, follow the instructions here to create you wandb account: https://docs.wandb.ai/quickstart

Make sure to perform the `wandb.login()` step from your local computer.

Finally, use a web browser to go to the wandb IP (defaults to `localhost:3001`)
