# LongNav
This repository provides the implementation of LongNav. 

### 🚧 🚧 under construction 🚧🚧
## Installation 🛠️
### RL trainer
Create the main conda environment responsible for vlm trainer
```
conda create -n longnav_vlm python=3.10.16
```

Install with pip:
```
pip install -e .
pip install flash-attn --no-build-isolation
```

Make sure that the verl submodule is installed. 
```
git submodule update --init --recursive
cd verl && pip install --no-dependencies -e .
```

You may optionally install flash attention in the same env.
### Environment
Install habitat lab in a separate conda env named vln. Then run:
```
pip install --no-dependencies -e .
```
This allows ray to resolve the interface correctly.

### Testing the Install
Two basic tests are currently available to validate your install:

[eval smoke test](tests/eval_smoke.py) (tests the rollout collection)
```
python3 tests/eval_smoke.py
```

[rl smoke test](tests/rl_smoke.py) (tests one RL training step)
```
python3 tests/rl_smoke.py
```

These may be referenced for integrating new Envs.

## Quickstart
Run our framework with a dummy environment (doesn't require habitat dependencies)
```
python3 -m longnav.training_scripts.train_dummy.py
```
## ObjectNav Training 🚀

```
python3 -m longnav.training_scripts.train_rl.py +experiment=<experiment_name>
```
- experiment_name must be a config that exists in src/conf/experiment.

## Eval ⏱️
```
python3 -m longnav.eval.py +experiment=<experiment_name>
```
## Sim to Real Serving 🤖
Serve the FAST API:
```
python3 -m longnav.serve
```

See the [client example](tools/client.py) for API usage.
## Project Structure:
### Configuration Management
Config schemas are defined via data class in [config_schema.py](src/longnav/config_schema.py). The 
### RL Env Interface
See [dummy env structure](src/longnav/env/env_base.py) for reference. Any compatible Env may be used by the [rollout collection orchestration](src/longnav/utils/rollout_core.py) to produce rollouts for training steps.

At a high level, the Env Actors return dictionaries of standard RL outputs. However, RGB observation is separated out for better Ray performance.
