# LongNav
This repository provides the implementation of LongNav. 

### 🚧 🚧 under construction 🚧🚧
## Installation 🛠️
Create the main conda environment responsible for vlm trainer (simulator may need separate conda env).
```
conda create -n longnav python=3.10.16
```

Install with pip:
```
pip install -e .
```

Make sure that the verl submodule is installed. 

#### Note
- verl has minor incompatibility with latest transformers. you may have to patch the verl code.

- flash attention is supported, install is left as exercise to reader (wheel availability depends on machine)
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
## Training 🚀
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
