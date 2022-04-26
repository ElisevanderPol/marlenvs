## Installation 

```
pip install -e .
```

## Use environment
Imports:
```
import gym

import marlenvs
```
For Wildlife env:
```
gym.make("WildlifeEnv-v0")
```
For Traffic env:
```
gym.make("JunctionEnv-v0")
```

### Cite
If you use this code in your own work, please cite our paper:
```
@inproceedings{van2022multi,
  title={Multi-Agent {MDP} Homomorphic Networks},
  author={van der Pol, Elise and van Hoof, Herke and Oliehoek, Frans A. and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

The Robert Bosch GmbH is acknowledged for financial support.
