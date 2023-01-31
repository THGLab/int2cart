# Int2Cart
An algorithm for predicting protein backbone bond lengths and bond angles from torsion angles and amino acid sequence, so that reconstruction of protein structure with torsion angles only can be more accurate.

## Authors
* Jie Li
* Oufan Zhang
* Seokyoung Lee
* Ashley Namini
* Zi Hao Liu
* João Miguel Correia Teixeira
* Julie D Forman-Kay
* Teresa Head-Gordon

## Installation
### Required packages
* Python (3.8.0)
* numpy (1.23.5)
* yaml (6.0)
* pytorch (1.8.1)
* pandas (1.5.2)
* SidechainNet (https://github.com/THGLab/sidechainnet, modified from [jonathanking's version](https://github.com/jonathanking/sidechainnet))
* pdbtools (https://github.com/JerryJohnsonLee/pdbtools, modified from [Mike Harm's version](https://github.com/harmslab/pdbtools)), only needed for using `--from_pdb` option

### Installation steps
Go to the root folder of the repository (where `setup.py` resides) and run `pip install -e .`. Make sure to switch to the target virtual environment before installation if you want to install the code in a separate environment.

## Using pretrained model
Go into the `run_scripts` folder and run with:
`python predict.py [seq] [torsion_addr] [output]`

If we want to rebuild from a pdb file, we can use the `--from_pdb` option:

`python predict.py --from_pdb ../example/5jmb.pdb --build`

This will automatically extract the sequence and torsion angles from the pdb file and rebuild the structure.
Results will be saved to the same folder as the pdb file by default

<pre>
positional arguments:
  seq                   the one-letter amino acid sequence of the protein to
                        be predicted
  torsion_addr          the address to a numpy file that stores the backbone
                        torsion angles for the protein to be predicted.
                        Expected to be in size of Nx3, where N is the sequence
                        length, and 3 corresponds to phi, psi and omega.
  output                The output file name, excluding file extension

optional arguments:
  --from_pdb FROM_PDB   extract amino acid sequence and torsion angles from
                        the pdb file instead of providing them through
                        arguments
  --build, -b           build the structure after predicting bond lengths and
                        bond angles
  --units UNITS, -u UNITS
                        the units for angles used for inputs and outputs.
                        Should be one of [degree](default) or [radian]
  --device DEVICE, -d DEVICE
                        the device used for running the model, could be
                        [cpu](default) or [cuda:x] for GPU
  --config CONFIG, -c CONFIG
                        configuration file specifying model structure
  --model_addr MODEL_ADDR, -m MODEL_ADDR
                        a path specifying the position of trained model
</pre>         

## Using Int2Cart as a rosetta mover
The Int2Cart algorithm has also been implemented as a mover in pyRosetta. The mover can be used in the following way (pyRosetta installation is required):
```python 
from pyrosetta import *
init()
import sys
sys.path.append('[int2cart_base_path]/rosetta')
pose = pose_from_pdb("example/5jmb.pdb")
mover = Int2CartMover(pose.sequence(), device="cpu")
mover.apply(pose)
pose.dump_pdb("pose_after_int2cart_move.pdb")
```

## Training new models
* Make sure SidechainNet package is installed and patching script (under `tools` folder have been executed)
* Prepare a yml file similar to `configs/debug.yml` and change parameters as needed
* run `run_scripts/train.py [config_file]` to train the model