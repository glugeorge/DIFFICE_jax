# Examples

This folder provides four example codes that use `DIFFICE_jax` package to assimilate remote-sensing velocity
and thickness data and infer effective ice viscosity under either **isotropic** or **anisotropic** assumption
via regular PINNs or extended-PINNs (XPINNs). 

The mathematical formulation for inferring **isotropic** ice viscosity
via regular PINNs are provided in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/paper.md).  The 
description for inferring **anisotropic** viscosity is given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/examples/Anisotropic.md),
and the description of **XPINNs** settings is given in [this link](https://github.com/YaoGroup/DIFFICE_jax/blob/main/model/XPINNs.md).


## `DIFFICE_jax/examples/train_pinns_iso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **isotropic**
assumption via **regular PINNs**. The code are computationally-efficient and accurate enough to study ice shelves
of size close or smaller than Amery or Larce C Ice Shelves. An companion Colab Notebook of this script is 
provided in the `colab` subfolder. View it by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_pinns_iso.ipynb)


## `DIFFICE_jax/examples/train_pinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **anisotropic**
assumption via **regular PINNs**. Different from isotropic viscosity inversion, this code infers two viscosity 
components, one in the horizontal and the other in the vertical direction.  An companion Colab Notebook of this script is 
provided in the `colab` subfolder. View it by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_pinns_aniso.ipynb)


## `DIFFICE_jax/examples/train_xpinns_iso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **isotropic**
assumption via **extended-PINNs (XPINNs)**. This code are required to study several largest ice shelves around the
Antarctica, such as Ross and Ronne-Filchner Ice Shelves, which involve many local structural regions with dense 
spatial variation that are difficult to be captured by one single neural network due to the spectral biases.
An companion Colab Notebook of this script is provided in the `colab` subfolder. View it by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_xpinns_iso.ipynb)


## `DIFFICE_jax/examples/train_xpinns_aniso.py`

A python script that assimilate remote-sensing data and infer the effective ice viscosity under **anisotropic**
assumption via **extended-PINNs (XPINNs)**. Different from isotropic viscosity inversion, this code infers two viscosity 
components, one in the horizontal and the other in the vertical direction.  An companion Colab Notebook of this script is 
provided in the `colab` subfolder. View it by clicking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YaoGroup/DIFFICE_jax/blob/main/examples/colab/train_xpinns_aniso.ipynb)

