# Henbun
[![Build status](https://app.codeship.com/projects/e36941d0-90fc-0134-ef8a-6ae3641f140d/status?branch=master)](https://codeship.com/projects/185829)
[![codecov](https://codecov.io/gh/fujii-team/Henbun/branch/master/graph/badge.svg)](https://codecov.io/gh/fujii-team/Henbun)


Variational Bayesian inference for large-scale data.

# What is Henbun
Henbun is a Python library to make a large-scale Bayesian inference based on variational approximation.

Henbun is built on top of **TensorFlow** and thanks to TensorFlow's functionality,
GPU computation becomes very easy without changing Python code.

In large-scale Bayesian inference usually contains both the *local*
and *global* parameters, where *local* parameters are unique for each data while
*global* parameters are common for all the data.  
Henbun makes it possible to construct a feed-forward network to encode data into
the variational local parameters.  
This encoding replaces *local* parameters to the feed-forward network written
only by *global* parameters and therefore the model can be optimized
stochastically.

# Examples
Some examples can be found in [**notebooks**](notebooks/).
+ Regression problems
  + [Gaussian Process regression](notebooks/GaussianProcess.ipynb)  
  A very simple tutorial for simple variational Bayesian inference.

  + [Expert model with Gaussian Process](notebooks/Expert_GPR.ipynb)

+ Tomographic reconstruction (coming soon)
+ Spectroscopic tomography (coming soon)  
An example as an inverse problem solver

+ Auto-encoder (coming soon)
An example as an deep-learning framework.

+ Variational auto-encoder (coming soon)
An example as an large scale Bayesian inference with feed-forward network.


# Structure of Henbun
In the following notebooks, some structures of Henbun are described.
+ Brief description about optimization [(Henbun_structure)](notebooks/Henbun_structure.ipynb).
+ Brief description about variational inference [(Henbun_structure2)](notebooks/Henbun_structure2.ipynb).


# Dependencies and Installation
**Henbun** heavily depends on
+ [**TensorFlow**](https://www.tensorflow.org/): a Large-Scale Machine Learning library.

Before installing **Henbun**, **TensorFlow** must be installed.

See [**here**](https://www.tensorflow.org/versions/master/get_started/os_setup.html).

For the installation of **Henbun**, execute
> `python setup.py install`


# Acknowledgements
We learned a code structure from **GPflow**
(https://https://github.com/GPflow/GPflow).
