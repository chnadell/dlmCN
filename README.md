# Deep Learning for Materials
## Installation
Pre-requisite:
- Python 3
- Numpy
- Matplotlib
- Tensorflow = 1.8
- [TFplot](https://github.com/wookayin/tensorflow-plot)

## Environment
This code should work just fine on TensorFlow CPU, but is best run from a TensorFlow GPU environment. If running from the command line, be sure to activate an environment with TensorFlow GPU installed to see faster results. Similarly, if running from a code/text editor or IDE, make sure the interpreter option points to a python instance that will access a TensorFlow GPU installation. 

## Summary
#### 1. train.py
Trains a model and saves it. 

#### 2. evaluate.py
Evaluates a trained model.

#### 3. batch_plot.py
Randomly samples the output of a trained model on the validation set, to get an idea of what the network predictions generally look like.

#### 4. data_reader.py
Main function is to read the data and convert to TensorFlow Dataset. Handles shuffling and batching data. 
Also contains functions for adding new columns (i.e., derived input values) to the dataset (see addColumns()), for splitting the data into new validation/training sets based on (geometric) constraints (see gridShape()), and for checking that the data is uniformly randomly distributed across the hyperspace and generating some plots to visualize this (see check_data()). 

#### 5. lookup.py
Function for generating the large list of geometries for which the model will predict spectra in order to build the lookup table (gen_data()). 

Function for generating network predictions based on the geometry grid data and saving to a set of files (main()). 
A few versions of functions for searching through the lookup table using our naive linear algorithm--I should get around to cleaning these up at some point. The latest one, and the one we used in our manuscript is called lookupBin2().

#### 8. network_maker.py
Defines a high-level network class that stores meta-information about the given network, like how the loss is defined, which optimizer should be used, how the model should be saved. 

#### 6. utils.py
Wrapper functions for some types of layers. Also, functions that define the shape/type of neural network that will be passed into the network class. The most important one is my_model_fn_tense(), which actually has the tensor module portion turned off since we found it was detrimental to performance. 

#### 7. network_helper.py
Functions for getting good tensorboard results. Codes what values to save (like validation MSE, training MSE, etc.) and when to save them. 

Function for extracting hyperparameter values from a saved file. 

## Usage (from editor)
1. put training data files into `./dataIn` folder, evalutation data files into './dataIn/eval'
2. adjust hyperparameters in train.py
3. run train.py
4. Model will be stored in `./models` with a timestamp as its folder name. The function of the model and the parameters used will be recorded in `./[timestamp]/model_meta.txt`
5. To evaluate the model, run `evaluate.py` with `MODEL_NAME` the name of the model (should be a timestamp) you want to evaluate, then run `batch_plot.py` and set the corresponding model name to get all curves on the validation data

## Usage (from command line)
1. put training data files into `./dataIn` folder, evalutation data files into './dataIn/eval'
2. run ```train.py --input-size=[input dimension] --fc-filters=[#neurons at each fc layer] --tconv-dims=[upsampled dimension after each layer] --tconv-filters=[#filters for each tconv layer] --learn-rate=[your learn rate]```
3. run ```evaluate.py```, the models will be evaluated with results written in `./data/test_pred.csv`
4. Training process can be monitored by the [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard#launching_tensorboard)
5. Model will be stored in `./models` with a timestamp as its folder name. The function of the model and the parameters used will be recorded in `./[timestamp]/model_meta.txt`
6. To evaluate the model, run `evaluate.py` with `MODEL_NAME` the name of the model (should be a timestamp) you want to evaluate, then run `batch_plot.py` and set the corresponding model name to get all curves on the validation data
## Customize
1. To use your customized network, modify function `my_model_fn` in `utils.py` or redefine a new function and pass it to the third parameter in line 67 of `train.py`
2. For other options for running the model, check function `read_flag()` in `train.py`
## Resources
1. TensorFlow [input pipeline](https://www.tensorflow.org/programmers_guide/datasets) (TF>=1.4 is required)
2. A *Hook* class inspired by [tf.train.SessionRunHook](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook) is used in this framework
