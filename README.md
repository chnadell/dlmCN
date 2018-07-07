# Deep Learning for Materials
## Installation
Pre-requisite:
- Python 3
- Numpy
- Matplotlib
- Tensorflow = 1.8
- [TFplot](https://github.com/wookayin/tensorflow-plot)
## Usage
1. put data files into `./data` folder
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