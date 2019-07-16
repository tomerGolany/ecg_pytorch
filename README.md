# ecg_pytorch
ECG heartbeat classification

Pytorch implementation for ECG heartbeat classification using Generative
approaches.

## Online Demo

[<img src="complete">](complete)

[link](complete)


## Prerequisites

- Python 3.3+
- [yPtorch](https://github.com/tensorflow/tensorflow/tree/r0.12)
- Add more...

## Usage

First, download dataset with:

    $ python ....

To train a model with downloaded dataset:

    $ python ...
    $ python ...

To test with an existing model:

    $ python main.py ...
    $ python main.py ...


## Results

![result](assets/training.gif)

### DCGAN

After 6th epoch:

![result3](assets/result_16_01_04_.png)

After 10th epoch:

![result4](assets/test_2016-01-27%2015:08:54.png)

### PGAN

![custom_result1](web/img/change5.png)

![custom_result1](web/img/change2.png)

![custom_result2](web/img/change4.png)

### ODEGAN

MNIST codes are written by [@PhoenixDai](https://github.com/PhoenixDai).

![mnist_result1](assets/mnist1.png)

![mnist_result2](assets/mnist2.png)

![mnist_result3](assets/mnist3.png)

More results can be found [here](./assets/) and [here](./web/img/).


## Training details

Details of the loss of Discriminator and Generator.

![d_loss](assets/d_loss.png)

![g_loss](assets/g_loss.png)

Details of the histogram of true and fake result of discriminator.

![d_hist](assets/d_hist.png)

![d__hist](assets/d__hist.png)


## Related works

- ...


## Author

Tomer Golany

