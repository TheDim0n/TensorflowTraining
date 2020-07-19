# TensorflowTraining
Additional python package for train models with [Tensorflow](https://www.tensorflow.org/).

# Installation
```pip install --upgrade tf-train```

# Usage
```import tf-train as tft```

In tft directly are available **keras**:
* tft.models
* tft.losses
* tft.optimizers
* tft.metrics

Use the ```tft.train()``` to train model.
```
Args:
    
    train_dataset: Tensorflow Dataset object for train the model.

    model: Keras trainable model.

    epochs (int): Num of epochs of training.

    loss: Loss function.

    optimizer: Tensorflow optimizer. Default Adam with learning_rate=0.001.

    metrics: List of tensorflow mertics. Default contains Recall and Precision.
```