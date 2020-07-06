import tf_training.layers
import tensorflow.keras.models as models
from tensorflow import GradientTape
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer, Adam


def train(train_dataset, model, epochs, loss: Loss, optimizer=Adam, metrics=None):
    
    def calculate_loss(x, y, training=True):
        y_pred = model(x, training=training)
        return loss(y_true=y, y_pred=y_pred)

    def grad(x, y):
        with GradientTape as tape:
            loss_value = calculate_loss(x, y)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


    for epoch in range(epochs + 1):
        epoch_loss = loss

        for x, y in train_dataset:
            loss_value, grads = grad(x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            

            for metric in metrics:
                metric.update_state(loss_value)