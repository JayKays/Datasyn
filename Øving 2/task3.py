import utils
import matplotlib.pyplot as plt
import numpy as np
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
from timeit import default_timer as timer 


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    #Set tricks to be comparedin the two models
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    neurons_per_layer = [64] * 1 + [10] #Layers for first network

    learning_rate = 0.02 if use_momentum else .1 #Adjusting learning rate for momentum

    model_1 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_1 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_1, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_1, val_history_1 = trainer_1.train(num_epochs)

    # Creating comparison model: change True <-> False to keep/turn off tricks used in prev. model
    use_improved_weight_init &= True
    use_improved_sigmoid &= True
    use_momentum &= False

    neurons_per_layer = [64] * 1 + [10] #Layers for the second network

    learning_rate = 0.02 if use_momentum else .1 #Adjusting learning rate for momentum

    model_2= SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_2, val_history_2 = trainer_2.train(num_epochs)

    #Third model used in task 4ab
    # neurons_per_layer = [32] * 1 + [10] 
    # model_3 = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_3 = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_3, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )

    # train_history_3, val_history_3 = trainer_3.train(
    #     num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history_1["loss"],"", npoints_to_average=10)
    utils.plot_loss(train_history_2["loss"], "", npoints_to_average=10)
    # utils.plot_loss(train_history_3["loss"], "32 hidden units", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.ylabel("Cross entropy losss")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history_1["accuracy"], "With momemtum")
    utils.plot_loss(val_history_2["accuracy"], "No Momentum")
    # utils.plot_loss(val_history_3["accuracy"], "32 hidden units")
    # utils.plot_loss(train_history_1["accuracy"], "Single layer train")
    # utils.plot_loss(train_history_2["accuracy"], "Double layer train")
    # utils.plot_loss(train_history_3["accuracy"], "32 Training accuracy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
