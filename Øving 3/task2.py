import pathlib
import torch
import matplotlib.pyplot as plt
import utils
from torch import nn
import torchvision
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy

#The final task 3 model
class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,32,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(32),
            
            nn.Dropout2d(p=0.1),
            
            nn.Conv2d(32,32,5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32,64,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
            
            nn.Dropout(p=0.2),
                    
            nn.Conv2d(64,64,5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Dropout(p=0.2),

            nn.Conv2d(64,128,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(128),
        )

        # Initialize our last fully connected layer
        self.num_output_features = 4*4*128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout2d(p=0.3),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]

        feat = self.feature_extractor(x)    #Feature extractor
        out = self.classifier(feat)         #Classifier

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
#function to print out loss and accuracy for the best performing model of a trainer
def print_best_model(trainer: Trainer):

    trainer.load_best_model()

    trainer.model.eval()
    train_loss, train_acc = compute_loss_and_accuracy(
        trainer.dataloader_train, model, trainer.loss_criterion)
    val_loss, val_acc = compute_loss_and_accuracy(
        trainer.dataloader_val, model, trainer.loss_criterion)
    test_loss, test_acc = compute_loss_and_accuracy(
        trainer.dataloader_test, model, trainer.loss_criterion)
    
    print("Best model values:")
    print(f"Train:\t Loss: {train_loss:.3f} \t Acc: {train_acc:.3f}")
    print(f"Val  :\t Loss: {val_loss:.3f} \t Acc: {val_acc:.3f}")
    print(f"Test :\t Loss: {test_loss:.3f} \t Acc: {test_acc:.3f}")

#Model for ResNet18 network
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)
        # No need to apply softmax,
        # # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected layer
            param.requires_grad = True
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional layer
            param.requires_grad = True
        
    def forward(self, x):
        x = self.model(x)
        return x

#The model from Task 2, used as a comparison to changes made in task 3
class Model_task2(nn.Module):
    def __init__(self, image_channels,num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_features = 4*4*128
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,32,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32,64,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, x):
        feat = self.feature_extractor(x) 
        out = self.classifier(feat)
        return out

#Function to create a comparison plot between two trained models    
def create_comp_plots(trainer1: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.train_history["loss"], label="BatchNorm Train", npoints_to_average=10)
    utils.plot_loss(trainer2.train_history["loss"], label="Task2 Train", npoints_to_average=10)
    utils.plot_loss(trainer1.validation_history["loss"], label="BatchNorm Validation")
    utils.plot_loss(trainer2.validation_history["loss"], label="Task2 Validation")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer1.validation_history["accuracy"], label="BatchNorm")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Task2")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

#Function to plot loss and accuracy over training
def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)

    #Task 2/3 network parameters
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    model2 = Model_task2(image_channels=3, num_classes=10) 
    
    #Task2 model trainer
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    #trainer2.train()
    #print_best_model(trainer2)

    #Final task3 model trainer
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    #trainer.train()
    #print_best_model(trainer)
    
    #Task 4 parameters (Remember to change optimizer in trainer.py, and mean/std in dataloaders.py)
    epochs = 5
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model()
    
    trainer4 = Trainer(
            batch_size,
            learning_rate,
            early_stop_count,
            epochs,
            model,
            dataloaders
        )
    # trainer4.train()
    # print_best_model(trainer4)
    
    
