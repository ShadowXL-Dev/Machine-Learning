import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nnfs
import os
from nnfs.datasets import sine_data, spiral_data
from sklearn.model_selection import train_test_split
# import Testing_Pandas as tp
# from Testing_Pandas import *

nnfs.init()


class Layer_Dense: # Dense layer
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0): # Layer initialization
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training): # Forward pass
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues): # Backward pass
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout: # Dropout
    def __init__(self, rate): # Init
        # Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training): # Forward pass
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues): # Backward pass
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Layer_Input: # Input "layer"
    
    def forward(self, inputs, training): # Forward pass
        self.output = inputs


class Activation_ReLU: # ReLU activation
    def forward(self, inputs, training): # Forward pass
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues): # Backward pass
        # Since we need to modify original variable, let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs): # Calculate predictions for outputs
        return outputs


class Activation_Softmax: # Softmax activation
    def forward(self, inputs, training): # Forward pass
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues): # Backward pass

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs): # Calculate predictions for outputs
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid: # Sigmoid activation
    def forward(self, inputs, training): # Forward pass
        # Save input and calculate/save output of the sigmoid function
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues): # Backward pass
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs): # Calculate predictions for outputs
        return (outputs > 0.5) * 1


class Activation_Linear: # Linear activation
    def forward(self, inputs, training): # Forward pass
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues): # Backward pass
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    def predictions(self, outputs): # Calculate predictions for outputs
        return outputs


class Optimizer_Adam: # Adam optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999): # Initialize optimizer - set settings
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_parameters(self): # Call once before any parameter updates
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_parameters(self, layer): # Update parameters

        # If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # Get corrected momentum self.iteration is 0 at first pass and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_parameters(self): # Call once after any parameter updates
        self.iterations += 1


class Loss: # Common loss class
    def regularization_loss(self): # Regularization loss calculation

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            # L1 regularization - biases calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers): # Set/remember trainable layers
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False): # Calculates the data and regularization losses given model output and ground truth values

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()


class Loss_CategoricalCrossentropy(Loss): # Cross-entropy loss
    def forward(self, y_pred, y_true): # Forward pass

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0 Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true): # Backward pass

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy(): # Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
    def backward(self, dvalues, y_true): # Backward pass

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss): # Binary cross-entropy loss
    def forward(self, y_pred, y_true): # Forward pass 

        # Clip data to prevent division by 0 Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true): # Backward pass

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0 Clip both sides to not drag mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):  # L2 loss # Mean Squared Error loss
    def forward(self, y_pred, y_true): # Forward pass

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true): # Backward pass

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):  # L1 loss # Mean Absolute Error loss

    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true): # Backward pass

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


class Accuracy: # Common accuracy class
    def calculate(self, predictions, y): # Calculates an accuracy given predictions and ground truth values

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Return accuracy
        return accuracy


class Accuracy_Categorical(Accuracy): # Accuracy calculation for classification model
    def init(self, y): # No initialization is needed
        pass

    def compare(self, predictions, y): # Compares predictions to the ground truth values
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        
        return predictions == y


class Accuracy_Regression(Accuracy): # Accuracy calculation for regression model
    def __init__(self):
        # Create precision property
        self.precision = None

    def init(self, y, reinit=False): # Calculates precision value based on passed in ground truth values
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y): # Compares predictions to the ground truth values
        return np.absolute(predictions - y) < self.precision


class Model: # Model class
    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    def add(self, layer): # Add objects to the model
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy): # Set loss, optimizer and accuracy
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self): # Finalize the model

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss Also let's save aside the reference to the last object whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights", it's a trainable layer - 
            # add it to the list of trainable layers We don't need to check for biases - checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and loss function is Categorical Cross-Entropy create an object of combined activation and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation and loss functions
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
 
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None): # Train the model

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs+1):

            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_parameters()
            for layer in self.trainable_layers:
                self.optimizer.update_parameters(layer)
            self.optimizer.post_update_parameters()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' +
                      f'lr: {self.optimizer.current_learning_rate:.6f}')

        # If there is the validation data
        if validation_data is not None:

            # For better readability
            X_val, y_val = validation_data

            # Perform the forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print(f'validation| ' + f'|acc: {accuracy:.3f}| ' + f'|loss: {loss:.3f}')

    def evaluate(self, X_val, y_val, *, batch_size=None): # Evaluates the model using passed-in dataset

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining data but not a full batch, this won't include it Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()


        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')


    def predict(self, X, *, batch_size=None): # Predicts on the samples
        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining data, but not a full batch, this won't include it Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    def forward(self, X, training): # Performs forward pass
        # Call forward method on the input layer this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list, return its output
        return layer.output

    def backward(self, output, y):# Performs backward pass

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method on the combined activation/loss this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer which is Softmax activation as we used combined activation/loss object, let's set dinputs in this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through all the objects but last in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


class ReadData():
    stored = []
    sheetnames = []
    
    def readNames(self, path, fileType):
        if fileType == 'excel':
            df = pd.ExcelFile(path)
            self.sheetnames = df.sheet_names
            for i in range(len(self.sheetnames)):
                self.stored.append(df.parse(self.sheetnames[i]))

            df1 = self.stored[0]
            df2 = df.parse(usecols="C, D")
            return df2
            #print(df2)
        # for i in range(len(self.stored)):
        #     print(self.stored[2])
        #     print()
    
    def getExpected(self):
        return self.df2

    def getData(self, ndx):
        return stored[ndx]

    def getSheet(self, ndx):
        return sheetnames[ndx]

    # def length(self):
    #     return len(sheetnames)

    def findNDX(self, searchString):
        for i in range(len(self.sheetnames)):
            if self.sheetnames[i] == searchString:
                return i
            else:
                print("Not Found, Returning NULL")
                pass

rd = ReadData()

rd.readNames(r"C:\Users\cfournier\NeuralDS.xls", 'excel')



# Create dataset
# X, y = spiral_data(samples=1000, classes=3)
# X_test, y_test = spiral_data(samples=100, classes=3)

print(rd.readNames(r"C:\Users\cfournier\NeuralDS.xls", 'excel'))

X, y, X_test, y_test = train_test_split(rd.readNames(r"C:\Users\cfournier\NeuralDS.xls", 'excel'), test_size=0.2)

# X = df[['Part Number', 'Recevied']]
# y = df[['Expected']]


# tp.ReadData.readNames()
# tp.ReadData.getSheet()
# tp.ReadData.getData()
# tp.ReadData.findNDX()


# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 128, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128, 3))
model.add(Activation_Softmax())


# confidences = model.predcit()
# predictions = model.output_layer_activation.predictions(confidences)

# print expected
# print(predictions)
# print received
# total = 9
# print(total)



# Set loss, optimizer and accuracy objects
model.set(loss=Loss_CategoricalCrossentropy(), optimizer=Optimizer_Adam(learning_rate=0.07, decay=6e-6), accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()


# print received
# print difference


# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)



