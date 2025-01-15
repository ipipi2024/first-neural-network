import numpy
import scipy.special as sp

#neural network class definition
class neuralNetwork:
    #initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices
        #using the normal distribution to keep the variance constant
        self.wih = numpy.random.normal(0.0,pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        #learning rate
        self.lr = learningrate

        #activation function is the sigmoid function 1/(1+e^-x)
        self.activation_function = lambda x: sp.expit(x)
        pass

    #train the neural network
    def train(self, inputs_list, targets_list):
        #convert inputs list to 2d array and transpose to be a column vector
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

         #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target - actual)
        output_errors = targets - final_outputs

        #hidden layer error is the output_errors, split proportionaly to thier weight
        #and recombined at the hidden node
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #update the weights for the link between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))

        #update the weights for the links between the input and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))

        

        pass

    #query the neural network
    def query(self, inputs_list):
        #convert inputs list to 2d array and transpose to have input as column vectors
        inputs = numpy.array(inputs_list, ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)

        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signal into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        #calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
    
     # Evaluate performance using Mean Squared Error (MSE)
    def evaluate(self, test_data):
        total_error = 0
        for inputs, targets in test_data:
            outputs = self.query(inputs)
            total_error += numpy.mean((numpy.array(targets, ndmin=2).T - outputs) ** 2)
        mse = total_error / len(test_data)
        return mse


#number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#learning rate
learning_rate = 0.3

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_data = [
    ([0.3, 0.2, 0.1], [0.6, 0.4, 0.2]),
    ([0.25, 0.4, 0.3], [0.5, 0.8, 0.6]),
    ([0.1, 0.35, 0.15], [0.2, 0.7, 0.3])
]

# Testing data
test_data = [
    ([0.32, 0.22, 0.12], [0.64, 0.44, 0.24]),
    ([0.28, 0.38, 0.29], [0.56, 0.76, 0.58])
]

# Number of training iterations
iterations = 5000

# Train the network
for _ in range(iterations):
    for inputs, targets in train_data:
        n.train(inputs, targets)

# Evaluate performance
mse = n.evaluate(test_data)
print("Mean Squared Error:", mse)


# Query the network with a specific input
result = n.query([0.11,0.19,0.05])
print(result)



