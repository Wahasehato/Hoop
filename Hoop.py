class Heb:
    def p1(self):
        print('''
class McCullochPittsNeuron: 
    def __init__(self, weights, threshold): 
        self.weights = weights 
        self.threshold = threshold 

    def activate(self, inputs): 
        weighted_sum = sum([inputs[i] * self.weights[i] for i in range(len(inputs))]) 
        return 1 if weighted_sum >= self.threshold else 0 

# AND Logic Function 
and_weights = [1, 1] 
and_threshold = 2 
and_neuron = McCullochPittsNeuron(and_weights, and_threshold) 

# OR Logic Function 
or_weights = [1, 1] 
or_threshold = 1 
or_neuron = McCullochPittsNeuron(or_weights, or_threshold) 

# Test AND logic function 
input_values_and = [(0, 0), (0, 1), (1, 0), (1, 1)] 
print("AND Logic Function:") 
for inputs in input_values_and: 
    output = and_neuron.activate(inputs) 
    print(f"Input: {inputs}, Output: {output}") 

# Test OR logic function 
input_values_or = [(0, 0), (0, 1), (1, 0), (1, 1)] 
print("\n OR Logic Function:") 
for inputs in input_values_or: 
    output = or_neuron.activate(inputs) 
    print(f"Input: {inputs}, Output: {output}")
    
        ''')
    def p2(self):
        print(''' 
class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold

    def activate(self, inputs):
        weighted_sum = sum([inputs[i] * self.weights[i] for i in range(len(inputs))])
        return 1 if weighted_sum >= self.threshold else 0

and_not_weights = [1, -1]
and_not_threshold = 1
and_not_neuron = McCullochPittsNeuron(and_not_weights, and_not_threshold)

input_values_and_not = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("And not")
for inputs in input_values_and_not:
    outputs = and_not_neuron.activate(inputs)
    print(f"Input={inputs}, Output={outputs}")

xor_weights = [1, 1]
xor_threshold = 2
xor_neuron = McCullochPittsNeuron(xor_weights, xor_threshold)

or_weights = [1, 1]
or_threshold = 1
or_neuron = McCullochPittsNeuron(or_weights, or_threshold)

nand_weights = [-1, -1]
nand_threshold = -1
nand_neuron = McCullochPittsNeuron(nand_weights, nand_threshold)

input_values_xor = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("XOR")
for inputs in input_values_xor:
    nand_output = nand_neuron.activate(inputs)
    or_output = or_neuron.activate(inputs)
    outputs = xor_neuron.activate((nand_output, or_output))
    print(f"Input={inputs}, Output={outputs}")
              
              ''')
    def p3(self):
        print(''' 
#Implement the Perceptron Learning single layer Algorithm by Initializing the weights and threshold.
#Execute the code and check, how many iterations are needed, until the network converge.
import numpy as np

def initialize_weights_and_bias(num_features):
    # Initialize weights with small random values and bias to 0
    weights = np.random.rand(num_features)
    bias = np.random.rand(1)
    return weights, bias

def perceptron_train(inputs, targets, weights, bias, learning_rate=0.1, max_iterations=1000):
    num_iterations = 0
    converged = False

    while not converged and num_iterations < max_iterations:
        num_iterations += 1
        converged = True

        for input_data, target in zip(inputs, targets):
            # Calculate the weighted sum
            weighted_sum = np.dot(weights, input_data) + bias[0]

            # Apply the step function
            output = 1 if weighted_sum >= 0 else 0

            # Update weights and bias if necessary
            if output != target:
                converged = False
                weights += learning_rate * (target - output) * input_data
                bias += learning_rate * (target - output) * 1  # Update bias

    return weights, bias, num_iterations

# Example usage:
# Assuming you have 'inputs' and 'targets' defined somewhere before this point
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])

# Initialize weights and bias
num_features = len(inputs[0])
weights, bias = initialize_weights_and_bias(num_features)

# Train the perceptron
trained_weights, trained_bias, num_iterations = perceptron_train(inputs, targets, weights, bias)

# Print the results
print(f"Converged in {num_iterations} iterations")
print("Trained Weights:", trained_weights)
print("Trained Bias:", trained_bias[0])

              
              ''')
    def p4(self):
        print('''
import numpy as np
import matplotlib.pyplot as plt

class HebbianNetwork:
    def __init__(self, input_size):
        self.weights = np.zeros((input_size, input_size))

    def train(self, input_patterns):
        for pattern in input_patterns:
            self.weights += np.outer(pattern, pattern)

    def classify(self, input_pattern):
        output = np.dot(input_pattern, self.weights)
        return np.sign(output)

def plot_patterns(input_patterns, title):
    for pattern in input_patterns:
        plt.scatter(pattern[0], pattern[1], color='b')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.show()

def main():
    input_size = 2
    hebb_net = HebbianNetwork(input_size)

    # Define input patterns
    pattern1 = np.array([1, 1])
    pattern2 = np.array([1, -1])
    pattern3 = np.array([-1, 1])
    pattern4 = np.array([-1, -1])

    input_patterns = [pattern1, pattern2, pattern3, pattern4]

    # Train the Hebbian network
    hebb_net.train(input_patterns)

    # Classify new patterns
    test_pattern1 = np.array([0.5, 0.5])
    test_pattern2 = np.array([0.5, -0.5])
    test_pattern3 = np.array([-0.5, 0.5])
    test_pattern4 = np.array([-0.5, -0.5])

    result1 = hebb_net.classify(test_pattern1)
    result2 = hebb_net.classify(test_pattern2)
    result3 = hebb_net.classify(test_pattern3)
    result4 = hebb_net.classify(test_pattern4)

    print(f"Test Pattern 1 Result: {result1}")
    print(f"Test Pattern 2 Result: {result2}")
    print(f"Test Pattern 3 Result: {result3}")
    print(f"Test Pattern 4 Result: {result4}")

    # Plot input patterns and test patterns
    plot_patterns(input_patterns, 'Input Patterns')
    plot_patterns([test_pattern1, test_pattern2, test_pattern3, test_pattern4], 'Test Patterns')

if __name__ == "__main__":
    main()
''')
    def p5(self):
        print('''
        import numpy as np


class DiscreteFieldNetwork:
    def __init__(self,num_nueron):
        self.num_nueron=num_nueron
        self.weight=np.zeros((num_nueron,num_nueron))
    def train(self,patterns):
        pattern=np.array(patterns)
        outer_product=np.outer(pattern,pattern)
        np.fill_diagonal(outer_product,0)
        self.weight+=outer_product
    def energy(self,state):
        state=np.array(state)
        return 0.5*np.sign(np.dot(self.weight,state))
    def update_rule(self,state):
        new_state=np.sign(np.dot(self.weight,state))
        new_state[new_state>=0]=1
        new_state[new_state<0]=0
        return new_state
    def run(self,initial_state,max_iteration=100):
        current_state=np.array(initial_state)
        for _ in range(max_iteration):
            new_state=self.update_rule(current_state)
            if np.array_equal(new_state,current_state):
                break
        current_state=new_state
        return current_state

hopfield_network=DiscreteFieldNetwork(4)
training_pattern=[[1,1,1,-1]]
hopfield_network.train(training_pattern)
initial_state=[0,0,1,0]
result=hopfield_network.run(initial_state)
print(result)
print("energy:",hopfield_network.energy(result))

''')
    def p6(self):
        print('''
import numpy as np
import matplotlib.pyplot as plt

class KohonenSOM:
    def __init__(self, input_size, map_size):
        self.input_size = input_size
        self.map_size = map_size
        self.weights = np.random.rand(map_size[0], map_size[1], input_size)

    def update_weights(self, input_vector, winner, learning_rate, neighborhood_radius):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                weight_vector = self.weights[i, j, :]
                distance = np.linalg.norm(np.array([i, j]) - np.array(winner))
                influence = np.exp(-(distance*2) / (2 * neighborhood_radius*2))
                self.weights[i, j, :] += learning_rate * influence * (input_vector - weight_vector)

    def train(self, data, epochs, initial_learning_rate=0.1, initial_radius=None):
        if initial_radius is None:
            initial_radius = max(self.map_size) / 2

        for epoch in range(epochs):
            learning_rate = initial_learning_rate * np.exp(-epoch / epochs)
            neighborhood_radius = initial_radius * np.exp(-epoch / epochs)

            for input_vector in data:
                winner = self.find_winner(input_vector)
                self.update_weights(input_vector, winner, learning_rate, neighborhood_radius)

    def find_winner(self, input_vector):
        min_distance = float('inf')
        winner = (0, 0)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                weight_vector = self.weights[i, j, :]
                distance = np.linalg.norm(input_vector - weight_vector)

                if distance < min_distance:
                    min_distance = distance
                    winner = (i, j)

        return winner

    def visualize(self, data):
        colors = ['r', 'g', 'b', 'y', 'c', 'm']

        for input_vector in data:
            winner = self.find_winner(input_vector)
            plt.scatter(winner[0], winner[1], color=colors[np.random.randint(len(colors))])

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                plt.scatter(i, j, color='k', marker='x')

        plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate some random 2D data points
    data = np.random.rand(100, 2)

    # Create a Kohonen SOM with input size 2 and a 10x10 map
    som = KohonenSOM(input_size=2, map_size=(10, 10))

    # Train the SOM for 100 epochs
    som.train(data, epochs=100)

    # Visualize the result
    som.visualize(data)''')
    def p7(self):
        print('''
        import numpy as np

class LVQ:
    def __init__(self, num_prototypes=2, learning_rate=0.1, epochs=100):
        self.num_prototypes = num_prototypes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.prototypes = None

    def fit(self, X, y):
        num_features = X.shape[1]
        self.prototypes = np.random.rand(self.num_prototypes, num_features)

        for _ in range(self.epochs):
            for i, x in enumerate(X):
                # Find the closest prototype
                closest_prototype_idx = np.argmin(np.linalg.norm(self.prototypes - x, axis=1))

                # Update the closest prototype based on the learning rate and the class label
                if y[i] == 1:
                    self.prototypes[closest_prototype_idx] += self.learning_rate * (x - self.prototypes[closest_prototype_idx])
                else:
                    self.prototypes[closest_prototype_idx] -= self.learning_rate * (x - self.prototypes[closest_prototype_idx])

    def predict(self, X):
        y_pred = []
        for x in X:
            closest_prototype_idx = np.argmin(np.linalg.norm(self.prototypes - x, axis=1))
            y_pred.append(closest_prototype_idx)
        return np.array(y_pred)

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

lvq = LVQ(num_prototypes=2, learning_rate=0.1, epochs=100)
lvq.fit(X, y)

print("Predictions:", lvq.predict(X))


''')
     def p8(self):
        print('''
     import numpy as np

def bipolar_sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(hidden_size, input_size) - 0.5
    weights_hidden_output = np.random.rand(output_size, hidden_size) - 0.5
    return weights_input_hidden, weights_hidden_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output):
    hidden_inputs = np.dot(weights_input_hidden, inputs)
    hidden_outputs = bipolar_sigmoid(hidden_inputs)

    final_inputs = np.dot(weights_hidden_output, hidden_outputs)
    final_outputs = bipolar_sigmoid(final_inputs)

    return hidden_outputs, final_outputs

def backward_propagation(inputs, targets, hidden_outputs, final_outputs, weights_hidden_output):
    output_errors = targets - final_outputs
    output_gradients = bipolar_sigmoid_derivative(final_outputs) * output_errors

    hidden_errors = np.dot(weights_hidden_output.T, output_gradients)
    hidden_gradients = bipolar_sigmoid_derivative(hidden_outputs) * hidden_errors

    return output_gradients, hidden_gradients

def update_weights(inputs, hidden_outputs, output_gradients, hidden_gradients,
                   weights_input_hidden, weights_hidden_output, learning_rate):
    weights_hidden_output += learning_rate * np.outer(output_gradients, hidden_outputs)
    weights_input_hidden += learning_rate * np.outer(hidden_gradients, inputs)

def train_xor_network(inputs, targets, hidden_size, epochs, learning_rate):
    input_size = len(inputs[0])
    output_size = len(targets[0])

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(inputs)):
            input_data = inputs[i]
            target_data = targets[i]

            hidden_outputs, final_outputs = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)

            # Calculate output error and gradient
            output_errors = target_data - final_outputs
            output_gradients = bipolar_sigmoid_derivative(final_outputs) * output_errors

            # Calculate hidden layer error and gradient
            hidden_errors = np.dot(weights_hidden_output.T, output_gradients)
            hidden_gradients = bipolar_sigmoid_derivative(hidden_outputs) * hidden_errors

            # Update weights
            weights_hidden_output += learning_rate * np.outer(output_gradients, hidden_outputs)
            weights_input_hidden += learning_rate * np.outer(hidden_gradients, input_data)

            # Accumulate error for reporting
            total_error += 0.5 * np.sum(output_errors**2)

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Error: {total_error}")

    return weights_input_hidden, weights_hidden_output

def test_xor_network(inputs, weights_input_hidden, weights_hidden_output):
    for i in range(len(inputs)):
        input_data = inputs[i]
        _, output = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)
        print(f"Input: {input_data}, Output: {output}")

if __name__ == "__main__":
    # XOR function inputs and corresponding bipolar outputs
    xor_inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    xor_targets = np.array([[1], [-1], [-1], [1]])

    hidden_layer_size = 2
    training_epochs = 10000
    learning_rate = 0.1

    trained_weights_input_hidden, trained_weights_hidden_output = train_xor_network(
        xor_inputs, xor_targets, hidden_layer_size, training_epochs, learning_rate)

    print("\nTrained XOR Network:")
    test_xor_network(xor_inputs, trained_weights_input_hidden, trained_weights_hidden_output)


''')

    