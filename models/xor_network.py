import math
import random
from typing import List

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x: float) -> float:
    # x ya es la salida del sigmoid, así que su derivada es x*(1-x)
    return x * (1 - x)

class XORNeuralNetwork:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 20000):
        # 2 neuronas en capa oculta, 1 en la capa de salida
        self.lr = learning_rate
        self.epochs = epochs

        # Pesos y biases aleatorios (se pueden inicializar en un rango más pequeño)
        self.hidden_weights = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]  # 2x2
        self.hidden_bias = [random.uniform(-1, 1) for _ in range(2)]

        self.output_weights = [random.uniform(-1, 1) for _ in range(2)]  # 2 → 1
        self.output_bias = random.uniform(-1, 1)

    def feedforward(self, inputs: List[int]) -> float:
        # Capa oculta
        self.hidden_layer = []
        for i in range(2):
            total = sum(w * inp for w, inp in zip(self.hidden_weights[i], inputs)) + self.hidden_bias[i]
            self.hidden_layer.append(sigmoid(total))

        # Capa de salida
        total_output = sum(w * h for w, h in zip(self.output_weights, self.hidden_layer)) + self.output_bias
        self.output = sigmoid(total_output)
        return self.output

    def train(self, X: List[List[int]], y: List[int], verbose: bool = False):
        self.training_log = []

        for epoch in range(self.epochs):
            # Mezclar el orden de los ejemplos en cada época
            indices = list(range(len(X)))
            random.shuffle(indices)
            for idx in indices:
                inputs = X[idx]
                expected = y[idx]
                output = self.feedforward(inputs)
                error = expected - output
                d_output = error * sigmoid_derivative(output)

                d_hidden = []
                for i in range(2):
                    err = d_output * self.output_weights[i]
                    d_hidden.append(err * sigmoid_derivative(self.hidden_layer[i]))

                # Registro antes de la actualización (si verbose está activado)
                log = {
                    "epoch": epoch,
                    "inputs": inputs,
                    "expected": expected,
                    "predicted": output,
                    "error": error,
                    "output_weights_before": self.output_weights.copy(),
                    "hidden_weights_before": [h.copy() for h in self.hidden_weights],
                    "log_hidden_outputs": self.hidden_layer.copy(),
                }

                # Actualización de la capa de salida
                for i in range(2):
                    self.output_weights[i] += self.lr * d_output * self.hidden_layer[i]
                self.output_bias += self.lr * d_output

                # Actualización de la capa oculta
                for i in range(2):
                    for j in range(2):
                        self.hidden_weights[i][j] += self.lr * d_hidden[i] * inputs[j]
                    self.hidden_bias[i] += self.lr * d_hidden[i]

                # Registro después de la actualización
                log["output_weights_after"] = self.output_weights.copy()
                log["hidden_weights_after"] = [h.copy() for h in self.hidden_weights]

                if verbose:
                    self.training_log.append(log)

    def predict(self, inputs: List[int]) -> int:
        output = self.feedforward(inputs)
        return 1 if output > 0.5 else 0

    def get_training_log(self) -> List[dict]:
        return self.training_log
