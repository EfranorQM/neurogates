from typing import List

class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1, epochs: int = 10):
        self.weights = [0.0] * (input_size + 1)  # +1 para el bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x: float) -> int:
        return 1 if x >= 0 else 0  # Función escalón

    def predict(self, inputs: List[int]) -> int:
        # Añadimos el bias como un 1 al final
        total = sum(w * i for w, i in zip(self.weights, inputs + [1]))
        return self.activation(total)

    # Dentro de la clase Perceptron

    def fit(self, X: List[List[int]], y: List[int], verbose: bool = False):
        self.training_log = []

        for epoch in range(self.epochs):
            for inputs, expected in zip(X, y):
                prediction = self.predict(inputs)
                error = expected - prediction

                log = {
                    "epoch": epoch,
                    "inputs": inputs,
                    "expected": expected,
                    "predicted": prediction,
                    "error": error,
                    "weights_before": self.weights.copy()
                }

                for i in range(len(inputs)):
                    self.weights[i] += self.learning_rate * error * inputs[i]
                self.weights[-1] += self.learning_rate * error  # bias

                log["weights_after"] = self.weights.copy()

                if verbose:
                    self.training_log.append(log)

    def get_weights(self) -> List[float]:
        return self.weights

    def get_training_log(self) -> List[dict]:
        return self.training_log
