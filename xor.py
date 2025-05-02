from typing import List
from perceptron import Perceptron

class XORNetwork:
    def __init__(self, lr: float = 0.1, epochs: int = 10):
        # Capas
        self.hidden_or = Perceptron(2, lr, epochs)
        self.hidden_nand = Perceptron(2, lr, epochs)
        self.output_and = Perceptron(2, lr, epochs)

    def train(self, X: List[List[int]], y: List[int]):
        # salidas
        or_targets = [0, 1, 1, 1]
        nand_targets = [1, 1, 1, 0]
    
        self.hidden_or.train(X, or_targets)
        self.hidden_nand.train(X, nand_targets)

        # Salidas de la capa oculta
        hidden_outputs = []
        for inputs in X:
            output_or = self.hidden_or.predict(inputs)
            output_nand = self.hidden_nand.predict(inputs)
            hidden_outputs.append([output_or, output_nand])

        # Entrenar perceptron de salida (AND(OR,NAND))
        self.output_and.train(hidden_outputs, y)

    def predict(self, x: List[int]) -> int:
        # 1) Calculamos las activaciones ocultas
        h1 = self.hidden_or.predict(x)
        h2 = self.hidden_nand.predict(x)
        # 2) Calculamos la salida final
        return self.output_and.predict([h1, h2])
