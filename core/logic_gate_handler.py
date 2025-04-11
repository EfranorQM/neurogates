from typing import Tuple, List
from models.perceptron import Perceptron
from models.xor_network import XORNeuralNetwork

class LogicGateHandler:
    def __init__(self):
        self.models = {
            "AND": Perceptron(input_size=2),
            "OR": Perceptron(input_size=2),
            "XOR": XORNeuralNetwork()
        }

        self._train_all()

    def _train_all(self):
        X = [[0,0], [0,1], [1,0], [1,1]]
        self.models["AND"].fit(X, [0, 0, 0, 1], verbose=True)
        self.models["OR"].fit(X, [0, 1, 1, 1], verbose=True)
        self.models["XOR"].train(X, [0, 1, 1, 0], verbose=True)


    def predict(self, gate: str, a: int, b: int) -> Tuple[int, List[float]]:
        gate = gate.upper()
        model = self.models.get(gate)

        if not model:
            raise ValueError(f"Compuerta no soportada: {gate}")

        result = model.predict([a, b])

        if gate in ("AND", "OR"):
            weights = model.get_weights()
        else:
            # Red neuronal -> mostrar pesos de capa de salida (como ejemplo)
            weights = model.output_weights + [model.output_bias]

        return result, weights

    def get_training_log(self, gate: str) -> List[dict]:
        gate = gate.upper()
        model = self.models.get(gate)

        if not model:
            raise ValueError(f"Compuerta no soportada: {gate}")

        return model.get_training_log()
