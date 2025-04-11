from models.perceptron import Perceptron
from models.xor_network import XORNeuralNetwork

class LogicGateIdentifier:
    def __init__(self, X, y):
        """
        Inicializa la clase con los datos de entrada X y salida y.
        Se identifica automáticamente la compuerta lógica según los datos.
        """
        self.X = X
        self.y = y
        self.gate = self.identify_gate()  # Identifica la compuerta
        self.model = self.initialize_model()
    
    def identify_gate(self):
        """
        Identifica la compuerta lógica basada en el vector de salidas.
        Se compara con los patrones conocidos para AND, OR y XOR.
        """
        if self.y == [0, 0, 0, 1]:
            return "AND"
        elif self.y == [0, 1, 1, 1]:
            return "OR"
        elif self.y == [0, 1, 1, 0]:
            return "XOR"
        else:
            raise ValueError("El patrón de salida no corresponde a ninguna compuerta soportada (AND, OR, XOR)")
    
    def initialize_model(self):
        """
        Inicializa el modelo según la compuerta identificada.
        Se utiliza Perceptron para compuertas linealmente separables (AND, OR)
        y XORNeuralNetwork para la compuerta XOR.
        """
        if self.gate in ["AND", "OR"]:
            return Perceptron(input_size=2)
        elif self.gate == "XOR":
            return XORNeuralNetwork()
    
    def train(self, verbose=True):
        """
        Entrena el modelo con los datos proporcionados.
        Dependiendo de la compuerta, se llama a fit (perceptron) o train (red XOR).
        """
        if self.gate in ["AND", "OR"]:
            self.model.fit(self.X, self.y, verbose=verbose)
        elif self.gate == "XOR":
            self.model.train(self.X, self.y, verbose=verbose)
    
    def predict(self, inputs):
        """
        Realiza una predicción para un conjunto de entradas.
        """
        return self.model.predict(inputs)
    
    def get_details(self):
        """
        Devuelve un diccionario con detalles del modelo entrenado,
        incluyendo los pesos y el log del entrenamiento.
        """
        details = {"gate": self.gate}
        if self.gate in ["AND", "OR"]:
            details["weights"] = self.model.get_weights()
            details["training_log"] = self.model.get_training_log()
        elif self.gate == "XOR":
            details["hidden_weights"] = self.model.hidden_weights
            details["hidden_bias"] = self.model.hidden_bias
            details["output_weights"] = self.model.output_weights
            details["output_bias"] = self.model.output_bias
            details["training_log"] = self.model.get_training_log()
        return details

if __name__ == "__main__":
    # Datos de entrada estándar para compuertas lógicas
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    # Podemos probar con las tres compuertas. Aquí te muestro ejemplos.
    
    # --- Prueba con la compuerta AND ---
    print("Prueba con la compuerta AND:")
    y_and = [0, 0, 0, 1]
    identifier_and = LogicGateIdentifier(X, y_and)
    identifier_and.train(verbose=False)
    details_and = identifier_and.get_details()
    print("Compuerta identificada:", details_and["gate"])
    print("Pesos:", details_and["weights"])
    print("Número de pasos en el log de entrenamiento:", len(details_and["training_log"]))
    print("-" * 50)
    
    # --- Prueba con la compuerta OR ---
    print("Prueba con la compuerta OR:")
    y_or = [0, 1, 1, 1]
    identifier_or = LogicGateIdentifier(X, y_or)
    identifier_or.train(verbose=False)
    details_or = identifier_or.get_details()
    print("Compuerta identificada:", details_or["gate"])
    print("Pesos:", details_or["weights"])
    print("Número de pasos en el log de entrenamiento:", len(details_or["training_log"]))
    print("-" * 50)
    
    # --- Prueba con la compuerta XOR ---
    print("Prueba con la compuerta XOR:")
    y_xor = [0, 1, 1, 0]
    identifier_xor = LogicGateIdentifier(X, y_xor)
    identifier_xor.train(verbose=False)
    details_xor = identifier_xor.get_details()
    print("Compuerta identificada:", details_xor["gate"])
    print("Pesos capa oculta:", details_xor["hidden_weights"])
    print("Bias capa oculta:", details_xor["hidden_bias"])
    print("Pesos capa de salida:", details_xor["output_weights"])
    print("Bias de salida:", details_xor["output_bias"])
    print("Número de pasos en el log de entrenamiento:", len(details_xor["training_log"]))
    print("-" * 50)
    
    # Ejemplo de predicción para XOR
    test_input = [1, 0]
    prediction = identifier_xor.predict(test_input)
    print("Predicción para la entrada", test_input, "en compuerta XOR:", prediction)
