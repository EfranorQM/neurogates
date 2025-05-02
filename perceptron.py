class Perceptron:
    def __init__(self, n_inputs, lr=0.1, epochs=10):
        # Creamos n_inputs pesos (0) y uno extra para el bias
        self.w = [0.0] * (n_inputs + 1)
        self.lr = lr        # tasa de aprendizaje
        self.epochs = epochs  # cuántas veces repasamos el dataset

    def predict(self, inputs):
        # suma ponderada de entradas
        total = 0.0
        for i in range(len(inputs)):
            total += self.w[i] * inputs[i]

        # bias (último peso * 1)
        total += self.w[-1] * 1

        if total >= 0:
            return 1
        else:
            return 0

    def train(self, X, y):
        for i in range(self.epochs):
            for index in range(len(X)):
                inputs = X[index]      # [0,1]
                expected = y[index]    # 1

                prediction = self.predict(inputs)

                error = expected - prediction

                # w_new = w_old + lr * error * input
                for j in range(len(inputs)):
                    self.w[j] = self.w[j] + self.lr * error * inputs[j]

                # ajustar el bias
                self.w[-1] = self.w[-1] + self.lr * error

