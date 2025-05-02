from xor import XORNetwork


def main():
    print("=== Simulador XOR ===")

    # Permitir al estudiante elegir lr y epochs
    try:
        lr_input = input("Ingresa learning rate (por ejemplo 0.1) o presiona Enter para 0.1: ")
        lr = float(lr_input) if lr_input.strip() != "" else 0.1
    except ValueError:
        print("Valor inválido para learning rate. Usando lr=0.1 por defecto.")
        lr = 0.1

    try:
        ep_input = input("Ingresa número de epochs (por ejemplo 10) o presiona Enter para 10: ")
        epochs = int(ep_input) if ep_input.strip() != "" else 10
    except ValueError:
        print("Valor inválido para epochs. Usando epochs=10 por defecto.")
        epochs = 10

    print(f"Usando lr={lr} y epochs={epochs}\n")

    net = XORNetwork(lr=lr, epochs=epochs)
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    net.train(X, y)
    print(f"Red XOR entrenada con lr={lr} y epochs={epochs}.\n")

    while True:
        entrada = input("Ingresa dos bits (0/1) separados por coma (ej: 1,0) o 'salir': ")
        if entrada.strip().lower() in ('salir', 'exit', 'q'):
            print("¡Hasta luego!")
            break
        partes = entrada.split(',')
        if len(partes) != 2 or any(p.strip() not in ('0', '1') for p in partes):
            print("Formato inválido. Usa '0,1' o '1,0'.")
            continue
        a, b = int(partes[0]), int(partes[1])
        resultado = net.predict([a, b])
        print(f"XOR({a}, {b}) = {resultado}\n")

if __name__ == '__main__':
    main()