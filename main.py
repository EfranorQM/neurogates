from LogicGateIdentifier import LogicGateIdentifier

def parse_inputs(input_string):
    """
    Convierte el string ingresado (por ejemplo: "0,0;0,1;1,0;1,1")
    en una lista de listas de enteros, por ejemplo: [[0, 0], [0, 1], [1, 0], [1, 1]].
    """
    entries = input_string.strip().split(";")
    X = []
    for entry in entries:
        try:
            # Separa cada combinación por coma y convierte a entero
            values = [int(num.strip()) for num in entry.split(",")]
            X.append(values)
        except ValueError:
            raise ValueError(f"Error al procesar la entrada: {entry}. Asegúrate de usar números 0 y 1 separados por comas.")
    return X

def parse_outputs(output_string):
    """
    Convierte el string de salidas (por ejemplo: "0,1,1,0")
    en una lista de enteros: [0, 1, 1, 0].
    """
    try:
        return [int(num.strip()) for num in output_string.strip().split(",")]
    except ValueError:
        raise ValueError("Error al procesar las salidas. Asegúrate de usar números 0 y 1 separados por comas.")

def show_entorno_cases():
    """
    Muestra tres ejemplos de casos de entorno en los que se pueden aplicar las compuertas lógicas.
    """
    print("\n=== Ejemplos de Casos de Entorno ===")
    print("1) Compuerta AND:")
    print("   Ejemplo: En un sistema de seguridad, se requiere que dos sensores de movimiento se activen simultáneamente para disparar una alarma.")
    print("2) Compuerta OR:")
    print("   Ejemplo: En un sistema de confort, se enciende una luz si alguno de dos interruptores es presionado.")
    print("3) Compuerta XOR:")
    print("   Ejemplo: En un sistema de control de acceso, un dispositivo se activa solo si uno de dos códigos es correcto, pero no ambos al mismo tiempo.")

if __name__ == "__main__":
    print("Elige el modo de ejecución:")
    print("  T: Modo Técnico (Entradas, entrenamiento, predicción y detalles técnicos)")
    print("  E: Modo Casos de Entorno (Muestra ejemplos de aplicaciones en el mundo real)")
    mode = input("Escribe 'T' o 'E': ").strip().lower()
    if mode not in ['t', 'e']:
        print("Opción no válida. Se ejecutará el modo técnico por defecto.\n")
        mode = "t"

    # Solicitar los datos de entrada hasta que sean válidos
    while True:
        entradas = input("\nIngresa las 4 combinaciones de entrada separadas por punto y coma (ej: 0,0;0,1;1,0;1,1): ")
        salidas = input("Ingresa los 4 valores de salida separados por coma (ej: 0,1,1,0): ")
        try:
            X = parse_inputs(entradas)
            y = parse_outputs(salidas)
        except ValueError as e:
            print(f"Error: {e}. Por favor, inténtalo nuevamente.\n")
            continue
        break

    # Repetir la creación de la instancia hasta que se identifique correctamente la compuerta
    while True:
        try:
            identifier = LogicGateIdentifier(X, y)
        except ValueError as e:
            print(f"Identificación fallida: {e}. Por favor, revisa los datos ingresados e inténtalo nuevamente.\n")
            # Volvemos a solicitar los datos desde cero.
            entradas = input("Ingresa las 4 combinaciones de entrada separadas por punto y coma: ")
            salidas = input("Ingresa los 4 valores de salida separados por coma: ")
            try:
                X = parse_inputs(entradas)
                y = parse_outputs(salidas)
            except ValueError as e_inner:
                print(f"Error: {e_inner}. Por favor, inténtalo nuevamente.\n")
                continue
            continue
        break

    print(f"\nCompuerta lógica identificada: {identifier.gate}")
    print("Entrenando el modelo, por favor espera...\n")
    identifier.train(verbose=True)

    # Mostrar detalles del modelo entrenado
    details = identifier.get_details()
    print("=== Detalles del Modelo Entrenado ===")
    if identifier.gate in ["AND", "OR"]:
        print("Pesos finales:", details["weights"])
        print("Cantidad de pasos en el log de entrenamiento:", len(details["training_log"]))
    elif identifier.gate == "XOR":
        print("Pesos de la capa oculta:")
        for i, neuron_weights in enumerate(details["hidden_weights"]):
            print(f" Neurona {i+1}: Pesos: {neuron_weights}, Bias: {details['hidden_bias'][i]}")
        print("Pesos de la capa de salida:")
        print(f" Pesos: {details['output_weights']}, Bias: {details['output_bias']}")
        print("Cantidad de pasos en el log de entrenamiento:", len(details["training_log"]))

    if mode == "t":
        # Modo Técnico: permite además probar una predicción
        while True:
            prueba = input("\nIngresa una combinación de entrada para predecir (ejemplo: 0,1): ")
            try:
                prueba_parsed = [int(num.strip()) for num in prueba.strip().split(",")]
            except ValueError:
                print("Error al procesar la combinación de prueba. Usa el formato: 0,1. Inténtalo nuevamente.\n")
                continue
            break

        resultado = identifier.predict(prueba_parsed)
        print(f"\nPredicción para la entrada {prueba_parsed}: {resultado}")
    elif mode == "e":
        # Modo Casos de Entorno: además de lo técnico, mostramos ejemplos de aplicación
        show_entorno_cases()
