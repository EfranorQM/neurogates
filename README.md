# Proyecto NeuroGates

Este proyecto implementa compuertas lógicas utilizando **perceptrones** y **redes neuronales simples**. Se abordan tres compuertas lógicas fundamentales: **AND**, **OR** y **XOR**. La compuerta XOR, que no es linealmente separable, se resuelve mediante una red neuronal con una capa oculta. Además, el proyecto incorpora una clase llamada `LogicGateIdentifier` que permite identificar automáticamente qué compuerta se debe entrenar y, posteriormente, visualizar tanto los detalles técnicos como ejemplos de aplicación en el mundo real.

## Contenido del Proyecto

- **perceptron.py:**  
  Implementa el modelo de perceptrón básico, incluyendo la función de activación (escalón), la actualización de pesos y el registro del proceso de entrenamiento.

- **xor_network.py:**  
  Define una red neuronal simple para resolver la compuerta XOR. Se utiliza una capa oculta con dos neuronas y una capa de salida, junto con funciones para el feedforward, el entrenamiento y la predicción.

- **LogicGateIdentifier.py:**  
  Gestiona los distintos modelos (perceptrones para AND y OR, y la red neuronal para XOR). Se encarga de entrenar todos los modelos y permite obtener predicciones y detalles técnicos (pesos y logs de entrenamiento).

- **main.py:**  
  Programa interactivo que permite al usuario elegir entre dos modos de ejecución:
  - **Modo Técnico (T):**  
    El usuario ingresa manualmente las combinaciones de entrada y los valores de salida. El sistema entrena el modelo correspondiente, muestra detalles técnicos y permite realizar predicciones.
  - **Modo Casos de Entorno (E):**  
    Se muestran ejemplos reales de aplicación de las compuertas lógicas, sin solicitar datos al usuario.

- **LogicGateIdentifier:**  
  Una clase que, a partir de un conjunto de entradas y salidas, identifica la compuerta lógica a utilizar y crea el modelo adecuado (perceptrón para AND y OR, red neuronal para XOR).

## Instalación y Requisitos

- **Python 3.x:**  
  Asegúrate de tener instalado Python 3.
  
- **Librerías estándar:**  
  Este proyecto utiliza únicamente módulos estándar de Python: `math`, `random` y `typing`.

No es necesaria la instalación de librerías adicionales.

## Uso

1. **Ejecutar el Proyecto**  
   Desde la terminal, navega hasta el directorio del proyecto y ejecuta:
   ```bash
   python main.py

# Selección de Modo

Al iniciar, se te pedirá elegir entre dos modos:

## Modo Técnico (T)
- **Se solicita ingresar:**
  - Las 4 combinaciones de entrada (ejemplo: `0,0;0,1;1,0;1,1`).
  - Los 4 valores de salida correspondientes (ejemplo: `0,1,1,0` para la compuerta XOR).
- **Después de entrenar el modelo:**
  - Se muestran los detalles técnicos (pesos, log del entrenamiento) y se permite probar una predicción.

## Modo Casos de Entorno (E)
- Se muestran ejemplos prácticos donde se aplican las compuertas lógicas, sin solicitar datos al usuario.

# Interacción y Resultados

## Modo Técnico
- Se muestra el proceso de entrenamiento, los detalles técnicos y se realiza una predicción basada en la entrada de prueba.

## Modo Casos de Entorno
- Se exhiben ejemplos de aplicación de compuertas lógicas en situaciones reales, tales como:

  - **Compuerta AND:**
    - *Ejemplo:* En un sistema de seguridad, se requiere que dos sensores de movimiento se activen simultáneamente para disparar una alarma.
  
  - **Compuerta OR:**
    - *Ejemplo:* En un sistema de confort, se enciende una luz si alguno de dos interruptores es presionado.
  
  - **Compuerta XOR:**
    - *Ejemplo:* En un sistema de control de acceso, un dispositivo se activa solo si uno de dos códigos es correcto, pero no ambos.

# Ejemplos de Aplicación (Casos de Entorno)

- **Compuerta AND:**
  - *Ejemplo:* Se requiere la activación simultánea de dos sensores de movimiento para que se dispare una alarma en un sistema de seguridad.
  
- **Compuerta OR:**
  - *Ejemplo:* En un sistema de confort, se enciende una luz cuando se presiona alguno de dos interruptores.
  
- **Compuerta XOR:**
  - *Ejemplo:* En un sistema de control de acceso, se valida la entrada únicamente si uno de dos códigos es correcto (exclusividad).

# Consideraciones y Mejoras

- **Inicialización Aleatoria y Convergencia:**
  - Los pesos y biases se inicializan de manera aleatoria.
  - Se han ajustado parámetros como la tasa de aprendizaje y el número de épocas para mejorar la convergencia, aunque la red puede comportarse de forma ligeramente distinta en cada ejecución.
  
- **Validación de Datos:**
  - El programa incluye validaciones interactivas que evitan que errores en el formato de entrada o salida detengan la ejecución.
  - En caso de error, se muestran mensajes útiles y se vuelven a solicitar los datos.
  
- **Extensibilidad:**
  - La estructura modular permite integrar nuevas compuertas o funcionalidades adicionales en el futuro.

# Conclusión

Este proyecto combina conceptos de lógica digital y redes neuronales para implementar compuertas lógicas básicas. La integración del `LogicGateIdentifier` y la posibilidad de elegir entre un modo técnico o de casos de entorno brindan una herramienta educativa interactiva para comprender el funcionamiento de perceptrones y redes neuronales, así como sus aplicaciones prácticas.
