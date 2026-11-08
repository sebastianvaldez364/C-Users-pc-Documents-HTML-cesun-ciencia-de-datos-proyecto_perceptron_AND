"""
Lógica del Perceptrón Monocapa para aprender la compuerta lógica AND.
Este módulo no depende de Streamlit: contiene únicamente el algoritmo.
"""

AND_DATA = [
    {"x1": 0, "x2": 0, "target": 0},
    {"x1": 0, "x2": 1, "target": 0},
    {"x1": 1, "x2": 0, "target": 0},
    {"x1": 1, "x2": 1, "target": 1},
]


def step_function(z):
    """Función escalón: devuelve 1 si z >= 0; de lo contrario devuelve 0."""
    return 1 if z >= 0 else 0


def predict(x1, x2, w1, w2, bias):
    """Calcula la suma ponderada y la salida del perceptrón."""
    z = (x1 * w1) + (x2 * w2) + bias
    y = step_function(z)
    return z, y


def train_perceptron(w1, w2, bias, learning_rate, max_epochs):
    """
    Entrena el perceptrón con la regla de actualización:
      e = objetivo - predicción
      w1 = w1 + eta * e * x1
      w2 = w2 + eta * e * x2
      b  = b  + eta * e

    Devuelve pesos finales, historial por época y detalle por muestra.
    """
    if learning_rate <= 0:
        raise ValueError("La tasa de aprendizaje debe ser mayor que cero.")
    if max_epochs < 1:
        raise ValueError("El número máximo de épocas debe ser al menos 1.")

    history = []
    steps = []
    converged = False

    for epoch in range(1, max_epochs + 1):
        errors = 0

        for sample_number, sample in enumerate(AND_DATA, start=1):
            x1 = sample["x1"]
            x2 = sample["x2"]
            target = sample["target"]

            old_w1, old_w2, old_bias = w1, w2, bias
            z, prediction_value = predict(x1, x2, w1, w2, bias)
            error = target - prediction_value

            if error != 0:
                errors += 1

            delta_w1 = learning_rate * error * x1
            delta_w2 = learning_rate * error * x2
            delta_bias = learning_rate * error

            w1 += delta_w1
            w2 += delta_w2
            bias += delta_bias

            steps.append({
                "epoch": epoch,
                "sample": sample_number,
                "x1": x1,
                "x2": x2,
                "target": target,
                "w1_before": old_w1,
                "w2_before": old_w2,
                "bias_before": old_bias,
                "z": z,
                "prediction": prediction_value,
                "error": error,
                "delta_w1": delta_w1,
                "delta_w2": delta_w2,
                "delta_bias": delta_bias,
                "w1_after": w1,
                "w2_after": w2,
                "bias_after": bias,
            })

        history.append({
            "epoch": epoch,
            "w1": w1,
            "w2": w2,
            "bias": bias,
            "errors": errors,
        })

        if errors == 0:
            converged = True
            break

    return {
        "w1": w1,
        "w2": w2,
        "bias": bias,
        "history": history,
        "steps": steps,
        "converged": converged,
        "epochs_completed": len(history),
    }
