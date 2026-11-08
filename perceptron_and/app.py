import streamlit as st
import pandas as pd

from perceptron import train_perceptron, predict, AND_DATA
from pdf_generator import create_training_pdf


st.set_page_config(
    page_title="Perceptrón AND | Laboratorio IA",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label,
    .stApp span, .stApp div {
        color: #FFFFFF;
    }

    div[data-testid="stMetric"] {
        background-color: #1B222C;
        padding: 15px;
        border-radius: 8px;
    }

    .stButton > button {
        background-color: #30363D;
        color: white;
        border: 1px solid #484F58;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title"> Laboratorio de Perceptrón Monocapa</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Aprendizaje supervisado de la compuerta lógica AND</div>', unsafe_allow_html=True)
st.divider()

# ------------------------- CONFIGURACIÓN -------------------------
with st.sidebar:
    st.header(" Configuración")
    st.caption("Define los valores iniciales antes de entrenar.")
    initial_w1 = st.number_input("Peso inicial w1", value=0.0, step=0.1, format="%.3f")
    initial_w2 = st.number_input("Peso inicial w2", value=0.0, step=0.1, format="%.3f")
    initial_bias = st.number_input("Bias inicial (b)", value=0.0, step=0.1, format="%.3f")
    learning_rate = st.number_input(
        "Tasa de aprendizaje (η)", min_value=0.001, max_value=10.0,
        value=0.1, step=0.05, format="%.3f"
    )
    max_epochs = st.number_input(
        "Máximo de épocas", min_value=1, max_value=10000,
        value=100, step=10
    )
    train_button = st.button(" Entrenar perceptrón", type="primary", use_container_width=True)
    st.caption("La tasa de aprendizaje debe ser mayor que cero.")

if train_button:
    try:
        result = train_perceptron(
            float(initial_w1), float(initial_w2), float(initial_bias),
            float(learning_rate), int(max_epochs)
        )
        st.session_state["training_result"] = result
        st.session_state["initial_values"] = {
            "w1": float(initial_w1), "w2": float(initial_w2),
            "bias": float(initial_bias)
        }
        st.session_state["learning_rate"] = float(learning_rate)
        st.session_state["max_epochs"] = int(max_epochs)
        st.session_state["has_trained"] = True
    except ValueError as exc:
        st.error(str(exc))

# ------------------------- TABLA AND -------------------------
left, right = st.columns([1, 1.5])
with left:
    st.subheader("Tabla de verdad AND")
    truth_df = pd.DataFrame(AND_DATA)
    truth_df.columns = ["Entrada x1", "Entrada x2", "Objetivo"]
    st.dataframe(truth_df, hide_index=True, use_container_width=True)

with right:
    st.subheader("Estructura del perceptrón")
    current = st.session_state.get("training_result")
    if current:
        w1_show, w2_show, b_show = current["w1"], current["w2"], current["bias"]
    else:
        w1_show, w2_show, b_show = initial_w1, initial_w2, initial_bias

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="node"><strong>x₁ → w₁</strong><br><small>Peso: {w1_show:.3f}</small></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="node"><strong>x₂ → w₂</strong><br><small>Peso: {w2_show:.3f}</small></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="node"><strong>Bias (b)</strong><br><small>{b_show:.3f}</small></div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#60758B;padding:7px'>↓ suma ponderada: z = x₁w₁ + x₂w₂ + b ↓</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="node"><strong>Σ</strong><br><small>Sumatoria</small></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="node"><strong>Escalón</strong><br><small>1 si z ≥ 0; si no, 0</small></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="node"><strong>Salida y</strong><br><small>Predicción</small></div>', unsafe_allow_html=True)

st.divider()

# ------------------------- RESULTADOS -------------------------
if st.session_state.get("has_trained", False):
    result = st.session_state["training_result"]
    st.subheader(" Resultados del entrenamiento")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Épocas ejecutadas", result["epochs_completed"])
    m2.metric("Peso w1 final", f'{result["w1"]:.4f}')
    m3.metric("Peso w2 final", f'{result["w2"]:.4f}')
    m4.metric("Bias final", f'{result["bias"]:.4f}')

    if result["converged"]:
        st.success("¡Entrenamiento completado! El perceptrón alcanzó cero errores en una época.")
    else:
        st.warning("Se alcanzó el límite de épocas. El perceptrón todavía presenta errores; intenta aumentar el límite o ajustar los parámetros.")

    st.markdown("#### Evolución por épocas")
    history_df = pd.DataFrame(result["history"])
    history_display = history_df.rename(columns={
        "epoch": "Época", "w1": "Peso w1", "w2": "Peso w2",
        "bias": "Bias", "errors": "Número de errores"
    })
    st.dataframe(history_display, hide_index=True, use_container_width=True)

    st.markdown("#### Procedimiento matemático por muestra")
    st.latex(r"z = (x_1 \cdot w_1) + (x_2 \cdot w_2) + b")
    st.latex(r"y = f(z) = \begin{cases}1 & z \geq 0\\0 & z < 0\end{cases}")
    st.latex(r"e = t-y")
    st.latex(r"w_1^{nuevo}=w_1^{anterior}+\eta \cdot e \cdot x_1")
    st.latex(r"w_2^{nuevo}=w_2^{anterior}+\eta \cdot e \cdot x_2")
    st.latex(r"b^{nuevo}=b^{anterior}+\eta \cdot e")

    steps_df = pd.DataFrame(result["steps"])
    for epoch in sorted(steps_df["epoch"].unique()):
        with st.expander(f"Época {epoch}: operaciones de sus cuatro muestras"):
            epoch_steps = steps_df[steps_df["epoch"] == epoch]
            for _, s in epoch_steps.iterrows():
                st.markdown(f"**Muestra {int(s['sample'])}:** entrada ({int(s['x1'])}, {int(s['x2'])}), objetivo t = {int(s['target'])}")
                st.latex(
                    rf"z=({s['x1']}\cdot {s['w1_before']:.4f})+"
                    rf"({s['x2']}\cdot {s['w2_before']:.4f})+"
                    rf"{s['bias_before']:.4f}={s['z']:.4f}"
                )
                st.latex(rf"y=f(z)={int(s['prediction'])},\quad e=t-y={int(s['target'])}-{int(s['prediction'])}={int(s['error'])}")
                st.latex(
                    rf"w_1={s['w1_before']:.4f}+{learning_rate:.4f}\cdot"
                    rf"{int(s['error'])}\cdot {int(s['x1'])}={s['w1_after']:.4f}"
                )
                st.latex(
                    rf"w_2={s['w2_before']:.4f}+{learning_rate:.4f}\cdot"
                    rf"{int(s['error'])}\cdot {int(s['x2'])}={s['w2_after']:.4f}"
                )
                st.latex(
                    rf"b={s['bias_before']:.4f}+{learning_rate:.4f}\cdot"
                    rf"{int(s['error'])}={s['bias_after']:.4f}"
                )
                st.caption(
                    f"Después de esta muestra: w1={s['w1_after']:.4f}, "
                    f"w2={s['w2_after']:.4f}, b={s['bias_after']:.4f}."
                )
                st.divider()

    st.markdown("#### Descargar informe")
    pdf_bytes = create_training_pdf(
        result, st.session_state["initial_values"],
        st.session_state["learning_rate"], st.session_state["max_epochs"]
    )
    st.download_button(
        "📄 Descargar informe PDF",
        data=pdf_bytes,
        file_name="informe_perceptron_AND.pdf",
        mime="application/pdf",
        type="primary"
    )

    # ------------------------- INFERENCIA -------------------------
    st.divider()
    st.subheader(" Módulo de inferencia / prueba")
    st.write("Selecciona una combinación de entradas. Se utilizarán los pesos finales del entrenamiento.")
    test1, test2 = st.columns(2)
    with test1:
        input_x1 = st.selectbox("Entrada x1", [0, 1], key="inference_x1")
    with test2:
        input_x2 = st.selectbox("Entrada x2", [0, 1], key="inference_x2")

    z_test, y_test = predict(
        input_x1, input_x2, result["w1"], result["w2"], result["bias"]
    )
    out1, out2, out3 = st.columns(3)
    out1.metric("Suma ponderada (z)", f"{z_test:.4f}")
    out2.metric("Predicción (y)", y_test)
    out3.metric("Resultado AND esperado", input_x1 & input_x2)
    if y_test == (input_x1 & input_x2):
        st.success("La predicción coincide con la tabla de verdad AND.")
    else:
        st.error("La predicción no coincide. El modelo necesita más entrenamiento o una configuración diferente.")

else:
    st.info("Configura los hiperparámetros en el menú izquierdo y presiona «Entrenar perceptrón» para ver los resultados, el procedimiento matemático, el PDF y el módulo de inferencia.")

st.caption("Perceptrón monocapa · Compuerta lógica AND · Python 3.x + Streamlit")
