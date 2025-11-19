import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import *

# ------------------------------
# Función para evaluar f(x) dada como texto
# ------------------------------
def build_function(func_str):
    """
    Construye una función f(x) a partir de un string.
    Permite usar funciones de numpy y de math.
    """
    allowed_names = {}

    # Importar funciones de numpy
    import numpy as np
    allowed_names.update({k: getattr(np, k) for k in dir(np) if not k.startswith("_")})

    # Importar funciones de math
    import math
    allowed_names.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})

    def f(x):
        return eval(func_str, {"__builtins__": {}}, {**allowed_names, "x": x})

    return f

# ------------------------------
# Métodos de integración
# ------------------------------
def trapecio(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    I = h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1])
    return I, x, y, h

def simpson_tercio(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3, n debe ser par.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    I = (h / 3) * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum())
    return I, x, y, h

# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.set_page_config(page_title="Integración numérica", layout="wide")

st.title("Aplicativo Web: Regla del Trapecio y Simpson 1/3")
st.write(
    """
    Este aplicativo permite aproximar una integral definida usando:
    - **Regla del trapecio**
    - **Regla de Simpson 1/3**

    Ingresa la función, los límites de integración y el número de subintervalos.
    """
)

# ---- Entradas del usuario ----
col1, col2 = st.columns(2)

with col1:
    metodo = st.radio(
        "Método de integración",
        ("Regla del trapecio", "Regla de Simpson 1/3")
    )
    func_str = st.text_input("Función f(x):", value="x**2")
    a = st.number_input("Límite inferior a:", value=0.0)
    b = st.number_input("Límite superior b:", value=2.0)
    n = st.number_input("Número de subintervalos n:", min_value=1, value=4, step=1)

with col2:
    mostrar_paso_a_paso = st.checkbox("Mostrar paso a paso", value=True)
    mostrar_grafica = st.checkbox("Mostrar gráfica", value=True)
    st.info(
        "Puedes usar funciones como `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, etc.\n"
        "Ejemplos: `x**2`, `sin(x)`, `exp(-x**2)`"
    )

# Botón para calcular
if st.button("Calcular integral aproximada"):
    try:
        f = build_function(func_str)
        # Verificación rápida de la función
        _ = f(np.array([a, b]))

        if metodo == "Regla del trapecio":
            I, x, y, h = trapecio(f, a, b, int(n))
        else:
            if int(n) % 2 != 0:
                st.error("Para Simpson 1/3, el número de subintervalos n debe ser **par**.")
                st.stop()
            I, x, y, h = simpson_tercio(f, a, b, int(n))

        # ---- Resultado principal ----
        st.subheader("Resultado de la integral aproximada")
        st.write(
            f"Integral aproximada de `f(x) = {func_str}` en el intervalo "
            f"[{a}, {b}] con n = {int(n)} subintervalos:"
        )
        st.success(f"**I ≈ {I:.6f}**")

        # ---- Paso a paso ----
        if mostrar_paso_a_paso:
            st.subheader("Paso a paso")

            st.markdown(
                f"- Ancho de cada subintervalo: "
                f"`h = (b - a) / n = ({b} - {a}) / {int(n)} = {h}`"
            )

            # Tabla con xi y f(xi)
            st.markdown("**Puntos de evaluación:**")
            datos = {
                "i": list(range(len(x))),
                "x_i": x,
                "f(x_i)": y
            }

            # Añadir coeficientes según el método
            coef = []
            if metodo == "Regla del trapecio":
                for i in range(len(x)):
                    if i == 0 or i == len(x) - 1:
                        coef.append(0.5)
                    else:
                        coef.append(1.0)
                st.latex(
                    r"\int_a^b f(x)\,dx \approx h\left[\frac{f(x_0)}{2} + f(x_1) + "
                    r"\dots + f(x_{n-1}) + \frac{f(x_n)}{2}\right]"
                )
            else:
                # Simpson 1/3
                for i in range(len(x)):
                    if i == 0 or i == len(x) - 1:
                        coef.append(1)
                    elif i % 2 == 1:
                        coef.append(4)
                    else:
                        coef.append(2)
                st.latex(
                    r"""\int_a^b f(x)\,dx \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2)
                    + \dots + 4f(x_{n-1}) + f(x_n)\right]"""
                )

            datos["Coeficiente"] = coef
            datos["Coef * f(x_i)"] = np.array(coef) * y

            st.dataframe(datos)

            if metodo == "Regla del trapecio":
                st.markdown(
                    f"""
                    **Cálculo final (Trapecio):**

                    \\[
                    I \\approx h \\left( \\tfrac{{1}}{{2}} f(x_0) +
                    f(x_1) + \\dots + f(x_{{n-1}}) + \\tfrac{{1}}{{2}} f(x_n) \\right)
                    \\]
                    """
                )
            else:
                st.markdown(
                    f"""
                    **Cálculo final (Simpson 1/3):**

                    \\[
                    I \\approx \\frac{{h}}{{3}} \\left( f(x_0) + 4f(x_1) + 2f(x_2)
                    + \\dots + 4f(x_{{n-1}}) + f(x_n) \\right)
                    \\]

                    \\[
                    I \\approx \\frac{{{h}}}{{3}} \\times \\left( {datos["Coef * f(x_i)"].sum()} \\right)
                    \\]
                    """
                )

        # ---- Gráfica ----
        if mostrar_grafica:
            st.subheader("Gráfica de la función y puntos de integración")

            fig, ax = plt.subplots()
            xs = np.linspace(a, b, 400)
            ys = f(xs)
            ax.plot(xs, ys, label="f(x)")
            ax.scatter(x, y, label="Puntos de integración")

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.set_title("Integración numérica")
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Hubo un error al evaluar la función o el método: {e}")
