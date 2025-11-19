import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from math import *


# ------------------------------
# Función para evaluar f(x)
# ------------------------------
def build_function(func_str):
    """
    Construye una función f(x) a partir de un string.
    Permite usar funciones de numpy (para trabajar con arreglos)
    y de math (para constantes como pi, e, etc.).
    """
    import numpy as np
    import math

    allowed_names = {
        "pi": math.pi,
        "π": math.pi,
        "e": math.e,
    }

    # Funciones de math
    allowed_names.update({
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    })

    # Funciones de numpy (sobrescriben math.sin, math.cos, etc.)
    allowed_names.update({
        k: getattr(np, k)
        for k in dir(np)
        if not k.startswith("_")
    })

    def f(x):
        return eval(func_str, {"__builtins__": {}}, {**allowed_names, "x": x})

    return f


# ------------------------------
# Métodos numéricos
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
# Parseador de límites tipo pi, pi/3, 2*pi, etc.
# ------------------------------
def parse_limit(expr):
    import math
    import numpy as np

    allowed = {
        "pi": math.pi,
        "π": math.pi,
        "e": math.e,
    }

    allowed.update({
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    })
    allowed.update({
        k: getattr(np, k)
        for k in dir(np)
        if not k.startswith("_")
    })

    return float(eval(expr, {"__builtins__": {}}, allowed))


# ------------------------------
# Interfaz Streamlit
# ------------------------------
st.set_page_config(page_title="Integración numérica", layout="wide")

st.title("Aplicativo Web: Regla del Trapecio y Simpson 1/3")
st.write(
    """
    Este aplicativo permite aproximar una integral definida usando:
    - **Regla del Trapecio**
    - **Regla de Simpson 1/3**

    Puedes usar funciones como `sin(x)`, `cos(x)`, `exp(x)`,  
    y límites con expresiones como `pi`, `2*pi`, `pi/3`, etc.
    """
)

# ---- Entradas ----
col1, col2 = st.columns(2)

with col1:
    metodo = st.radio(
        "Método de integración",
        ("Regla del trapecio", "Regla de Simpson 1/3")
    )
    func_str = st.text_input("Función f(x):", value="sin(x)")
    a_str = st.text_input("Límite inferior a:", value="0")
    b_str = st.text_input("Límite superior b:", value="pi")
    n = st.number_input("Número de subintervalos n (Simpson requiere par):",
                        min_value=1, value=4, step=1)

with col2:
    mostrar_paso_a_paso = st.checkbox("Mostrar paso a paso", value=True)
    mostrar_grafica = st.checkbox("Mostrar gráfica", value=True)
    st.info(
        "Ejemplos válidos:\n"
        "`sin(x)` | `cos(x)` | `exp(-x**2)` | `x**3 + sin(x)`\n"
        "Límites: `0`, `pi`, `pi/2`, `2*pi`, etc."
    )

# ------------------------------
# Botón principal
# ------------------------------
if st.button("Calcular integral aproximada"):
    try:
        # Convertir límites escritos como texto
        a = parse_limit(a_str)
        b = parse_limit(b_str)

        # Crear función evaluable
        f = build_function(func_str)
        _ = f(np.array([a, b]))  # prueba

        # Calcular integral
        if metodo == "Regla del trapecio":
            I, x, y, h = trapecio(f, a, b, int(n))
        else:
            if int(n) % 2 != 0:
                st.error("Simpson 1/3 requiere que n sea PAR.")
                st.stop()
            I, x, y, h = simpson_tercio(f, a, b, int(n))

        # Resultado
        st.subheader("Resultado")
        st.success(f"I ≈ {I:.6f}")

        # ------------------------------
        # Paso a paso
        # ------------------------------
        if mostrar_paso_a_paso:
            st.subheader("Paso a paso")

            st.markdown(
                f"1. **Cálculo del ancho:**  \n"
                f"h = (b - a) / n = ({b} - {a}) / {int(n)} = **{h}**"
            )

            datos = {
                "i": list(range(len(x))),
                "x_i": x,
                "f(x_i)": y
            }

            coef = []

            if metodo == "Regla del trapecio":
                st.markdown("2. **Coeficientes (Trapecio)**")
                for i in range(len(x)):
                    coef.append(0.5 if i == 0 or i == len(x)-1 else 1)

                st.latex(r"I \approx h\left[\frac{f(x_0)}{2} + f(x_1) + \dots + f(x_{n-1}) + \frac{f(x_n)}{2}\right]")

            else:
                st.markdown("2. **Coeficientes (Simpson 1/3)**")
                for i in range(len(x)):
                    if i == 0 or i == len(x)-1:
                        coef.append(1)
                    elif i % 2 == 1:
                        coef.append(4)
                    else:
                        coef.append(2)

                st.latex(r"I \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2)+ \dots + 4f(x_{n-1}) + f(x_n)\right]")

            datos["Coeficiente"] = coef
            datos["Coef * f(x_i)"] = np.array(coef) * y
            st.dataframe(datos)

            S = datos["Coef * f(x_i)"].sum()

            st.markdown("3. **Suma ponderada:**")
            st.latex(f"S = {S}")

            if metodo == "Regla del trapecio":
                st.markdown("4. **Resultado final:**")
                st.latex(f"I = h · S = {h}·{S} = {I}")
            else:
                st.markdown("4. **Resultado final:**")
                st.latex(f"I = (h/3) · S = {h}/3 · {S} = {I}")

        # ------------------------------
        # Gráfica mejorada
        # ------------------------------
        if mostrar_grafica:
            st.subheader("Gráfica de la función y la aproximación")

            fig, ax = plt.subplots()

            # Curva real
            xs = np.linspace(a, b, 400)
            ys = f(xs)
            ax.plot(xs, ys, label="f(x) (función real)", linewidth=2)

            # Aproximación por tramos
            ax.plot(x, y, "o--", label="Aproximación por tramos", color="orange")

            # Líneas verticales
            for xi in x:
                ax.vlines(xi, 0, f(xi), linestyle="dashed", linewidth=0.7)

            # Sombreado del área
            for i in range(len(x) - 1):
                xx = [x[i], x[i+1]]
                yy = [y[i], y[i+1]]
                ax.fill_between(xx, yy, [0, 0], alpha=0.2, color="orange")

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True)
            ax.legend()

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Ocurrió un error: {e}")
