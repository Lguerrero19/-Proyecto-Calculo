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
        # constantes útiles
        "pi": math.pi,
        "π": math.pi,
        "e": math.e,
    }

    # Funciones y constantes de math (asin, atan, etc.)
    allowed_names.update({
        k: getattr(math, k)
        for k in dir(math)
        if not k.startswith("_")
    })

    # Funciones de numpy: sin, cos, exp, log, etc. (versiones que aceptan arrays)
    allowed_names.update({
        k: getattr(np, k)
        for k in dir(np)
        if not k.startswith("_")
    })

    def f(x):
        # Evalúa usando solo los nombres permitidos + la variable x
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
# Parsear límites tipo "pi/2", "2*pi", etc.
# ------------------------------
def parse_limit(expr):
    """
    Convierte una cadena como 'pi', 'pi/2', '2*pi', '3.5', etc. a un número float.
    Solo usa nombres y funciones permitidas (pi, e, sin, cos, etc.).
    """
    import math
    import numpy as np

    allowed = {
        "pi": math.pi,
        "π": math.pi,
        "e": math.e,
    }

    # Por si quisieras usar algo como sin(pi/2) en un límite
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
    a_str = st.text_input("Límite inferior a:", value="0")
    b_str = st.text_input("Límite superior b:", value="pi")
    n = st.number_input("Número de subintervalos n:", min_value=1, value=4, step=1)

with col2:
    mostrar_paso_a_paso = st.checkbox("Mostrar paso a paso", value=True)
    mostrar_grafica = st.checkbox("Mostrar gráfica", value=True)
    st.info(
        "Puedes usar funciones como `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, etc.\n"
        "Ejemplos: `x**2`, `sin(x)`, `exp(-x**2)`, `sin(x*pi)`"
    )

# ------------------------------
# Botón para calcular
# ------------------------------
if st.button("Calcular integral aproximada"):
    try:
        # Convertir los límites de texto a número (aceptando pi, pi/2, etc.)
        a = parse_limit(a_str)
        b = parse_limit(b_str)

        f = build_function(func_str)
        _ = f(np.array([a, b]))  # prueba rápida

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
            f"[{a_str}, {b_str}] con n = {int(n)} subintervalos:"
        )
        st.success(f"**I ≈ {I:.6f}**")

        # ---- Paso a paso ----
        if mostrar_paso_a_paso:
            st.subheader("Paso a paso")

            st.markdown(
                f"1. **Cálculo del ancho de subintervalo**  \n"
                f"h = (b - a) / n = ({b} - {a}) / {int(n)} = **{h}**"
            )

            # Tabla con xi y f(xi)
            st.markdown("2. **Puntos de evaluación y valores de la función**")
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

                st.markdown("3. **Fórmula de la regla del trapecio**")
                st.latex(
                    r"I \approx h\left[\frac{f(x_0)}{2} + f(x_1) + \dots + f(x_{n-1}) + \frac{f(x_n)}{2}\right]"
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

                st.markdown("3. **Fórmula de Simpson 1/3**")
                st.latex(
                    r"I \approx \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2)"
                    r" + \dots + 4f(x_{n-1}) + f(x_n)\right]"
                )

            datos["Coeficiente"] = coef
            datos["Coef * f(x_i)"] = np.array(coef) * y

            st.dataframe(datos)

            # Paso final: suma y resultado
            S = datos["Coef * f(x_i)"].sum()

            st.markdown("4. **Cálculo numérico de la suma ponderada**")
            st.latex(r"S = \sum_{i=0}^{n} c_i\,f(x_i)")
            st.latex(f"S \\approx {S}")

            if metodo == "Regla del trapecio":
                st.markdown("5. **Resultado final (Trapecio)**")
                st.latex(
                    f"I \\approx h \\cdot S = {h} \\cdot {S} \\approx {I}"
                )
            else:
                st.markdown("5. **Resultado final (Simpson 1/3)**")
                st.latex(
                    f"I \\approx \\frac{{h}}{{3}} \\cdot S"
                    f" = \\frac{{{h}}}{{3}} \\cdot {S} \\approx {I}"
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
