# ============================================================
# App: Pruebas de Hipótesis No Paramétricas
# Autores: Martin Bastidas y Andres Erazo
# ============================================================
# TABLA DE CONTENIDOS:
#   1. Imports y configuración de página
#   2. Funciones utilitarias compartidas
#   3. Funciones específicas de Kruskal-Wallis
#   4. Render: Kruskal-Wallis
#   5. Render: Wilcoxon una muestra
#   6. Render: Wilcoxon pareadas
#   7. Render: Mann-Whitney U
#   8. Render: Chi-cuadrado Independencia
#   9. Orquestador principal con tabs
# ============================================================

# ============================================================
# 1. Imports y configuración de página
# ============================================================
import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pruebas No Paramétricas",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# 2. Funciones utilitarias compartidas
# ============================================================
def parsear_valores(texto: str):
    """Convierte texto separado por comas o espacios a lista de floats.
    Lanza ValueError si hay no-numéricos."""
    if not texto or not texto.strip():
        return []
    partes = texto.replace(",", " ").split()
    return [float(x) for x in partes]


def graficar_chi2(estadistico, chi_critico, gl, alpha):
    """Figura de región de rechazo para χ² (misma apariencia que KW)."""
    x_max = max(chi_critico * 1.5, stats.chi2.ppf(0.999, df=gl))
    est_plot = min(estadistico, x_max)
    x = np.linspace(0, x_max, 500)
    y = stats.chi2.pdf(x, df=gl)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label=f"χ² con gl = {gl}", color="steelblue")
    ax.fill_between(x, y, where=(x >= chi_critico),
                    color="salmon", alpha=0.5, label="Región de rechazo")
    etiqueta = f"Estadístico = {estadistico:.3f}"
    if estadistico > x_max:
        etiqueta += " (fuera de rango)"
    ax.axvline(est_plot, color="green", linestyle="--", label=etiqueta)
    ax.axvline(chi_critico, color="red", linestyle=":",
               label=f"χ² crítico = {chi_critico:.3f}")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.legend()
    return fig


def graficar_normal(estadistico, alpha, tipo):
    """Figura de región de rechazo para la distribución normal estándar.
    tipo ∈ {'bilateral', 'derecha', 'izquierda'}."""
    x_min = min(-4.0, estadistico - 0.5)
    x_max = max(4.0, estadistico + 0.5)
    x = np.linspace(x_min, x_max, 500)
    y = stats.norm.pdf(x)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label="N(0,1)", color="steelblue")

    if tipo == "bilateral":
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ax.fill_between(x, y, where=(x >= z_crit),
                        color="salmon", alpha=0.5, label="Región de rechazo")
        ax.fill_between(x, y, where=(x <= -z_crit),
                        color="salmon", alpha=0.5)
        ax.axvline(z_crit, color="red", linestyle=":",
                   label=f"z crítico = ±{z_crit:.3f}")
        ax.axvline(-z_crit, color="red", linestyle=":")
    elif tipo == "derecha":
        z_crit = stats.norm.ppf(1 - alpha)
        ax.fill_between(x, y, where=(x >= z_crit),
                        color="salmon", alpha=0.5, label="Región de rechazo")
        ax.axvline(z_crit, color="red", linestyle=":",
                   label=f"z crítico = {z_crit:.3f}")
    else:  # izquierda
        z_crit = stats.norm.ppf(alpha)
        ax.fill_between(x, y, where=(x <= z_crit),
                        color="salmon", alpha=0.5, label="Región de rechazo")
        ax.axvline(z_crit, color="red", linestyle=":",
                   label=f"z crítico = {z_crit:.3f}")

    ax.axvline(estadistico, color="green", linestyle="--",
               label=f"Z = {estadistico:.3f}")
    ax.set_xlabel("Z")
    ax.set_ylabel("Densidad")
    ax.legend()
    return fig


def bloque_decision(rechaza_H0, p_valor, alpha, estadistico, critico, nombre_est,
                    direccion="bilateral"):
    """Renderiza el bloque unificado de decisión y conclusión.
    direccion ∈ {'bilateral', 'derecha', 'izquierda'} define cómo se compara
    el estadístico con el valor crítico."""
    coinciden = None
    if critico is not None:
        if direccion == "bilateral":
            rechaza_por_critico = abs(estadistico) > abs(critico)
        elif direccion == "derecha":
            rechaza_por_critico = estadistico > critico
        else:  # izquierda
            rechaza_por_critico = estadistico < critico
        coinciden = rechaza_por_critico == (p_valor < alpha)
        if not coinciden:
            st.warning(
                "⚠️ El criterio del valor crítico y el p-valor no coinciden; "
                "se usa el **p-valor** como criterio final."
            )

    detalle_crit = ""
    if critico is not None:
        detalle_crit = f"; {nombre_est} = {estadistico:.4f} vs crítico = {critico:.4f}"

    if rechaza_H0:
        st.success(
            f"**Se RECHAZA H₀** (p-valor = {p_valor:.4f} < α = {alpha}{detalle_crit})."
        )
        st.markdown(
            f"**Conclusión:** Con un nivel de significancia del {int(alpha*100)}%, "
            "existe evidencia estadística suficiente para rechazar la hipótesis nula."
        )
    else:
        st.info(
            f"**NO se rechaza H₀** (p-valor = {p_valor:.4f} ≥ α = {alpha}{detalle_crit})."
        )
        st.markdown(
            f"**Conclusión:** Con un nivel de significancia del {int(alpha*100)}%, "
            "**no** existe evidencia estadística suficiente para rechazar la hipótesis nula."
        )


def _tipo_a_alternative(tipo_label: str) -> str:
    """Mapea la etiqueta de hipótesis al alternative de scipy."""
    if tipo_label.startswith("Bilateral"):
        return "two-sided"
    if "derecha" in tipo_label:
        return "greater"
    return "less"


def _tipo_a_normal(tipo_label: str) -> str:
    """Mapea la etiqueta a la clave de graficar_normal."""
    if tipo_label.startswith("Bilateral"):
        return "bilateral"
    if "derecha" in tipo_label:
        return "derecha"
    return "izquierda"


# ============================================================
# 3. Funciones específicas de Kruskal-Wallis
# ============================================================
def calcular_kw(datos_grupos):
    """H sin corrección, factor C, H corregido, p-valor, rangos por grupo."""
    N = sum(len(g) for g in datos_grupos)
    todos = np.concatenate(datos_grupos)
    rangos = stats.rankdata(todos)

    suma_rangos = []
    prom_rangos = []
    idx = 0
    for g in datos_grupos:
        r = rangos[idx: idx + len(g)]
        suma_rangos.append(r.sum())
        prom_rangos.append(r.mean())
        idx += len(g)

    H_sin = (12.0 / (N * (N + 1))) * sum(
        (Ri ** 2) / len(g) for Ri, g in zip(suma_rangos, datos_grupos)
    ) - 3 * (N + 1)

    _, counts = np.unique(todos, return_counts=True)
    tie_term = sum((t ** 3 - t) for t in counts if t > 1)
    C = 1 - tie_term / (N ** 3 - N) if (N ** 3 - N) > 0 else 1.0
    H_corr = H_sin / C if C > 0 else H_sin

    H_scipy, p_valor = stats.kruskal(*datos_grupos)

    return {
        "N": N,
        "suma_rangos": suma_rangos,
        "prom_rangos": prom_rangos,
        "H_sin": H_sin,
        "C": C,
        "H_corr": H_corr,
        "H_scipy": H_scipy,
        "p_valor": p_valor,
        "tie_term": tie_term,
    }


def chequear_forma_similar(datos_grupos):
    """Avisa si la razón IQR_max / IQR_min sugiere formas muy distintas."""
    iqrs = [float(stats.iqr(g)) for g in datos_grupos]
    positivos = [x for x in iqrs if x > 0]
    if not positivos:
        return iqrs, np.inf
    min_iqr = min(positivos)
    max_iqr = max(iqrs)
    ratio = max_iqr / min_iqr if min_iqr > 0 else np.inf
    return iqrs, ratio


# ============================================================
# 4. Render: Kruskal-Wallis
# ============================================================
def render_kw():
    st.header("Prueba de Kruskal-Wallis")
    st.markdown(
        "Prueba **no paramétrica** para comparar si **tres o más grupos independientes** "
        "provienen de la misma distribución."
    )
    st.divider()

    # 1. Configuración
    st.subheader("1️⃣ Configuración de la prueba")
    col1, col2 = st.columns(2)
    with col1:
        k = st.number_input(
            "Número de grupos (k)",
            min_value=3, max_value=10, value=3, step=1,
            key="kw_k"
        )
    with col2:
        alpha = st.selectbox(
            "Nivel de significancia (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="kw_alpha"
        )

    # Limpia claves obsoletas si se reduce k
    prev_k = st.session_state.get("kw_prev_k", k)
    if prev_k > k:
        for i in range(k + 1, prev_k + 1):
            st.session_state.pop(f"kw_grupo_{i}", None)
    st.session_state["kw_prev_k"] = k

    st.divider()

    # 2. Ingreso de datos
    st.subheader("2️⃣ Ingreso de datos")
    st.caption("Ingresa los valores de cada grupo separados por **comas** o **espacios**.")

    grupos = {}
    nombres_grupos = []
    datos_grupos = []
    errores = []

    for i in range(1, k + 1):
        texto = st.text_input(
            f"Grupo {i}",
            placeholder="Ej: 12, 15, 18, 20, 22",
            key=f"kw_grupo_{i}"
        )
        nombre = f"Grupo {i}"
        if not texto.strip():
            errores.append(f"El {nombre} está vacío.")
            continue
        try:
            valores = [float(x) for x in texto.replace(",", " ").split()]
        except ValueError:
            st.error(f"❌ El {nombre} contiene valores no numéricos.")
            errores.append(f"{nombre} no numérico.")
            continue
        if len(valores) < 2:
            st.warning(f"⚠️ El {nombre} debe tener al menos 2 observaciones.")
            errores.append(f"{nombre} con <2 observaciones.")
            continue
        grupos[nombre] = valores
        nombres_grupos.append(nombre)
        datos_grupos.append(np.asarray(valores, dtype=float))

    entradas_validas = (len(errores) == 0) and (len(grupos) == k)

    st.divider()

    # 3. Ejecución
    ejecutar = st.button(
        "🔬 Ejecutar prueba de Kruskal-Wallis",
        type="primary",
        key="kw_ejecutar"
    )

    if ejecutar:
        if not entradas_validas:
            st.error("Debes completar correctamente los datos de **todos** los grupos.")
            st.session_state.pop("resultado_kw", None)
        else:
            res = calcular_kw(datos_grupos)
            iqrs, ratio_iqr = chequear_forma_similar(datos_grupos)
            gl = k - 1
            chi_critico = stats.chi2.ppf(1 - alpha, df=gl)
            rechaza_H0 = res["p_valor"] < alpha

            st.session_state["resultado_kw"] = {
                "grupos": grupos,
                "nombres": nombres_grupos,
                "datos": [g.tolist() for g in datos_grupos],
                "k": k,
                "alpha": alpha,
                "gl": gl,
                "chi_critico": chi_critico,
                "rechaza_H0": rechaza_H0,
                "iqrs": iqrs,
                "ratio_iqr": ratio_iqr,
                **res,
            }

    # 4. Reporte
    if "resultado_kw" in st.session_state:
        R = st.session_state["resultado_kw"]
        nombres_grupos = R["nombres"]
        datos_grupos = [np.asarray(g) for g in R["datos"]]
        alpha_r = R["alpha"]
        gl = R["gl"]
        chi_critico = R["chi_critico"]
        N = R["N"]
        H_sin = R["H_sin"]
        C = R["C"]
        H_corr = R["H_corr"]
        p_valor = R["p_valor"]

        st.header("3️⃣ Reporte completo")

        st.subheader("📌 Planteamiento de hipótesis")
        st.markdown(
            "- **H₀:** Las distribuciones de los *k* grupos son iguales "
            "(bajo forma similar: medianas iguales).\n"
            "- **H₁:** Al menos una distribución difiere de las demás."
        )

        st.subheader("📋 Supuestos de la prueba y chequeos")
        st.markdown(
            "- Las muestras son **independientes** entre sí.\n"
            "- La variable es al menos **ordinal** (o continua).\n"
            "- Las distribuciones tienen **forma similar** (para interpretar en términos de medianas)."
        )
        tamaños = [len(g) for g in datos_grupos]
        if any(n < 5 for n in tamaños):
            st.warning(
                f"⚠️ Hay grupos con nᵢ < 5 (tamaños: {tamaños}). La aproximación χ² "
                "puede ser poco confiable; interpretar el p-valor con cautela."
            )
        if R["ratio_iqr"] == np.inf or R["ratio_iqr"] > 2:
            iqrs_fmt = [round(x, 3) for x in R["iqrs"]]
            st.warning(
                f"⚠️ Los rangos intercuartílicos difieren notablemente "
                f"(IQR = {iqrs_fmt}, razón max/min ≈ {R['ratio_iqr']:.2f}). "
                "El supuesto de **forma similar** puede no cumplirse: formular la "
                "conclusión sobre **distribuciones**, no sobre medianas."
            )

        st.subheader("📊 Resumen de los grupos")
        resumen = pd.DataFrame({
            "n": tamaños,
            "Mediana": [np.median(g) for g in datos_grupos],
            "IQR": R["iqrs"],
            "Suma de rangos (Rᵢ)": R["suma_rangos"],
            "Rango promedio": R["prom_rangos"],
        }, index=nombres_grupos)
        st.dataframe(resumen.style.format({
            "Mediana": "{:.3f}",
            "IQR": "{:.3f}",
            "Suma de rangos (Rᵢ)": "{:.2f}",
            "Rango promedio": "{:.3f}",
        }))

        todos_concat = np.concatenate(datos_grupos)
        rangos_concat = stats.rankdata(todos_concat)
        rangos_por_grupo = []
        _idx = 0
        for g in datos_grupos:
            rangos_por_grupo.append(rangos_concat[_idx: _idx + len(g)])
            _idx += len(g)

        with st.expander("🔽 Ver tabla de observaciones ordenadas con sus rangos"):
            etiquetas_grupo = []
            for nombre, g in zip(nombres_grupos, datos_grupos):
                etiquetas_grupo.extend([nombre] * len(g))
            df_orden = pd.DataFrame({
                "Valor": todos_concat,
                "Grupo": etiquetas_grupo,
                "Rango": rangos_concat,
            }).sort_values("Valor", kind="mergesort").reset_index(drop=True)
            df_orden.index = df_orden.index + 1
            df_orden.index.name = "Orden"
            st.dataframe(df_orden.style.format({
                "Valor": "{:.3f}",
                "Rango": "{:.2f}",
            }))

        st.subheader("📑 Datos y rangos por grupo")
        max_n = max(len(g) for g in datos_grupos)
        columnas = {}
        for nombre, g, r in zip(nombres_grupos, datos_grupos, rangos_por_grupo):
            pad = max_n - len(g)
            columnas[f"{nombre} — Valor"] = list(g) + [np.nan] * pad
            columnas[f"{nombre} — Rango"] = list(r) + [np.nan] * pad
        df_datos_rangos = pd.DataFrame(columnas)
        df_datos_rangos.index = [str(i) for i in range(1, max_n + 1)]

        fila_suma = {}
        for nombre, r in zip(nombres_grupos, rangos_por_grupo):
            fila_suma[f"{nombre} — Valor"] = np.nan
            fila_suma[f"{nombre} — Rango"] = float(r.sum())
        df_suma = pd.DataFrame(fila_suma, index=["Σ Rᵢ"])

        df_datos_full = pd.concat([df_datos_rangos, df_suma])
        df_datos_full.index.name = "i"
        st.dataframe(df_datos_full.style.format(precision=3, na_rep=""))

        st.subheader("🧮 Cálculo del estadístico")
        st.markdown("**Fórmula sin corrección por empates:**")
        st.latex(r"H_{sin} = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)")
        st.markdown("**Factor de corrección por empates:**")
        st.latex(r"C = 1 - \frac{\sum_j (t_j^3 - t_j)}{N^3 - N} \qquad\Rightarrow\qquad H = \frac{H_{sin}}{C}")

        st.markdown(f"- **N total** = {N}")
        st.markdown(f"- **Grados de libertad (gl)** = k − 1 = {gl}")
        st.markdown(f"- **H sin corrección** = `{H_sin:.4f}`")
        st.markdown(
            f"- **Factor C** = `{C:.4f}` "
            + ("(hay empates → corrección aplicada)" if C < 1 else "(sin empates)")
        )
        st.markdown(f"- **H corregido** = `{H_corr:.4f}`")
        st.markdown(f"- **Valor crítico χ²** (α = {alpha_r}, gl = {gl}) = `{chi_critico:.4f}`")
        st.markdown(f"- **p-valor (scipy)** = `{p_valor:.4f}`")

        st.subheader("📈 Región de rechazo")
        x_max = max(chi_critico * 1.5, stats.chi2.ppf(0.999, df=gl))
        H_plot = min(H_corr, x_max)
        x = np.linspace(0, x_max, 500)
        y = stats.chi2.pdf(x, df=gl)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, label=f"χ² con gl = {gl}", color="steelblue")
        ax.fill_between(x, y, where=(x >= chi_critico),
                        color="salmon", alpha=0.5, label="Región de rechazo")
        etiqueta_H = f"H = {H_corr:.3f}"
        if H_corr > x_max:
            etiqueta_H += " (fuera de rango)"
        ax.axvline(H_plot, color="green", linestyle="--", label=etiqueta_H)
        ax.axvline(chi_critico, color="red", linestyle=":",
                   label=f"χ² crítico = {chi_critico:.3f}")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Densidad")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("✅ Decisión y conclusión")
        coinciden = (H_corr > chi_critico) == (p_valor < alpha_r)
        if not coinciden:
            st.warning(
                "⚠️ La aproximación χ² y el p-valor exacto no coinciden; "
                "se usa el **p-valor** como criterio final."
            )

        if R["rechaza_H0"]:
            st.success(
                f"**Se RECHAZA H₀** (p-valor = {p_valor:.4f} < α = {alpha_r}; "
                f"H = {H_corr:.4f} vs χ² crítico = {chi_critico:.4f})."
            )
            st.markdown(
                f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
                "existe evidencia estadística suficiente para afirmar que **al menos una** "
                "de las distribuciones difiere de las demás."
            )
        else:
            st.info(
                f"**NO se rechaza H₀** (p-valor = {p_valor:.4f} ≥ α = {alpha_r}; "
                f"H = {H_corr:.4f} vs χ² crítico = {chi_critico:.4f})."
            )
            st.markdown(
                f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
                "**no** existe evidencia estadística suficiente para afirmar que las "
                "distribuciones de los grupos sean diferentes."
            )


# ============================================================
# 5. Render: Wilcoxon una muestra
# ============================================================
def render_wilcoxon_1():
    st.header("Prueba de Wilcoxon — una muestra")
    st.markdown(
        "Prueba **no paramétrica** para contrastar si la **mediana** de una muestra "
        "difiere de un valor hipotético μ₀."
    )
    st.divider()

    # 1. Configuración
    st.subheader("1️⃣ Configuración de la prueba")
    col1, col2, col3 = st.columns(3)
    with col1:
        mu0 = st.number_input(
            "Mediana hipotética μ₀",
            value=0.0, step=1.0, format="%.4f",
            key="w1_mu0"
        )
    with col2:
        tipo = st.selectbox(
            "Tipo de hipótesis",
            options=["Bilateral (≠)", "Unilateral derecha (>)", "Unilateral izquierda (<)"],
            index=0,
            key="w1_tipo"
        )
    with col3:
        alpha = st.selectbox(
            "Nivel de significancia (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="w1_alpha"
        )

    st.divider()

    # 2. Ingreso de datos
    st.subheader("2️⃣ Ingreso de datos")
    st.caption("Valores separados por **comas** o **espacios**.")
    texto = st.text_input(
        "Muestra",
        placeholder="Ej: 12.3, 15.1, 17.8, 14.2, 16.9, 13.5",
        key="w1_muestra"
    )

    datos_validos = False
    valores = []
    try:
        valores = parsear_valores(texto)
        if valores and len(valores) < 6:
            st.warning(f"⚠️ Se recomienda n ≥ 6 (actual n = {len(valores)}).")
        if len(valores) >= 1:
            datos_validos = True
    except ValueError:
        st.error("❌ La muestra contiene valores no numéricos.")

    st.divider()

    # 3. Ejecución
    ejecutar = st.button(
        "🔬 Ejecutar Wilcoxon (una muestra)",
        type="primary",
        key="w1_ejecutar"
    )

    if ejecutar:
        if not datos_validos or len(valores) < 1:
            st.error("Debes ingresar la muestra con valores numéricos.")
            st.session_state.pop("resultado_w1", None)
        else:
            x = np.asarray(valores, dtype=float)
            diferencias = x - mu0
            n_ceros = int(np.sum(diferencias == 0))
            dif_nz = diferencias[diferencias != 0]
            n_ef = len(dif_nz)

            if n_ef < 1:
                st.error("Todas las diferencias son cero: la prueba no puede ejecutarse.")
                st.session_state.pop("resultado_w1", None)
            else:
                alternative = _tipo_a_alternative(tipo)
                W_scipy, p_valor = stats.wilcoxon(diferencias, alternative=alternative,
                                                  zero_method="wilcox", correction=False)

                rangos_abs = stats.rankdata(np.abs(dif_nz))
                signos = np.sign(dif_nz)
                W_pos = float(np.sum(rangos_abs[signos > 0]))
                W_neg = float(np.sum(rangos_abs[signos < 0]))

                # Aproximación normal
                mu_w = n_ef * (n_ef + 1) / 4.0
                _, counts_abs = np.unique(np.abs(dif_nz), return_counts=True)
                tie_adj = sum((t ** 3 - t) for t in counts_abs if t > 1)
                var_w = n_ef * (n_ef + 1) * (2 * n_ef + 1) / 24.0 - tie_adj / 48.0
                sigma_w = np.sqrt(var_w) if var_w > 0 else np.nan
                W_est = min(W_pos, W_neg)
                z_stat = (W_pos - mu_w) / sigma_w if sigma_w and sigma_w > 0 else np.nan

                rechaza_H0 = p_valor < alpha

                st.session_state["resultado_w1"] = {
                    "x": x.tolist(),
                    "mu0": mu0,
                    "tipo": tipo,
                    "alpha": alpha,
                    "diferencias": diferencias.tolist(),
                    "dif_nz": dif_nz.tolist(),
                    "rangos_abs": rangos_abs.tolist(),
                    "signos": signos.tolist(),
                    "W_pos": W_pos,
                    "W_neg": W_neg,
                    "W_est": W_est,
                    "W_scipy": float(W_scipy),
                    "p_valor": float(p_valor),
                    "n_ceros": n_ceros,
                    "n_ef": n_ef,
                    "z_stat": float(z_stat) if not np.isnan(z_stat) else None,
                    "mu_w": mu_w,
                    "sigma_w": float(sigma_w) if sigma_w and not np.isnan(sigma_w) else None,
                    "rechaza_H0": rechaza_H0,
                }

    # 4. Reporte
    if "resultado_w1" in st.session_state:
        R = st.session_state["resultado_w1"]
        x = np.asarray(R["x"])
        mu0 = R["mu0"]
        tipo = R["tipo"]
        alpha_r = R["alpha"]
        diferencias = np.asarray(R["diferencias"])
        dif_nz = np.asarray(R["dif_nz"])
        rangos_abs = np.asarray(R["rangos_abs"])
        signos = np.asarray(R["signos"])
        n_ef = R["n_ef"]
        n_ceros = R["n_ceros"]
        W_pos = R["W_pos"]
        W_neg = R["W_neg"]
        p_valor = R["p_valor"]
        z_stat = R["z_stat"]
        sigma_w = R["sigma_w"]
        mu_w = R["mu_w"]

        st.header("3️⃣ Reporte completo")

        # Hipótesis
        st.subheader("📌 Planteamiento de hipótesis")
        if tipo.startswith("Bilateral"):
            st.markdown(f"- **H₀:** mediana = {mu0}\n- **H₁:** mediana ≠ {mu0}")
        elif "derecha" in tipo:
            st.markdown(f"- **H₀:** mediana ≤ {mu0}\n- **H₁:** mediana > {mu0}")
        else:
            st.markdown(f"- **H₀:** mediana ≥ {mu0}\n- **H₁:** mediana < {mu0}")

        # Supuestos
        st.subheader("📋 Supuestos de la prueba y chequeos")
        st.markdown(
            "- Observaciones **independientes**.\n"
            "- Las diferencias dᵢ = xᵢ − μ₀ son **simétricas** respecto a la mediana.\n"
            "- Escala al menos ordinal/intervalar."
        )
        if n_ceros > 0:
            st.info(f"ℹ️ Se eliminaron **{n_ceros}** diferencias iguales a cero (n efectivo = {n_ef}).")
        if n_ef < 6:
            st.warning(f"⚠️ n efectivo = {n_ef} (< 6); interpretar con cautela.")

        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        ax_h.hist(dif_nz, bins=min(10, max(3, n_ef // 2 + 1)),
                  color="steelblue", edgecolor="black", alpha=0.7)
        ax_h.axvline(0, color="red", linestyle="--", label="0")
        ax_h.set_xlabel("dᵢ = xᵢ − μ₀")
        ax_h.set_ylabel("Frecuencia")
        ax_h.set_title("Chequeo visual de simetría de las diferencias")
        ax_h.legend()
        st.pyplot(fig_h)
        plt.close(fig_h)

        # Resumen
        st.subheader("📊 Resumen descriptivo")
        resumen = pd.DataFrame({
            "Valor": [len(x), float(np.median(x)), float(np.mean(x)),
                      float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                      n_ef, n_ceros, W_pos, W_neg]
        }, index=["n", "Mediana muestral", "Media", "Desv. estándar",
                  "n efectivo (sin ceros)", "Ceros eliminados", "W⁺", "W⁻"])
        st.dataframe(resumen.style.format({"Valor": "{:.4f}"}))

        # Tabla de detalle
        st.subheader("📑 Tabla de detalle")
        df_det = pd.DataFrame({
            "xᵢ": x,
            "dᵢ = xᵢ − μ₀": diferencias,
        })
        # Rellenamos rango/signo solo para no-ceros
        rangos_full = np.full(len(x), np.nan)
        signos_full = [""] * len(x)
        idx_nz = np.where(diferencias != 0)[0]
        for k_i, pos in enumerate(idx_nz):
            rangos_full[pos] = rangos_abs[k_i]
            signos_full[pos] = "+" if signos[k_i] > 0 else "−"
        df_det["|dᵢ|"] = np.abs(diferencias)
        df_det["Rango"] = rangos_full
        df_det["Signo"] = signos_full
        df_det.index = np.arange(1, len(x) + 1)
        df_det.index.name = "i"

        fila_suma = pd.DataFrame({
            "xᵢ": [np.nan], "dᵢ = xᵢ − μ₀": [np.nan],
            "|dᵢ|": [np.nan], "Rango": [f"W⁺={W_pos:.2f} / W⁻={W_neg:.2f}"],
            "Signo": [""]
        }, index=["Σ"])
        st.dataframe(pd.concat([df_det, fila_suma]).style.format(
            precision=4, na_rep=""
        ))

        # Cálculo
        st.subheader("🧮 Cálculo del estadístico")
        st.latex(r"d_i = x_i - \mu_0")
        st.latex(r"W^{+} = \sum_{d_i>0} R(|d_i|) \qquad W^{-} = \sum_{d_i<0} R(|d_i|)")
        st.latex(r"W = \min(W^{+},\ W^{-})")
        st.markdown(f"- **n efectivo** = {n_ef}")
        st.markdown(f"- **W⁺** = `{W_pos:.4f}`")
        st.markdown(f"- **W⁻** = `{W_neg:.4f}`")
        st.markdown(f"- **W = min(W⁺, W⁻)** = `{min(W_pos, W_neg):.4f}`")
        st.markdown(f"- **p-valor (scipy)** = `{p_valor:.4f}`")

        usa_normal = n_ef >= 20
        z_critico = None
        if usa_normal and z_stat is not None:
            st.latex(r"Z = \frac{W^{+} - \mu_W}{\sigma_W},\quad \mu_W=\frac{n(n+1)}{4},\quad "
                     r"\sigma_W=\sqrt{\frac{n(n+1)(2n+1)}{24}}")
            st.markdown(f"- **μ_W** = `{mu_w:.4f}`")
            st.markdown(f"- **σ_W** = `{sigma_w:.4f}`")
            st.markdown(f"- **Z** = `{z_stat:.4f}`")
            if tipo.startswith("Bilateral"):
                z_critico = stats.norm.ppf(1 - alpha_r / 2)
            elif "derecha" in tipo:
                z_critico = stats.norm.ppf(1 - alpha_r)
            else:
                z_critico = stats.norm.ppf(alpha_r)
            st.markdown(f"- **Z crítico** (α = {alpha_r}) = `{z_critico:.4f}`")
        else:
            st.info("ℹ️ n efectivo < 20: se usa la **distribución exacta de Wilcoxon** "
                    "(p-valor reportado por scipy).")

        # Región de rechazo
        st.subheader("📈 Región de rechazo")
        if usa_normal and z_stat is not None:
            fig = graficar_normal(z_stat, alpha_r, _tipo_a_normal(tipo))
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig_b, ax_b = plt.subplots(figsize=(6, 3))
            ax_b.bar(["W⁺", "W⁻"], [W_pos, W_neg],
                     color=["steelblue", "salmon"], edgecolor="black")
            ax_b.set_ylabel("Suma de rangos")
            ax_b.set_title("W⁺ vs W⁻ (distribución exacta)")
            st.pyplot(fig_b)
            plt.close(fig_b)

        # Decisión
        st.subheader("✅ Decisión y conclusión")
        if usa_normal and z_stat is not None and z_critico is not None:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            z_stat, z_critico, "Z",
                            direccion=_tipo_a_normal(tipo))
        else:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            min(W_pos, W_neg), None, "W")


# ============================================================
# 6. Render: Wilcoxon muestras pareadas
# ============================================================
def render_wilcoxon_pareadas():
    st.header("Prueba de Wilcoxon — muestras pareadas")
    st.markdown(
        "Prueba **no paramétrica** para pares dependientes (ej. antes/después). "
        "Evalúa si la mediana de las diferencias dᵢ = xᵢ − yᵢ es cero."
    )
    st.divider()

    # 1. Configuración
    st.subheader("1️⃣ Configuración de la prueba")
    col1, col2 = st.columns(2)
    with col1:
        tipo = st.selectbox(
            "Tipo de hipótesis",
            options=["Bilateral (≠)", "Unilateral derecha (>)", "Unilateral izquierda (<)"],
            index=0,
            key="wp_tipo"
        )
    with col2:
        alpha = st.selectbox(
            "Nivel de significancia (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="wp_alpha"
        )

    st.divider()

    # 2. Datos
    st.subheader("2️⃣ Ingreso de datos")
    st.caption("Ambas muestras deben tener la **misma longitud**.")
    texto_x = st.text_input(
        "Muestra X (antes)",
        placeholder="Ej: 10, 12, 14, 11, 13, 15",
        key="wp_x"
    )
    texto_y = st.text_input(
        "Muestra Y (después)",
        placeholder="Ej: 12, 13, 13, 14, 15, 16",
        key="wp_y"
    )

    datos_ok = False
    x_vals, y_vals = [], []
    try:
        x_vals = parsear_valores(texto_x)
        y_vals = parsear_valores(texto_y)
        if x_vals and y_vals:
            if len(x_vals) != len(y_vals):
                st.error(f"❌ Longitudes distintas: n(X) = {len(x_vals)}, n(Y) = {len(y_vals)}.")
            else:
                datos_ok = True
    except ValueError:
        st.error("❌ Hay valores no numéricos en alguna de las muestras.")

    st.divider()

    # 3. Ejecución
    ejecutar = st.button(
        "🔬 Ejecutar Wilcoxon pareadas",
        type="primary",
        key="wp_ejecutar"
    )

    if ejecutar:
        if not datos_ok:
            st.error("Debes ingresar las dos muestras con la misma longitud.")
            st.session_state.pop("resultado_wp", None)
        else:
            x = np.asarray(x_vals, dtype=float)
            y = np.asarray(y_vals, dtype=float)
            diferencias = x - y
            n_ceros = int(np.sum(diferencias == 0))
            dif_nz = diferencias[diferencias != 0]
            n_ef = len(dif_nz)

            if n_ef < 6:
                st.warning(f"⚠️ n efectivo = {n_ef} (< 6); interpretar con cautela.")
            if n_ef < 1:
                st.error("Todas las diferencias son cero: la prueba no puede ejecutarse.")
                st.session_state.pop("resultado_wp", None)
            else:
                alternative = _tipo_a_alternative(tipo)
                T_scipy, p_valor = stats.wilcoxon(x, y, alternative=alternative,
                                                  zero_method="wilcox", correction=False)

                rangos_abs = stats.rankdata(np.abs(dif_nz))
                signos = np.sign(dif_nz)
                T_pos = float(np.sum(rangos_abs[signos > 0]))
                T_neg = float(np.sum(rangos_abs[signos < 0]))

                mu_t = n_ef * (n_ef + 1) / 4.0
                _, counts_abs = np.unique(np.abs(dif_nz), return_counts=True)
                tie_adj = sum((t ** 3 - t) for t in counts_abs if t > 1)
                var_t = n_ef * (n_ef + 1) * (2 * n_ef + 1) / 24.0 - tie_adj / 48.0
                sigma_t = np.sqrt(var_t) if var_t > 0 else np.nan
                z_stat = (T_pos - mu_t) / sigma_t if sigma_t and sigma_t > 0 else np.nan

                rechaza_H0 = p_valor < alpha

                st.session_state["resultado_wp"] = {
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "tipo": tipo,
                    "alpha": alpha,
                    "diferencias": diferencias.tolist(),
                    "dif_nz": dif_nz.tolist(),
                    "rangos_abs": rangos_abs.tolist(),
                    "signos": signos.tolist(),
                    "T_pos": T_pos,
                    "T_neg": T_neg,
                    "T_scipy": float(T_scipy),
                    "p_valor": float(p_valor),
                    "n_ceros": n_ceros,
                    "n_ef": n_ef,
                    "mu_t": mu_t,
                    "sigma_t": float(sigma_t) if sigma_t and not np.isnan(sigma_t) else None,
                    "z_stat": float(z_stat) if not np.isnan(z_stat) else None,
                    "rechaza_H0": rechaza_H0,
                }

    # 4. Reporte
    if "resultado_wp" in st.session_state:
        R = st.session_state["resultado_wp"]
        x = np.asarray(R["x"])
        y = np.asarray(R["y"])
        tipo = R["tipo"]
        alpha_r = R["alpha"]
        diferencias = np.asarray(R["diferencias"])
        dif_nz = np.asarray(R["dif_nz"])
        rangos_abs = np.asarray(R["rangos_abs"])
        signos = np.asarray(R["signos"])
        n_ef = R["n_ef"]
        n_ceros = R["n_ceros"]
        T_pos = R["T_pos"]
        T_neg = R["T_neg"]
        p_valor = R["p_valor"]
        z_stat = R["z_stat"]
        sigma_t = R["sigma_t"]
        mu_t = R["mu_t"]

        st.header("3️⃣ Reporte completo")

        st.subheader("📌 Planteamiento de hipótesis")
        if tipo.startswith("Bilateral"):
            st.markdown("- **H₀:** mediana(X − Y) = 0\n- **H₁:** mediana(X − Y) ≠ 0")
        elif "derecha" in tipo:
            st.markdown("- **H₀:** mediana(X − Y) ≤ 0\n- **H₁:** mediana(X − Y) > 0")
        else:
            st.markdown("- **H₀:** mediana(X − Y) ≥ 0\n- **H₁:** mediana(X − Y) < 0")

        st.subheader("📋 Supuestos de la prueba y chequeos")
        st.markdown(
            "- Los pares son **dependientes** (mediciones sobre la misma unidad).\n"
            "- Las diferencias son **simétricas** respecto a su mediana.\n"
            "- Escala al menos ordinal."
        )
        if n_ceros > 0:
            st.info(f"ℹ️ Se eliminaron **{n_ceros}** pares con diferencia cero (n efectivo = {n_ef}).")
        if n_ef < 6:
            st.warning(f"⚠️ n efectivo = {n_ef} (< 6); interpretar con cautela.")

        fig_h, ax_h = plt.subplots(figsize=(6, 3))
        ax_h.hist(dif_nz, bins=min(10, max(3, n_ef // 2 + 1)),
                  color="steelblue", edgecolor="black", alpha=0.7)
        ax_h.axvline(0, color="red", linestyle="--", label="0")
        ax_h.set_xlabel("dᵢ = xᵢ − yᵢ")
        ax_h.set_ylabel("Frecuencia")
        ax_h.set_title("Chequeo visual de simetría de las diferencias")
        ax_h.legend()
        st.pyplot(fig_h)
        plt.close(fig_h)

        st.subheader("📊 Resumen descriptivo")
        resumen = pd.DataFrame({
            "X": [len(x), float(np.median(x)), float(np.mean(x))],
            "Y": [len(y), float(np.median(y)), float(np.mean(y))],
            "d = X − Y": [len(diferencias), float(np.median(diferencias)),
                          float(np.mean(diferencias))],
        }, index=["n", "Mediana", "Media"])
        st.dataframe(resumen.style.format("{:.4f}"))

        st.subheader("📑 Tabla de detalle")
        rangos_full = np.full(len(x), np.nan)
        signos_full = [""] * len(x)
        idx_nz = np.where(diferencias != 0)[0]
        for k_i, pos in enumerate(idx_nz):
            rangos_full[pos] = rangos_abs[k_i]
            signos_full[pos] = "+" if signos[k_i] > 0 else "−"

        df_det = pd.DataFrame({
            "xᵢ": x,
            "yᵢ": y,
            "dᵢ": diferencias,
            "|dᵢ|": np.abs(diferencias),
            "Rango": rangos_full,
            "Signo": signos_full,
        })
        df_det.index = np.arange(1, len(x) + 1)
        df_det.index.name = "i"
        fila_suma = pd.DataFrame({
            "xᵢ": [np.nan], "yᵢ": [np.nan], "dᵢ": [np.nan], "|dᵢ|": [np.nan],
            "Rango": [f"T⁺={T_pos:.2f} / T⁻={T_neg:.2f}"], "Signo": [""]
        }, index=["Σ"])
        st.dataframe(pd.concat([df_det, fila_suma]).style.format(
            precision=4, na_rep=""
        ))

        st.subheader("🧮 Cálculo del estadístico")
        st.latex(r"d_i = x_i - y_i")
        st.latex(r"T^{+} = \sum_{d_i>0} R(|d_i|) \qquad T^{-} = \sum_{d_i<0} R(|d_i|)")
        st.latex(r"T = \min(T^{+},\ T^{-})")
        st.markdown(f"- **n efectivo** = {n_ef}")
        st.markdown(f"- **T⁺** = `{T_pos:.4f}`")
        st.markdown(f"- **T⁻** = `{T_neg:.4f}`")
        st.markdown(f"- **T = min(T⁺, T⁻)** = `{min(T_pos, T_neg):.4f}`")
        st.markdown(f"- **p-valor (scipy)** = `{p_valor:.4f}`")

        usa_normal = n_ef >= 20
        z_critico = None
        if usa_normal and z_stat is not None:
            st.latex(r"Z = \frac{T^{+} - \mu_T}{\sigma_T},\quad \mu_T=\frac{n(n+1)}{4},\quad "
                     r"\sigma_T=\sqrt{\frac{n(n+1)(2n+1)}{24}}")
            st.markdown(f"- **μ_T** = `{mu_t:.4f}`")
            st.markdown(f"- **σ_T** = `{sigma_t:.4f}`")
            st.markdown(f"- **Z** = `{z_stat:.4f}`")
            if tipo.startswith("Bilateral"):
                z_critico = stats.norm.ppf(1 - alpha_r / 2)
            elif "derecha" in tipo:
                z_critico = stats.norm.ppf(1 - alpha_r)
            else:
                z_critico = stats.norm.ppf(alpha_r)
            st.markdown(f"- **Z crítico** (α = {alpha_r}) = `{z_critico:.4f}`")
        else:
            st.info("ℹ️ n efectivo < 20: se usa la **distribución exacta de Wilcoxon**.")

        st.subheader("📈 Región de rechazo")
        if usa_normal and z_stat is not None:
            fig = graficar_normal(z_stat, alpha_r, _tipo_a_normal(tipo))
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig_b, ax_b = plt.subplots(figsize=(6, 3))
            ax_b.bar(["T⁺", "T⁻"], [T_pos, T_neg],
                     color=["steelblue", "salmon"], edgecolor="black")
            ax_b.set_ylabel("Suma de rangos")
            ax_b.set_title("T⁺ vs T⁻ (distribución exacta)")
            st.pyplot(fig_b)
            plt.close(fig_b)

        st.subheader("✅ Decisión y conclusión")
        if usa_normal and z_stat is not None and z_critico is not None:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            z_stat, z_critico, "Z",
                            direccion=_tipo_a_normal(tipo))
        else:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            min(T_pos, T_neg), None, "T")


# ============================================================
# 7. Render: Mann-Whitney U
# ============================================================
def render_mann_whitney():
    st.header("Prueba U de Mann-Whitney")
    st.markdown(
        "Prueba **no paramétrica** para comparar dos grupos **independientes**."
    )
    st.divider()

    # 1. Configuración
    st.subheader("1️⃣ Configuración de la prueba")
    col1, col2, col3 = st.columns(3)
    with col1:
        tipo = st.selectbox(
            "Tipo de hipótesis",
            options=["Bilateral (≠)", "Unilateral derecha (>)", "Unilateral izquierda (<)"],
            index=0,
            key="mw_tipo"
        )
    with col2:
        alpha = st.selectbox(
            "Nivel de significancia (α)",
            options=[0.01, 0.05, 0.10],
            index=1,
            key="mw_alpha"
        )
    with col3:
        use_cont = st.checkbox(
            "Corrección por continuidad",
            value=True,
            key="mw_cont"
        )

    st.divider()

    # 2. Datos
    st.subheader("2️⃣ Ingreso de datos")
    st.caption("Los tamaños de ambos grupos pueden ser distintos.")
    texto_1 = st.text_input(
        "Grupo 1", placeholder="Ej: 12, 15, 17, 14, 16",
        key="mw_g1"
    )
    texto_2 = st.text_input(
        "Grupo 2", placeholder="Ej: 18, 20, 19, 22, 21, 23",
        key="mw_g2"
    )

    datos_ok = False
    g1, g2 = [], []
    try:
        g1 = parsear_valores(texto_1)
        g2 = parsear_valores(texto_2)
        if g1 and g2:
            if len(g1) < 2 or len(g2) < 2:
                st.error("❌ Cada grupo debe tener al menos 2 observaciones.")
            else:
                datos_ok = True
                if len(g1) < 5 or len(g2) < 5:
                    st.warning(
                        f"⚠️ Se recomienda nᵢ ≥ 5 (actual: n₁ = {len(g1)}, n₂ = {len(g2)})."
                    )
    except ValueError:
        st.error("❌ Hay valores no numéricos en alguno de los grupos.")

    st.divider()

    # 3. Ejecución
    ejecutar = st.button(
        "🔬 Ejecutar U de Mann-Whitney",
        type="primary",
        key="mw_ejecutar"
    )

    if ejecutar:
        if not datos_ok:
            st.error("Debes ingresar los dos grupos correctamente.")
            st.session_state.pop("resultado_mw", None)
        else:
            x1 = np.asarray(g1, dtype=float)
            x2 = np.asarray(g2, dtype=float)
            n1, n2 = len(x1), len(x2)
            N = n1 + n2

            todos = np.concatenate([x1, x2])
            rangos = stats.rankdata(todos)
            r1 = rangos[:n1]
            r2 = rangos[n1:]
            R1 = float(r1.sum())
            R2 = float(r2.sum())

            U1 = n1 * n2 + n1 * (n1 + 1) / 2.0 - R1
            U2 = n1 * n2 - U1
            U_min = min(U1, U2)

            alternative = _tipo_a_alternative(tipo)
            U_scipy, p_valor = stats.mannwhitneyu(
                x1, x2, alternative=alternative, use_continuity=use_cont
            )

            # Z con corrección por empates
            mu_U = n1 * n2 / 2.0
            _, counts = np.unique(todos, return_counts=True)
            tie_term = sum((t ** 3 - t) for t in counts if t > 1)
            var_U = (n1 * n2 / 12.0) * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else np.nan
            sigma_U = np.sqrt(var_U) if var_U and var_U > 0 else np.nan
            z_stat = ((U2 - mu_U) / sigma_U) if sigma_U and sigma_U > 0 else np.nan

            # Chequeo de forma similar reutilizando función de KW
            iqrs, ratio_iqr = chequear_forma_similar([x1, x2])

            rechaza_H0 = p_valor < alpha

            st.session_state["resultado_mw"] = {
                "x1": x1.tolist(), "x2": x2.tolist(),
                "n1": n1, "n2": n2, "N": N,
                "R1": R1, "R2": R2,
                "U1": U1, "U2": U2, "U_min": U_min,
                "U_scipy": float(U_scipy), "p_valor": float(p_valor),
                "mu_U": mu_U,
                "sigma_U": float(sigma_U) if sigma_U and not np.isnan(sigma_U) else None,
                "z_stat": float(z_stat) if not np.isnan(z_stat) else None,
                "iqrs": iqrs, "ratio_iqr": ratio_iqr,
                "alpha": alpha, "tipo": tipo, "use_cont": use_cont,
                "rechaza_H0": rechaza_H0,
                "rangos_todos": rangos.tolist(),
            }

    # 4. Reporte
    if "resultado_mw" in st.session_state:
        R = st.session_state["resultado_mw"]
        x1 = np.asarray(R["x1"])
        x2 = np.asarray(R["x2"])
        n1, n2, N = R["n1"], R["n2"], R["N"]
        R1, R2 = R["R1"], R["R2"]
        U1, U2, U_min = R["U1"], R["U2"], R["U_min"]
        p_valor = R["p_valor"]
        z_stat = R["z_stat"]
        mu_U = R["mu_U"]
        sigma_U = R["sigma_U"]
        alpha_r = R["alpha"]
        tipo = R["tipo"]

        st.header("3️⃣ Reporte completo")

        st.subheader("📌 Planteamiento de hipótesis")
        if tipo.startswith("Bilateral"):
            st.markdown(
                "- **H₀:** Las distribuciones de los dos grupos son iguales.\n"
                "- **H₁:** Las distribuciones difieren."
            )
        elif "derecha" in tipo:
            st.markdown(
                "- **H₀:** P(X₁ > X₂) ≤ 1/2.\n"
                "- **H₁:** P(X₁ > X₂) > 1/2 (Grupo 1 tiende a ser mayor)."
            )
        else:
            st.markdown(
                "- **H₀:** P(X₁ > X₂) ≥ 1/2.\n"
                "- **H₁:** P(X₁ > X₂) < 1/2 (Grupo 1 tiende a ser menor)."
            )

        st.subheader("📋 Supuestos de la prueba y chequeos")
        st.markdown(
            "- Los dos grupos son **independientes**.\n"
            "- Escala al menos **ordinal**.\n"
            "- Distribuciones con **forma similar** (para interpretar en términos de medianas)."
        )
        if n1 < 5 or n2 < 5:
            st.warning(
                f"⚠️ Grupo pequeño: n₁ = {n1}, n₂ = {n2}. La aproximación normal "
                "puede ser poco confiable."
            )
        if R["ratio_iqr"] == np.inf or R["ratio_iqr"] > 2:
            iqrs_fmt = [round(x, 3) for x in R["iqrs"]]
            st.warning(
                f"⚠️ IQR muy distintos (IQR = {iqrs_fmt}, razón ≈ {R['ratio_iqr']:.2f}). "
                "Formular la conclusión sobre **distribuciones**, no sobre medianas."
            )

        st.subheader("📊 Resumen por grupo")
        resumen = pd.DataFrame({
            "n": [n1, n2],
            "Mediana": [float(np.median(x1)), float(np.median(x2))],
            "IQR": R["iqrs"],
            "Suma de rangos": [R1, R2],
            "Rango promedio": [R1 / n1, R2 / n2],
        }, index=["Grupo 1", "Grupo 2"])
        st.dataframe(resumen.style.format({
            "Mediana": "{:.3f}", "IQR": "{:.3f}",
            "Suma de rangos": "{:.2f}", "Rango promedio": "{:.3f}",
        }))

        with st.expander("🔽 Ver tabla de observaciones ordenadas con sus rangos"):
            etiquetas = ["Grupo 1"] * n1 + ["Grupo 2"] * n2
            todos = np.concatenate([x1, x2])
            df_ord = pd.DataFrame({
                "Valor": todos,
                "Grupo": etiquetas,
                "Rango": R["rangos_todos"],
            }).sort_values("Valor", kind="mergesort").reset_index(drop=True)
            df_ord.index = df_ord.index + 1
            df_ord.index.name = "Orden"
            st.dataframe(df_ord.style.format({"Valor": "{:.3f}", "Rango": "{:.2f}"}))

        st.subheader("🧮 Cálculo del estadístico")
        st.latex(r"U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1,\qquad U_2 = n_1 n_2 - U_1")
        st.latex(r"U = \min(U_1, U_2)")
        st.markdown(f"- **n₁** = {n1}, **n₂** = {n2}, **N** = {N}")
        st.markdown(f"- **R₁** = `{R1:.2f}`, **R₂** = `{R2:.2f}`")
        st.markdown(f"- **U₁** = `{U1:.4f}`, **U₂** = `{U2:.4f}`")
        st.markdown(f"- **U = min(U₁, U₂)** = `{U_min:.4f}`")
        st.markdown(f"- **p-valor (scipy)** = `{p_valor:.4f}`")

        usa_normal = N >= 20
        z_critico = None
        if usa_normal and z_stat is not None:
            st.latex(r"Z = \frac{U_2 - \mu_U}{\sigma_U},\quad \mu_U=\frac{n_1 n_2}{2}")
            st.latex(r"\sigma_U = \sqrt{\frac{n_1 n_2}{12}\left[(N+1) - \frac{\sum(t^3-t)}{N(N-1)}\right]}")
            st.markdown(f"- **μ_U** = `{mu_U:.4f}`")
            st.markdown(f"- **σ_U** = `{sigma_U:.4f}` (con corrección por empates)")
            st.markdown(f"- **Z** = `{z_stat:.4f}`")
            if tipo.startswith("Bilateral"):
                z_critico = stats.norm.ppf(1 - alpha_r / 2)
            elif "derecha" in tipo:
                z_critico = stats.norm.ppf(1 - alpha_r)
            else:
                z_critico = stats.norm.ppf(alpha_r)
            st.markdown(f"- **Z crítico** (α = {alpha_r}) = `{z_critico:.4f}`")
        else:
            st.info("ℹ️ N total < 20: se usa la **distribución exacta** de scipy.")

        st.subheader("📈 Región de rechazo")
        if usa_normal and z_stat is not None:
            fig = graficar_normal(z_stat, alpha_r, _tipo_a_normal(tipo))
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig_b, ax_b = plt.subplots(figsize=(6, 3))
            ax_b.bar(["U₁", "U₂"], [U1, U2],
                     color=["steelblue", "salmon"], edgecolor="black")
            ax_b.set_ylabel("Valor de U")
            ax_b.set_title("U₁ vs U₂ (distribución exacta)")
            st.pyplot(fig_b)
            plt.close(fig_b)

        st.subheader("✅ Decisión y conclusión")
        if usa_normal and z_stat is not None and z_critico is not None:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            z_stat, z_critico, "Z",
                            direccion=_tipo_a_normal(tipo))
        else:
            bloque_decision(R["rechaza_H0"], p_valor, alpha_r,
                            U_min, None, "U")


# ============================================================
# 8. Render: Chi-cuadrado Independencia
# ============================================================
def render_chi_independencia():
    st.header("Prueba χ² de Independencia")
    st.markdown(
        "Evalúa si dos variables categóricas son **independientes** usando una "
        "tabla de contingencia."
    )
    st.divider()

    # 1. Configuración
    st.subheader("1️⃣ Configuración de la prueba")
    col1, col2, col3 = st.columns(3)
    with col1:
        r = st.number_input("Número de filas (r)", min_value=2, max_value=10,
                            value=2, step=1, key="chi_r")
    with col2:
        c = st.number_input("Número de columnas (c)", min_value=2, max_value=10,
                            value=2, step=1, key="chi_c")
    with col3:
        alpha = st.selectbox("Nivel de significancia (α)",
                             options=[0.01, 0.05, 0.10], index=1,
                             key="chi_alpha")

    yates_disabled = not (r == 2 and c == 2)
    yates = st.checkbox(
        "Aplicar corrección de Yates (solo 2×2)",
        value=False,
        disabled=yates_disabled,
        key="chi_yates"
    )
    if yates_disabled:
        st.caption("La corrección de Yates solo se aplica a tablas 2×2.")

    st.divider()

    # 2. Datos
    st.subheader("2️⃣ Ingreso de datos")
    nombres_filas_txt = st.text_input(
        "Nombres de filas (separados por comas)",
        placeholder="Ej: Hombre, Mujer",
        key="chi_nom_f"
    )
    nombres_cols_txt = st.text_input(
        "Nombres de columnas (separados por comas)",
        placeholder="Ej: Sí, No",
        key="chi_nom_c"
    )

    def _parse_nombres(texto, n, prefijo):
        if not texto.strip():
            return [f"{prefijo} {i+1}" for i in range(n)]
        nombres = [x.strip() for x in texto.split(",") if x.strip()]
        if len(nombres) < n:
            nombres += [f"{prefijo} {i+1}" for i in range(len(nombres), n)]
        return nombres[:n]

    nombres_f = _parse_nombres(nombres_filas_txt, r, "Fila")
    nombres_c = _parse_nombres(nombres_cols_txt, c, "Col")

    st.caption("Ingresa las frecuencias observadas en la tabla:")
    df_init = pd.DataFrame(
        np.zeros((r, c), dtype=int),
        index=nombres_f,
        columns=nombres_c
    )
    tabla_editada = st.data_editor(
        df_init,
        key="chi_tabla",
        num_rows="fixed",
        use_container_width=True
    )

    datos_ok = False
    tabla_arr = None
    try:
        tabla_arr = tabla_editada.to_numpy(dtype=float)
        if np.any(tabla_arr < 0) or not np.all(np.isfinite(tabla_arr)):
            st.error("❌ Todas las frecuencias deben ser enteros ≥ 0.")
        elif not np.allclose(tabla_arr, np.round(tabla_arr)):
            st.error("❌ Las frecuencias deben ser enteros.")
        elif tabla_arr.sum() <= 0:
            st.error("❌ La tabla debe tener al menos una frecuencia > 0.")
        else:
            datos_ok = True
            tabla_arr = tabla_arr.astype(int)
    except (TypeError, ValueError):
        st.error("❌ La tabla contiene valores no numéricos.")

    st.divider()

    # 3. Ejecución
    ejecutar = st.button(
        "🔬 Ejecutar χ² de Independencia",
        type="primary",
        key="chi_ejecutar"
    )

    if ejecutar:
        if not datos_ok:
            st.error("Debes ingresar una tabla de contingencia válida.")
            st.session_state.pop("resultado_chi", None)
        else:
            correction = yates and (r == 2 and c == 2)
            chi2_val, p_valor, gl, esperadas = stats.chi2_contingency(
                tabla_arr, correction=correction
            )
            n = int(tabla_arr.sum())
            denom = n * max(1, min(r - 1, c - 1))
            v_cramer = float(np.sqrt(chi2_val / denom)) if denom > 0 else np.nan
            chi_critico = stats.chi2.ppf(1 - alpha, df=gl)
            rechaza_H0 = p_valor < alpha

            st.session_state["resultado_chi"] = {
                "tabla": tabla_arr.tolist(),
                "nombres_f": nombres_f,
                "nombres_c": nombres_c,
                "r": r, "c": c, "n": n,
                "alpha": alpha,
                "correction": correction,
                "chi2": float(chi2_val),
                "p_valor": float(p_valor),
                "gl": int(gl),
                "esperadas": esperadas.tolist(),
                "v_cramer": v_cramer,
                "chi_critico": float(chi_critico),
                "rechaza_H0": rechaza_H0,
            }

    # 4. Reporte
    if "resultado_chi" in st.session_state:
        R = st.session_state["resultado_chi"]
        tabla = np.asarray(R["tabla"])
        esperadas = np.asarray(R["esperadas"])
        nombres_f = R["nombres_f"]
        nombres_c = R["nombres_c"]
        r_r, c_r = R["r"], R["c"]
        n = R["n"]
        chi2_val = R["chi2"]
        p_valor = R["p_valor"]
        gl = R["gl"]
        v_cramer = R["v_cramer"]
        chi_critico = R["chi_critico"]
        alpha_r = R["alpha"]

        st.header("3️⃣ Reporte completo")

        st.subheader("📌 Planteamiento de hipótesis")
        st.markdown(
            "- **H₀:** Las dos variables categóricas son **independientes**.\n"
            "- **H₁:** Las dos variables **no** son independientes (existe asociación)."
        )

        st.subheader("📋 Supuestos de la prueba y chequeos")
        st.markdown(
            "- Observaciones **independientes**.\n"
            "- Variables **categóricas**.\n"
            "- Frecuencias esperadas suficientemente grandes (Eᵢⱼ ≥ 5 en ≥ 80% de las celdas)."
        )
        n_celdas = r_r * c_r
        celdas_lt5 = int(np.sum(esperadas < 5))
        celdas_lt1 = int(np.sum(esperadas < 1))
        pct_lt5 = celdas_lt5 / n_celdas * 100
        st.markdown(
            f"- Celdas con Eᵢⱼ < 5: **{celdas_lt5}/{n_celdas}** ({pct_lt5:.1f}%)."
        )
        if pct_lt5 > 20 or celdas_lt1 > 0:
            sugerencia = (
                "la **prueba exacta de Fisher**" if (r_r == 2 and c_r == 2)
                else "**agrupar categorías**"
            )
            st.warning(
                f"⚠️ Más del 20% de celdas tienen Eᵢⱼ < 5"
                + (f" (o hay celdas con Eᵢⱼ < 1)" if celdas_lt1 > 0 else "")
                + f". Considerar {sugerencia}."
            )

        st.subheader("📊 Resumen descriptivo")
        totales_f = tabla.sum(axis=1)
        totales_c = tabla.sum(axis=0)
        resumen_df = pd.DataFrame(tabla, index=nombres_f, columns=nombres_c)
        resumen_df["Total fila"] = totales_f
        total_col_row = list(totales_c) + [n]
        resumen_df.loc["Total columna"] = total_col_row
        st.dataframe(resumen_df)

        st.subheader("📑 Tablas de detalle")
        df_obs = pd.DataFrame(tabla, index=nombres_f, columns=nombres_c)
        df_esp = pd.DataFrame(esperadas, index=nombres_f, columns=nombres_c)
        contrib = (tabla - esperadas) ** 2 / np.where(esperadas > 0, esperadas, np.nan)
        df_con = pd.DataFrame(contrib, index=nombres_f, columns=nombres_c)

        with st.expander("🔽 Frecuencias observadas (Oᵢⱼ)"):
            st.dataframe(df_obs)
        with st.expander("🔽 Frecuencias esperadas (Eᵢⱼ)"):
            st.dataframe(df_esp.style.format("{:.3f}"))
        with st.expander("🔽 Contribuciones (Oᵢⱼ − Eᵢⱼ)² / Eᵢⱼ"):
            st.dataframe(df_con.style.format("{:.4f}"))

        st.subheader("🧮 Cálculo del estadístico")
        st.latex(r"E_{ij} = \frac{(\text{total fila}_i)(\text{total col}_j)}{n}")
        st.latex(r"\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}")
        st.latex(r"gl = (r-1)(c-1)")
        st.latex(r"V_{Cram\'er} = \sqrt{\frac{\chi^2}{n\cdot \min(r-1,\ c-1)}}")

        st.markdown(f"- **n total** = {n}")
        st.markdown(f"- **gl** = ({r_r}−1)({c_r}−1) = {gl}")
        if R["correction"]:
            st.markdown("- **Corrección de Yates:** aplicada (tabla 2×2).")
        st.markdown(f"- **χ²** = `{chi2_val:.4f}`")
        st.markdown(f"- **χ² crítico** (α = {alpha_r}, gl = {gl}) = `{chi_critico:.4f}`")
        st.markdown(f"- **p-valor** = `{p_valor:.4f}`")
        st.markdown(f"- **V de Cramér** = `{v_cramer:.4f}`")

        st.subheader("📈 Región de rechazo")
        fig = graficar_chi2(chi2_val, chi_critico, gl, alpha_r)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("✅ Decisión y conclusión")
        coinciden = (chi2_val > chi_critico) == (p_valor < alpha_r)
        if not coinciden:
            st.warning(
                "⚠️ El criterio del χ² crítico y el p-valor no coinciden; "
                "se usa el **p-valor** como criterio final."
            )

        if R["rechaza_H0"]:
            if v_cramer < 0.2:
                mag = "pequeño"
            elif v_cramer < 0.4:
                mag = "mediano"
            else:
                mag = "grande"
            st.success(
                f"**Se RECHAZA H₀** (p-valor = {p_valor:.4f} < α = {alpha_r}; "
                f"χ² = {chi2_val:.4f} vs χ² crítico = {chi_critico:.4f})."
            )
            st.markdown(
                f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
                "existe evidencia estadística suficiente para afirmar que **las variables "
                f"no son independientes**. El tamaño del efecto (V de Cramér = {v_cramer:.3f}) "
                f"es **{mag}** (referencia aproximada: ~0.1 pequeño, ~0.3 mediano, ~0.5 grande)."
            )
        else:
            st.info(
                f"**NO se rechaza H₀** (p-valor = {p_valor:.4f} ≥ α = {alpha_r}; "
                f"χ² = {chi2_val:.4f} vs χ² crítico = {chi_critico:.4f})."
            )
            st.markdown(
                f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
                "**no** existe evidencia estadística suficiente para afirmar que las "
                "variables estén asociadas."
            )


# ============================================================
# 9. Orquestador principal con tabs
# ============================================================
st.title("📊 Pruebas de Hipótesis No Paramétricas")
st.markdown("**Autores:** Martin Bastidas y Andres Erazo")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Kruskal-Wallis",
    "Wilcoxon 1 muestra",
    "Wilcoxon pareadas",
    "Mann-Whitney U",
    "χ² Independencia",
])

with tab1:
    render_kw()
with tab2:
    render_wilcoxon_1()
with tab3:
    render_wilcoxon_pareadas()
with tab4:
    render_mann_whitney()
with tab5:
    render_chi_independencia()
