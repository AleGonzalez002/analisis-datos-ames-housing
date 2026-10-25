from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

DATA = Path(__file__).parent / "data" / "AmesHousing_clean.csv"

PRIMARY = "#2F5D8C"
NEUTRAL = "#8A94A0"
ALERT = "#B45309"
GRID = "#E6E9ED"
SEQ = ["#DCE6F0", "#BCCFE2", "#9AB6D3", "#789DC4", "#5784B5", "#3F6B9B", "#2F5D8C", "#234A70", "#1A3754", "#11253A"]
DIVERGING = ["#B45309", "#D99B5C", "#F0DCC6", "#F2F2F2", "#C6D5E6", "#6E97C0", "#2F5D8C"]
MIN_N = 30
MIN_EST = 5
DERIVADAS = ["Precio x pie2", "Antiguedad", "Esperado", "Residual"]
ANIO_PARCIAL = 2010

st.set_page_config(page_title="Ames Housing | Cuadro de mando", page_icon="▦", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
      [data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 600;}
      [data-testid="stMetricLabel"] {color: #6B7480;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      hr {margin: 0.6rem 0 1.4rem 0; border-color: #E6E9ED;}
      .stTabs [data-baseweb="tab"] {padding: 0.4rem 1.1rem;}
      .nota {color:#6B7480; font-size:0.86rem; line-height:1.45;}
      .alerta {color:#B45309; font-size:0.86rem; line-height:1.45;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Estado de los filtros por selección directa (brushing and linking) ────────
if "nonce" not in st.session_state:
    st.session_state.nonce = {"rank": 0, "qual": 0, "hist": 0, "heat": 0}


def kc(name):
    return f"{name}_{st.session_state.nonce[name]}"


def limpiar(name):
    st.session_state.nonce[name] += 1


def limpiar_todo():
    for k in st.session_state.nonce:
        st.session_state.nonce[k] += 1


def seleccion(name):
    ev = st.session_state.get(kc(name))
    if not ev:
        return None
    sel = ev.get("selection") if isinstance(ev, dict) else getattr(ev, "selection", None)
    if not sel:
        return None
    return sel


def punto(name, campo):
    sel = seleccion(name)
    if not sel:
        return None
    pts = sel.get("points") or []
    if not pts:
        return None
    return pts[0].get(campo)


def rango_box(name):
    sel = seleccion(name)
    if not sel:
        return None
    cajas = sel.get("box") or []
    if cajas and cajas[0].get("x"):
        xs = cajas[0]["x"]
        return float(min(xs)), float(max(xs))
    pts = sel.get("points") or []
    if pts:
        xs = [p.get("x") for p in pts if p.get("x") is not None]
        if xs:
            return float(min(xs)), float(max(xs))
    return None


def tufte(fig, height=380, showlegend=False):
    fig.update_layout(
        template="simple_white",
        height=height,
        showlegend=showlegend,
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=8, r=8, t=34, b=8),
        font=dict(family="Helvetica Neue, Arial, sans-serif", size=12, color="#333A42"),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        plot_bgcolor="white",
        paper_bgcolor="white",
        clickmode="event+select",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside", tickcolor=GRID, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, ticks="", linecolor=GRID)
    return fig


def compact(x):
    return "—" if pd.isna(x) else f"${x/1000:,.0f}k"


@st.cache_data
def load():
    df = pd.read_csv(DATA)
    df["Fecha Venta"] = pd.to_datetime(
        dict(year=df["Yr Sold"], month=df["Mo Sold"], day=1), errors="coerce"
    )
    df["Precio x pie2"] = df["SalePrice"] / df["Gr Liv Area"]
    df["Antiguedad"] = df["Yr Sold"] - df["Year Built"]
    return df


@st.cache_data(show_spinner=False)
def cerca_tukey(serie: pd.Series) -> float:
    q1, q3 = serie.quantile([0.25, 0.75])
    return q3 + 1.5 * (q3 - q1)


@st.cache_data(show_spinner=False)
def valorar(sub: pd.DataFrame) -> pd.DataFrame:
    """Precio esperado de cada vivienda dadas su ubicación, tamaño, calidad y antigüedad.

    Modelo log-lineal ajustado sobre el segmento filtrado. El residual es la diferencia
    porcentual entre el precio pagado y el esperado. Esta referencia sustituye a la
    comparación contra el precio por pie² del barrio, que solo señalaba viviendas
    pequeñas: su precio unitario es estructuralmente más alto.
    """
    out = sub.copy()
    m = out.rename(
        columns={
            "Gr Liv Area": "area", "Overall Qual": "calidad", "Overall Cond": "condicion",
            "Year Built": "anio", "Neighborhood": "barrio", "SalePrice": "precio",
        }
    )[["precio", "area", "calidad", "condicion", "anio", "barrio"]]

    formula = "np.log(precio) ~ np.log(area) + calidad + condicion + anio"
    if m["barrio"].nunique() > 1:
        formula += " + C(barrio)"
    try:
        if len(m) < 60:
            raise ValueError("muestra insuficiente")
        fit = smf.ols(formula, data=m).fit()
        out["Esperado"] = np.exp(fit.fittedvalues)
        out["Residual"] = out["SalePrice"] / out["Esperado"] - 1
        out.attrs["r2"] = fit.rsquared
        out.attrs["modelo"] = True
    except Exception:
        base = out.groupby("Neighborhood")["Precio x pie2"].transform("median")
        out["Esperado"] = base * out["Gr Liv Area"]
        out["Residual"] = out["SalePrice"] / out["Esperado"] - 1
        out.attrs["r2"] = np.nan
        out.attrs["modelo"] = False
    return out


df = load()
CERCA = cerca_tukey(df["SalePrice"])

# ─── Zoom and filter: filtros cruzados globales ────────────────────────────────
with st.sidebar:
    st.markdown("### Filtros")
    st.caption("Se aplican de forma sincronizada a todas las vistas.")

    anios = sorted(df["Yr Sold"].unique())
    r_anio = st.select_slider("Año de venta", options=anios, value=(anios[0], anios[-1]))

    q_min, q_max = int(df["Overall Qual"].min()), int(df["Overall Qual"].max())
    r_qual = st.slider("Calidad general (Overall Qual)", q_min, q_max, (q_min, q_max))

    a_min, a_max = int(df["Gr Liv Area"].min()), int(df["Gr Liv Area"].max())
    r_area = st.slider("Área habitable (pies²)", a_min, a_max, (a_min, a_max), step=50)

    cond = st.multiselect(
        "Condición de venta",
        sorted(df["Sale Condition"].unique()),
        default=[],
        help="Vacío = todas. Normal aísla el mercado abierto; Partial y Abnorml "
             "introducen precios no comparables.",
    )

    barrios = st.multiselect(
        "Barrio (Neighborhood)",
        sorted(df["Neighborhood"].unique()),
        default=[],
        help="Vacío = todos los barrios.",
    )

    st.markdown("---")
    log_scale = st.checkbox("Escala logarítmica en precio", value=False)
    st.markdown(
        f'<p class="nota">Con menos de {MIN_N} registros una vista se marca en ámbar; '
        f"con menos de {MIN_EST} no se calcula ningún estadístico.</p>",
        unsafe_allow_html=True,
    )

# Selecciones directas sobre las marcas
sel_barrio = punto("rank", "y")
sel_qual = punto("qual", "x")
sel_precio = rango_box("hist")
sel_par = (punto("heat", "x"), punto("heat", "y"))
sel_par = sel_par if all(sel_par) else None

m = (
    df["Yr Sold"].between(*r_anio)
    & df["Overall Qual"].between(*r_qual)
    & df["Gr Liv Area"].between(*r_area)
)
if cond:
    m &= df["Sale Condition"].isin(cond)
if barrios:
    m &= df["Neighborhood"].isin(barrios)
if sel_barrio:
    m &= df["Neighborhood"] == sel_barrio
if sel_qual is not None:
    m &= df["Overall Qual"] == sel_qual
if sel_precio:
    m &= df["SalePrice"].between(*sel_precio)

f = df[m]

st.title("Mercado residencial de Ames, Iowa")
st.markdown(
    '<p class="nota">Cuadro de mando sobre el conjunto depurado en la Fase I '
    "(2,925 observaciones · 82 variables · sin nulos). Ventas 2006–2010.</p>",
    unsafe_allow_html=True,
)

# ─── Estado visible y reversible ───────────────────────────────────────────────
chips = []
if sel_barrio:
    chips.append(("rank", f"Barrio: {sel_barrio}"))
if sel_qual is not None:
    chips.append(("qual", f"Calidad: {sel_qual}"))
if sel_precio:
    chips.append(("hist", f"Precio: {compact(sel_precio[0])} – {compact(sel_precio[1])}"))
if sel_par:
    chips.append(("heat", f"Par: {sel_par[0]} ↔ {sel_par[1]}"))

barra = st.columns([2.1, 1, 1, 1, 1.5])
barra[0].markdown(
    f"**n = {len(f):,} de {len(df):,}**  ·  {len(f)/len(df):.1%} del conjunto"
)
for i, (clave, etiqueta) in enumerate(chips[:3]):
    if barra[i + 1].button(f"✕ {etiqueta}", key=f"chip_{clave}", width="stretch"):
        limpiar(clave)
        st.rerun()
if chips:
    if barra[4].button("↺ Volver al panorama", width="stretch"):
        limpiar_todo()
        st.rerun()
else:
    barra[4].markdown(
        '<p class="nota">Haga clic en una barra, un nivel de calidad o una celda del '
        "heatmap para filtrar las demás vistas.</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

if f.empty:
    st.warning("La combinación de filtros no deja registros. Amplíe el filtro para continuar.")
    st.stop()

# Regla (2): con n < 5 no se calcula ningún estadístico
if len(f) < MIN_EST:
    st.warning(
        f"Solo {len(f)} registro(s) bajo la selección actual. Por debajo de {MIN_EST} no se "
        "calcula ningún estadístico: se muestran las observaciones individuales. "
        "**Acción: ampliar filtro.**"
    )
    st.dataframe(
        f[["PID", "Neighborhood", "SalePrice", "Gr Liv Area", "Overall Qual",
           "Year Built", "Yr Sold", "Sale Condition"]],
        width="stretch", hide_index=True,
    )
    st.stop()

if len(f) < MIN_N:
    st.markdown(
        f'<p class="alerta">▲ n bajo: solo {len(f)} registros bajo la selección actual. '
        "Las medianas y correlaciones de esta vista no son representativas.</p>",
        unsafe_allow_html=True,
    )

# ─── Overview first: cuatro tarjetas KPI ───────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Ventas en vista", f"{len(f):,}", f"{len(f)/len(df):.0%} del conjunto")
k2.metric("Precio mediano", compact(f["SalePrice"].median()),
          f"{f['SalePrice'].median()/df['SalePrice'].median()-1:+.1%} vs global")
k3.metric("Precio medio", compact(f["SalePrice"].mean()),
          f"{f['SalePrice'].mean()/f['SalePrice'].median()-1:+.1%} sobre la mediana")
k4.metric("Precio mediano por pie²", f"${f['Precio x pie2'].median():,.0f}")

# ─── Alertas activas, en ámbar ─────────────────────────────────────────────────
n_tukey = int((f["SalePrice"] > CERCA).sum())
n_nonorm = int((f["Sale Condition"] != "Normal").sum())
n_parcial = int((f["Yr Sold"] == ANIO_PARCIAL).sum())
vc = f["Neighborhood"].value_counts()
n_bajo = int(vc[vc < MIN_N].sum())

avisos = []
if n_bajo:
    avisos.append(f"{(vc < MIN_N).sum()} barrios con n &lt; {MIN_N} ({n_bajo} viviendas)")
if n_tukey:
    avisos.append(f"{n_tukey} viviendas sobre la cerca de Tukey ({compact(CERCA)})")
if n_nonorm:
    avisos.append(f"{n_nonorm} ventas no normales ({n_nonorm/len(f):.1%})")
if n_parcial:
    avisos.append(f"{n_parcial} ventas de {ANIO_PARCIAL}, año parcial (enero–julio)")
if avisos:
    st.markdown(
        '<p class="alerta">▲ Alertas activas: ' + "  ·  ".join(avisos) + "</p>",
        unsafe_allow_html=True,
    )

st.markdown("")

t1, t2, t3, t4 = st.tabs(
    ["1 · Diagnóstico", "2 · Patrones y anomalías", "3 · Causa raíz", "4 · Decisión"]
)

# ─── 1. Diagnóstico inicial ────────────────────────────────────────────────────
with t1:
    st.subheader("¿Cómo se comporta el precio en el segmento seleccionado?")
    c1, c2 = st.columns([3, 2])

    with c1:
        bins = np.histogram_bin_edges(f["SalePrice"], bins=50)
        centros = (bins[:-1] + bins[1:]) / 2
        alturas, _ = np.histogram(f["SalePrice"], bins=bins)
        colores = [ALERT if c > CERCA else PRIMARY for c in centros]
        fig = go.Figure(
            go.Bar(x=centros, y=alturas, marker_color=colores,
                   width=(bins[1] - bins[0]) * 0.92,
                   hovertemplate="Precio: %{x:$,.0f}<br>Viviendas: %{y}<extra></extra>")
        )
        med = f["SalePrice"].median()
        fig.add_vline(x=med, line_width=1.5, line_dash="dot", line_color=NEUTRAL,
                      annotation_text=f"mediana {compact(med)}",
                      annotation_font_color=NEUTRAL, annotation_position="top left")
        fig.add_vline(x=CERCA, line_width=1, line_dash="dash", line_color=ALERT,
                      annotation_text=f"cerca de Tukey {compact(CERCA)}",
                      annotation_font_color=ALERT, annotation_position="top right")
        fig.update_xaxes(title="Precio de venta", tickprefix="$",
                         type="log" if log_scale else "linear")
        fig.update_yaxes(title="Viviendas")
        fig = tufte(fig, 360)
        fig.update_layout(dragmode="select")
        st.plotly_chart(fig, width="stretch",
                        on_select="rerun", selection_mode="box", key=kc("hist"))
        st.markdown(
            f'<p class="nota">Asimetría {f["SalePrice"].skew():.3f}. En ámbar, las '
            f"{n_tukey} viviendas por encima de la cerca de Tukey. Arrastre sobre el gráfico "
            "para filtrar por intervalo de precio.</p>",
            unsafe_allow_html=True,
        )

    with c2:
        fig = px.box(f, y="SalePrice", points=False, color_discrete_sequence=[PRIMARY])
        fig.update_traces(width=0.35, hovertemplate="%{y:$,.0f}<extra></extra>")
        fig.update_yaxes(title="Precio de venta", tickprefix="$",
                         type="log" if log_scale else "linear")
        fig.update_xaxes(title="", showticklabels=False)
        st.plotly_chart(tufte(fig, 360), width="stretch")
        q1, q3 = f["SalePrice"].quantile([0.25, 0.75])
        st.markdown(
            f'<p class="nota">Rango intercuartílico: {compact(q1)} – {compact(q3)}. '
            f"Cerca superior de Tukey: {compact(CERCA)}.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Evolución del mercado, 2006–2010")
    serie = (
        f.groupby("Fecha Venta")
        .agg(mediana=("SalePrice", "median"), ventas=("SalePrice", "size"))
        .reset_index()
        .sort_values("Fecha Venta")
    )
    serie["suavizada"] = serie["mediana"].rolling(3, center=True, min_periods=2).median()
    previo = serie[serie["Fecha Venta"].dt.year < ANIO_PARCIAL]
    parcial = serie[serie["Fecha Venta"].dt.year == ANIO_PARCIAL]

    fig = go.Figure()
    fig.add_bar(x=serie["Fecha Venta"], y=serie["ventas"], name="Ventas en el mes",
                marker_color=GRID, yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>%{y} ventas<extra></extra>")
    fig.add_scatter(x=serie["Fecha Venta"], y=serie["mediana"], name="Mediana mensual",
                    mode="lines", line=dict(color=NEUTRAL, width=1),
                    hovertemplate="%{x|%b %Y}<br>Mediana %{y:$,.0f}<extra></extra>")
    if not previo.empty:
        fig.add_scatter(x=previo["Fecha Venta"], y=previo["suavizada"],
                        name="Tendencia (3 meses)", mode="lines",
                        line=dict(color=PRIMARY, width=2.5),
                        hovertemplate="%{x|%b %Y}<br>Tendencia %{y:$,.0f}<extra></extra>")
    if not parcial.empty:
        puente = pd.concat([previo.tail(1), parcial])
        fig.add_scatter(x=puente["Fecha Venta"], y=puente["suavizada"],
                        name=f"{ANIO_PARCIAL}: año parcial", mode="lines",
                        line=dict(color=ALERT, width=2.5),
                        hovertemplate="%{x|%b %Y}<br>Tendencia %{y:$,.0f}"
                                      "<br>año parcial<extra></extra>")
        fig.add_annotation(x=parcial["Fecha Venta"].iloc[-1], y=parcial["suavizada"].iloc[-1],
                           text=f"{ANIO_PARCIAL} solo cubre enero–julio", showarrow=False,
                           yshift=-26, font=dict(color=ALERT, size=11))
    fig.update_layout(
        yaxis=dict(title="Precio mediano", tickprefix="$"),
        yaxis2=dict(title="Ventas", overlaying="y", side="right", showgrid=False,
                    rangemode="tozero", tickformat="d"),
        hovermode="x unified",
    )
    st.plotly_chart(tufte(fig, 370, showlegend=True), width="stretch")
    st.markdown(
        '<p class="nota">La mediana mensual (gris) es volátil porque algunos meses registran '
        "pocas ventas; la tendencia a tres meses es la serie que debe leerse. El tramo ámbar "
        f"corresponde a {ANIO_PARCIAL}, que solo cubre enero–julio y no es comparable con los "
        "años completos. El eje vertical no parte de cero porque la línea codifica cambio y no "
        "magnitud absoluta.</p>",
        unsafe_allow_html=True,
    )

# ─── 2. Detección de patrones y anomalías ──────────────────────────────────────
with t2:
    st.subheader("¿Dónde y con qué calidad se concentra el valor?")
    c1, c2 = st.columns([3, 2])

    with c1:
        g = (
            f.groupby("Neighborhood")
            .agg(mediana=("SalePrice", "median"), n=("SalePrice", "size"))
            .reset_index()
        )
        sin_est = g[g["n"] < MIN_EST]
        g = g[g["n"] >= MIN_EST].sort_values("mediana")
        if g.empty:
            st.info("Ningún barrio alcanza el mínimo para calcular una mediana.")
        else:
            g["confiable"] = np.where(g["n"] >= MIN_N, "Representativo", f"n bajo (< {MIN_N})")
            orden = g["Neighborhood"].tolist()
            fig = px.bar(
                g, x="mediana", y="Neighborhood", orientation="h", color="confiable",
                color_discrete_map={"Representativo": PRIMARY, f"n bajo (< {MIN_N})": ALERT},
                custom_data=["n"],
            )
            fig.update_traces(hoverinfo="skip", hovertemplate=None)
            fig.add_scatter(
                x=g["mediana"] / 2, y=g["Neighborhood"], mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0)"), showlegend=False,
                customdata=np.stack([g["n"], g["mediana"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>Mediana: %{customdata[1]:$,.0f}"
                              "<br>Viviendas: %{customdata[0]}"
                              "<br><i>clic para filtrar</i><extra></extra>",
            )
            fig.add_vline(x=f["SalePrice"].median(), line_width=1, line_dash="dot",
                          line_color=NEUTRAL)
            fig.update_xaxes(title="Precio mediano", tickprefix="$")
            fig.update_yaxes(title="", categoryorder="array", categoryarray=orden)
            st.plotly_chart(tufte(fig, max(420, 21 * len(g)), showlegend=True),
                            width="stretch", on_select="rerun",
                            selection_mode="points", key=kc("rank"))
        nota = (f'Ordenado por mediana. Clic en una barra para filtrar todas las vistas por ese '
                f"barrio. En ámbar, los barrios con menos de {MIN_N} ventas: su mediana se "
                f"muestra pero no debe sustentar una decisión.")
        if not sin_est.empty:
            detalle = ", ".join(f"{r.Neighborhood} (n={r.n})" for r in sin_est.itertuples())
            nota += (f" Se omite la mediana de los barrios con menos de {MIN_EST} ventas: "
                     f"{detalle}.")
        st.markdown(f'<p class="nota">{nota}</p>', unsafe_allow_html=True)

    with c2:
        fig = px.box(f, x="Overall Qual", y="SalePrice", points=False,
                     color="Overall Qual", color_discrete_sequence=SEQ)
        fig.update_traces(hoverinfo="skip", hovertemplate=None)
        niveles = (
            f.groupby("Overall Qual")
            .agg(med=("SalePrice", "median"), n=("SalePrice", "size"))
            .reset_index()
        )
        fig.add_scatter(
            x=niveles["Overall Qual"], y=niveles["med"], mode="markers",
            marker=dict(size=26, color="rgba(0,0,0,0)"), showlegend=False,
            customdata=np.stack([niveles["n"], niveles["med"]], axis=-1),
            hovertemplate="Calidad %{x}<br>Mediana: %{customdata[1]:$,.0f}"
                          "<br>Viviendas: %{customdata[0]}"
                          "<br><i>clic para filtrar</i><extra></extra>",
        )
        fig.update_xaxes(title="Calidad general (1–10)", dtick=1)
        fig.update_yaxes(title="Precio de venta", tickprefix="$",
                         type="log" if log_scale else "linear")
        st.plotly_chart(tufte(fig, 430), width="stretch", on_select="rerun",
                        selection_mode="points", key=kc("qual"))
        st.markdown(
            '<p class="nota">La saturación codifica el orden de la variable ordinal. '
            "Clic en un nivel para filtrar por esa calidad.</p>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("Área habitable frente a precio")
    resaltar = st.checkbox(
        "Resaltar ventas no normales (Partial, Abnorml, Family, Alloca, AdjLand)",
        value=False,
    )
    fig = px.scatter(
        f, x="Gr Liv Area", y="SalePrice", color="Overall Qual",
        color_continuous_scale=SEQ, opacity=0.62,
        custom_data=["Neighborhood", "Year Built", "Overall Qual", "Precio x pie2", "PID"],
        trendline="ols", trendline_color_override=NEUTRAL,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b> · PID %{customdata[4]}<br>Área: %{x:,.0f} pies²"
            "<br>Precio: %{y:$,.0f}<br>Precio/pie²: $%{customdata[3]:,.0f}"
            "<br>Calidad: %{customdata[2]}<br>Construida en %{customdata[1]}<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )
    if resaltar:
        nn = f[f["Sale Condition"] != "Normal"]
        fig.add_scatter(
            x=nn["Gr Liv Area"], y=nn["SalePrice"], mode="markers", name="Venta no normal",
            marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(width=1.4, color=ALERT)),
            customdata=nn[["Neighborhood", "Sale Condition"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>%{y:$,.0f}<extra></extra>",
        )
    fig.update_xaxes(title="Área habitable (pies²)")
    fig.update_yaxes(title="Precio de venta", tickprefix="$",
                     type="log" if log_scale else "linear")
    fig.update_layout(coloraxis_colorbar=dict(title="Calidad", thickness=12, len=0.7))
    st.plotly_chart(tufte(fig, 440, showlegend=resaltar), width="stretch")
    st.markdown(
        '<p class="nota">Dos canales posicionales para las variables continuas y el color '
        "reservado a la calidad. La dispersión creciente alrededor del ajuste indica que el área "
        "sola no explica el precio, lo que motiva la pestaña siguiente.</p>",
        unsafe_allow_html=True,
    )

# ─── 3. Análisis de causa raíz ─────────────────────────────────────────────────
with t3:
    st.subheader("¿Qué variables explican el precio?")

    num = f.select_dtypes(include=[np.number]).drop(
        columns=[c for c in ["Order", "PID", "Yr Sold", "Mo Sold"] + DERIVADAS if c in f.columns]
    )
    corr = num.corr(method="spearman", numeric_only=True)["SalePrice"].drop("SalePrice").dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index)[:14].sort_values()

    c1, c2 = st.columns([2, 3])

    with c1:
        fig = px.bar(x=top.values, y=top.index, orientation="h",
                     color=top.values, color_continuous_scale=DIVERGING, range_color=[-1, 1])
        fig.update_traces(hovertemplate="<b>%{y}</b><br>ρ = %{x:.3f}<extra></extra>")
        fig.add_vline(x=0, line_width=1, line_color=NEUTRAL)
        fig.update_xaxes(title="Correlación de Spearman con el precio", range=[-1, 1])
        fig.update_yaxes(title="")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(tufte(fig, 470), width="stretch")
        st.markdown(
            '<p class="nota">Catorce variables de mayor correlación absoluta. Se usa Spearman '
            "por la asimetría del precio y porque varias predictoras son ordinales. Se excluyen "
            "las variables derivadas del propio precio, que darían correlaciones circulares.</p>",
            unsafe_allow_html=True,
        )

    with c2:
        cols = list(top.abs().sort_values(ascending=False).index[:8]) + ["SalePrice"]
        cm = f[cols].corr(method="spearman")
        fig = px.imshow(cm, color_continuous_scale=DIVERGING, zmin=-1, zmax=1,
                        aspect="auto", text_auto=".2f")
        fig.update_traces(hoverinfo="skip", hovertemplate=None, textfont_size=10)
        gx, gy = np.meshgrid(cm.columns, cm.index)
        rho_plano = cm.to_numpy().ravel()
        fig.add_scatter(
            x=gx.ravel(), y=gy.ravel(), mode="markers",
            marker=dict(size=34, color="rgba(0,0,0,0)"), showlegend=False,
            customdata=np.stack([rho_plano, np.full(rho_plano.shape, len(f))], axis=-1),
            hovertemplate="%{x} ↔ %{y}<br>ρ = %{customdata[0]:.3f}"
                          "<br>n = %{customdata[1]:,} pares"
                          "<br><i>clic para abrir la dispersión</i><extra></extra>",
        )
        fig.update_layout(coloraxis_colorbar=dict(title="ρ", thickness=12, len=0.7))
        st.plotly_chart(tufte(fig, 470), width="stretch", on_select="rerun",
                        selection_mode="points", key=kc("heat"))
        st.markdown(
            '<p class="nota">Coeficiente de Spearman y número de pares en el tooltip. '
            "Clic en una celda para abrir la dispersión de ese par. Las correlaciones altas "
            "entre predictoras (Garage Cars y Garage Area) advierten colinealidad.</p>",
            unsafe_allow_html=True,
        )

    if sel_par:
        vx, vy = sel_par
        if vx in f.columns and vy in f.columns and vx != vy:
            st.markdown("")
            st.subheader(f"Dispersión del par seleccionado: {vx} ↔ {vy}")
            fig = px.scatter(f, x=vx, y=vy, opacity=0.6,
                             color_discrete_sequence=[PRIMARY],
                             custom_data=["Neighborhood", "PID"])
            fig.update_traces(marker=dict(size=6, line=dict(width=0)),
                              hovertemplate="<b>%{customdata[0]}</b> · PID %{customdata[1]}"
                                            "<br>" + vx + ": %{x:,.0f}<br>" + vy +
                                            ": %{y:,.0f}<extra></extra>")
            st.plotly_chart(tufte(fig, 380), width="stretch")
            rho = f[[vx, vy]].corr(method="spearman").iloc[0, 1]
            st.markdown(
                f'<p class="nota">ρ de Spearman = {rho:.3f} sobre {len(f):,} pares.</p>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.subheader("Descomposición: el precio es área por precio unitario")
    g = (
        f.groupby("Neighborhood")
        .agg(area=("Gr Liv Area", "median"), unit=("Precio x pie2", "median"),
             precio=("SalePrice", "median"), n=("SalePrice", "size"))
        .reset_index()
    )
    g = g[g["n"] >= 10]
    if g.empty:
        st.info("Ningún barrio alcanza diez ventas con la selección actual.")
    else:
        fig = px.scatter(g, x="area", y="unit", size="n", color="precio",
                         color_continuous_scale=SEQ, text="Neighborhood",
                         custom_data=["Neighborhood", "n", "precio"], size_max=34)
        fig.update_traces(
            textposition="top center", textfont=dict(size=9, color=NEUTRAL),
            hovertemplate=("<b>%{customdata[0]}</b><br>Área mediana: %{x:,.0f} pies²"
                           "<br>Precio/pie² mediano: $%{y:,.0f}"
                           "<br>Precio mediano: %{customdata[2]:$,.0f}"
                           "<br>Ventas: %{customdata[1]}<extra></extra>"),
        )
        fig.add_hline(y=f["Precio x pie2"].median(), line_width=1, line_dash="dot",
                      line_color=NEUTRAL)
        fig.add_vline(x=f["Gr Liv Area"].median(), line_width=1, line_dash="dot",
                      line_color=NEUTRAL)
        fig.update_xaxes(title="Área habitable mediana (pies²)")
        fig.update_yaxes(title="Precio mediano por pie²", tickprefix="$")
        fig.update_layout(coloraxis_colorbar=dict(title="Precio", thickness=12, len=0.7))
        st.plotly_chart(tufte(fig, 470), width="stretch")
        st.markdown(
            '<p class="nota">Separar el precio en sus dos componentes distingue el barrio caro '
            "porque sus casas son grandes (derecha abajo) del barrio caro porque su suelo vale "
            "más (izquierda arriba). Se excluyen los barrios con menos de diez ventas.</p>",
            unsafe_allow_html=True,
        )

# ─── 4. Recomendación de acción ────────────────────────────────────────────────
with t4:
    st.subheader("Viviendas que se apartan de su valor esperado")

    val = valorar(f)
    usa_modelo = val.attrs.get("modelo", False)
    r2 = val.attrs.get("r2", np.nan)

    if usa_modelo:
        st.markdown(
            '<p class="nota">El precio esperado de cada vivienda se estima con un modelo '
            "log-lineal ajustado sobre el segmento seleccionado, que controla barrio, área "
            "habitable, calidad, condición y año de construcción. El residual es la diferencia "
            "porcentual entre lo pagado y lo esperado. Esta referencia sustituye a la "
            "comparación por precio por pie², que solo señalaba viviendas pequeñas: su precio "
            f"unitario es estructuralmente más alto. R² = {r2:.3f}.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="alerta">▲ Muestra insuficiente para ajustar el modelo: se usa como '
            "referencia el precio por pie² mediano del barrio, que tiende a señalar viviendas "
            "pequeñas. Amplíe los filtros.</p>",
            unsafe_allow_html=True,
        )

    umbral = st.slider("Umbral de desviación respecto al valor esperado (%)", 5, 60, 20, 5) / 100
    out = val[val["Residual"].abs() >= umbral].copy()
    out["Señal"] = np.where(out["Residual"] < 0, "Bajo su valor esperado",
                            "Sobre su valor esperado")

    c1, c2, c3 = st.columns(3)
    c1.metric("Viviendas señaladas", f"{len(out):,}", f"{len(out)/len(val):.1%} de la vista")
    c2.metric("Bajo el valor esperado", f"{(out['Residual'] < 0).sum():,}")
    c3.metric("Sobre el valor esperado", f"{(out['Residual'] > 0).sum():,}")

    if out.empty:
        st.info("Ninguna vivienda supera el umbral con la selección actual.")
    else:
        fig = px.scatter(
            out, x="Esperado", y="SalePrice", color="Señal",
            color_discrete_map={"Bajo su valor esperado": PRIMARY,
                                "Sobre su valor esperado": ALERT},
            custom_data=["Neighborhood", "Residual", "Overall Qual", "Year Built",
                         "Gr Liv Area", "PID"],
            opacity=0.75,
        )
        lim = [min(out["Esperado"].min(), out["SalePrice"].min()),
               max(out["Esperado"].max(), out["SalePrice"].max())]
        fig.add_scatter(x=lim, y=lim, mode="lines", name="Valor esperado",
                        line=dict(color=NEUTRAL, width=1, dash="dot"), hoverinfo="skip")
        fig.update_traces(marker=dict(size=8, line=dict(width=0)), selector=dict(mode="markers"))
        fig.update_traces(
            hovertemplate=("<b>%{customdata[0]}</b> · PID %{customdata[5]}"
                           "<br>Precio pagado: %{y:$,.0f}<br>Valor esperado: %{x:$,.0f}"
                           "<br>Desviación: %{customdata[1]:+.1%}"
                           "<br>%{customdata[4]:,.0f} pies² · calidad %{customdata[2]} · "
                           "%{customdata[3]}<extra></extra>"),
            selector=dict(mode="markers"),
        )
        fig.update_xaxes(title="Valor esperado", tickprefix="$",
                         type="log" if log_scale else "linear")
        fig.update_yaxes(title="Precio pagado", tickprefix="$",
                         type="log" if log_scale else "linear")
        st.plotly_chart(tufte(fig, 420, showlegend=True), width="stretch")
        st.markdown(
            '<p class="nota">La diagonal punteada representa la igualdad entre el precio observado y el precio estimado por el modelo.</p>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Tabla de inspección")
        tabla = out.copy()
        tabla["Desviación"] = tabla["Residual"] * 100
        cols = ["PID", "Neighborhood", "SalePrice", "Esperado", "Desviación", "Gr Liv Area",
                "Overall Qual", "Overall Cond", "Year Built", "Yr Sold", "Sale Condition"]
        tabla = tabla[cols].sort_values("Desviación")
        st.dataframe(
            tabla, width="stretch", height=340, hide_index=True,
            column_config={
                "PID": st.column_config.NumberColumn("PID", format="%d"),
                "Neighborhood": "Barrio",
                "SalePrice": st.column_config.NumberColumn("Precio pagado", format="$%d"),
                "Esperado": st.column_config.NumberColumn("Valor esperado", format="$%d"),
                "Desviación": st.column_config.NumberColumn("Desviación", format="%.1f%%"),
                "Gr Liv Area": st.column_config.NumberColumn("Área (pies²)", format="%d"),
                "Overall Qual": "Calidad",
                "Overall Cond": "Condición",
                "Year Built": "Construida",
                "Yr Sold": "Vendida",
                "Sale Condition": "Cond. venta",
            },
        )
        st.download_button(
            "Descargar selección (CSV)",
            tabla.to_csv(index=False).encode("utf-8"),
            file_name="ames_viviendas_senaladas.csv",
            mime="text/csv",
        )
        st.markdown(
            '<p class="nota">El PID permite rastrear cada fila en el conjunto original.</p>',
            unsafe_allow_html=True,
        )

        bajas = out[out["Residual"] < 0]
        if len(bajas) >= 5:
            top_b = bajas["Neighborhood"].value_counts().head(3)
            brecha = (bajas["Esperado"] - bajas["SalePrice"]).median()
            st.success(
                f"**Recomendación.** {len(bajas)} viviendas se vendieron al menos un "
                f"{umbral:.0%} por debajo de su valor esperado, con una brecha mediana de "
                f"{compact(brecha)}. La concentración más alta está en "
                f"{', '.join(top_b.index)} ({int(top_b.sum())} casos). El siguiente paso es "
                "revisar en la tabla si la condición general, el año de construcción o una "
                "condición de venta no normal explican la diferencia; cuando no la explican, "
                "el descuento es atribuible a la transacción y no al inmueble."
            )

st.markdown("---")
st.markdown(
    '<p class="nota">Grupo 10 · Fase II — Visualización Interactiva, UX & Data Storytelling · '
    "Universidad Evangélica de El Salvador. Fuente: Ames Housing (De Cock, 2011), "
    "depurado en la Fase I.</p>",
    unsafe_allow_html=True,
)
