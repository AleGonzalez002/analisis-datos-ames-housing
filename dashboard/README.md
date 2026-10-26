# Cuadro de mando — Mercado residencial de Ames, Iowa

**Grupo 10 · Fase II — Visualización Interactiva, UX & Data Storytelling**
Universidad Evangélica de El Salvador

Entregable 2.2: aplicación analítica interactiva conectada a la data depurada en la Fase I
(2,925 observaciones · 82 variables · sin valores nulos).

## Dashboard interactivo — Versión pública

La aplicación está desplegada en Streamlit Community Cloud y permite
explorar interactivamente el mercado residencial de Ames, Iowa.

**Acceso al dashboard:**
https://ames-housing-grupo10.streamlit.app

**Configuración del despliegue:**
- Repositorio: Juanjo-devfx/analisis-datos-ames-housing
- Rama: feature/dashboard-interactivo
- Archivo principal: dashboard/app.py

## Ejecutar en local

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`.

## Publicar en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio público de GitHub.
2. Entrar a [share.streamlit.io](https://share.streamlit.io) con la cuenta de GitHub.
3. *New app* → seleccionar el repositorio, rama `main` y archivo `app.py`.
4. *Deploy*. El enlace resultante es el que se entrega al docente.

No requiere variables de entorno ni credenciales: la data viaja en el repositorio.

---

## Datos

`data/AmesHousing_clean.csv` es la salida del pipeline de la Fase I
(`data/processed/AmesHousing.csv` del repositorio de análisis). No se aplica ninguna
limpieza adicional dentro de la aplicación. Las únicas columnas derivadas se calculan
al vuelo y están documentadas en `app.py`:

| Columna | Definición |
|---|---|
| `Fecha Venta` | `Yr Sold` + `Mo Sold` como fecha mensual |
| `Precio x pie2` | `SalePrice / Gr Liv Area` |
| `Antiguedad` | `Yr Sold − Year Built` |
| `Esperado` | Precio predicho por el modelo log-lineal (pestaña 4) |
| `Residual` | `SalePrice / Esperado − 1` |

---

## Estructura de la interfaz

La aplicación sigue la heurística de Shneiderman y el flujo narrativo exigido en las
indicaciones.

**Overview first.** El estado inicial muestra las 2,925 ventas sin ningún filtro aplicado.
Cuatro tarjetas KPI permanecen visibles sobre las pestañas: ventas en vista, precio
mediano, precio medio y precio mediano por pie². La mediana es el indicador principal por
la asimetría de la variable; el precio medio se muestra junto a ella precisamente para
hacer visible esa brecha (+12.8 % sobre la mediana en el conjunto completo).

**Zoom and filter.** Los cinco filtros de la barra lateral —año, calidad general, área
habitable, condición de venta y barrio— son globales y se aplican de forma sincronizada
a las cuatro pestañas. A ellos se suma la selección directa sobre las marcas (*brushing
and linking*): clic en una barra del ranking filtra por ese barrio, clic en un nivel del
boxplot filtra por esa calidad, arrastre sobre el histograma filtra por intervalo de
precio y clic en una celda del heatmap abre la dispersión de ese par de variables. Todas
las vistas leen un estado único compartido; no existen filtros locales ocultos.

**Estado visible y reversible.** Bajo el título se muestra el contador «n = X de 2,925»
junto a las selecciones activas como etiquetas removibles, y un botón «Volver al
panorama» que restablece todas las selecciones directas en un solo paso.

**Details on demand.** Todos los gráficos llevan tooltips enriquecidos con las variables
de contexto del registro; la pestaña 4 añade una tabla de inspección ordenable y
descargable en CSV, con el PID de cada vivienda para rastrear la fila en el conjunto
original.

### Flujo narrativo

| Pestaña | Pregunta | Salida |
|---|---|---|
| 1 · Diagnóstico | ¿Cómo se comporta el precio en este segmento? | Distribución, dispersión y tendencia mensual suavizada |
| 2 · Patrones y anomalías | ¿Dónde y con qué calidad se concentra el valor? | Ranking de barrios, precio por nivel de calidad, área frente a precio |
| 3 · Causa raíz | ¿Qué variables explican el precio? | Correlaciones, matriz de colinealidad y descomposición área × precio unitario |
| 4 · Decisión | ¿Sobre qué viviendas actuar? | Residuales del modelo de valoración, tabla de inspección y recomendación |

---

## Decisiones de diseño

**Codificación visual.** Posición y longitud para toda comparación de magnitud. El color
cumple tres funciones y ninguna decorativa: un tono primario único para las métricas de
precio, gris neutro para referencias y contexto, y ámbar reservado a alertas. Las escalas
secuenciales codifican variables ordinales y la divergente centra el cero en las
correlaciones. No hay gráficos 3D ni circulares, y los ejes de barras parten de cero; la
escala logarítmica de precio es opcional y queda rotulada en el eje.

**Carga cognitiva.** Sin bordes de panel, sombras ni fondos; cuadrícula horizontal tenue
y únicamente donde se requiere leer un valor. Tres niveles tipográficos y una sola
familia. Cada pestaña responde a una pregunta y agrupa entre dos y tres visualizaciones.

**Las cuatro alertas en ámbar.** El ámbar nunca decora: señala las cuatro condiciones que
pueden inducir a una lectura errónea. (1) *n bajo*: barrios con menos de 30 ventas,
coloreados en el ranking y contados en el tooltip. (2) *Cerca de Tukey*: las viviendas por
encima de Q3 + 1.5·RIC se dibujan en ámbar en el histograma, con la cerca rotulada sobre
el eje. (3) *Año parcial*: el tramo de 2010 de la serie temporal se traza en ámbar porque
solo cubre enero–julio y no es comparable con los años completos. (4) *Venta no normal*:
una casilla resalta en ámbar las ventas Partial, Abnorml, Family, Alloca y AdjLand sobre
la dispersión. El encabezado resume las cuatro alertas activas bajo la selección vigente.

**Reglas de baja densidad.** (1) Con n < 30 la vista se marca en ámbar en lugar de
ocultarse. (2) Con n < 5 no se calcula ningún estadístico: se muestran las observaciones
individuales y la acción «ampliar filtro»; la regla se aplica tanto a la vista completa
como a la mediana de cada barrio en el ranking. (3) Ninguna vista se presenta vacía sin
explicación. La descomposición por barrio excluye además los que no alcanzan diez ventas,
y la mediana mensual se acompaña siempre del volumen para que una caída por muestra
escasa no se lea como una caída de precio.

**Correlaciones.** Se usa el coeficiente de Spearman, no Pearson, por la asimetría del
precio y porque varias predictoras son ordinales. El tooltip del heatmap reporta ρ y el
número de pares. Las variables derivadas del propio precio se excluyen para no producir
correlaciones circulares.

**Valoración de la pestaña 4.** El precio esperado proviene de un modelo log-lineal
ajustado sobre el segmento filtrado, que controla barrio, área habitable, calidad,
condición y año de construcción (R² ≈ 0.88 sobre el conjunto completo). Se descartó la
comparación directa contra el precio por pie² mediano del barrio: al ser el precio
unitario estructuralmente más alto en viviendas pequeñas, ese indicador señalaba el
tamaño en lugar de la oportunidad. Cuando el segmento filtrado baja de 60 registros el
modelo no se ajusta y la aplicación lo advierte antes de mostrar resultados.

---

## Archivos

```
app.py                      Aplicación completa
requirements.txt            Dependencias con versión fija
.streamlit/config.toml      Tema (paleta y tipografía)
data/AmesHousing_clean.csv  Salida del pipeline de la Fase I
```

## Fuente

D. De Cock, "Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester
Regression Project," *Journal of Statistics Education*, vol. 19, no. 3, 2011.
