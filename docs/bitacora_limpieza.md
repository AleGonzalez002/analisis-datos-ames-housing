# Bitácora de limpieza — Fase III

**Archivo de entrada:** `data/raw/AmesHousing.csv` (inmutable)
**Archivo de salida:** `data/processed/AmesHousing.csv` (limpio)
**Notebook:** `notebooks/03_limpieza_AmesHousing.ipynb`

---

## 1. Resumen ejecutivo

| Indicador				| Antes		| Después	| Variación		|	
| Filas					| 2,930		| 2,925		| −5 (−0.17%)	|
| Columnas				| 82		| 82		| sin cambios	|
| Celdas nulas			| 15,749	| 0			| −100%			|
| % de celdas nulas		| 6.56%		| 0.00%		| −6.56 pp		|
| Columnas con nulos	| 27		| 0			| −27			|
| Duplicados exactos	| 0			| 0			| —				|
| Variables numéricas	| 39		| 39		| sin cambios	|
| Variables categóricas | 43		| 43		| sin cambios	|

**Total de operaciones registradas:** 33
**Celdas intervenidas:** 18,270

Ninguna variable fue eliminada, la limpieza se limitó a corregir contenido, no a
modificar la estructura del dataset

---

## 2. Hallazgo metodológico principal

El 93% de los valores faltantes de este dataset **no son datos perdidos**.

En la documentación original de De Cock (2011), un `NaN` en `Pool QC` significa que
la vivienda *no tiene piscina*, no que se desconozca la calidad de su piscina. Lo
mismo ocurre con callejón, cerca, chimenea, garaje, sótano y revestimiento de
mampostería. Aplicar imputación por moda o mediana sobre estas columnas habría
introducido atributos inexistentes en miles de observaciones.

La hipótesis se verificó contra una variable testigo en cada caso:

| Columna			| Nulos | Variable testigo		| Coincidencia |
| `Pool QC`			| 2,917 | `Pool Area == 0`		| 100% |
| `Fireplace Qu`	| 1,422 | `Fireplaces == 0`		| 100% |
| `Garage Type`		| 157	| `Garage Area == 0`	| 100% |
| `Bsmt Qual`		| 80	| `Total Bsmt SF == 0`	| 100% |

La coincidencia perfecta confirma que se trata de **ausencia del atributo**, no de
información perdida. Estos casos se codificaron como categoría explícita `"Ninguno"`
(categóricas) o como `0` (numéricas).

> **Nota técnica sobre la etiqueta.** No se usó `"None"`, aunque es la convención
> habitual en la literatura sobre este dataset. `pandas.read_csv()` incluye esa
> cadena en su lista de valores nulos por defecto: al exportar y volver a leer el
> archivo, la categoría se reconvertía en `NaN` y el trabajo de esta fase se perdía
> sin generar ningún error visible. La etiqueta `"Ninguno"` no colisiona con ningún
> token de valor faltante. El notebook incluye una verificación de ida y vuelta
> (sección 11) que releé el archivo exportado y confirma que conserva 0 nulos.

---

## 3. Tratamiento de valores faltantes

### 3.1 Nulos estructurales — categóricas (14 columnas)

`Pool QC`, `Misc Feature`, `Alley`, `Fence`, `Fireplace Qu`, `Garage Type`,
`Garage Finish`, `Garage Qual`, `Garage Cond`, `Bsmt Qual`, `Bsmt Cond`,
`Bsmt Exposure`, `BsmtFin Type 1`, `BsmtFin Type 2`.

- **Método:** codificación como categoría `"Ninguno"` (15,050 celdas).
- **Justificación:** la ausencia del atributo es información válida y con valor
  predictivo. Una vivienda sin sótano es un caso real del mercado, no un registro
  incompleto.

### 3.2 Nulos estructurales — numéricas (8 columnas)

`Garage Area`, `Garage Cars`, `Total Bsmt SF`, `Bsmt Unf SF`, `BsmtFin SF 1`,
`BsmtFin SF 2`, `Bsmt Full Bath`, `Bsmt Half Bath`.

- **Método:** imputación con `0`.
- **Justificación:** si la estructura no existe, su superficie es cero, no
  desconocida. Imputar la mediana habría creado sótanos inexistentes en el 2.7% de
  la muestra.

### 3.3 Revestimiento de mampostería — caso mixto

`Mas Vnr Type` presentaba 1,775 nulos, pero no todos son del mismo tipo:

- **1,768 casos** con área igual a cero o ausente → sin revestimiento.
  Tipo = `"Ninguno"`, área = `0`.
- **7 casos** con área mayor que cero → el revestimiento existe y solo falta su
  clasificación. Imputados con la moda.

Separar ambos grupos evita dos errores opuestos: inventar revestimientos donde no
los hay, y borrar los que sí existen.

### 3.4 Nulos reales — imputación justificada

| Variable					| Nulos | %			| Método									| Justificación |
| `Lot Frontage`			| 490	| 16.72%	| Mediana por `Neighborhood`				| El frente de lote lo determina el trazado 
																							  urbano del barrio. La mediana global 
																							  (68 pies) sobreestima barrios densos 
																							  como BrDale (21 pies) y subestima ClearCr 
																							  (80.5 pies) |
| `Garage Yr Blt`			| 159	| 5.43%		| `Year Built` de la vivienda				| El garaje se construye junto a la casa 
																							  salvo remodelación. Evita introducir 
																							  `0` como año, que distorsionaría cualquier 
																							  cálculo de antigüedad |
| `Garage Finish/Qual/Cond` | 1		| 0.03%		| Moda condicionada al mismo `Garage Type`	| Garaje real de 360 sqft con atributos 
																							  sin registrar. Condicionar por tipo 
																							  preserva la relación entre tipo y acabado |
| `Electrical`				| 1		| 0.03%		| Moda (`SBrkr`)							| Caso único; la moda concentra más 
																							  del 90% de las observaciones |

### 3.5 Inconsistencia interna detectada

Un registro declaraba `Garage Type = "Detchd"` sin superficie ni capacidad
asociadas. Un garaje sin área ni plazas no existe físicamente: el registro se
reclasificó como ausencia de garaje.

---

## 4. Errores de captura

| Variable | Valor detectado | Corrección | Justificación |
|---|---|---|---|
| `Garage Yr Blt` | 2207 | 2007 | Año posterior al último año de venta del dataset (2010). Error de transposición de dígitos: la vivienda se construyó en 2006 y se vendió en 2007 |

Se verificó además que ninguna variable de superficie, precio o valor contuviera
valores negativos. **Resultado: 0 casos.**

---

## 5. Estandarización de texto

Se revisaron las 43 columnas no numéricas en busca de espacios sobrantes,
inconsistencias de capitalización y categorías duplicadas por formato.

| Variable		| Problema							| Filas afectadas	| Método |
| `Sale Type`	| Valor `"WD "` con espacio final	| 2,536				| `strip()` |

Este caso es relevante porque `"WD "` y `"WD"` son cadenas distintas para pandas.
De no corregirse, el One-Hot Encoding de la Fase IV habría generado dos columnas
para la misma categoría, con el 87% de las observaciones repartido entre ambas.

También se validaron los campos de fecha: `Mo Sold` se mantiene en el rango 1–12 y
`Yr Sold` entre 2006 y 2010. No se construyó una variable de fecha combinada, por
corresponder a la creación de variables derivadas de la Fase IV.

---

## 6. Duplicados

Evaluados en tres niveles, porque `Order` y `PID` son identificadores únicos por
construcción y harían indetectable un registro repetido con distinto folio.

| Nivel							| Criterio							| Duplicados |
| Fila completa					| las 82 columnas					| 0 |
| Identificador de propiedad	| `PID`								| 0 |
| Parcial						| 80 columnas sin `Order` ni `PID`	| 0 |

**Conclusión:** el dataset no presenta duplicación. La ausencia de duplicados
parciales es el hallazgo relevante, ya que es el único de los tres niveles capaz de
detectar una doble captura de la misma vivienda con folio distinto.

---

## 7. Outliers

### 7.1 Comparativa de criterios

Se aplicaron ambos métodos sobre 13 variables continuas. Los resultados divergen de
forma sistemática:

| Variable			| IQR	| Z-score (>3)	| Skewness	| Curtosis |
| `Lot Area`		| 127	| 29			| 12.82		| 265.02 |
| `Misc Val`		| 103	| 19			| 22.00		| 566.20 |
| `Mas Vnr Area`	| 200	| 63			| 2.61		| 9.29 |
| `Lot Frontage`	| 187	| 22			| 1.50		| 11.23 |
| `SalePrice`		| 137	| 45			| 1.74		| 5.12 |
| `Gr Liv Area`		| 75	| 25			| 1.27		| 4.14 |
| `Garage Area`		| 42	| 17			| 0.24		| 0.95 |

**Interpretación:** el IQR detecta entre 2 y 5 veces más casos. La brecha crece con
la asimetría porque el Z-score depende de la media y la desviación estándar, y ambas
ya están contaminadas por los valores extremos que se pretende detectar. En
`Lot Area`, con skewness de 12.8, el Z-score identifica apenas el 23% de lo que
señala el IQR. El criterio intercuartílico es preferible en este dataset por no
asumir normalidad.

### 7.2 Decisión de tratamiento

**No se eliminaron los outliers detectados estadísticamente.** Una vivienda con lote
de 12 acres en Ames existe; no es un error de captura. Recortar por criterio
puramente estadístico habría destruido cerca del 4.7% de la muestra y sesgado el
análisis hacia la vivienda promedio, eliminando precisamente el segmento alto del
mercado que el dataset busca representar.

La única eliminación tiene fundamento **de dominio**:

| Criterio						| Filas | % de la muestra |
| `Gr Liv Area > 4,000 sqft`	| 5		| 0.17% |

De Cock (2011) documenta estas cinco viviendas como observaciones atípicas y
recomienda su exclusión. Tres corresponden a ventas en condición `Partial`
—inmuebles no terminados al momento de la transacción—, con precios por pie cuadrado
de entre 28 y 39 dólares frente a una mediana de 120 en el resto del dataset. Esos
precios no reflejan el valor de mercado del bien terminado.

Las otras dos (`Abnorml` y `Normal`, ambas con `Overall Qual = 10`) se eliminaron
por consistencia con el criterio documentado, aunque sus precios sí son coherentes
con su superficie.

La asimetría residual de las variables conservadas se atenderá mediante la elección
del escalador en la Fase IV, no mediante recorte de observaciones.

---

## 8. Efecto de la limpieza sobre la distribución

Variables donde el tratamiento modificó de forma apreciable la forma de la
distribución:

| Variable			| Media antes	| Media después | Δ %		| Skew antes	| Skew después |
| `Total Bsmt SF`	| 1,051.61		| 1,046.49		| −0.49%	| 1.156			| 0.395 |
| `BsmtFin SF 1`	| 442.63		| 437.95		| −1.06%	| 1.416			| 0.822 |
| `Lot Frontage`	| 69.22			| 69.30			| +0.11%	| 1.499			| 1.088 |
| `Mas Vnr Area`	| 101.90		| 99.92			| −1.94%	| 2.607			| 2.578 |
| `Garage Yr Blt`	| 1,978.13		| 1,976.17		| −0.10%	| −0.385		| −0.693 |

Ninguna media se desplazó más del 2%, lo que indica que la imputación **no**
introdujo sesgo apreciable**. La reducción de asimetría en `Total Bsmt SF`
(de 1.156 a 0.395) proviene de la eliminación de las cinco viviendas atípicas, no
de la imputación.

---

## 9. Reporte complementario para la Fase IV

Se identificaron 13 variables con más del 95% de concentración en un solo valor
(`Utilities` con 99.9%, `Street` con 99.6%, `Pool Area` con 99.6%, entre otras).

**No se eliminaron.** La baja varianza sugiere escaso poder discriminante, pero la
exclusión de variables pertenece a la selección de características, no a la
limpieza. El listado se entrega en `docs/reporte_baja_varianza.csv` como insumo.

---

## 10. Verificación de integridad del entregable

El notebook vuelve a leer desde disco el archivo exportado y valida cuatro
condiciones antes de darlo por bueno:

| Control										| Resultado |
| Forma idéntica a la del DataFrame en memoria	| 2,925 × 82 |
| Sin nulos tras la lectura						| 0 |
| Mismas columnas y en el mismo orden			| 82/82 |
| Categoría de ausencia conservada				| 15,050 celdas |

Este control se agregó tras detectar que la etiqueta inicial se perdía en la
exportación (ver nota técnica de la sección 2).

---

## 11. Entregables

| Archivo								| Contenido |
| `data/processed/AmesHousing.csv`		| Dataset limpio (2,925 × 82) |
| `docs/bitacora_limpieza.md`			| Este documento |
| `docs/bitacora_operaciones.csv`		| Registro detallado de las 33 operaciones |
| `docs/metricas_pre_post.csv`			| Descriptivos de las 39 numéricas, antes y después |
| `docs/variacion_por_variable.csv`		| Δ media, desviación, skewness y curtosis |
| `docs/comparativa_global.csv`			| Indicadores globales del dataset |
| `docs/reporte_outliers.csv`			| IQR vs Z-score por variable |
| `docs/reporte_baja_varianza.csv`		| Variables casi constantes |
| `docs/reporte_nulos_inicial.csv`		| Diagnóstico de nulos del crudo |

---

## 12. Referencia

De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data as an End of
Semester Regression Project*. Journal of Statistics Education, 19(3).
https://doi.org/10.1080/10691898.2011.11889627
