# IA Workshop - Modelos de Clasificación: Regresión Logística

## Descripción

Este repositorio contiene material educativo sobre modelos de clasificación en machine learning, específicamente centrado en **Regresión Logística**. El proyecto forma parte de un workshop de Inteligencia Artificial que enseña los fundamentos de los algoritmos de clasificación a través de ejemplos prácticos, explicaciones detalladas y visualizaciones interactivas.

## Fundamentos Teóricos

### ¿Qué es la Regresión Logística?

La **regresión logística** es un método de clasificación en machine learning que se utiliza para predecir si algo pertenece a una categoría (clase) o a otra. A diferencia de la regresión lineal, que predice un número (como la calificación que obtendrás en un examen), la regresión logística predice una probabilidad, es decir, la probabilidad de que algo ocurra o no.

**Analogía del mundo real**: Imagina que tienes que predecir si un estudiante va a aprobar o no un examen basándote en el número de horas que ha estudiado. La regresión logística te ayuda a determinar la probabilidad de que apruebe, y a clasificar el resultado en "aprobado" o "no aprobado".

### La Función Sigmoide

A nivel matemático, la regresión logística utiliza la función logística o sigmoide:

```
P(Y=1) = 1 / (1 + e^-(β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ))
```

**Donde:**
- `P(Y=1)` es la probabilidad de que el resultado sea la clase 1
- `β₀` es el término de sesgo o intercepto
- `β₁, β₂, ..., βₙ` son los coeficientes que se aplican a cada una de las características
- `X₁, X₂, ..., Xₙ` son las características (en nuestro ejemplo solo hay una: horas estudiadas)
- `e` es la base del logaritmo natural, aproximadamente igual a 2.718

**¿Por qué la función sigmoide?** Esta función tiene una forma de "S" que transforma cualquier número real en un valor entre 0 y 1, lo que es perfecto para representar probabilidades. Es como un interruptor suave que gradualmente pasa de "imposible" (0%) a "seguro" (100%).

## Ejercicio Paso a Paso: Predicción de Aprobación de Exámenes

### Paso 1: Entender el Problema

**Objetivo**: Predecir si un estudiante pasará un examen dependiendo de cuántas horas ha estudiado.

**La lógica**: Si ha estudiado más horas, es más probable que pase. Sin embargo, esto no es una relación lineal simple; estudiar una hora podría no tener un gran impacto, pero estudiar diez horas podría casi garantizar el éxito.

**Entrada (X)**: Número de horas estudiadas  
**Salida (Y)**: Resultado del examen (0 = No pasa, 1 = Pasa)

### Paso 2: Importar las Librerías Necesarias

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
```

**¿Por qué estas librerías?**
- **NumPy**: Para manejar arrays numéricos de forma eficiente
- **Pandas**: Para crear tablas (DataFrames) y manipular datos
- **Matplotlib**: Para crear gráficos y visualizaciones
- **Scikit-learn**: Contiene el algoritmo de regresión logística ya implementado

### Paso 3: Crear un Dataset Ficticio (Escenario Ideal)

```python
# Datos simulados: horas estudiadas y si pasaron o no el examen
horas_estudiadas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
resultado_examen = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
```

**¿Qué significan estos datos?**
- Los estudiantes que estudiaron 4 horas o menos NO pasaron el examen (0)
- Los estudiantes que estudiaron 5 horas o más SÍ pasaron el examen (1)
- Este es un escenario "ideal" porque hay una división clara

**Metáfora**: Es como un interruptor que se activa exactamente en 5 horas. En la vida real, esto raramente ocurre tan claramente.

### Paso 4: Entrenar el Modelo

```python
# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(horas_estudiadas.reshape(-1, 1), resultado_examen)
```

**¿Qué hace `reshape(-1, 1)`?**
El algoritmo de scikit-learn espera que los datos tengan una forma específica. Imagina que tienes una fila de números `[1, 2, 3, 4, 5]` pero el algoritmo necesita una columna:
```
[1]
[2]
[3]
[4]
[5]
```
El `reshape(-1, 1)` convierte la fila en una columna. El `-1` significa "calcula automáticamente cuántas filas necesitas" y el `1` significa "una columna".

**¿Qué pasa durante el entrenamiento?**
El algoritmo ajusta automáticamente los coeficientes β₀ (intercepto) y β₁ (pendiente) para que la curva sigmoide se ajuste lo mejor posible a nuestros datos. Es como encontrar la mejor curva en "S" que pase cerca de todos nuestros puntos.

### Paso 5: Hacer Predicciones

```python
# Predecir probabilidades para una gama de horas estudiadas
horas_nuevas = np.linspace(0, 12, 1000).reshape(-1, 1)
probabilidades = modelo.predict_proba(horas_nuevas)[:, 1]
```

**¿Qué hace `np.linspace(0, 12, 1000)`?**
Crea 1000 puntos uniformemente distribuidos entre 0 y 12 horas. Es como tener una regla muy precisa que mide cada milímetro entre 0 y 12.

**¿Qué hace `predict_proba`?**
Calcula la probabilidad de aprobar para cada una de esas 1000 horas diferentes. El `[:, 1]` toma solo las probabilidades de la clase 1 (aprobar).

### Paso 6: Visualizar los Resultados

```python
# Gráfico de los datos de entrenamiento
plt.scatter(horas_estudiadas, resultado_examen, color='red', label='Datos reales')
plt.xlabel('Horas Estudiadas')
plt.ylabel('Resultado del Examen')
plt.title('Regresión Logística: Muestra de entrenamiento')
plt.legend()
plt.show()

# Gráfico con la curva de probabilidad
plt.scatter(horas_estudiadas, resultado_examen, color='red', label='Datos reales')
plt.plot(horas_nuevas, probabilidades, color='blue', label='Probabilidad de pasar')
plt.xlabel('Horas Estudiadas')
plt.ylabel('Probabilidad de Pasar')
plt.title('Regresión Logística: Predicción de Resultados en un Examen')
plt.legend()
plt.show()
```

### Paso 7: Examinar los Coeficientes

```python
print("Intercepto (β0):", modelo.intercept_[0])
print("Coeficiente (β1):", modelo.coef_[0][0])
```

**¿Qué significan estos números?**

- **Intercepto (β₀)**: Es el punto de partida de nuestra ecuación. Un número negativo significa que cuando las horas de estudio son 0, la probabilidad de aprobar es muy baja.

- **Coeficiente (β₁)**: Es la "fuerza" del efecto. Un número positivo significa que más horas de estudio aumentan la probabilidad de aprobar. Un número más grande significa un efecto más fuerte.

**Metáfora**: El intercepto es como el nivel base de dificultad del examen, y el coeficiente es como qué tanto ayuda cada hora adicional de estudio.

### Paso 8: Crear una Tabla de Probabilidades

```python
# Crear un DataFrame con las probabilidades
probabilidades_originales = modelo.predict_proba(horas_estudiadas.reshape(-1, 1))[:, 1]
datos = pd.DataFrame({
    'Horas estudiadas': horas_estudiadas,
    'Probabilidad de aprobar': probabilidades_originales
})
print(datos)
```

Esta tabla muestra exactamente qué probabilidad de aprobar tiene cada estudiante según las horas que estudió.

## Interpretación de Resultados del Primer Ejemplo

### Análisis del Gráfico

**Puntos rojos (Datos reales)**: Representan los datos de entrenamiento. Los estudiantes que estudiaron 4 horas o menos no pasaron (etiquetados con 0), mientras que aquellos que estudiaron 5 horas o más pasaron (etiquetados con 1).

**Línea azul (Curva de probabilidad)**: Muestra la predicción de nuestro modelo. Esta es la famosa curva sigmoide en acción.

### ¿Qué nos dice la curva?

- **Con 1-2 horas de estudio**: La probabilidad es muy baja (menos del 5%)
- **Con 5-6 horas de estudio**: La probabilidad está alrededor del 50-85%
- **Con 8-10 horas de estudio**: La probabilidad es muy alta (más del 95%)

**La zona de transición**: Entre las 4 y 6 horas hay una zona donde la probabilidad cambia rápidamente. Esta es la característica más importante de la función sigmoide.

## Segundo Ejemplo: Datos con Variabilidad Real

### ¿Qué es la Variabilidad?

En el mundo real, los datos no son tan "perfectos" como en nuestro primer ejemplo. La **variabilidad** significa que hay excepciones y casos inesperados.

**Metáfora de la vida real**: Imagina que estás prediciendo si alguien llegará a tiempo al trabajo basándote en a qué hora sale de casa. En un mundo ideal, quien sale temprano siempre llega a tiempo. En el mundo real, puede haber tráfico, el metro puede retrasarse, o alguien puede tomar un atajo que no conocías.

### Ejercicio Paso a Paso con Variabilidad

#### Paso 1: Crear Datos con Más Variabilidad

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Nuevos datos simulados: horas estudiadas y si pasaron o no el examen (con más variabilidad)
horas_estudiadas = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 8, 5, 7, 9])
resultado_examen = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])
```

**¿Qué vemos aquí de diferente?**
- Un estudiante que estudió solo 1 hora no aprobó (normal)
- Un estudiante que estudió 8 horas no aprobó (inesperado)
- Un estudiante que estudió 5 horas sí aprobó (esperado)
- Un estudiante que estudió 9 horas no aprobó (muy inesperado)

**¿Por qué puede pasar esto en la vida real?**
- El examen era más difícil de lo esperado
- El estudiante estudió de forma ineficiente
- El estudiante tuvo un mal día
- El estudiante ya tenía conocimientos previos
- Problemas de salud, estrés, etc.

#### Paso 2: Entrenar el Modelo con los Nuevos Datos

```python
# Crear y entrenar el modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(horas_estudiadas.reshape(-1, 1), resultado_examen)

# Predecir probabilidades para una gama de horas estudiadas
horas_nuevas = np.linspace(0, 10, 1000).reshape(-1, 1)
probabilidades = modelo.predict_proba(horas_nuevas)[:, 1]
```

**¿Qué está pasando aquí?**
El modelo ahora tiene que ajustarse a datos más "ruidosos". Ya no puede hacer una división perfecta como en el ejemplo anterior. Tiene que encontrar la mejor curva posible que tenga en cuenta todas las excepciones.

#### Paso 3: Visualizar los Resultados con Variabilidad

```python
# Gráfico de los resultados
plt.scatter(horas_estudiadas, resultado_examen, color='red', label='Datos reales')
plt.plot(horas_nuevas, probabilidades, color='blue', label='Probabilidad de pasar')
plt.xlabel('Horas Estudiadas')
plt.ylabel('Probabilidad de Pasar')
plt.title('Regresión Logística: Predicción de Resultados en un Examen con Variabilidad')
plt.legend()
plt.show()
```

**¿Qué observamos en el gráfico?**
- Los puntos rojos están más "dispersos"
- La línea azul es menos pronunciada (menos empinada)
- No hay una transición tan clara como antes

#### Paso 4: Examinar los Nuevos Coeficientes

```python
print("Intercepto (β0):", modelo.intercept_[0])
print("Coeficiente (β1):", modelo.coef_[0][0])
```

**Resultados esperados**:
- Intercepto (β0): -2.37 (aproximadamente)
- Coeficiente (β1): 0.44 (aproximadamente)

**¿Por qué son diferentes a los del primer ejemplo?**
- **Intercepto más alto**: Como hay estudiantes que aprueban con pocas horas, el modelo es "menos pesimista"
- **Coeficiente más bajo**: Como el efecto de estudiar más horas es menos consistente, cada hora adicional tiene menos impacto garantizado

#### Paso 5: Crear Tabla de Probabilidades con Variabilidad

```python
import pandas as pd

# Obtener probabilidades para las horas estudiadas originales
probabilidades_originales = modelo.predict_proba(horas_estudiadas.reshape(-1, 1))[:, 1]

# Crear un DataFrame con los datos
datos = pd.DataFrame({
    'Horas estudiadas': horas_estudiadas,
    'Probabilidad de aprobar': probabilidades_originales
})

datos_ordenados = datos.sort_values(by=['Probabilidad de aprobar'])
print(datos_ordenados)
```

**¿Qué vemos en esta tabla?**
- Menos diferencia entre las probabilidades
- Rango de probabilidades más estrecho
- Mayor incertidumbre en general

### Interpretación de Resultados con Variabilidad

**Análisis de los Datos Reales con Variabilidad**:
- Los puntos rojos representan datos con más variabilidad
- **Algunos estudiantes aprobaron con pocas horas de estudio**
- **Algunos estudiantes no aprobaron a pesar de estudiar muchas horas**
- **Otros resultados son más esperables**: La mayoría de los estudiantes con 8 o más horas de estudio tienden a aprobar

**Curva de Probabilidad Ajustada**:
- La línea azul muestra la probabilidad ajustada a estos nuevos datos más realistas
- **Curva menos pronunciada**: Debido a la variabilidad, la curva es menos pronunciada que en el ejemplo anterior
- **Rango de incertidumbre más amplio**: La probabilidad de aprobar no sube tan rápidamente con las horas de estudio
- Hay una mayor zona de "incertidumbre" donde la probabilidad está cerca del 50%

**Interpretación Final**:
- **Influencia de la Variabilidad**: Este ejemplo muestra que en situaciones reales, no siempre hay una relación clara y directa entre dos variables
- **Otros factores influyen**: Como el nivel de dificultad del examen, cómo el estudiante estudió, etc.
- **Probabilidad en Decisiones Reales**: El modelo aún proporciona una probabilidad útil para la toma de decisiones
- **Umbral de decisión**: Si la probabilidad de aprobar es mayor al 70%, podrías considerarlo un buen indicador

**¿Qué aprendemos de esto?**
Este ejemplo destaca cómo la regresión logística maneja situaciones con datos más variados y menos predecibles, reflejando mejor los escenarios del mundo real. Los modelos perfectos rara vez existen en la práctica; la clave está en entender y trabajar con la incertidumbre.


## Conceptos Técnicos Explicados de Forma Simple

### ¿Por qué usar Reshape?

**El problema**: Los algoritmos de machine learning son como máquinas muy específicas que esperan que les entregues los datos de una forma exacta.

**La analogía**: Es como una máquina expendedora que solo acepta monedas puestas de cierta manera. Nuestros datos pueden estar en una "fila" pero la máquina necesita una "columna".

**La solución**: `reshape(-1, 1)` reorganiza nuestros datos del formato que tenemos al formato que necesita la máquina.

### ¿Qué significa predict_proba?

**predict_proba** devuelve dos columnas:
- Columna 0: Probabilidad de NO aprobar
- Columna 1: Probabilidad de SÍ aprobar

Cuando usamos `[:, 1]` estamos diciendo "dame solo la segunda columna" (probabilidad de aprobar).

### ¿Cómo interpretar los coeficientes?

**Intercepto negativo**: Significa que sin estudiar nada, la probabilidad de aprobar es muy baja.

**Coeficiente positivo**: Significa que cada hora adicional de estudio aumenta las posibilidades.

**Ejemplo práctico**: Si β₁ = 1.18, significa que por cada hora adicional de estudio, el "logaritmo de las probabilidades" aumenta en 1.18. En términos simples: estudiar más ayuda significativamente.

## Aplicaciones en el Mundo Real

La regresión logística se usa en muchísimas situaciones reales:

### Marketing
- **Problema**: ¿Comprará un cliente este producto?
- **Variables**: Edad, ingresos, historial de compras
- **Resultado**: Probabilidad de compra (0% a 100%)

### Medicina
- **Problema**: ¿Tiene el paciente esta enfermedad?
- **Variables**: Síntomas, edad, historial médico
- **Resultado**: Probabilidad de tener la enfermedad

### Tecnología
- **Problema**: ¿Es este email spam?
- **Variables**: Palabras clave, remitente, enlaces
- **Resultado**: Probabilidad de ser spam

### Finanzas
- **Problema**: ¿Pagará el cliente su préstamo?
- **Variables**: Ingresos, historial crediticio, empleo
- **Resultado**: Probabilidad de pago

## Lecciones Clave del Workshop

### 1. La Importancia de la Variabilidad
Los datos del mundo real nunca son perfectos. Siempre hay excepciones y casos inesperados. Un buen modelo debe poder manejar esta incertidumbre.

### 2. Probabilidades vs Predicciones Absolutas
La regresión logística no dice "sí" o "no" rotundamente. Te da una probabilidad que puedes usar para tomar decisiones informadas.

### 3. La Visualización es Clave
Los gráficos nos ayudan a entender qué está haciendo realmente nuestro modelo. Sin visualización, los números solos pueden ser engañosos.

### 4. Los Algoritmos Son Herramientas
La regresión logística es como una calculadora muy sofisticada. La parte inteligente está en saber qué datos usar y cómo interpretar los resultados.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **NumPy**: Operaciones numéricas y manejo de arrays
- **Matplotlib**: Visualización de datos y gráficos
- **Scikit-learn**: Implementación del algoritmo de regresión logística
- **Pandas**: Manipulación y análisis de datos
- **Jupyter Notebook**: Entorno de desarrollo interactivo

## Estructura del Proyecto

```
IA-Workshop-Modelos-Clasificacion/
├── ejemplo_regresion_logistica.ipynb    # Notebook principal con ejemplos
├── README.md                            # Este archivo
└── probabilidades_regresion_logistica.csv  # Archivo generado con resultados
```

## Objetivos de Aprendizaje

Al completar este workshop, serás capaz de:

1. **Entender** qué es la regresión logística y cuándo usarla
2. **Implementar** modelos básicos usando Python y Scikit-learn
3. **Interpretar** probabilidades y coeficientes del modelo
4. **Visualizar** datos de clasificación y entender las curvas sigmoide
5. **Reconocer** el impacto de la variabilidad en datos reales
6. **Aplicar** estos conceptos a problemas del mundo real

## Requisitos

Para ejecutar el notebook necesitas:
- Python 3.7+
- Jupyter Notebook o JupyterLab
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Instrucciones de Uso

1. Clona este repositorio
2. Instala las dependencias: `pip install numpy matplotlib scikit-learn pandas jupyter`
3. Abre `ejemplo_regresion_logistica.ipynb` en Jupyter
4. Ejecuta las celdas secuencialmente para ver los ejemplos
5. Experimenta modificando los datos para observar diferentes comportamientos

## Próximos Pasos

Después de este workshop, puedes explorar:
- Regresión logística con múltiples variables
- Otros algoritmos de clasificación (SVM, Random Forest)
- Validación cruzada y métricas de evaluación
- Preprocesamiento de datos reales
- Clasificación multiclase (más de 2 categorías)

## Autor

Este material forma parte del programa de workshops de Inteligencia Artificial desarrollado para Factoria F5 Madrid.

## Licencia

Material educativo de uso libre para fines académicos y de aprendizaje.