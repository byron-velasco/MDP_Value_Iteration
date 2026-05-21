# MDP Value Iteration Agent — 50×50 Maze

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-scientific-blue?logo=numpy&logoColor=white)
![Algorithm](https://img.shields.io/badge/Algorithm-Value_Iteration-orange)
![States](https://img.shields.io/badge/States-2500-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

> Proceso de Decisión de Markov (MDP) resuelto con Value Iteration en un laberinto de 2,500 estados.  
> Análisis de tradeoffs gamma/tolerancia, mapas Q, curvas de convergencia y extracción de política óptima.

---

## ¿Qué hace este proyecto?

Implementa un agente de Reinforcement Learning clásico basado en **Value Iteration** para encontrar la política óptima de navegación en un laberinto de 50×50 celdas con obstáculos aleatorios.

El agente aprende a maximizar la recompensa acumulada descontada — llegando a la meta en el menor número de pasos posible, evitando obstáculos y minimizando penalizaciones por movimientos innecesarios.

**Configuración del entorno:**
```
Cuadrícula: 50×50 = 2,500 estados
Inicio: S = (0, 0)      Meta: G = (49, 49)
Acciones: ↑ ↓ ← → (4 por estado)
Recompensa meta: +1      Penalización movimiento: -0.01
Penalización obstáculo/pared: -0.5
```

---

## ¿Por qué importa?

MDP + Value Iteration es el "Hello World" del Reinforcement Learning — pero implementarlo correctamente, a escala y con análisis riguroso de hiperparámetros, es lo que separa entender el concepto de dominarlo.

Lo que se trabaja aquí:

- **Representación eficiente del espacio de estados:** con 2,500 estados y 4 acciones, una matriz densa de transición requeriría 25M de entradas. Se usa representación tabular directa de `next_states` y `rewards` para mantener el problema manejable en memoria
- **Ecuación de Bellman aplicada iterativamente:** Value Iteration converge garantizadamente bajo condiciones de descuento — el proyecto muestra ese proceso de convergencia visualmente, no solo el resultado final
- **Análisis de hiperparámetros:** `gamma` y `tolerancia` tienen tradeoffs medibles en número de iteraciones y calidad de política — aquí se documentan con curvas reales, no con explicaciones teóricas
- **Interpretabilidad completa:** mapas de valores V(s), valores Q por acción y la política óptima visualizada sobre el laberinto — el agente es completamente interpretable

**Hallazgo central:** el nivel de agregación temporal (gamma) es el parámetro más determinante. Con gamma=0.90 el agente converge y encuentra rutas eficientes; con gamma=0.95 el número de iteraciones crece exponencialmente con tolerancias estrictas, evidenciando el tradeoff precisión/costo computacional.

---

## Algoritmo — Value Iteration

```
Inicializar V(s) = 0 para todos los estados
    ↓
Repetir hasta convergencia (max|V_nuevo - V_viejo| < tolerancia):
    Para cada estado s:
        V(s) = max_a [ R(s,a) + gamma · V(s') ]
    ↓
Extraer política: π(s) = argmax_a [ R(s,a) + gamma · V(s') ]
```

La **ecuación de Bellman** garantiza que iterando este proceso, V(s) converge al valor óptimo V*(s) para cualquier política greedy sobre él.

---

## Hiperparámetros analizados

| Parámetro | Rango evaluado | Efecto |
|-----------|---------------|--------|
| `gamma` (descuento) | 0.80 · 0.85 · 0.90 · 0.95 | Determina el peso de recompensas futuras; valores altos requieren más iteraciones pero producen políticas más eficientes en laberintos grandes |
| `tolerancia` | 1e-3 · 1e-5 · 1e-7 | Controla la precisión de convergencia; tolerancias estrictas aumentan iteraciones dramáticamente con gamma alto |

### Iteraciones requeridas según configuración

| Configuración | Iteraciones aprox. |
|--------------|-------------------|
| gamma=0.80, tol=1e-3 | ~25 |
| gamma=0.90, tol=1e-5 | ~130 |
| gamma=0.95, tol=1e-7 | ~305 |

---

## Outputs visuales generados

| Output | Descripción |
|--------|-------------|
| Mapa del laberinto | Cuadrícula 50×50 con inicio (verde), meta (amarillo) y obstáculos (magenta) |
| Ruta óptima | Camino trazado por el agente siguiendo la política aprendida |
| Mapa de valores V(s) | Estados cercanos a la meta con mayor valor esperado |
| Curva de convergencia | Iteraciones requeridas según gamma y tolerancia |
| Valores Q | Distribución de valores por acción en estados clave |

---

## Estructura del repositorio

```
mdp-value-iteration/
│
├── MDP_Value_Iteration.ipynb   ← Notebook principal (Colab-ready)
├── README.md
└── requirements.txt
```

---

## Instalación y ejecución

```bash
# Opción recomendada — Google Colab
# Abrir MDP_Value_Iteration.ipynb en Colab y ejecutar todas las celdas

# Local
git clone https://github.com/byron-velasco/mdp-value-iteration.git
cd mdp-value-iteration
pip install -r requirements.txt
jupyter notebook MDP_Value_Iteration.ipynb
```

**requirements.txt**
```
numpy
matplotlib
```

---

## Decisiones metodológicas

**¿Por qué representación tabular y no matriz de transición densa?**  
Una matriz densa P(s'|s,a) para 2,500 estados y 4 acciones requeriría almacenar 25 millones de probabilidades de transición. En un laberinto determinista, cada (s,a) lleva a exactamente un s' — almacenar directamente `next_state` y `reward` por par (s,a) reduce el espacio a 10,000 entradas y hace el código más legible.

**¿Por qué gamma=0.90 como configuración principal?**  
En laberintos grandes, la meta puede estar a decenas de pasos. Un gamma bajo (≤0.85) hace que el valor de la meta se desvanezca demasiado rápido al propagarse hacia los estados iniciales, produciendo políticas subóptimas que priorizan recompensas inmediatas. 0.90 balancea correctamente alcance temporal y velocidad de convergencia.

**¿Por qué incluir el análisis de gamma/tolerancia?**  
Porque Value Iteration sin análisis de hiperparámetros es una caja negra. Mostrar cómo cambian las iteraciones requeridas con distintas configuraciones convierte el notebook en una herramienta de aprendizaje real — no solo en un algoritmo que "funciona".

**¿Por qué penalizar movimientos además de obstáculos?**  
Sin penalización por movimiento, el agente es indiferente entre rutas cortas y largas que llegan a la meta. La penalización de -0.01 por paso introduce presión para encontrar caminos eficientes, produciendo políticas más realistas para aplicaciones de navegación autónoma.

---

## Licencia

MIT — código libre para uso, modificación y distribución.

