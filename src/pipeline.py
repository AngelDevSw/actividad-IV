"""Actividad IV: comparación de selección y cruce en un algoritmo genético.

El programa selecciona características del conjunto Breast Cancer Wisconsin
(Diagnostic) incluido en scikit-learn.  Compara nueve configuraciones: tres
operadores de selección y tres operadores de cruce.  La comparación se hace
solamente con validación cruzada sobre el conjunto de entrenamiento; el
conjunto de prueba se reserva para la evaluación final de la configuración
ganadora.

Ejecución local desde la raíz del proyecto:
    python src/pipeline.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Parámetros experimentales. Se pueden modificar desde un único lugar.
# ---------------------------------------------------------------------------
SEMILLA = 42
TAMANO_POBLACION = 30
NUMERO_GENERACIONES = 50
TAMANO_TORNEO = 3
PROBABILIDAD_CRUCE = 0.8
PROBABILIDAD_MUTACION = 1 / 30
LAMBDA_PENALIZACION = 0.02
NUMERO_ELITES = 2
NUMERO_REPETICIONES = 3
SEMILLAS_EXPERIMENTO = [42, 43, 44]
NUMERO_FOLDS_CV = 5
EPSILON_RULETA = 1e-12
INTERVALO_REPORTE_GENERACION = 5

Evaluacion = dict[str, float | int]
FuncionSeleccion = Callable[[np.ndarray, list[Evaluacion], np.random.Generator], np.ndarray]
FuncionCruce = Callable[[np.ndarray, np.ndarray, np.random.Generator], tuple[np.ndarray, np.ndarray]]


def validar_cromosoma(cromosoma: np.ndarray, numero_caracteristicas: int) -> None:
    """Comprueba que una máscara binaria sea válida y no esté vacía."""
    assert cromosoma.ndim == 1, "El cromosoma debe tener una sola dimensión."
    assert len(cromosoma) == numero_caracteristicas, "Longitud de cromosoma inválida."
    assert np.all(np.isin(cromosoma, [0, 1])), "Los genes deben ser binarios."
    assert bool(cromosoma.any()), "Un individuo no puede tener todos los genes en cero."


def validar_poblacion(poblacion: np.ndarray, numero_caracteristicas: int) -> None:
    """Valida tamaño, forma y contenido de una población."""
    assert poblacion.shape == (TAMANO_POBLACION, numero_caracteristicas), (
        "La población no tiene el tamaño esperado."
    )
    for individuo in poblacion:
        validar_cromosoma(individuo, numero_caracteristicas)


def mostrar_informacion_dataset(X: pd.DataFrame, y: pd.Series) -> None:
    """Imprime el resumen solicitado del conjunto de datos."""
    faltantes = int(X.isna().sum().sum()) + int(y.isna().sum())
    print("\n================ INFORMACIÓN DEL DATASET ================")
    print(f"Número total de registros: {len(X)}")
    print(f"Número de características: {X.shape[1]}")
    print(f"Casos benignos (0): {(y == 0).sum()}")
    print(f"Casos malignos (1): {(y == 1).sum()}")
    print(f"Valores faltantes: {faltantes}")


# ---------------------------------------------------------------------------
# ETAPA 1: INICIALIZACIÓN — inicialización binaria aleatoria.
# ---------------------------------------------------------------------------
def crear_individuo(
    generador: np.random.Generator, numero_caracteristicas: int
) -> np.ndarray:
    """Crea un cromosoma binario con al menos una característica activa."""
    individuo = generador.integers(0, 2, size=numero_caracteristicas, dtype=int)
    if not individuo.any():
        individuo[generador.integers(numero_caracteristicas)] = 1
    validar_cromosoma(individuo, numero_caracteristicas)
    return individuo


def crear_poblacion(
    generador: np.random.Generator,
    tamano_poblacion: int,
    numero_caracteristicas: int,
) -> np.ndarray:
    """Crea la población inicial usando inicialización binaria aleatoria."""
    assert tamano_poblacion == TAMANO_POBLACION, "Use el tamaño de población configurado."
    poblacion = np.array(
        [crear_individuo(generador, numero_caracteristicas) for _ in range(tamano_poblacion)],
        dtype=int,
    )
    validar_poblacion(poblacion, numero_caracteristicas)
    return poblacion


# ---------------------------------------------------------------------------
# ETAPA 2: EVALUACIÓN — F1 de CV y penalización por número de variables.
# ---------------------------------------------------------------------------
def crear_modelo_svm(semilla: int) -> Pipeline:
    """Construye el pipeline; el escalado queda contenido en cada fold."""
    return Pipeline(
        [
            ("escalador", StandardScaler()),
            ("svm", SVC(kernel="linear", random_state=semilla)),
        ]
    )


def evaluar_individuo(
    individuo: np.ndarray,
    X_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    validador_cruzado: StratifiedKFold,
    cache_evaluaciones: dict[tuple[int, ...], Evaluacion],
    semilla: int,
) -> Evaluacion:
    """Evalúa un cromosoma una sola vez por ejecución mediante un caché."""
    numero_caracteristicas_totales = X_entrenamiento.shape[1]
    validar_cromosoma(individuo, numero_caracteristicas_totales)
    clave = tuple(individuo.astype(int).tolist())
    if clave in cache_evaluaciones:
        return cache_evaluaciones[clave].copy()

    indices_seleccionados = np.flatnonzero(individuo)
    modelo = crear_modelo_svm(semilla)
    puntajes_f1 = cross_val_score(
        modelo,
        X_entrenamiento.iloc[:, indices_seleccionados],
        y_entrenamiento,
        cv=validador_cruzado,
        scoring="f1",
        n_jobs=None,
    )
    f1_promedio = float(np.mean(puntajes_f1))
    numero_seleccionadas = int(len(indices_seleccionados))
    fitness = f1_promedio - LAMBDA_PENALIZACION * (
        numero_seleccionadas / numero_caracteristicas_totales
    )
    assert np.isfinite(f1_promedio) and np.isfinite(fitness), "Evaluación no numérica."

    resultado: Evaluacion = {
        "f1": f1_promedio,
        "fitness": float(fitness),
        "num_caracteristicas": numero_seleccionadas,
    }
    cache_evaluaciones[clave] = resultado
    return resultado.copy()


def evaluar_poblacion(
    poblacion: np.ndarray,
    X_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    validador_cruzado: StratifiedKFold,
    cache_evaluaciones: dict[tuple[int, ...], Evaluacion],
    semilla: int,
) -> list[Evaluacion]:
    """Evalúa los individuos de una población."""
    return [
        evaluar_individuo(
            individuo,
            X_entrenamiento,
            y_entrenamiento,
            validador_cruzado,
            cache_evaluaciones,
            semilla,
        )
        for individuo in poblacion
    ]


def fitnesses(evaluaciones: list[Evaluacion]) -> np.ndarray:
    """Obtiene el vector de fitness y verifica que sea numérico."""
    valores = np.asarray([float(evaluacion["fitness"]) for evaluacion in evaluaciones])
    assert len(valores) == len(evaluaciones) and np.all(np.isfinite(valores))
    return valores


def validar_probabilidades(probabilidades: np.ndarray) -> None:
    """Verifica las probabilidades empleadas por ruleta y ranking."""
    assert np.all(np.isfinite(probabilidades)), "Las probabilidades deben ser finitas."
    assert np.all(probabilidades >= 0), "No se permiten probabilidades negativas."
    assert np.isclose(probabilidades.sum(), 1.0), "Las probabilidades deben sumar 1."


# ---------------------------------------------------------------------------
# ETAPA 3: SELECCIÓN — los tres operadores comparados.
# ---------------------------------------------------------------------------
def seleccion_torneo(
    poblacion: np.ndarray,
    evaluaciones: list[Evaluacion],
    generador: np.random.Generator,
) -> np.ndarray:
    """Elige el mejor de ``TAMANO_TORNEO`` individuos tomados al azar."""
    assert len(poblacion) >= TAMANO_TORNEO
    candidatos = generador.choice(len(poblacion), size=TAMANO_TORNEO, replace=False)
    ganador = max(candidatos, key=lambda indice: float(evaluaciones[indice]["fitness"]))
    return poblacion[int(ganador)].copy()


def seleccion_ruleta(
    poblacion: np.ndarray,
    evaluaciones: list[Evaluacion],
    generador: np.random.Generator,
) -> np.ndarray:
    """Selecciona proporcionalmente al fitness ajustado para evitar negativos."""
    valores_fitness = fitnesses(evaluaciones)
    fitness_ajustado = valores_fitness - valores_fitness.min() + EPSILON_RULETA
    probabilidades = fitness_ajustado / fitness_ajustado.sum()
    validar_probabilidades(probabilidades)
    indice = int(generador.choice(len(poblacion), p=probabilidades))
    return poblacion[indice].copy()


def seleccion_ranking(
    poblacion: np.ndarray,
    evaluaciones: list[Evaluacion],
    generador: np.random.Generator,
) -> np.ndarray:
    """Selecciona por rangos; el mejor fitness recibe el rango más alto."""
    valores_fitness = fitnesses(evaluaciones)
    orden_ascendente = np.argsort(valores_fitness, kind="stable")
    rangos = np.empty(len(poblacion), dtype=float)
    rangos[orden_ascendente] = np.arange(1, len(poblacion) + 1, dtype=float)
    probabilidades = rangos / rangos.sum()
    validar_probabilidades(probabilidades)
    indice = int(generador.choice(len(poblacion), p=probabilidades))
    return poblacion[indice].copy()


METODOS_SELECCION: dict[str, FuncionSeleccion] = {
    "Torneo": seleccion_torneo,
    "Ruleta": seleccion_ruleta,
    "Ranking": seleccion_ranking,
}


# ---------------------------------------------------------------------------
# ETAPA 4: CRUCE — los tres operadores comparados.
# ---------------------------------------------------------------------------
def sin_cruce_si_corresponde(
    padre_1: np.ndarray,
    padre_2: np.ndarray,
    generador: np.random.Generator,
) -> bool:
    """Devuelve si se omite el cruce según la probabilidad global común."""
    return bool(generador.random() >= PROBABILIDAD_CRUCE)


def cruce_un_punto(
    padre_1: np.ndarray, padre_2: np.ndarray, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Intercambia los segmentos posteriores a un punto de corte válido."""
    assert len(padre_1) == len(padre_2) >= 2
    if sin_cruce_si_corresponde(padre_1, padre_2, generador):
        return padre_1.copy(), padre_2.copy()
    punto = int(generador.integers(1, len(padre_1)))
    assert 0 < punto < len(padre_1)
    hijo_1 = np.concatenate((padre_1[:punto], padre_2[punto:])).astype(int)
    hijo_2 = np.concatenate((padre_2[:punto], padre_1[punto:])).astype(int)
    assert len(hijo_1) == len(padre_1) and len(hijo_2) == len(padre_1)
    return hijo_1, hijo_2


def cruce_dos_puntos(
    padre_1: np.ndarray, padre_2: np.ndarray, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Intercambia el segmento comprendido entre dos puntos de corte."""
    assert len(padre_1) == len(padre_2) >= 3
    if sin_cruce_si_corresponde(padre_1, padre_2, generador):
        return padre_1.copy(), padre_2.copy()
    punto_1, punto_2 = sorted(generador.choice(np.arange(1, len(padre_1)), size=2, replace=False))
    punto_1, punto_2 = int(punto_1), int(punto_2)
    assert 0 < punto_1 < punto_2 < len(padre_1), "Puntos de corte inválidos."
    hijo_1 = padre_1.copy()
    hijo_2 = padre_2.copy()
    hijo_1[punto_1:punto_2] = padre_2[punto_1:punto_2]
    hijo_2[punto_1:punto_2] = padre_1[punto_1:punto_2]
    assert len(hijo_1) == len(padre_1) and len(hijo_2) == len(padre_1)
    return hijo_1.astype(int), hijo_2.astype(int)


def cruce_uniforme(
    padre_1: np.ndarray, padre_2: np.ndarray, generador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Combina ambos padres con una máscara independiente de probabilidad 0.5."""
    assert len(padre_1) == len(padre_2)
    if sin_cruce_si_corresponde(padre_1, padre_2, generador):
        return padre_1.copy(), padre_2.copy()
    mascara = generador.random(len(padre_1)) < 0.5
    hijo_1 = np.where(mascara, padre_1, padre_2).astype(int)
    hijo_2 = np.where(mascara, padre_2, padre_1).astype(int)
    assert len(hijo_1) == len(padre_1) and len(hijo_2) == len(padre_1)
    return hijo_1, hijo_2


METODOS_CRUCE: dict[str, FuncionCruce] = {
    "Un punto": cruce_un_punto,
    "Dos puntos": cruce_dos_puntos,
    "Uniforme": cruce_uniforme,
}


# ---------------------------------------------------------------------------
# ETAPA 5: MUTACIÓN — bit-flip con la misma tasa para todos los experimentos.
# ---------------------------------------------------------------------------
def mutacion_bit_flip(
    individuo: np.ndarray, generador: np.random.Generator
) -> np.ndarray:
    """Invierte cada bit con probabilidad 1 / número de características."""
    hijo = individuo.copy().astype(int)
    probabilidad_mutacion = 1 / len(hijo)
    # La constante documenta el valor inicial solicitado (1/30).
    assert np.isclose(probabilidad_mutacion, PROBABILIDAD_MUTACION) or len(hijo) != 30
    genes_a_mutar = generador.random(len(hijo)) < probabilidad_mutacion
    hijo[genes_a_mutar] = 1 - hijo[genes_a_mutar]
    if not hijo.any():
        hijo[generador.integers(len(hijo))] = 1
    validar_cromosoma(hijo, len(hijo))
    return hijo


# ---------------------------------------------------------------------------
# ETAPA 6: REEMPLAZO — generacional con elitismo fijo.
# ---------------------------------------------------------------------------
def reemplazo_generacional_con_elitismo(
    poblacion: np.ndarray,
    evaluaciones: list[Evaluacion],
    generador: np.random.Generator,
    funcion_seleccion: FuncionSeleccion,
    funcion_cruce: FuncionCruce,
) -> np.ndarray:
    """Conserva las élites y genera el resto usando los operadores recibidos."""
    numero_caracteristicas = poblacion.shape[1]
    indices_elite = np.argsort(fitnesses(evaluaciones))[::-1][:NUMERO_ELITES]
    nueva_poblacion = [poblacion[int(indice)].copy() for indice in indices_elite]

    while len(nueva_poblacion) < TAMANO_POBLACION:
        padre_1 = funcion_seleccion(poblacion, evaluaciones, generador)
        padre_2 = funcion_seleccion(poblacion, evaluaciones, generador)
        hijo_1, hijo_2 = funcion_cruce(padre_1, padre_2, generador)
        nueva_poblacion.append(mutacion_bit_flip(hijo_1, generador))
        if len(nueva_poblacion) < TAMANO_POBLACION:
            nueva_poblacion.append(mutacion_bit_flip(hijo_2, generador))

    resultado = np.asarray(nueva_poblacion, dtype=int)
    validar_poblacion(resultado, numero_caracteristicas)
    return resultado


# ---------------------------------------------------------------------------
# ETAPA 7: PARO — número máximo de generaciones.
# ---------------------------------------------------------------------------
def ejecutar_algoritmo_genetico(
    X_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    funcion_seleccion: FuncionSeleccion,
    funcion_cruce: FuncionCruce,
    semilla: int,
    etiqueta_experimento: str = "AG",
) -> tuple[np.ndarray, Evaluacion, list[dict[str, float | int]]]:
    """Ejecuta un motor genérico de AG con operadores intercambiables."""
    generador = np.random.default_rng(semilla)
    numero_caracteristicas = X_entrenamiento.shape[1]
    poblacion = crear_poblacion(generador, TAMANO_POBLACION, numero_caracteristicas)
    validador_cruzado = StratifiedKFold(
        n_splits=NUMERO_FOLDS_CV, shuffle=True, random_state=semilla
    )
    cache_evaluaciones: dict[tuple[int, ...], Evaluacion] = {}
    mejor_individuo_global: np.ndarray | None = None
    mejor_resultado_global: Evaluacion | None = None
    historial: list[dict[str, float | int]] = []

    for generacion in range(1, NUMERO_GENERACIONES + 1):
        evaluaciones = evaluar_poblacion(
            poblacion,
            X_entrenamiento,
            y_entrenamiento,
            validador_cruzado,
            cache_evaluaciones,
            semilla,
        )
        valores_fitness = fitnesses(evaluaciones)
        indice_mejor = int(np.argmax(valores_fitness))
        mejor_generacion = evaluaciones[indice_mejor]

        if (
            mejor_resultado_global is None
            or float(mejor_generacion["fitness"]) > float(mejor_resultado_global["fitness"])
        ):
            mejor_individuo_global = poblacion[indice_mejor].copy()
            mejor_resultado_global = mejor_generacion.copy()
            mejor_resultado_global["generacion_mejor"] = generacion

        assert mejor_resultado_global is not None
        fila_historial: dict[str, float | int] = {
            "generacion": generacion,
            "mejor_fitness": float(mejor_generacion["fitness"]),
            "f1_mejor_individuo": float(mejor_generacion["f1"]),
            "fitness_promedio": float(np.mean(valores_fitness)),
            "numero_caracteristicas": int(mejor_generacion["num_caracteristicas"]),
            "mejor_fitness_global": float(mejor_resultado_global["fitness"]),
        }
        historial.append(fila_historial)

        if generacion == 1 or generacion % INTERVALO_REPORTE_GENERACION == 0 or generacion == NUMERO_GENERACIONES:
            print(
                f"{etiqueta_experimento} | Generación {generacion:02d} | "
                f"Mejor fitness: {mejor_generacion['fitness']:.4f} | "
                f"F1: {mejor_generacion['f1']:.4f} | "
                f"Características: {mejor_generacion['num_caracteristicas']}"
            )

        if generacion < NUMERO_GENERACIONES:
            poblacion = reemplazo_generacional_con_elitismo(
                poblacion,
                evaluaciones,
                generador,
                funcion_seleccion,
                funcion_cruce,
            )

    assert mejor_individuo_global is not None and mejor_resultado_global is not None
    assert len(historial) == NUMERO_GENERACIONES
    print(f"{etiqueta_experimento} | Cromosomas únicos evaluados: {len(cache_evaluaciones)}")
    return mejor_individuo_global, mejor_resultado_global, historial


def crear_configuraciones() -> list[tuple[str, str, str, FuncionSeleccion, FuncionCruce]]:
    """Genera E1--E9 en el orden establecido para la comparación."""
    configuraciones = []
    numero_experimento = 1
    for nombre_seleccion, funcion_seleccion in METODOS_SELECCION.items():
        for nombre_cruce, funcion_cruce in METODOS_CRUCE.items():
            configuraciones.append(
                (
                    f"E{numero_experimento}",
                    nombre_seleccion,
                    nombre_cruce,
                    funcion_seleccion,
                    funcion_cruce,
                )
            )
            numero_experimento += 1
    assert len(METODOS_SELECCION) == 3
    assert len(METODOS_CRUCE) == 3
    assert len(configuraciones) == 9
    return configuraciones


def ejecutar_experimentos(
    X_entrenamiento: pd.DataFrame, y_entrenamiento: pd.Series
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], list[dict[str, float | int]]]]:
    """Ejecuta las 27 corridas y registra el resultado de cada una."""
    assert NUMERO_REPETICIONES == len(SEMILLAS_EXPERIMENTO), (
        "Debe haber una semilla por repetición."
    )
    filas: list[dict[str, Any]] = []
    historiales: dict[tuple[str, str, int], list[dict[str, float | int]]] = {}
    configuraciones = crear_configuraciones()

    print("\n================ EJECUCIÓN DE EXPERIMENTOS ================")
    for experimento, nombre_seleccion, nombre_cruce, funcion_seleccion, funcion_cruce in configuraciones:
        for repeticion, semilla in enumerate(SEMILLAS_EXPERIMENTO, start=1):
            etiqueta = (
                f"{experimento} | Selección: {nombre_seleccion} | "
                f"Cruce: {nombre_cruce} | Semilla: {semilla}"
            )
            print("\n" + "=" * 72)
            print(etiqueta)
            print("=" * 72)
            inicio = perf_counter()
            cromosoma, resultado, historial = ejecutar_algoritmo_genetico(
                X_entrenamiento,
                y_entrenamiento,
                funcion_seleccion,
                funcion_cruce,
                semilla,
                experimento,
            )
            tiempo = perf_counter() - inicio
            numero_seleccionadas = int(resultado["num_caracteristicas"])
            reduccion = (1 - numero_seleccionadas / X_entrenamiento.shape[1]) * 100
            fila = {
                "Experimento": experimento,
                "Selección": nombre_seleccion,
                "Cruce": nombre_cruce,
                "Repetición": repeticion,
                "Semilla": semilla,
                "Mejor fitness": float(resultado["fitness"]),
                "F1 de validación": float(resultado["f1"]),
                "Número de características": numero_seleccionadas,
                "Reducción (%)": float(reduccion),
                "Generación mejor solución": int(resultado["generacion_mejor"]),
                "Tiempo de ejecución (s)": float(tiempo),
                "Cromosoma ganador": cromosoma.copy(),
            }
            valores_numericos = [
                fila["Mejor fitness"],
                fila["F1 de validación"],
                fila["Reducción (%)"],
                fila["Tiempo de ejecución (s)"],
            ]
            assert np.all(np.isfinite(valores_numericos)), "Se generó un resultado inválido."
            filas.append(fila)
            historiales[(nombre_seleccion, nombre_cruce, semilla)] = historial
            print(
                f"Resumen: fitness={fila['Mejor fitness']:.4f}, "
                f"F1={fila['F1 de validación']:.4f}, "
                f"características={numero_seleccionadas}, tiempo={tiempo:.2f} s"
            )

    df_ejecuciones = pd.DataFrame(filas)
    assert len(df_ejecuciones) == 9 * NUMERO_REPETICIONES
    return df_ejecuciones, historiales


def resumen_por_configuracion(df_ejecuciones: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las repeticiones de las nueve configuraciones."""
    df_configuraciones = (
        df_ejecuciones.groupby(["Experimento", "Selección", "Cruce"], as_index=False)
        .agg(
            **{
                "Fitness promedio": ("Mejor fitness", "mean"),
                "Desv. Fitness": ("Mejor fitness", "std"),
                "F1 promedio": ("F1 de validación", "mean"),
                "Desv. F1": ("F1 de validación", "std"),
                "Características promedio": ("Número de características", "mean"),
                "Reducción promedio (%)": ("Reducción (%)", "mean"),
                "Tiempo promedio (s)": ("Tiempo de ejecución (s)", "mean"),
            }
        )
        .sort_values(
            ["Fitness promedio", "F1 promedio", "Características promedio"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    )
    assert len(df_configuraciones) == 9
    return df_configuraciones


def resumen_por_operador(df_ejecuciones: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Resume métricas para métodos de selección o de cruce."""
    resultado = (
        df_ejecuciones.groupby(columna, as_index=False)
        .agg(
            **{
                "Fitness promedio": ("Mejor fitness", "mean"),
                "Desv. Fitness": ("Mejor fitness", "std"),
                "F1 promedio": ("F1 de validación", "mean"),
                "Desv. F1": ("F1 de validación", "std"),
                "Características promedio": ("Número de características", "mean"),
                "Reducción promedio (%)": ("Reducción (%)", "mean"),
                "Tiempo promedio (s)": ("Tiempo de ejecución (s)", "mean"),
            }
        )
        .sort_values(
            ["Fitness promedio", "F1 promedio", "Características promedio"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    )
    assert len(resultado) == 3
    return resultado


def evaluar_modelo_final(
    X_entrenamiento: pd.DataFrame,
    X_prueba: pd.DataFrame,
    y_entrenamiento: pd.Series,
    y_prueba: pd.Series,
) -> dict[str, Any]:
    """Entrena en el conjunto de entrenamiento y evalúa una sola vez en prueba."""
    modelo = crear_modelo_svm(SEMILLA)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(X_prueba)
    return {
        "accuracy": float(accuracy_score(y_prueba, predicciones)),
        "precision": float(
            precision_score(y_prueba, predicciones, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_prueba, predicciones, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_prueba, predicciones, pos_label=1, zero_division=0)),
        "matriz_confusion": confusion_matrix(y_prueba, predicciones, labels=[0, 1]),
    }


def imprimir_metricas(nombre: str, metricas: dict[str, Any]) -> None:
    """Imprime las cuatro métricas y la matriz de confusión de un modelo."""
    print(f"\n{nombre}")
    print(f"Accuracy:  {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall:    {metricas['recall']:.4f}")
    print(f"F1:        {metricas['f1']:.4f}")
    print("Matriz de confusión [[TN, FP], [FN, TP]]:")
    print(metricas["matriz_confusion"])


def guardar_csv(
    df_ejecuciones: pd.DataFrame,
    df_configuraciones: pd.DataFrame,
    df_comparacion_seleccion: pd.DataFrame,
    df_comparacion_cruce: pd.DataFrame,
    df_caracteristicas: pd.DataFrame,
    directorio_resultados: Path,
) -> None:
    """Almacena las cinco tablas solicitadas en el directorio de resultados."""
    # Los cromosomas se convierten en texto legible para su persistencia en CSV.
    ejecuciones_csv = df_ejecuciones.copy()
    ejecuciones_csv["Cromosoma ganador"] = ejecuciones_csv["Cromosoma ganador"].apply(
        lambda cromosoma: "[" + ", ".join(map(str, np.asarray(cromosoma, dtype=int))) + "]"
    )
    ejecuciones_csv.to_csv(directorio_resultados / "resultados_ejecuciones.csv", index=False)
    df_configuraciones.to_csv(
        directorio_resultados / "resultados_configuraciones.csv", index=False
    )
    df_comparacion_seleccion.to_csv(
        directorio_resultados / "comparacion_metodos_seleccion.csv", index=False
    )
    df_comparacion_cruce.to_csv(
        directorio_resultados / "comparacion_metodos_cruce.csv", index=False
    )
    df_caracteristicas.to_csv(
        directorio_resultados / "caracteristicas_seleccionadas.csv", index=False
    )


def guardar_y_mostrar(figura: plt.Figure, ruta: Path) -> None:
    """Guarda una gráfica; ``plt.show`` se llama una vez tras crearlas todas."""
    figura.tight_layout()
    figura.savefig(ruta, dpi=150, bbox_inches="tight")


def graficar_matriz_confusion(matriz: np.ndarray, titulo: str, ruta: Path) -> None:
    """Construye y guarda una matriz de confusión anotada."""
    figura, eje = plt.subplots(figsize=(6, 5))
    imagen = eje.imshow(matriz, interpolation="nearest", cmap="Blues")
    figura.colorbar(imagen, ax=eje)
    eje.set(
        title=titulo,
        xlabel="Predicción",
        ylabel="Valor real",
        xticks=[0, 1],
        xticklabels=["Benigno (0)", "Maligno (1)"],
        yticks=[0, 1],
        yticklabels=["Benigno (0)", "Maligno (1)"],
    )
    umbral = matriz.max() / 2 if matriz.max() else 0
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            eje.text(
                columna,
                fila,
                str(matriz[fila, columna]),
                ha="center",
                va="center",
                color="white" if matriz[fila, columna] > umbral else "black",
            )
    guardar_y_mostrar(figura, ruta)


def etiquetas_configuraciones(df_configuraciones: pd.DataFrame) -> list[str]:
    """Produce etiquetas compactas para las nueve barras."""
    return [
        f"{fila.Experimento}\n{fila.Selección} / {fila.Cruce}"
        for fila in df_configuraciones.itertuples(index=False)
    ]


def graficar_barras_configuraciones(
    df_configuraciones: pd.DataFrame,
    columna: str,
    titulo: str,
    etiqueta_y: str,
    ruta: Path,
) -> None:
    """Grafica una métrica agregada para las nueve configuraciones."""
    figura, eje = plt.subplots(figsize=(13, 6))
    etiquetas = etiquetas_configuraciones(df_configuraciones)
    barras = eje.bar(etiquetas, df_configuraciones[columna], color="tab:blue")
    eje.set(title=titulo, xlabel="Configuración experimental", ylabel=etiqueta_y)
    eje.tick_params(axis="x", rotation=25)
    eje.grid(axis="y", alpha=0.3)
    eje.bar_label(barras, fmt="%.3f", padding=2, fontsize=8)
    guardar_y_mostrar(figura, ruta)


def graficar_comparacion_operadores(
    df_comparacion: pd.DataFrame,
    columna_operador: str,
    titulo: str,
    ruta: Path,
) -> None:
    """Compara Fitness y F1 medios de los tres operadores de una etapa."""
    figura, eje = plt.subplots(figsize=(9, 5))
    posiciones = np.arange(len(df_comparacion))
    ancho = 0.36
    barras_fitness = eje.bar(
        posiciones - ancho / 2,
        df_comparacion["Fitness promedio"],
        ancho,
        label="Fitness promedio",
    )
    barras_f1 = eje.bar(
        posiciones + ancho / 2,
        df_comparacion["F1 promedio"],
        ancho,
        label="F1 promedio",
    )
    eje.set(
        title=titulo,
        xlabel=columna_operador,
        ylabel="Puntuación",
        xticks=posiciones,
        xticklabels=df_comparacion[columna_operador],
        ylim=(0, 1.05),
    )
    eje.legend()
    eje.grid(axis="y", alpha=0.3)
    eje.bar_label(barras_fitness, fmt="%.3f", padding=2)
    eje.bar_label(barras_f1, fmt="%.3f", padding=2)
    guardar_y_mostrar(figura, ruta)


def generar_graficas(
    df_configuraciones: pd.DataFrame,
    df_comparacion_seleccion: pd.DataFrame,
    df_comparacion_cruce: pd.DataFrame,
    historial_ganador: list[dict[str, float | int]],
    etiqueta_historial: str,
    metricas_original: dict[str, Any],
    metricas_optimizado: dict[str, Any],
    directorio_graficas: Path,
) -> None:
    """Crea, guarda y muestra las nueve gráficas solicitadas con Matplotlib."""
    graficar_barras_configuraciones(
        df_configuraciones,
        "Fitness promedio",
        "Fitness promedio de las nueve configuraciones",
        "Fitness promedio",
        directorio_graficas / "fitness_configuraciones.png",
    )
    graficar_barras_configuraciones(
        df_configuraciones,
        "F1 promedio",
        "F1 promedio de las nueve configuraciones",
        "F1 promedio",
        directorio_graficas / "f1_configuraciones.png",
    )
    graficar_barras_configuraciones(
        df_configuraciones,
        "Características promedio",
        "Número promedio de características por configuración",
        "Características promedio",
        directorio_graficas / "caracteristicas_configuraciones.png",
    )
    graficar_comparacion_operadores(
        df_comparacion_seleccion,
        "Selección",
        "Comparación de métodos de selección",
        directorio_graficas / "comparacion_seleccion.png",
    )
    graficar_comparacion_operadores(
        df_comparacion_cruce,
        "Cruce",
        "Comparación de métodos de cruce",
        directorio_graficas / "comparacion_cruce.png",
    )

    datos_historial = pd.DataFrame(historial_ganador)
    figura, eje = plt.subplots(figsize=(10, 5))
    eje.plot(
        datos_historial["generacion"],
        datos_historial["mejor_fitness"],
        marker="o",
        label="Mejor fitness por generación",
    )
    eje.plot(
        datos_historial["generacion"],
        datos_historial["fitness_promedio"],
        marker="s",
        label="Fitness promedio de población",
    )
    eje.plot(
        datos_historial["generacion"],
        datos_historial["mejor_fitness_global"],
        linestyle="--",
        label="Mejor fitness global",
    )
    eje.set(
        title=f"Evolución del fitness: configuración ganadora ({etiqueta_historial})",
        xlabel="Generación",
        ylabel="Fitness",
    )
    eje.legend()
    eje.grid(alpha=0.3)
    guardar_y_mostrar(figura, directorio_graficas / "convergencia_ganador.png")

    nombres_metricas = ["Accuracy", "Precision", "Recall", "F1"]
    claves_metricas = ["accuracy", "precision", "recall", "f1"]
    posiciones = np.arange(len(claves_metricas))
    ancho = 0.36
    figura, eje = plt.subplots(figsize=(9, 5))
    barras_original = eje.bar(
        posiciones - ancho / 2,
        [metricas_original[clave] for clave in claves_metricas],
        ancho,
        label="SVM original",
    )
    barras_optimizado = eje.bar(
        posiciones + ancho / 2,
        [metricas_optimizado[clave] for clave in claves_metricas],
        ancho,
        label="SVM optimizado",
    )
    eje.set(
        title="SVM original frente a SVM optimizado",
        xlabel="Métrica",
        ylabel="Puntuación",
        xticks=posiciones,
        xticklabels=nombres_metricas,
        ylim=(0, 1.08),
    )
    eje.legend()
    eje.bar_label(barras_original, fmt="%.3f", padding=2)
    eje.bar_label(barras_optimizado, fmt="%.3f", padding=2)
    guardar_y_mostrar(figura, directorio_graficas / "comparacion_svm.png")

    graficar_matriz_confusion(
        metricas_original["matriz_confusion"],
        "Matriz de confusión: SVM original",
        directorio_graficas / "matriz_confusion_original.png",
    )
    graficar_matriz_confusion(
        metricas_optimizado["matriz_confusion"],
        "Matriz de confusión: SVM optimizado",
        directorio_graficas / "matriz_confusion_optimizado.png",
    )

    # En ejecución local muestra las nueve figuras. Con backend no interactivo
    # (por ejemplo, un servidor) la llamada es inocua y los PNG ya fueron guardados.
    plt.show()
    plt.close("all")


def main() -> int:
    """Coordina la carga, el diseño experimental y la evaluación final."""
    assert len(METODOS_SELECCION) == 3
    assert len(METODOS_CRUCE) == 3
    assert NUMERO_REPETICIONES == len(SEMILLAS_EXPERIMENTO) == 3
    assert len(crear_configuraciones()) == 9

    # ``pipeline.py`` vive en ``src/``; los artefactos se conservan en la
    # raíz del proyecto, dentro de ``resultados/``.
    directorio_base = Path(__file__).resolve().parent.parent
    directorio_resultados = directorio_base / "resultados"
    directorio_graficas = directorio_resultados / "graficas"
    directorio_graficas.mkdir(parents=True, exist_ok=True)

    # Carga directa del dataset incorporado; no se requiere CSV externo.
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = (1 - dataset.target).astype(int)  # 0 = benigno, 1 = maligno.
    mostrar_informacion_dataset(X, y)

    # Única división. X_prueba/y_prueba no participan en los experimentos.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    print(
        f"\nPartición: entrenamiento={X_entrenamiento.shape}, prueba={X_prueba.shape}."
    )
    print("El conjunto de prueba queda reservado hasta la evaluación final.")

    df_ejecuciones, historiales = ejecutar_experimentos(X_entrenamiento, y_entrenamiento)
    df_configuraciones = resumen_por_configuracion(df_ejecuciones)
    df_comparacion_seleccion = resumen_por_operador(df_ejecuciones, "Selección")
    df_comparacion_cruce = resumen_por_operador(df_ejecuciones, "Cruce")

    print("\n================ RESULTADOS POR CONFIGURACIÓN ================")
    print(df_configuraciones.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    print("\n================ COMPARACIÓN: SELECCIÓN ================")
    print(df_comparacion_seleccion.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
    print("\n================ COMPARACIÓN: CRUCE ================")
    print(df_comparacion_cruce.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))

    # El ganador se determina exclusivamente a partir de CV en entrenamiento.
    mejor_configuracion = df_configuraciones.iloc[0]
    mascara_configuracion_ganadora = (
        (df_ejecuciones["Selección"] == mejor_configuracion["Selección"])
        & (df_ejecuciones["Cruce"] == mejor_configuracion["Cruce"])
    )
    ejecuciones_ganadoras = df_ejecuciones.loc[mascara_configuracion_ganadora].sort_values(
        ["Mejor fitness", "F1 de validación", "Número de características"],
        ascending=[False, False, True],
    )
    mejor_ejecucion = ejecuciones_ganadoras.iloc[0]
    cromosoma_ganador = np.asarray(mejor_ejecucion["Cromosoma ganador"], dtype=int)
    validar_cromosoma(cromosoma_ganador, X.shape[1])
    indices_seleccionados = np.flatnonzero(cromosoma_ganador)
    nombres_seleccionados = X.columns[indices_seleccionados].tolist()
    nombres_eliminados = X.columns[cromosoma_ganador == 0].tolist()
    reduccion_ganadora = (1 - len(indices_seleccionados) / X.shape[1]) * 100
    df_caracteristicas = pd.DataFrame(
        {
            "Índice original": indices_seleccionados,
            "Característica seleccionada": nombres_seleccionados,
        }
    )

    # Primera y única utilización del conjunto de prueba reservado.
    metricas_original = evaluar_modelo_final(
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    )
    metricas_optimizado = evaluar_modelo_final(
        X_entrenamiento.iloc[:, indices_seleccionados],
        X_prueba.iloc[:, indices_seleccionados],
        y_entrenamiento,
        y_prueba,
    )

    guardar_csv(
        df_ejecuciones,
        df_configuraciones,
        df_comparacion_seleccion,
        df_comparacion_cruce,
        df_caracteristicas,
        directorio_resultados,
    )
    historial_ganador = historiales[
        (
            str(mejor_ejecucion["Selección"]),
            str(mejor_ejecucion["Cruce"]),
            int(mejor_ejecucion["Semilla"]),
        )
    ]
    etiqueta_historial = f"{mejor_ejecucion['Experimento']}, semilla {mejor_ejecucion['Semilla']}"
    generar_graficas(
        df_configuraciones,
        df_comparacion_seleccion,
        df_comparacion_cruce,
        historial_ganador,
        etiqueta_historial,
        metricas_original,
        metricas_optimizado,
        directorio_graficas,
    )

    print("\n================= ACTIVIDAD IV =================")
    print("\nMEJOR CONFIGURACIÓN (seleccionada con CV sobre entrenamiento)")
    print(f"Selección: {mejor_configuracion['Selección']}")
    print(f"Cruce: {mejor_configuracion['Cruce']}")
    print(f"Fitness promedio: {mejor_configuracion['Fitness promedio']:.4f}")
    print(f"F1 promedio: {mejor_configuracion['F1 promedio']:.4f}")
    print(f"Características promedio: {mejor_configuracion['Características promedio']:.2f}")
    print(f"Reducción promedio: {mejor_configuracion['Reducción promedio (%)']:.2f} %")

    print("\nMEJOR SOLUCIÓN INDIVIDUAL DE ESA CONFIGURACIÓN")
    print(f"Vector binario ganador: {cromosoma_ganador.tolist()}")
    print(f"Características originales: {X.shape[1]}")
    print(f"Características seleccionadas: {len(indices_seleccionados)}")
    print(f"Características eliminadas: {len(nombres_eliminados)}")
    print(f"Reducción: {reduccion_ganadora:.2f} %")
    print("Nombres de las características seleccionadas:")
    for nombre in nombres_seleccionados:
        print(f"  - {nombre}")
    print(
        "Estas variables fueron seleccionadas por el algoritmo bajo las condiciones "
        "del experimento; no se interpretan como importancia clínica."
    )

    print("\nEVALUACIÓN FINAL EN PRUEBA")
    imprimir_metricas("SVM ORIGINAL", metricas_original)
    imprimir_metricas("SVM OPTIMIZADO", metricas_optimizado)
    print(f"\nResultados guardados en: {directorio_resultados}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
