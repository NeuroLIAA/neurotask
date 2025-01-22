from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from neurotask.tmt.metrics import get_correct_and_incorrect_segments
from neurotask.tmt.model.tmt_model import TMTExperiment


def average_correct_and_incorrect_segments(experiment: TMTExperiment) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Calcula el promedio de segmentos correctos e incorrectos en los ensayos de prueba de todos los sujetos.
    """
    correct_avg_by_subject = {}
    incorrect_avg_by_subject = {}

    for subject_id, subject in experiment.subjects.items():
        total_correct = 0
        total_incorrect = 0
        num_trials = 0

        for trial in subject.testing_trials:
            correct_segments, incorrect_segments = get_correct_and_incorrect_segments(trial, subject.target_radius)

            # Sumar segmentos correctos e incorrectos
            total_correct += len(correct_segments)
            total_incorrect += len(incorrect_segments)
            num_trials += 1

        if num_trials > 0:
            correct_avg_by_subject[subject_id] = total_correct / num_trials
            incorrect_avg_by_subject[subject_id] = total_incorrect / num_trials

    return correct_avg_by_subject, incorrect_avg_by_subject


def average_performance_by_age_group(experiment: TMTExperiment, age_ranges: List[Tuple[int, int]]) -> (
        Dict)[str, Dict[str, float]]:
    """
    Calcula el rendimiento promedio (correctos e incorrectos) agrupado por rango de edad.

    :param experiment: TMTExperiment que contiene los datos de todos los sujetos.
    :param age_ranges: Lista de tuplas que definen los rangos de edad, e.g., [(10, 20), (21, 30), ...].
    :return: Un diccionario que mapea cada rango de edad a los promedios correctos e incorrectos.
    """
    correct_avg_by_subject, incorrect_avg_by_subject = average_correct_and_incorrect_segments(experiment)

    # Inicializar diccionario para almacenar métricas por grupo de edad
    performance_by_age_group = defaultdict(lambda: {"average_correct": [], "average_incorrect": []})

    # Calcular rendimiento por sujeto y agrupar por rango de edad
    for subject_id, subject in experiment.subjects.items():
        age = subject.age()

        # Encontrar en qué grupo de edad cae el sujeto
        for age_range in age_ranges:
            if age_range[0] <= age <= age_range[1]:
                performance_by_age_group[f"{age_range[0]}-{age_range[1]}"]["average_correct"].append(
                    correct_avg_by_subject.get(subject_id, 0))
                performance_by_age_group[f"{age_range[0]}-{age_range[1]}"]["average_incorrect"].append(
                    incorrect_avg_by_subject.get(subject_id, 0))
                break

    # Calcular el promedio por cada grupo de edad
    result = {}
    for age_group, metrics in performance_by_age_group.items():
        avg_correct = np.mean(metrics["average_correct"]) if metrics["average_correct"] else 0
        avg_incorrect = np.mean(metrics["average_incorrect"]) if metrics["average_incorrect"] else 0
        result[age_group] = {"average_correct": avg_correct, "average_incorrect": avg_incorrect}

    return result


def average_performance_by_five_by_five_year_age_groups(experiment: TMTExperiment) -> Dict[str, Dict[str, float]]:
    """
    Calcula el rendimiento promedio (correctos e incorrectos) agrupado por rangos de edad de 5 en 5 años.

    :param experiment: TMTExperiment que contiene los datos de todos los sujetos.
    :return: Un diccionario que mapea cada rango de edad a los promedios correctos e incorrectos.
    """
    age_ranges = [(age, age + 4) for age in range(0, 100, 5)]
    return average_performance_by_age_group(experiment, age_ranges)

def calculate_median_segments(experiment: TMTExperiment):
    """
    Calcula la mediana de los segmentos correctos e incorrectos en los ensayos de prueba de todos los sujetos.

    Parameters:
    - experiment: TMTExperiment, la instancia que contiene los sujetos y sus ensayos.

    Returns:
    - mediana_correctos: La mediana de los segmentos correctos.
    - mediana_incorrectos: La mediana de los segmentos incorrectos.
    """
    correct_segments_counts = []
    incorrect_segments_counts = []

    for subject in experiment.subjects.values():
        for trial in subject.testing_trials:
            correct_segments, incorrect_segments = get_correct_and_incorrect_segments(trial, subject.target_radius)

            # Contamos los segmentos correctos e incorrectos
            correct_segments_counts.append(len(correct_segments))
            incorrect_segments_counts.append(len(incorrect_segments))

    # Calculamos la mediana de los segmentos correctos e incorrectos
    mediana_correctos = np.median(correct_segments_counts)
    mediana_incorrectos = np.median(incorrect_segments_counts)

    return mediana_correctos, mediana_incorrectos

