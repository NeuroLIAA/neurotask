from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from ..metrics import get_correct_and_incorrect_segments
from ..model.tmt_model import TMTExperiment


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

