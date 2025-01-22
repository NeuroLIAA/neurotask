from typing import Dict

from matplotlib import pyplot as plt

from neurotask.tmt.metrics import calculate_speeds_between_cursor_positions, \
    calculate_accelerations_between_cursor_positions, get_correct_and_incorrect_segments
from neurotask.tmt.model.tmt_model import TMTTrial, TMTExperiment, TrialType
from neurotask.tmt.visualization.utils import calculate_median_segments, \
    average_performance_by_five_by_five_year_age_groups


def plot_with_color(trial: TMTTrial, canvas_size: int, target_radius: float, color_by='time'):
    """
    Plotea la trayectoria del cursor junto con los targets en un gráfico,
    ajustando el tamaño del gráfico al tamaño del canvas y dibujando círculos
    alrededor de los targets con su contenido. La trayectoria del cursor se
    dibuja con puntos cuyos colores cambian en función del tiempo, la velocidad o la aceleración.
    El primer clic en el target se destaca con un marcador especial.

    Parameters:
    - trial: TMTTrial, el trial con la trayectoria del cursor y los targets.
    - canvas_size: int, tamaño del canvas (ancho y alto).
    - target_radius: float, radio de los círculos que rodean los targets.
    - color_by: str, 'time', 'speed' o 'acceleration', determina si el color de los puntos cambia en función del tiempo, velocidad o aceleración.
    """

    # Extraer la posición de los targets
    target_x = [target.position.x for target in trial.stimuli]
    target_y = [target.position.y for target in trial.stimuli]
    target_contents = [target.content for target in trial.stimuli]

    cursor_trail_from_first_click = trial.get_cursor_trail_from_start()
    # Extraer la trayectoria del cursor
    cursor_x = [cursor_info.position.x for cursor_info in cursor_trail_from_first_click]
    cursor_y = [cursor_info.position.y for cursor_info in cursor_trail_from_first_click]
    cursor_times = [cursor_info.time for cursor_info in cursor_trail_from_first_click]

    if color_by == 'time':
        # Normalizar tiempos para que estén en el rango [0, 1]
        norm = plt.Normalize(min(cursor_times), max(cursor_times))
        colors = plt.cm.viridis(norm(cursor_times))  # Usar un mapa de colores para el tiempo
    elif color_by == 'speed':
        speeds = calculate_speeds_between_cursor_positions(trial)
        speeds = [0] + speeds  # Para igualar el número de puntos con las velocidades calculadas
        norm = plt.Normalize(min(speeds), max(speeds))
        colors = plt.cm.viridis(norm(speeds))  # Usar un mapa de colores para la velocidad
    elif color_by == 'acceleration':
        accelerations = calculate_accelerations_between_cursor_positions(trial)
        accelerations = [0, 0] + accelerations  # Igualar el número de puntos (2 primeros puntos sin aceleración)
        norm = plt.Normalize(min(accelerations), max(accelerations))
        colors = plt.cm.viridis(norm(accelerations))  # Usar un mapa de colores para la aceleración
    else:
        raise ValueError("El parámetro color_by debe ser 'time', 'speed' o 'acceleration'.")

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 8))

    # Dibujar la trayectoria del cursor como líneas coloreadas por tiempo o velocidad
    for i in range(len(cursor_x) - 1):
        plt.plot([cursor_x[i], cursor_x[i + 1]],
                 [cursor_y[i], cursor_y[i + 1]],
                 color=colors[i], linewidth=2, zorder=4)

    # Dibujar los targets como círculos con el contenido dentro
    for x, y, content in zip(target_x, target_y, target_contents):
        circle = plt.Circle((x, y), target_radius, color='red', alpha=0.3, zorder=5)
        plt.gca().add_patch(circle)
        # Añadir el contenido del target en el centro del círculo
        plt.text(x, y, content, color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Destacar el primer clic
    if trial.start:
        fc_x = trial.start.position.x
        fc_y = trial.start.position.y
        plt.scatter(fc_x, fc_y, color='cyan', edgecolor='black', s=100, label='First Click', zorder=7,
                    marker='o', alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar_label = 'Time' if color_by == 'time' else 'Speed' if color_by == 'speed' else 'Acceleration'
    cbar = fig.colorbar(sm, ax=ax, label=cbar_label)

    # Set plot limits based on canvas size
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)  # Inverted Y-axis if needed

    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Cursor Trail with Targets')

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Add legend
    ax.legend()

    return fig


def get_correct_and_incorrect_segment_counts_by_subject(experiment: TMTExperiment, average_per_trial: bool = False):
    """
    Devuelve cuatro diccionarios:
    1. Cantidad de segmentos correctos por sujeto en ensayos de tipo A.
    2. Cantidad de segmentos incorrectos por sujeto en ensayos de tipo A.
    3. Cantidad de segmentos correctos por sujeto en ensayos de tipo B.
    4. Cantidad de segmentos incorrectos por sujeto en ensayos de tipo B.
    """
    correct_counts_A = {}
    incorrect_counts_A = {}
    correct_counts_B = {}
    incorrect_counts_B = {}

    for subject_id, subject in experiment.subjects.items():
        correct_count_A = 0
        incorrect_count_A = 0
        correct_count_B = 0
        incorrect_count_B = 0
        trial_count_A = 0
        trial_count_B = 0

        # Iterar sobre los trials de entrenamiento y prueba
        for trial in subject.testing_trials:
            correct_segments, incorrect_segments = get_correct_and_incorrect_segments(trial, subject.target_radius)

            if trial.trial_type == TrialType.PART_A:
                correct_count_A += len(correct_segments)
                incorrect_count_A += len(incorrect_segments)
                trial_count_A += 1
            elif trial.trial_type == TrialType.PART_B:
                correct_count_B += len(correct_segments)
                incorrect_count_B += len(incorrect_segments)
                trial_count_B += 1

        # Si average_per_trial es True, calcular el promedio de errores por ensayo
        if average_per_trial:
            correct_counts_A[subject_id] = correct_count_A / trial_count_A if trial_count_A > 0 else 0
            incorrect_counts_A[subject_id] = incorrect_count_A / trial_count_A if trial_count_A > 0 else 0
            correct_counts_B[subject_id] = correct_count_B / trial_count_B if trial_count_B > 0 else 0
            incorrect_counts_B[subject_id] = incorrect_count_B / trial_count_B if trial_count_B > 0 else 0
        else:
            correct_counts_A[subject_id] = correct_count_A
            incorrect_counts_A[subject_id] = incorrect_count_A
            correct_counts_B[subject_id] = correct_count_B
            incorrect_counts_B[subject_id] = incorrect_count_B

    return correct_counts_A, incorrect_counts_A, correct_counts_B, incorrect_counts_B


def plot_correct_incorrect_segments_by_subject(experiment: TMTExperiment, average_per_trial: bool = False):
    """
    Genera gráficos de barras para los segmentos correctos e incorrectos por sujeto,
    separados por tipo de ensayo (A y B). Si average_per_trial es True, calcula el
    promedio de errores por ensayo.
    """
    correct_counts_A, incorrect_counts_A, correct_counts_B, incorrect_counts_B = \
        get_correct_and_incorrect_segment_counts_by_subject(experiment, average_per_trial)

    # Truncar los IDs de los sujetos a las primeras 4 letras para que no sea tan largo
    subject_ids_A = [subject_id[:4] for subject_id in correct_counts_A.keys()]
    subject_ids_B = [subject_id[:4] for subject_id in correct_counts_B.keys()]

    # Gráfico para segmentos correctos en ensayos de tipo A
    plt.figure(figsize=(10, 6))
    plt.bar(subject_ids_A, correct_counts_A.values(), color='green')
    plt.title('Cantidad de segmentos correctos por sujeto (Tipo A - Ensayos de prueba)')
    plt.xlabel('Sujeto')
    plt.ylabel('Cantidad de segmentos correctos' + (' (Promedio por ensayo)' if average_per_trial else ''))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Gráfico para segmentos incorrectos en ensayos de tipo A
    plt.figure(figsize=(10, 6))
    plt.bar(subject_ids_A, incorrect_counts_A.values(), color='red')
    plt.title('Cantidad de segmentos incorrectos por sujeto (Tipo A - Ensayos de prueba)')
    plt.xlabel('Sujeto')
    plt.ylabel('Cantidad de segmentos incorrectos' + (' (Promedio por ensayo)' if average_per_trial else ''))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Gráfico para segmentos correctos en ensayos de tipo B
    plt.figure(figsize=(10, 6))
    plt.bar(subject_ids_B, correct_counts_B.values(), color='green')
    plt.title('Cantidad de segmentos correctos por sujeto (Tipo B - Ensayos de prueba)')
    plt.xlabel('Sujeto')
    plt.ylabel('Cantidad de segmentos correctos' + (' (Promedio por ensayo)' if average_per_trial else ''))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Gráfico para segmentos incorrectos en ensayos de tipo B
    plt.figure(figsize=(10, 6))
    plt.bar(subject_ids_B, incorrect_counts_B.values(), color='red')
    plt.title('Cantidad de segmentos incorrectos por sujeto (Tipo B - Ensayos de prueba)')
    plt.xlabel('Sujeto')
    plt.ylabel('Cantidad de segmentos incorrectos' + (' (Promedio por ensayo)' if average_per_trial else ''))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_number_of_testing_trials_per_subject(experiment: TMTExperiment):
    """
    Plotea la cantidad de ensayos de prueba (testing trials) que tiene cada sujeto en un experimento.

    Parameters:
    - experiment: TMTExperiment, la instancia que contiene los sujetos y sus ensayos.
    """
    # Obtener los ids de los sujetos y la cantidad de trials de prueba por sujeto
    subject_ids = []
    num_testing_trials = []

    for subject_id, subject in experiment.subjects.items():
        subject_ids.append(subject_id[:4])  # Usamos solo los primeros 4 caracteres del ID del sujeto
        num_testing_trials.append(len(subject.testing_trials))  # Contamos los ensayos de prueba

    # Crear el gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(subject_ids, num_testing_trials, color='blue')
    plt.xlabel('Subject ID')
    plt.ylabel('Number of Testing Trials')
    plt.title('Number of Testing Trials per Subject')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def plot_median_segments(experiment: TMTExperiment):
    """
    Plotea la mediana de los segmentos correctos e incorrectos en los ensayos de prueba de todos los sujetos.

    Parameters:
    - experiment: TMTExperiment, la instancia que contiene los sujetos y sus ensayos.
    """
    mediana_correctos, mediana_incorrectos = calculate_median_segments(experiment)

    # Ploteamos los resultados
    labels = ['Correctos', 'Incorrectos']
    medians = [mediana_correctos, mediana_incorrectos]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, medians, color=['green', 'red'])
    plt.ylabel('Mediana de segmentos')
    plt.title('Mediana de segmentos correctos e incorrectos')
    plt.show()


def plot_segment_ranges(experiment: TMTExperiment):
    """
    Calcula y plotea el porcentaje de sujetos en diferentes rangos de segmentos correctos e incorrectos.

    Parameters:
    - experiment: TMTExperiment, la instancia que contiene los sujetos y sus ensayos.
    """
    # Inicialización de contadores
    correct_ranges = {
        '0-5': 0,
        '6-12': 0,
        '13-15': 0,
        '16-20': 0
    }

    incorrect_ranges = {
        '0-5': 0,
        '6-12': 0,
        '13-15': 0,
        '16-20': 0
    }

    for subject in experiment.subjects.values():
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
            avg_correct = total_correct / num_trials
            avg_incorrect = total_incorrect / num_trials

            # Contar segmentos correctos
            if 0 <= avg_correct <= 5:
                correct_ranges['0-5'] += 1
            elif 5 < avg_correct <= 12:
                correct_ranges['6-12'] += 1
            elif 12 < avg_correct <= 15:
                correct_ranges['13-15'] += 1
            elif 15 < avg_correct <= 20:
                correct_ranges['16-20'] += 1
            else:
                raise ValueError("El rango de segmentos correctos no es válido.")

            # Contar segmentos incorrectos
            if 0 <= avg_incorrect <= 5:
                incorrect_ranges['0-5'] += 1
            elif 5 < avg_incorrect <= 12:
                incorrect_ranges['6-12'] += 1
            elif 12 < avg_incorrect <= 15:
                incorrect_ranges['13-15'] += 1
            elif 15 < avg_incorrect <= 20:
                incorrect_ranges['16-20'] += 1
            else:
                raise ValueError("El rango de segmentos correctos no es válido.")

    # Plotear los resultados
    x_labels = list(correct_ranges.keys())
    correct_values = list(correct_ranges.values())
    incorrect_values = list(incorrect_ranges.values())

    x = range(len(x_labels))  # Posiciones para las barras

    plt.figure(figsize=(10, 6))
    plt.bar(x, correct_values, width=0.4, label='Promedio de correctos en base a todos los trials de un sujeto',
            color='green', alpha=0.7, align='center')

    plt.ylabel('Cantidad de sujetos')
    plt.title('Cantidad de sujetos por rangos de segmentos correctos')
    plt.xticks([p + 0.2 for p in x], x_labels)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar([p + 0.4 for p in x], incorrect_values, width=0.4,
            label='Promedio de Incorrectos en base a todos los trials de un sujeto', color='red', alpha=0.7,
            align='center')
    plt.ylabel('Cantidad de sujetos')
    plt.title('Cantidad de sujetos por rangos de segmentos incorrectos')
    plt.xticks([p + 0.2 for p in x], x_labels)
    plt.legend()
    plt.grid(axis='y')
    plt.show()


import numpy as np


def plot_performance_by_age_group(experiment: TMTExperiment):
    """
    Grafica el rendimiento promedio (correctos e incorrectos) para cada rango de edad, con los rangos de edad ordenados.

    Parameters:
    - experiment: TMTExperiment, la instancia que contiene los sujetos y sus ensayos.
    """
    performance_by_age_group = average_performance_by_five_by_five_year_age_groups(experiment)

    # Convertir las llaves del diccionario (rangos de edad) en tuplas numéricas para poder ordenarlas
    sorted_performance_by_age_group = sorted(performance_by_age_group.items(), key=lambda x: int(x[0].split('-')[0]))

    age_labels = []
    correct_means = []
    incorrect_means = []

    # Extraer datos para graficar después de ordenar
    for age_range, performance in sorted_performance_by_age_group:
        age_labels.append(age_range)
        correct_means.append(performance['average_correct'])
        incorrect_means.append(performance['average_incorrect'])

    # Definir el ancho de las barras
    bar_width = 0.35
    index = np.arange(len(age_labels))

    # Crear el gráfico de barras para correctos e incorrectos
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_correct = ax.bar(index, correct_means, bar_width, label='Correctos', color='green', alpha=0.7)
    bar_incorrect = ax.bar(index + bar_width, incorrect_means, bar_width, label='Incorrectos', color='red', alpha=0.7)

    # Etiquetas y formato del gráfico
    ax.set_xlabel('Rango de Edad')
    ax.set_ylabel('Promedio de Segments')
    ax.set_title('Rendimiento Promedio por Rangos de Edad (Correctos e Incorrectos)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(age_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()
