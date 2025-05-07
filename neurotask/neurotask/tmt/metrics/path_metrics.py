import logging
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import scipy.stats as stats
import itertools


import numpy as np
from neurotask.tmt.metrics.base_metric import BaseMetricCalculator
# from neurotask.tmt.metrics.metrics import calculate_distance
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo


# Work in progress # 

# # --- Función para calcular amplitud del serrucho ---
# class ZigZagAmplitude(BaseMetricCalculator): #(items, times):
#     def add_metrics(self, metrics: dict, trial, **params) -> dict:
#         """Calcula la amplitud promedio del 'serrucho' en ensayos tipo B"""
#         time_deltas = np.diff([0] + list(trial.rt))
        
#         num_times = []
#         let_times = []

#         for item, dt in zip(trial.stimuli, time_deltas):
#             if item.isnumeric():
#                 num_times.append(dt)
#             elif item.isalpha():
#                 let_times.append(dt)

#         min_len = min(len(num_times), len(let_times))
#         num_times = num_times[:min_len]
#         let_times = let_times[:min_len]

#         zigzag_amplitudes = np.abs(np.array(num_times) - np.array(let_times))
#         metrics['zigzag_amplitude'] = np.mean(zigzag_amplitudes)
#         return metrics
    


