# Cómo agregar una nueva métrica al repositorio

1. **Definir la clase calculadora**

   * Crea un nuevo fichero (o agrégalo en `metrics_calculators.py`) con tu clase.
   * Hereda de la interfaz base (`BaseMetricCalculator`) y sobrescribe `add_metrics()`:

   ```python
   from metrics_base import BaseMetricCalculator

   class MiNuevaMetricaCalculator(BaseMetricCalculator):
       def __init__(self, parametro1, parametro2=42):
           self.param1 = parametro1
           self.param2 = parametro2

       def add_metrics(self, metrics: dict, trial, **params) -> dict:
           """
           Calcula y añade las claves/valores de esta métrica en `metrics`.
           - `trial`: objeto `TMTTrial` de entrada.
           - `params`: otros parámetros comunes (p. ej. `speed_threshold`).
           """
           # Ejemplo de cálculo:
           valor = alguna_funcion(trial.data, self.param1, params.get('otro_param'))
           metrics['mi_nueva_metrica'] = valor
           return metrics
   ```

2. **Registrar la calculadora en el pipeline**

   Abre el módulo donde defines la función `get_metric_calculators()` y añade tu instancia:

   ```python
   from metrics_calculators import (
       TotalDistanceCalculator,
       ReactionTimeCalculator,
       SpeedMetricsCalculator,
       SegmentationMetricCalculator,
       TargetsTouchesCalculator,
       CrossesMetricCalculator,
       MiNuevaMetricaCalculator,  # <- importa tu clase
   )

   def get_metric_calculators():
       return [
           TotalDistanceCalculator(),
           ReactionTimeCalculator(),
           SpeedMetricsCalculator(),
           SegmentationMetricCalculator(),
           TargetsTouchesCalculator(),
           CrossesMetricCalculator(),
           MiNuevaMetricaCalculator(param1=..., param2=...),  # <- añádela aquí
       ]
   ```

3. **Actualizar la llamada a `compute_trial_metrics`**

   Si tu calculadora necesita parámetros adicionales pasados por `**params`, inclúyelos al invocar:

   ```python
   trial_metrics = compute_trial_metrics(
       trial,
       get_metric_calculators(),
       speed_threshold=0.5,
       param1=123,            # coincide con tu __init__
       otro_param='valor_opcional'
   )
   ```

4. **Documentar (Work in Progress)**

   * Actualiza el README (o la sección de “Métricas disponibles”) con:

     * Una breve descripción de la métrica.
     * Sus parámetros.
     * Significado del resultado.
   * Si utilizas un changelog, anota la adición.
