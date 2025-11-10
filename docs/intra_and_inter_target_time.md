# Intra-Target Time e Inter-Target Time

## Descripción General

Estas dos métricas complementarias permiten analizar cómo los participantes distribuyen su tiempo durante la ejecución de un trial del Trail Making Test (TMT).

## Intra-Target Time (Tiempo Intra-Target)

### Definición
El **intra-target time** es el tiempo total que el cursor pasa **dentro** de los targets correctos durante un trial.

### Cálculo
Se calcula sumando la duración de todos los intervalos continuos donde el cursor se encuentra dentro del radio de un target correcto (en el orden esperado).

### Fórmula
```
intra_target_time = Σ (tiempo_salida - tiempo_entrada) para cada intervalo en target correcto
```

### Ejemplo
Para un trial donde el cursor:
- Entra al target 0 en t=0.0s y sale en t=1.5s → **1.5s**
- Entra al target 1 en t=3.0s y sale en t=4.0s → **1.0s**
- Entra al target 2 en t=6.0s y sale en t=8.0s → **2.0s**

**Intra-target time = 1.5 + 1.0 + 2.0 = 4.5s**

### Interpretación
- **Valores altos**: El participante pasa más tiempo "sobre" los targets, lo que puede indicar:
  - Mayor precisión al tocar los targets
  - Posible dificultad para salir de los targets
  - Movimientos más lentos
  
- **Valores bajos**: El participante pasa poco tiempo sobre los targets, lo que puede indicar:
  - Movimientos rápidos y directos
  - Posible menor precisión

### Consideraciones Importantes
- Solo se cuentan los targets tocados **en el orden correcto**
- Si se toca un target incorrecto (fuera de orden), ese tiempo **NO se cuenta**
- Si el cursor vuelve a un target ya completado, ese tiempo **NO se cuenta**

---

## Inter-Target Time (Tiempo Inter-Target)

### Definición
El **inter-target time** es el tiempo total que el cursor pasa **entre** targets, es decir, el tiempo de movimiento cuando NO está sobre ningún target correcto.

### Cálculo
Se calcula como el complemento del intra-target time:

```
inter_target_time = tiempo_total - intra_target_time
```

Donde:
- `tiempo_total = tiempo_último_cursor - tiempo_primer_cursor`

### Ejemplo
Continuando con el ejemplo anterior, si el trial completo dura **8.0s**:
- Intra-target time = 4.5s
- **Inter-target time = 8.0 - 4.5 = 3.5s**

### Interpretación
- **Valores altos**: El participante pasa más tiempo moviéndose entre targets, lo que puede indicar:
  - Movimientos más lentos o indirectos
  - Búsqueda visual de targets
  - Errores que requieren corrección
  - Dificultad para planificar la ruta
  
- **Valores bajos**: El participante pasa poco tiempo entre targets, lo que puede indicar:
  - Movimientos rápidos y eficientes
  - Buena planificación de la ruta
  - Familiaridad con la tarea

---

## Relación Entre Ambas Métricas

### Invariante Fundamental
Para cualquier trial, siempre se cumple:

```
intra_target_time + inter_target_time = tiempo_total
```

Esta relación permite validar que ambas métricas se calculan correctamente.

### Proporción (Ratio)
Una métrica derivada útil es la **proporción de tiempo en targets**:

```
ratio = intra_target_time / tiempo_total
```

- **ratio ≈ 0.2-0.4**: Típico para ejecuciones rápidas y eficientes
- **ratio > 0.5**: Puede indicar movimientos muy lentos o dificultades

---

## Casos Especiales

### Trial con Custom Start
Cuando un trial tiene un punto de inicio personalizado (`with_custom_start=True`), solo se considera el tiempo **después** del punto de inicio.

```
tiempo_total = tiempo_último_cursor - tiempo_start_personalizado
```

### Targets Superpuestos
Con radios de target grandes, múltiples targets pueden superponerse. En estos casos:
- Solo se cuenta el tiempo en el **target esperado** (según el orden)
- Si el cursor está sobre targets 0 y 1, pero el esperado es 0, solo cuenta como tiempo en target 0

### Targets Incorrectos
El tiempo pasado en targets tocados fuera de orden se considera **inter-target time**, no intra-target time, ya que no contribuyen al progreso correcto del trial.

---

## Uso en Análisis

Estas métricas son útiles para:

1. **Análisis de rendimiento**: Identificar si los problemas están en la ejecución motora (inter-target) o en la precisión (intra-target)

2. **Comparación entre grupos**: Comparar patrones de ejecución entre grupos clínicos y controles

3. **Análisis temporal**: Estudiar cómo cambia la distribución del tiempo a lo largo del trial

4. **Detección de estrategias**: Identificar diferentes estrategias de ejecución basadas en el ratio intra/inter

---

## Implementación

### Funciones Disponibles

```python
from neurotask.tmt.metrics.intra_and_inter_target_time import (
    calculate_intra_target_time,
    calculate_inter_target_time,
    get_intra_target_intervals
)

# Calcular intra-target time
intra_time = calculate_intra_target_time(trial, subject)

# Calcular inter-target time (requiere intra_time pre-calculado)
inter_time = calculate_inter_target_time(trial, intra_time)

# Obtener intervalos detallados
intervals = get_intra_target_intervals(trial, subject)
# Returns: [(target, start_time, end_time), ...]
```

### Ejemplo de Uso

```python
trial = TMTTrial(...)
subject = TMTSubject(target_radius=5.0, ...)

# Calcular métricas
intra = calculate_intra_target_time(trial, subject)
inter = calculate_inter_target_time(trial, intra)
total = trial.get_cursor_trail_from_start()[-1].time - trial.get_cursor_trail_from_start()[0].time

# Calcular ratio
ratio = intra / total if total > 0 else 0

print(f"Intra-target time: {intra:.2f}s ({ratio*100:.1f}%)")
print(f"Inter-target time: {inter:.2f}s ({(1-ratio)*100:.1f}%)")
print(f"Total time: {total:.2f}s")
```

---

## Referencias

- Ver `test_intra_target_time.py` para ejemplos detallados de cálculo
- Ver `how_to_add_new_metric.md` para agregar nuevas métricas relacionadas

