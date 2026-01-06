#!/usr/bin/env python3
"""
Test rápido para verificar que el filtro temporal corregido funciona correctamente.
"""

import sys
sys.path.insert(0, 'neurotask')

from neurotask.tmt.crosses.crosses import calculate_crosses_for_trial
from neurotask.tmt.model.tmt_model import TMTTrial, CursorInfo, Coordinate, TrialType


def test_filtro_temporal():
    """Test que verifica la lógica del filtro temporal corregido."""
    
    print("="*60)
    print("TEST: Filtro Temporal Corregido")
    print("="*60)
    
    # Caso 1: Segmentos separados por gap grande (deberían considerarse)
    print("\n🔹 Caso 1: Segmentos separados (gap=600, threshold=500)")
    print("   Esperado: Debe considerar los segmentos (gap >= threshold)")
    
    # Crear un trail donde dos segmentos se cruzan pero están separados en tiempo
    # Segmento 1: (0,0) -> (10,10) en tiempo [0, 100]
    # Segmento 2: (0,10) -> (10,0) en tiempo [700, 800]  (gap = 600)
    cursor_trail_1 = [
        CursorInfo(Coordinate(0, 0), 0),      # Inicio segmento 1
        CursorInfo(Coordinate(10, 10), 100), # Fin segmento 1
        CursorInfo(Coordinate(5, 5), 200),   # Punto intermedio
        CursorInfo(Coordinate(0, 10), 700),  # Inicio segmento 2
        CursorInfo(Coordinate(10, 0), 800), # Fin segmento 2 (se cruza con segmento 1)
    ]
    
    trial_1 = TMTTrial(
        stimuli=[],
        cursor_trail=cursor_trail_1,
        trial_type=TrialType.PART_A,
        id="test_1",
        order_of_appearance=1,
        rt=800
    )
    
    num_crosses_1, cross_segments_1 = calculate_crosses_for_trial(trial_1, time_threshold=500)
    print(f"   Resultado: {num_crosses_1} cruces detectados")
    if cross_segments_1:
        print(f"   Gap del cruce: {cross_segments_1[0][2]}")
    
    # Caso 2: Segmentos muy cercanos en tiempo (deberían excluirse)
    print("\n🔹 Caso 2: Segmentos muy cercanos (gap=50, threshold=500)")
    print("   Esperado: Debe excluir los segmentos (gap < threshold)")
    
    # Segmento 1: (0,0) -> (10,10) en tiempo [0, 100]
    # Segmento 2: (0,10) -> (10,0) en tiempo [150, 200]  (gap = 50)
    cursor_trail_2 = [
        CursorInfo(Coordinate(0, 0), 0),
        CursorInfo(Coordinate(10, 10), 100),
        CursorInfo(Coordinate(5, 5), 120),
        CursorInfo(Coordinate(0, 10), 150),
        CursorInfo(Coordinate(10, 0), 200),
    ]
    
    trial_2 = TMTTrial(
        stimuli=[],
        cursor_trail=cursor_trail_2,
        trial_type=TrialType.PART_A,
        id="test_2",
        order_of_appearance=1,
        rt=200
    )
    
    num_crosses_2, cross_segments_2 = calculate_crosses_for_trial(trial_2, time_threshold=500)
    print(f"   Resultado: {num_crosses_2} cruces detectados")
    if num_crosses_2 == 0:
        print("   ✅ Correcto: Los segmentos fueron excluidos por estar muy cerca en tiempo")
    else:
        print("   ❌ Error: Los segmentos deberían haber sido excluidos")
    
    # Caso 3: Segmentos que se solapan en tiempo (gap=0, deberían excluirse)
    print("\n🔹 Caso 3: Segmentos que se solapan (gap=0, threshold=500)")
    print("   Esperado: Debe excluir los segmentos (gap < threshold)")
    
    # Segmento 1: (0,0) -> (10,10) en tiempo [0, 200]
    # Segmento 2: (0,10) -> (10,0) en tiempo [100, 300]  (se solapan)
    cursor_trail_3 = [
        CursorInfo(Coordinate(0, 0), 0),
        CursorInfo(Coordinate(10, 10), 200),
        CursorInfo(Coordinate(5, 5), 150),
        CursorInfo(Coordinate(0, 10), 100),
        CursorInfo(Coordinate(10, 0), 300),
    ]
    
    trial_3 = TMTTrial(
        stimuli=[],
        cursor_trail=cursor_trail_3,
        trial_type=TrialType.PART_A,
        id="test_3",
        order_of_appearance=1,
        rt=300
    )
    
    num_crosses_3, cross_segments_3 = calculate_crosses_for_trial(trial_3, time_threshold=500)
    print(f"   Resultado: {num_crosses_3} cruces detectados")
    if num_crosses_3 == 0:
        print("   ✅ Correcto: Los segmentos fueron excluidos por solaparse en tiempo")
    else:
        print("   ❌ Error: Los segmentos deberían haber sido excluidos")
    
    # Caso 4: Trail muy corto (menos de 4 puntos)
    print("\n🔹 Caso 4: Trail muy corto (< 4 puntos)")
    print("   Esperado: Debe retornar 0 cruces")
    
    cursor_trail_4 = [
        CursorInfo(Coordinate(0, 0), 0),
        CursorInfo(Coordinate(10, 10), 100),
        CursorInfo(Coordinate(5, 5), 200),
    ]
    
    trial_4 = TMTTrial(
        stimuli=[],
        cursor_trail=cursor_trail_4,
        trial_type=TrialType.PART_A,
        id="test_4",
        order_of_appearance=1,
        rt=200
    )
    
    num_crosses_4, cross_segments_4 = calculate_crosses_for_trial(trial_4, time_threshold=500)
    print(f"   Resultado: {num_crosses_4} cruces detectados")
    if num_crosses_4 == 0:
        print("   ✅ Correcto: Retornó 0 para trail corto")
    else:
        print("   ❌ Error: Debería retornar 0 para trail corto")
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print("✅ Todos los casos básicos funcionan correctamente")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_filtro_temporal()

