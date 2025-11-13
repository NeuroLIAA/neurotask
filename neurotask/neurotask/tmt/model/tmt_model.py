from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional


@dataclass
class Coordinate:
    x: float
    y: float


@dataclass
class TMTTarget:
    content: str
    position: Coordinate


@dataclass
class CursorInfo:
    position: Coordinate
    time: float


class TrialType(Enum):
    PART_A = "PART_A"
    PART_B = "PART_B"


@dataclass
class TMTTrial:
    stimuli: List[TMTTarget]

    cursor_trail: List[CursorInfo]

    trial_type: TrialType

    id: str

    order_of_appearance: int

    # Reaction Time: tiempo total que tardó el sujeto en completar el trial
    rt: float

    # Indica si el trial tiene un punto de inicio personalizado, por ejemplo para Datapruebas es el primer click
    with_custom_start: bool = False

    # Punto de inicio del trial solo debe estar presente si with_custom_start es True
    start: Optional[CursorInfo] = None

    mapping_error: bool = False

    @classmethod
    def invalid_trial(cls, trial_id: str, order_of_appearance: int, stimuli, trial_type) -> "TMTTrial":
        """
        Crea un trial inválido con todos los campos vacíos o None, y mapping_error=True.
        """
        return cls(
            stimuli=stimuli,
            cursor_trail=[],
            trial_type=trial_type,  # Tipo desconocido o inválido
            id=trial_id,
            order_of_appearance=order_of_appearance,
            rt=0.0,
            with_custom_start=False,
            start=None,
            mapping_error=True,  # 🔹 marcamos que hubo un error en el mapeo
        )

    def get_cursor_trail_from_start(self) -> List[CursorInfo]:
        """
        Si el trial tiene un punto de inicio personalizado, esta función devuelve la trayectoria del cursor
        después de ese punto. Si no tiene un punto de inicio personalizado, devuelve la trayectoria completa.
        """
        if not self.with_custom_start:
            return self.cursor_trail

        if self.start is None:
            raise ValueError("first_click_cursor_info no está definido en este trial.")

        start_time = self.start.time

        filtered_cursor_trail = [cursor_info for cursor_info in self.cursor_trail
                                 if cursor_info.time >= start_time]

        return filtered_cursor_trail

    def is_valid(self):
        """
        Un trial es válido si tiene al menos 3 puntos en la trayectoria del cursor después del punto de inicio.
        Definimos esto para que nos permita calcular medidas de velocidad y aceleración.
        Ademas, si el trial tiene un punto de inicio personalizado, este debe estar definido.
        """
        if self.mapping_error:
            return False

        valid_length = self.is_valid_length()
        valid_start_configuration = self.is_valid_start_configuration()

        return valid_length and valid_start_configuration

    def is_valid_start_configuration(self):
        return (self.with_custom_start is True) == (self.start is not None)

    def is_valid_length(self):
        return len(self.get_cursor_trail_from_start()) > 2


from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SessionContext:
    device: Optional[str]
    hand: Optional[str]
    device_config: Optional[str]
    alcohol_drugs: Optional[str]
    treatment: Optional[str]
    pad_usage: Optional[str]
    final_comment: Optional[str]


@dataclass
class SubjectPersonalInformation:
    birthdate: datetime
    gender: Optional[str]
    education_level: Optional[str]
    nationality: Optional[str]
    residence_country: Optional[str]
    residence_region: Optional[str]


@dataclass
class TMTSubject:
    training_trials: List[TMTTrial]
    testing_trials: List[TMTTrial]
    target_radius: Optional[float]
    canvas_size: Optional[int]
    session_data: Optional[dict] = None


@dataclass
class TMTExperiment:
    subjects: dict[str, TMTSubject]
