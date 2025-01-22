from abc import abstractmethod, ABC
from typing import Optional

from ..model.tmt_model import TMTExperiment


class TMTMapper(ABC):

    @abstractmethod
    def map(self, data_path: str, metadata_path: Optional[str] = None) -> TMTExperiment:
        """
        Método abstracto para mapear los datos (y opcionalmente la metadata) a una instancia de TMTExperiment.

        Parameters:
        - data_path: str -> La ruta al archivo de datos.
        - metadata_path: Optional[str] -> La ruta al archivo de metadata. (Opcional)

        Returns:
        - TMTExperiment -> Instancia de TMTExperiment mapeada a partir de los datos (y la metadata si aplica).
        """
        pass
