"""Contrato de extracción estructural, pendiente de implementar.

No se calcula estructura desde Tg. Las fracciones de coordinación, la selección
de fotogramas y las condiciones periódicas deberán validarse con trayectorias.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StructuralDescriptors:
    composition_fingerprint: str
    source_file: str
    simulation_method: str
    cn_te_o: float
    mean_te_o_angstrom: float
    mean_te_o_te_deg: float
    fraction_te_o3: float
    fraction_te_o4: float
    fraction_te_o5: float


def extract_descriptors(trajectory: Path, analysis_protocol: Path) -> StructuralDescriptors:
    """Pendiente: vecinos, periodicidad, promedios y coordinación Te--O."""
    raise NotImplementedError(
        "Se necesitan trayectorias reales y un protocolo de análisis estructural validado."
    )
