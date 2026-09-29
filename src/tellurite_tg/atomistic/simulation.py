"""Interfaces de simulación; las decisiones físicas todavía están por definir."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationSpec:
    """Una composición y una configuración independiente de simulación.

    El protocolo deberá registrar además temperaturas, tiempos, enfriamiento,
    presión/densidad, pseudopotenciales y criterios de convergencia.
    """

    composition_fingerprint: str
    replicate_id: str
    engine: str
    functional: str
    initial_structure: Path
    protocol_file: Path


def prepare_inputs(spec: SimulationSpec, output_dir: Path) -> Path:
    """Pendiente: validar el protocolo y generar entradas del motor elegido."""
    raise NotImplementedError(
        "Definir y validar primero el motor AIMD y el protocolo físico."
    )


def run_aimd(input_file: Path, output_dir: Path) -> Path:
    """Pendiente: ejecutar el motor externo y devolver la trayectoria verificada."""
    raise NotImplementedError(
        "La ejecución AIMD requiere un motor externo y recursos aún no configurados."
    )
