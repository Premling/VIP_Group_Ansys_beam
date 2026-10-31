"""Beam-theory evaluator for the VIP individual design project.

Inputs are beam width and height in millimeters. The model represents a
simply supported rectangular steel beam under a central static point load,
with small deflections and linear elastic behavior. It excludes local
load/support contact effects, self-weight, and impact behavior.
"""

import math
from numbers import Real


# Consistent N, mm, MPa, kg system (1 MPa = 1 N/mm**2).
SPAN_MM = 1000.0
LOAD_N = 10_000.0
ELASTIC_MODULUS_MPA = 200_000.0
DENSITY_KG_PER_MM3 = 7.85e-6

WIDTH_BOUNDS_MM = (20.0, 100.0)
HEIGHT_BOUNDS_MM = (40.0, 120.0)
MAX_DEFLECTION_MM = 1.0
MAX_VON_MISES_STRESS_MPA = 120.0


def evaluate_beam(width_mm: float, height_mm: float) -> dict[str, float | bool]:
    """Evaluate one design and return its mass, deflection, stress, and feasibility.

    Args:
        width_mm: Rectangular cross-section width, in mm (20 to 100).
        height_mm: Rectangular cross-section height, in mm (40 to 120).

    Returns:
        A dictionary with ``mass_kg``, ``max_deflection_mm``,
        ``max_von_mises_stress_MPa``, and ``feasible``. The stress is the
        bending-surface von Mises stress at midspan. At this surface the
        shear traction from elementary beam theory is zero, so the von
        Mises stress equals the magnitude of the bending normal stress.

    Raises:
        ValueError: If either input is not a finite real number or is outside
            the specified design bounds. Validate agent proposals before
            passing them to this function.
    """
    for name, value, bounds in (
        ("width_mm", width_mm, WIDTH_BOUNDS_MM),
        ("height_mm", height_mm, HEIGHT_BOUNDS_MM),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number in mm")
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]} mm")

    width_mm = float(width_mm)
    height_mm = float(height_mm)

    second_moment_mm4 = width_mm * height_mm**3 / 12.0
    max_moment_n_mm = LOAD_N * SPAN_MM / 4.0

    mass_kg = DENSITY_KG_PER_MM3 * width_mm * height_mm * SPAN_MM
    max_deflection_mm = LOAD_N * SPAN_MM**3 / (
        48.0 * ELASTIC_MODULUS_MPA * second_moment_mm4
    )
    max_von_mises_stress_mpa = max_moment_n_mm * (height_mm / 2.0) / second_moment_mm4

    return {
        "mass_kg": mass_kg,
        "max_deflection_mm": max_deflection_mm,
        "max_von_mises_stress_MPa": max_von_mises_stress_mpa,
        "feasible": (
            max_deflection_mm <= MAX_DEFLECTION_MM
            and max_von_mises_stress_mpa <= MAX_VON_MISES_STRESS_MPA
        ),
    }
