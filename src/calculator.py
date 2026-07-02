"""
Carbon emission and EcoScore calculation utilities.

This module provides mathematical functions for calculating:
- Energy-related greenhouse gas emissions
- EcoScore classification based on carbon intensity

The implementation follows Clean Code principles and is designed
to be easily testable and maintainable.
"""

from typing import Dict

# ============================================================================
# Emission Factors (kg CO2e)
# ============================================================================

NATURAL_GAS_EMISSION_FACTOR: float = 1.90  # kg CO2e / m³
ELECTRICITY_EMISSION_FACTOR: float = 0.45  # kg CO2e / kWh


def _validate_non_negative(value: float, field_name: str) -> None:
    """Validate that a numeric value is not negative.

    Args:
        value: Numeric value to validate.
        field_name: Name of the validated field.

    Raises:
        ValueError: If the value is negative.
    """
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")


def _validate_positive_tonnage(urun_tonaj: float) -> None:
    """Validate that production tonnage is greater than zero.

    Args:
        urun_tonaj: Production amount in tons.

    Raises:
        ValueError: If tonnage is negative or zero.
    """
    if urun_tonaj < 0:
        raise ValueError("urun_tonaj cannot be negative.")

    if urun_tonaj == 0:
        raise ValueError("urun_tonaj cannot be zero.")


def _round(value: float) -> float:
    """Round a numeric value to two decimal places.

    Args:
        value: Numeric value.

    Returns:
        Rounded float.
    """
    return round(value, 2)


def calculate_energy_emissions(
    dogalgaz_m3: float,
    elektrik_kwh: float,
) -> Dict[str, float]:
    """Calculate greenhouse gas emissions from energy consumption.

    Calculates emissions generated from natural gas and electricity
    consumption using predefined emission factors.

    Args:
        dogalgaz_m3: Natural gas consumption in cubic meters.
        elektrik_kwh: Electricity consumption in kilowatt-hours.

    Returns:
        Dictionary containing natural gas emissions, electricity
        emissions, and total emissions (kg CO2e).

    Raises:
        ValueError:
            If natural gas or electricity consumption is negative.
    """
    _validate_non_negative(dogalgaz_m3, "dogalgaz_m3")
    _validate_non_negative(elektrik_kwh, "elektrik_kwh")

    # Calculate individual emission sources.
    dogalgaz_emisyonu = (
        dogalgaz_m3 * NATURAL_GAS_EMISSION_FACTOR
    )
    elektrik_emisyonu = (
        elektrik_kwh * ELECTRICITY_EMISSION_FACTOR
    )

    # Total greenhouse gas emissions.
    toplam_emisyon = (
        dogalgaz_emisyonu + elektrik_emisyonu
    )

    return {
        "dogalgaz_emisyonu": _round(dogalgaz_emisyonu),
        "elektrik_emisyonu": _round(elektrik_emisyonu),
        "toplam_emisyon": _round(toplam_emisyon),
    }


def _determine_ecoscore(co2_per_ton: float) -> tuple[str, str]:
    """Determine EcoScore based on carbon intensity.

    Args:
        co2_per_ton: Carbon intensity in kg CO2e/ton.

    Returns:
        Tuple containing EcoScore and description.
    """
    if co2_per_ton < 200:
        return "A++", "Mükemmel Yeşil"

    if co2_per_ton < 400:
        return "A", "Düşük Karbon"

    if co2_per_ton < 600:
        return "B", "İyi"

    if co2_per_ton < 800:
        return "C", "Orta"

    if co2_per_ton <= 1000:
        return "D", "Geliştirilmeli"

    return "E", "Yüksek Karbon Yoğunluğu"


def calculate_ecoscore(
    toplam_co2_kg: float,
    urun_tonaj: float,
) -> Dict[str, float | str]:
    """Calculate EcoScore from total emissions and production tonnage.

    Carbon intensity is calculated as:

        kg CO2e / ton

    The resulting value is classified into an EcoScore category.

    Args:
        toplam_co2_kg: Total greenhouse gas emissions (kg CO2e).
        urun_tonaj: Production amount in tons.

    Returns:
        Dictionary containing carbon intensity, EcoScore,
        and descriptive classification.

    Raises:
        ValueError:
            If total emissions are negative.
            If production tonnage is negative.
            If production tonnage is zero.
    """
    _validate_non_negative(toplam_co2_kg, "toplam_co2_kg")
    _validate_positive_tonnage(urun_tonaj)

    # Carbon intensity calculation.
    co2_per_ton = toplam_co2_kg / urun_tonaj

    ecoscore, description = _determine_ecoscore(co2_per_ton)

    return {
        "co2_per_ton": _round(co2_per_ton),
        "ecoscore": ecoscore,
        "description": description,
    }
# ============================================================================
# Compatibility functions for Streamlit App
# ============================================================================

def calculate_carbon_footprint(
    dogalgaz_m3: float,
    elektrik_kwh: float,
) -> float:
    """
    Calculate total carbon footprint.

    Returns:
        Total emissions in kg CO2e.
    """
    result = calculate_energy_emissions(
        dogalgaz_m3,
        elektrik_kwh,
    )

    return result["toplam_emisyon"]


def get_ecoscore_grade(
    co2_per_ton: float,
) -> str:
    """
    Return only EcoScore grade.

    Args:
        co2_per_ton:
            Carbon intensity (kg CO2e/ton)

    Returns:
        EcoScore letter.
    """

    grade, _ = _determine_ecoscore(co2_per_ton)

    return grade
