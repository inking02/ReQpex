"""
File containing the pulses that can be used on the QMIS utils.
"""

from pulser.waveforms import (
    InterpolatedWaveform,
    RampWaveform,
    ConstantWaveform,
    CompositeWaveform,
    BlackmanWaveform,
)
from pulser import Pulse


def Waveform_Pulse(
    Omega: float, T: float, delta_0: float = -5, delta_f: float = 5
) -> Pulse:
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float): The maximal value of Omega for the pulse in rad/µs.
    - T (float): The total time during which the pulse is applied.
    - delta_0 (float=-5): The initial decoherence value in rad/µs.
    - delta_f (float=5): The final decoherence value in rad/µs.

    Returns:
    - Pulse: A pulser pulse object.
    """
    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    return adiabatic_pulse


def Rise_Fall_Waveform(Omega: float, T: float, delta_0: float = -5, delta_f: float = 5):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float): The maximal value of Omega for the pulse in rad/µs.
    - T (float): The total time during which the pulse is applied.
    - delta_0 (float=-5): The initial decoherence value in rad/µs.
    - delta_f (float=5): The final decoherence value in rad/µs.

    Returns:
    - Pulse: A pulser pulse object.
    """
    up = RampWaveform(T / 2, 0, Omega)
    down = RampWaveform(T / 2, Omega, 0)
    d_up = RampWaveform(T / 2, delta_0, 0)
    d_down = RampWaveform(T / 2, 0, delta_f)

    rise_fall_Pulse = Pulse(
        CompositeWaveform(up, down), CompositeWaveform(d_up, d_down), 0
    )

    return rise_fall_Pulse


def Blackman_Waveform_Pulse(
    Omega: float, T: float, delta_0: float = -5, delta_f: float = 5
):
    """
    Creates a waveform pulse object. The blackman waveform is a pulse with a normal form with an area given by Omega.

    Parameters:
    - Omega (float): The maximal value of Omega for the pulse in rad/µs.
    - T (float): The total time during which the pulse is applied.
    - delta_0 (float=-5): The initial decoherence value in rad/µs.
    - delta_f (float=5): The final decoherence value in rad/µs.

    Returns:
    - Pulse: A pulser pulse object.
    """
    Blackman_Pulse = Pulse(
        BlackmanWaveform(T, Omega), InterpolatedWaveform(T, [delta_0, 0, delta_f]), 0
    )

    return Blackman_Pulse


def Constant_pulse_pyramid(
    Omega: float,
    T: float,
    T_pyramid: float,
    delta: float,
    delta_0: float = -5,
    delta_f: float = 5,
):
    """
    Creates a waveform pulse object. The constant_pulse_pyramid is a constant pulse with a value of Omega - delta. There's a little part of rise_fall in the middle of the pulse.

    Parameters:
    - Omega (float): The maximal value of Omega for the pulse in rad/µs.
    - T (float): The total time during which the pulse is applied.
    - T_pyramid (float): The time for the rise_fall part in the middle in µs.
    - delta (float) : The difference between Omega and the constant parts of the pulse.
    - delta_0 (float=-5): The initial decoherence value in rad/µs.
    - delta_f (float=5): The final decoherence value in rad/µs.

    Returns:
    - Pulse: A pulser pulse object.
    """

    Constant_1 = ConstantWaveform((T - T_pyramid) / 2, Omega - delta)
    up = RampWaveform(T_pyramid / 2, Omega - delta, Omega)
    down = RampWaveform(T_pyramid / 2, Omega, Omega - delta)
    Constant_2 = ConstantWaveform((T - T_pyramid) / 2, Omega - delta)

    r_Pulse = Pulse(
        CompositeWaveform(Constant_1, up, down, Constant_2),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),
        0,
    )
    return r_Pulse


def rise_sweep_fall(Omega: float, T: float, delta_0: float = -5, delta_f: float = 5):
    """
    Creates a waveform pulse object. The rise_sweep_fall pulse rises, stays constant and finally drop back to 0.

    Parameters:
    - Omega (float): The maximal value of Omega for the pulse in rad/µs.
    - T (float): The total time during which the pulse is applied.
    - delta_0 (float): The initial decoherence value in rad/µs.
    - delta_f (float): The final decoherence value in rad/µs.


    Returns:
    - Pulse: A pulser pulse object.
    """
    rise = RampWaveform(T / 4, 0, Omega)
    sweep = ConstantWaveform(T / 2, Omega)
    fall = RampWaveform(T / 4, Omega, 0)
    Omega_Wave = CompositeWaveform(rise, sweep, fall)

    constant1_d = ConstantWaveform(T / 4, delta_0)
    rise_d = RampWaveform(T / 2, delta_0, delta_f)
    constant2_d = ConstantWaveform(T / 4, delta_f)

    detuning = CompositeWaveform(constant1_d, rise_d, constant2_d)
    return Pulse(Omega_Wave, detuning, 0)


def pulse_constructor(
    T: float,
    Pulse_type: str,
    T_pyramid: float = 0,
    delta: float = 0,
    delta_0: float = -5,
    delta_f: float = 5,
):
    """
    Creates a waveform pulse object.

    Parameters:
    - T (float): The total time during which the pulse is applied.
    - Pulse_type (str): The type of pulse chosen. The five types implemented are 'Waveform', 'Blackman', 'Rise_fall', 'Pyramid', and 'Rise_sweep_fall
    - T_pyramid (float=0): The time for the rise_fall part in the middle in µs, only for the 'Pyramid' pulse.
    - delta (float=0) : The difference between Omega and the constant parts of the pulse.
    - delta_0 (float=-5): The initial decoherence value in rad/µs.
    - delta_f (float=5): The final decoherence value in rad/µs.

    Returns:
    - Pulse: A pulser pulse callable with a parameter Omega to be defined.
    """
    assert Pulse_type in [
        "Waveform",
        "Blackman",
        "Rise_fall",
        "Pyramid",
        "Rise_sweep_fall",
    ], "The five types implemented are 'Waveform', 'Blackman', 'Rise_fall', 'Pyramid', and 'Rise_sweep_fall', please use one of them."

    if Pulse_type == "Waveform":
        return lambda Omega: Waveform_Pulse(Omega, T, delta_0, delta_f)

    if Pulse_type == "Rise_fall":
        return lambda Omega: Rise_Fall_Waveform(Omega, T, delta_0, delta_f)

    if Pulse_type == "Blackman":
        return lambda Omega: Blackman_Waveform_Pulse(Omega, T, delta_0, delta_f)

    if Pulse_type == "Pyramid":
        return lambda Omega: Constant_pulse_pyramid(
            Omega, T, T_pyramid, delta, delta_0, delta_f
        )

    if Pulse_type == "Rise_sweep_fall":
        return lambda Omega: rise_sweep_fall(Omega, T, delta_0, delta_f)
