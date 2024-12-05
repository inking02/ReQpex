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


def Waveform_Pulse(Omega: float, T: float, delta_0 : float = -5, delta_f:float = 5) -> Pulse:
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - delta_0 (float):
    - delta_f (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),0
    )
    return adiabatic_pulse


def Rise_Fall_Waveform(Omega: float, T: float, delta_0 : float = -5, delta_f:float = 5):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    up = RampWaveform(T/2, 0, Omega)
    down = RampWaveform(T/2, Omega, 0)
    d_up = RampWaveform(T/2, delta_0, 0)
    d_down = RampWaveform(T/2, 0, delta_f)

    rise_fall_Pulse = Pulse(
        CompositeWaveform(up, down),
        CompositeWaveform(d_up, d_down),0
        )
    
    return rise_fall_Pulse

def Blackman_Waveform_Pulse(Omega: float, T: float, delta_0 : float = -5, delta_f:float = 5):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    Blackman_Pulse = Pulse(
        BlackmanWaveform(T, Omega),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),0
    )

    return Blackman_Pulse


def Constant_pulse_pyramide(
    Omega: float,
    T: float,
    T_pyramide: float,
    delta: float, 
    delta_0 : float = -5, 
    delta_f:float = 5
):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):
    - T_pyramide (float):

    Returns:
    Pulse: A pulser pusle object.
    """

    Constant_1 = ConstantWaveform((T-T_pyramide)/2, Omega-delta)
    up = RampWaveform(T_pyramide/2, Omega-delta, Omega)
    down = RampWaveform(T_pyramide/2, Omega, Omega-delta)
    Constant_2 = ConstantWaveform((T-T_pyramide)/2, Omega-delta)
    
    r_Pulse = Pulse(
        CompositeWaveform(Constant_1, up, down, Constant_2),
        InterpolatedWaveform(T, [delta_0, 0, delta_f]),0
    )
    return r_Pulse


def rise_sweep_fall(Omega: float, T: float, delta_0 : float = -5, delta_f:float = 5):
    """
    Creates a waveform pulse object.

    Parameters:
    - Omega (float):
    - T (float):


    Returns:
    Pulse: A pulser pusle object.
    """
    rise = RampWaveform(T/4, 0, Omega)
    sweep = ConstantWaveform(T/2, Omega)
    fall = RampWaveform(T/4, Omega, 0)
    Omega_Wave = CompositeWaveform(rise, sweep, fall)

    constant1_d = ConstantWaveform(T/4, delta_0)
    rise_d = RampWaveform(T/2, delta_0, delta_f)
    constant2_d = ConstantWaveform(T/4, delta_f)

    detuning = CompositeWaveform(constant1_d, rise_d, constant2_d)
    return Pulse(Omega_Wave, detuning, 0)


def Pulse_constructor(
    T: float,
    Pulse_type: str,
    T_pyramide: float = 0,
    delta: float = 0
):
    """
    Creates a waveform pulse object.

    Parameters:
    - T (float):
    - Pulse_type (str):
    - T_pyramide (float):

    Returns:
    Pulse: A pulser pusle object.
    """
    if Pulse_type == "Waveform":
        return lambda Omega: Waveform_Pulse(Omega, T)

    if Pulse_type == "Rise_fall":
        return lambda Omega: Rise_Fall_Waveform(Omega, T)

    if Pulse_type == "Blackman":
        return lambda Omega: Blackman_Waveform_Pulse(Omega, T)

    if Pulse_type == "Pyramide":
        return lambda Omega: Constant_pulse_pyramide(
            Omega, T, T_pyramide, delta
        )
    
    if Pulse_type == "Rise_sweep_fall":
        return lambda Omega: rise_sweep_fall(Omega, T)
