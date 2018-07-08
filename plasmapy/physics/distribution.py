"""
Common distribution functions for plasmas, such as the Maxwelian or
Kappa distributions. Functionality is intended to include generation,
fitting and calculation.
"""
import astropy as astropy
from astropy import units as u
from plasmapy.physics import parameters
import numpy as np
from scipy.special import gamma

__all__ = ["Maxwellian_1D",
           "Maxwellian_velocity_2D",
           "Maxwellian_velocity_3D",
           "Maxwellian_speed_1D",
           "Maxwellian_speed_2D",
           "Maxwellian_speed_3D",
           "kappa_velocity_1D",
           "kappa_velocity_3D"]

def _v_drift_units(v_drift):
    # Helper method to assign units to  v_drift if it takes a default value
    if (v_drift == 0 and
        not isinstance(v_drift, astropy.units.quantity.Quantity)):
        v_drift = v_drift * u.m / u.s
    else:
        v_drift = v_drift.to(u.m / u.s)
    return v_drift

def Maxwellian_1D(v,
                  T,
                  particle="e",
                  v_drift=0,
                  vTh=np.nan,
                  units="units"):
    r"""
    Probability distribution function of velocity for a Maxwellian 
    distribution in 1D.

    Returns the probability density function at the velocity `v` in m/s
    to find a particle `particle` in a plasma of temperature `T`
    following the Maxwellian distribution function.

    Parameters
    ----------
    v: ~astropy.units.Quantity
        The velocity in units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature in Kelvin.

    particle: str, optional
        Representation of the particle species(e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    v_drift: ~astropy.units.Quantity, optional
        The drift velocity in units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable velocity) in m/s. This is used for
        optimization purposes to avoid re-calculating vTh, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in units of Velocity^-1, normalized so that
        :math:`\int_{-\infty}^{+\infty} f(v) dv = 1`.

    Raises
    ------
    TypeError
        The parameter arguments are not Quantities and
        cannot be converted into Quantities.

    ~astropy.units.UnitConversionError
        If the parameters are not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In one dimension, the Maxwellian distribution function for a particle of
    mass m, velocity v, a drift velocity V and with temperature T is:

    .. math::

        f = \sqrt{\frac{m}{2 \pi k_B T}} e^{-\frac{m}{2 k_B T} (v-V)^2}
        \equiv \frac{1}{\sqrt{\pi v_{Th}^2}} e^{-(v - v_{drift})^2 / v_{Th}^2}

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed

    Examples
    --------
    >>> from plasmapy.physics import Maxwellian_1D
    >>> from astropy import units as u
    >>> v=1*u.m/u.s
    >>> Maxwellian_1D(v=v, T=30000 * u.K, particle='e', v_drift=0 * u.m / u.s)
    <Quantity 5.91632969e-07 s / m>
    """

    if units == "units":
        # unit checks and conversions
        # checking velocity units
        v = v.to(u.m / u.s)
        # Catching case where drift velocities have default values, they
        # need to be assigned units
        v_drift = _v_drift_units(v_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = (parameters.thermal_speed(T * u.K,
                                        particle=particle,
                                        method="most_probable")).si.value
    # Get thermal velocity squared
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = (v - v_drift) ** 2
    # calculating distribution function
    coeff = (vThSq * np.pi) ** (-1 / 2)
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff * expTerm
    if units == "units":
        return distFunc.to(u.s / u.m)
    elif units == "unitless":
        return distFunc


def Maxwellian_velocity_2D(vx,
                           vy,
                           T,
                           particle="e",
                           vx_drift=0,
                           vy_drift=0,
                           vTh=np.nan,
                           units="units"):
    r"""
    Probability distribution function of velocity for a Maxwellian 
    distribution in 2D.

    Return the probability density function for finding a particle with 
    velocity components `vx` and `vy` in m/s in an equilibrium plasma of 
    temperature `T` which follows the 2D Maxwellian distribution function. 
    This function assumes Cartesian coordinates.

    Parameters
    ----------
    vx: ~astropy.units.Quantity
        The velocity in x-direction units convertible to m/s.

    vy: ~astropy.units.Quantity
        The velocity in y-direction units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    particle: str, optional
        Representation of the particle species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    vx_drift: ~astropy.units.Quantity, optional
        The drift velocity in x-direction units convertible to m/s.

    vy_drift: ~astropy.units.Quantity, optional
        The drift velocity in y-direction units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating `vTh`, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in Velocity^-1, normalized so that
        :math:`\iiint_{0}^{\infty} f(\vec{v}) d\vec{v} = 1`.

    Raises
    ------
    TypeError
        A parameter argument is not a `~astropy.units.Quantity` and
        cannot be converted into a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In 2D, the Maxwellian velocity distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` is given by:

    .. math::

        f = (\pi v_{Th}^2)^{-1} \exp \left [-(\vec{v} -
        \vec{V}_{drift})^2 / v_{Th}^2 \right ]

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed.

    See also
    --------
    Maxwellian_1D

    Example
    -------
    >>> from plasmapy.physics import Maxwellian_velocity_2D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> Maxwellian_velocity_2D(vx=v,
    ... vy=v,
    ... T=30000*u.K,
    ... particle='e',
    ... vx_drift=0 * u.m / u.s,
    ... vy_drift=0 * u.m / u.s)
    <Quantity 3.5002957e-13 s2 / m2>


    """
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        vx = vx.to(u.m / u.s)
        vy = vy.to(u.m / u.s)
        # catching case where drift velocities have default values, they
        # need to be assigned units
        vx_drift = _v_drift_units(vx_drift)
        vy_drift = _v_drift_units(vy_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = parameters.thermal_speed(T * u.K,
                                       particle=particle,
                                       method="most_probable").si.value
    # accounting for thermal velocity in 2D
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = ((vx - vx_drift) ** 2 + (vy - vy_drift) ** 2)
    # calculating distribution function
    coeff = (vThSq * np.pi) ** (-1)
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff * expTerm
    if units == "units":
        return distFunc.to((u.s / u.m)**2)
    elif units == "unitless":
        return distFunc


def Maxwellian_velocity_3D(vx,
                           vy,
                           vz,
                           T,
                           particle="e",
                           vx_drift=0,
                           vy_drift=0,
                           vz_drift=0,
                           vTh=np.nan,
                           units="units"):
    r"""
    Probability distribution function of velocity for a Maxwellian 
    distribution in 3D.

    Return the probability density function for finding a particle with 
    velocity components `vx`, `vy`, and `vz` in m/s in an equilibrium 
    plasma of temperature `T` which follows the 3D Maxwellian distribution 
    function. This function assumes Cartesian coordinates.

    Parameters
    ----------
    vx: ~astropy.units.Quantity
        The velocity in x-direction units convertible to m/s.

    vy: ~astropy.units.Quantity
        The velocity in y-direction units convertible to m/s.

    vz: ~astropy.units.Quantity
        The velocity in z-direction units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    particle: str, optional
        Representation of the particle species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    vx_drift: ~astropy.units.Quantity, optional
        The drift velocity in x-direction units convertible to m/s.

    vy_drift: ~astropy.units.Quantity, optional
        The drift velocity in y-direction units convertible to m/s.

    vz_drift: ~astropy.units.Quantity, optional
        The drift velocity in z-direction units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating `vTh`, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in Velocity^-1, normalized so that
        :math:`\iiint_{0}^{\infty} f(\vec{v}) d\vec{v} = 1`.

    Raises
    ------
    TypeError
        A parameter argument is not a `~astropy.units.Quantity` and
        cannot be converted into a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In 3D, the Maxwellian speed distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` is given by:

    .. math::

        f = (\pi v_{Th}^2)^{-3/2} \exp \left [-(\vec{v} -
        \vec{V}_{drift})^2 / v_{Th}^2 \right ]

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed.

    See also
    --------
    Maxwellian_1D

    Example
    -------
    >>> from plasmapy.physics import Maxwellian_velocity_3D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> Maxwellian_velocity_3D(vx=v,
    ... vy=v,
    ... vz=v,
    ... T=30000 * u.K,
    ... particle='e',
    ... vx_drift=0 * u.m / u.s,
    ... vy_drift=0 * u.m / u.s,
    ... vz_drift=0 * u.m / u.s)
    <Quantity 2.07089033e-19 s3 / m3>


    """
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        vx = vx.to(u.m / u.s)
        vy = vy.to(u.m / u.s)
        vz = vz.to(u.m / u.s)
        # catching case where drift velocities have default values, they
        # need to be assigned units
        vx_drift = _v_drift_units(vx_drift)
        vy_drift = _v_drift_units(vy_drift)
        vz_drift = _v_drift_units(vz_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = parameters.thermal_speed(T * u.K,
                                       particle=particle,
                                       method="most_probable").si.value
    # accounting for thermal velocity in 3D
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = ((vx - vx_drift) ** 2 + (vy - vy_drift) ** 2 + (vz - vz_drift) ** 2)
    # calculating distribution function
    coeff = (vThSq * np.pi) ** (-3 / 2)
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff * expTerm
    if units == "units":
        return distFunc.to((u.s / u.m)**3)
    elif units == "unitless":
        return distFunc


def Maxwellian_speed_1D(v,
                        T,
                        particle="e",
                        v_drift=0,
                        vTh=np.nan,
                        units="units"):
    r"""
    Probability distribution function of speed for a Maxwellian distribution
    in 1D.

    Return the probability density function for finding a particle with 
    speed `v` in m/s in an equilibrium plasma of temperature `T` which 
    follows the Maxwellian distribution function.

    Parameters
    ----------
    v: ~astropy.units.Quantity
        The speed in units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    particle: str, optional
        Representation of the particle species(e.g., `'p'` for protons, `'D+'`
        for deuterium, or `'He-4 +1'` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    v_drift: ~astropy.units.Quantity
        The drift speed in units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating vTh, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in speed^-1, normalized so that
        :math:`\int_{0}^{\infty} f(v) dv = 1`.

    Raises
    ------
    TypeError
        The parameter arguments are not Quantities and
        cannot be converted into Quantities.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In one dimension, the Maxwellian speed distribution function describing
    the distribution of particles with speed v in a plasma with temperature T
    is given by:

    .. math::

       f(v) = 2 \frac{1}{(\pi v_{Th}^2)^{1/2}} \exp(-(v - V_{drift})^2 / v_{Th}^2 )

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed.

    Example
    -------
    >>> from plasmapy.physics import Maxwellian_speed_1D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> Maxwellian_speed_1D(v=v, T=30000 * u.K, particle='e', v_drift=0 * u.m / u.s)
    <Quantity 1.18326594e-06 s / m>

    """
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        v = v.to(u.m / u.s)
        # Catching case where drift velocities have default values, they
        # need to be assigned units
        v_drift = _v_drift_units(v_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = (parameters.thermal_speed(T * u.K,
                                        particle=particle,
                                        method="most_probable")).si.value
    # Get thermal velocity squared
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = (v - v_drift) ** 2
    # calculating distribution function
    coeff = 2 * (vThSq * np.pi) ** (-1 / 2)
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff * expTerm
    if units == "units":
        return distFunc.to(u.s / u.m)
    elif units == "unitless":
        return distFunc


def Maxwellian_speed_2D(v,
                        T,
                        particle="e",
                        v_drift=0,
                        vTh=np.nan,
                        units="units"):
    r"""
    Probability distribution function of speed for a Maxwellian distribution
    in 2D.

    Return the probability density function of finding a particle with speed components
    `vx` and `vy` in m/s in an equilibrium plasma of temperature
    `T` which follows the 2D Maxwellian distribution function. This
    function assumes Cartesian coordinates.

    Parameters
    ----------
    v: ~astropy.units.Quantity
        The speed in units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    particle: str, optional
        Representation of the particle species(e.g., `'p'` for protons, `'D+'`
        for deuterium, or `'He-4 +1'` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    v_drift: ~astropy.units.Quantity
        The drift speed in units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating vTh, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in speed^-1, normalized so that:
        :math:`\iiint_{0}^{\infty} f(\vec{v}) d\vec{v} = 1`.

    Raises
    ------
    TypeError
        A parameter argument is not a `~astropy.units.Quantity` and
        cannot be converted into a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In 2D, the Maxwellian speed distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` is given by:

    .. math::

       f = 2 \pi \vec{v} (\pi v_{Th}^2)^{-1} \exp(-(\vec{v} -
       \vec{V}_{drift})^2 / v_{Th}^2)

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed.

    See also
    --------
    Maxwellian_speed_1D

    Example
    -------
    >>> from plasmapy.physics import Maxwellian_speed_2D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> Maxwellian_speed_2D(v=v, T=30000 * u.K, particle='e', v_drift=0 * u.m / u.s)
    <Quantity 2.19930065e-12 s / m>

    """
    if v_drift != 0:
        raise NotImplementedError("Non-zero drift speed is work in progress.")
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        v = v.to(u.m / u.s)
        # Catching case where drift velocity has default value, and
        # needs to be assigned units
        v_drift = _v_drift_units(v_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = (parameters.thermal_speed(T * u.K,
                                        particle=particle,
                                        method="most_probable")).si.value
    # getting square of thermal speed
    vThSq = vTh ** 2
    # get square of relative particle speed
    vSq = (v - v_drift) ** 2
    # calculating distribution function
    coeff1 = (np.pi * vThSq) ** (-1)
    coeff2 = 2 * np.pi * (v - v_drift)
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff1 * coeff2 * expTerm
    if units == "units":
        return distFunc.to(u.s / u.m)
    elif units == "unitless":
        return distFunc


def Maxwellian_speed_3D(v,
                        T,
                        particle="e",
                        v_drift=0,
                        vTh=np.nan,
                        units="units"):
    r"""
    Probability distribution function of speed for a Maxwellian
    distribution in 3D.

    Return the probability density function for finding a particle with 
    speed components `vx`, `vy`, and `vz` in m/s in an equilibrium 
    plasma of temperature `T` which follows the 3D Maxwellian 
    distribution function. This function assumes Cartesian coordinates.

    Parameters
    ----------
    v: ~astropy.units.Quantity
        The speed in units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    particle: str, optional
        Representation of the particle species(e.g., `'p'` for protons, `'D+'`
        for deuterium, or `'He-4 +1'` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    v_drift: ~astropy.units.Quantity
        The drift speed in units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating vTh, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in speed^-1, normalized so that:
        :math:`\iiint_{0}^{\infty} f(\vec{v}) d\vec{v} = 1`.

    Raises
    ------
    TypeError
        A parameter argument is not a `~astropy.units.Quantity` and
        cannot be converted into a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In 3D, the Maxwellian speed distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` is given by:

    .. math::

       f = 4 \pi \vec{v}^2 (\pi v_{Th}^2)^{-3/2} \exp(-(\vec{v} -
       \vec{V}_{drift})^2 / v_{Th}^2)

    where :math:`v_{Th} = \sqrt{2 k_B T / m}` is the thermal speed.

    See also
    --------
    Maxwellian_speed_1D

    Example
    -------
    >>> from plasmapy.physics import Maxwellian_speed_3D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> Maxwellian_speed_3D(v=v, T=30000*u.K, particle='e', v_drift=0 * u.m / u.s)
    <Quantity 2.60235754e-18 s / m>

    """
    if v_drift != 0:
        raise NotImplementedError("Non-zero drift speed is work in progress.")
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        v = v.to(u.m / u.s)
        # Catching case where drift velocity has default value, and
        # needs to be assigned units
        v_drift = _v_drift_units(v_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = (parameters.thermal_speed(T,
                                            particle=particle,
                                            method="most_probable"))
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = (parameters.thermal_speed(T * u.K,
                                        particle=particle,
                                        method="most_probable")).si.value
    # getting square of thermal speed
    vThSq = vTh ** 2
    # get square of relative particle speed
    vSq = (v - v_drift) ** 2
    # calculating distribution function
    coeff1 = (np.pi * vThSq) ** (-3 / 2)
    coeff2 = 4 * np.pi * vSq
    expTerm = np.exp(-vSq / vThSq)
    distFunc = coeff1 * coeff2 * expTerm
    if units == "units":
        return distFunc.to(u.s / u.m)
    elif units == "unitless":
        return distFunc


def kappa_velocity_1D(v,
                      T,
                      kappa,
                      particle="e",
                      v_drift=0,
                      vTh=np.nan,
                      units="units"):
    r"""
    Return the probability density at the velocity `v` in m/s
    to find a particle `particle` in a plasma of temperature `T`
    following the Kappa distribution function in 1D. The slope of the
    tail of the Kappa distribution function is set by 'kappa', which
    must be greater than :math:`1/2`.

    Parameters
    ----------
    v: ~astropy.units.Quantity
        The velocity in units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature in Kelvin.

    kappa: ~astropy.units.Quantity
        The kappa parameter is a dimensionless number which sets the slope
        of the energy spectrum of suprathermal particles forming the tail
        of the Kappa velocity distribution function. Kappa must be greater
        than :math:`3/2`.

    particle: str, optional
        Representation of the particle species(e.g., `'p` for protons, `'D+'`
        for deuterium, or `'He-4 +1'` for :math:`He_4^{+1}`
        (singly ionized helium-4)), which defaults to electrons.

    v_drift: ~astropy.units.Quantity, optional
        The drift velocity in units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating `vTh`, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in Velocity^-1, normalized so that
        :math:`\int_{-\infty}^{+\infty} f(v) dv = 1`.

    Raises
    ------
    TypeError
        A parameter argument is not a `~astropy.units.Quantity` and
        cannot be converted into a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In one dimension, the Kappa velocity distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` and suprathermal parameter :math:`\kappa` is
    given by:

    .. math::

       f = A_\kappa \left(1 + \frac{(\vec{v} -
       \vec{V_{drift}})^2}{\kappa v_{Th},\kappa^2}\right)^{-\kappa}

    where :math:`v_{Th},\kappa` is the kappa thermal speed
    and :math:`A_\kappa = \frac{1}{\sqrt{\pi} \kappa^{3/2} v_{Th},\kappa^2
    \frac{\Gamma(\kappa + 1)}{\Gamma(\kappa - 1/2)}}`
    is the normalization constant.

    As :math:`\kappa` approaches infinity, the kappa distribution function
    converges to the Maxwellian distribution function.

    Examples
    --------
    >>> from plasmapy.physics import kappa_velocity_1D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> kappa_velocity_1D(v=v, T=30000*u.K, kappa=4, particle='e', v_drift=0 * u.m / u.s)
    <Quantity 6.75549854e-07 s / m>

    See Also
    --------
    kappa_velocity_3D
    kappa_thermal_speed
    """
    # must have kappa > 3/2 for distribution function to be valid
    if kappa <= 3 / 2:
        raise ValueError(f"Must have kappa > 3/2, instead of {kappa}.")
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        v = v.to(u.m / u.s)
        # catching case where drift velocities have default values, they
        # need to be assigned units
        if v_drift == 0:
            if not isinstance(v_drift, astropy.units.quantity.Quantity):
                v_drift = v_drift * u.m / u.s
        # checking units of drift velocities
        v_drift = v_drift.to(u.m / u.s)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = parameters.kappa_thermal_speed(T,
                                                 kappa,
                                                 particle=particle)
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = (parameters.kappa_thermal_speed(T * u.K,
                                              kappa,
                                              particle=particle)).si.value
    # Get thermal velocity squared and accounting for 1D instead of 3D
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = (v - v_drift) ** 2
    # calculating distribution function
    expTerm = (1 + vSq / (kappa * vThSq)) ** (-kappa)
    coeff1 = 1 / (np.sqrt(np.pi) * kappa ** (3 / 2) * vTh)
    coeff2 = gamma(kappa + 1) / (gamma(kappa - 1 / 2))
    distFunc = coeff1 * coeff2 * expTerm
    if units == "units":
        return distFunc.to(u.s / u.m)
    elif units == "unitless":
        return distFunc


def kappa_velocity_3D(vx,
                      vy,
                      vz,
                      T,
                      kappa,
                      particle="e",
                      vx_drift=0,
                      vy_drift=0,
                      vz_drift=0,
                      vTh=np.nan,
                      units="units"):
    r"""
    Return the probability density function for finding a particle with 
    velocity components `v_x`, `v_y`, and `v_z`in m/s in a suprathermal 
    plasma of temperature `T` and parameter 'kappa' which follows the 
    3D Kappa distribution function. This function assumes Cartesian 
    coordinates.

    Parameters
    ----------
    vx: ~astropy.units.Quantity
        The velocity in x-direction units convertible to m/s.

    vy: ~astropy.units.Quantity
        The velocity in y-direction units convertible to m/s.

    vz: ~astropy.units.Quantity
        The velocity in z-direction units convertible to m/s.

    T: ~astropy.units.Quantity
        The temperature, preferably in Kelvin.

    kappa: ~astropy.units.Quantity
        The kappa parameter is a dimensionless number which sets the slope
        of the energy spectrum of suprathermal particles forming the tail
        of the Kappa velocity distribution function. Kappa must be greater
        than :math:`3/2`.

    particle: str, optional
        Representation of the particle species(e.g., 'p' for protons, 'D+'
        for deuterium, or 'He-4 +1' for :math:`He_4^{+1}` : singly ionized
        helium-4)), which defaults to electrons.

    vx_drift: ~astropy.units.Quantity, optional
        The drift velocity in x-direction units convertible to m/s.

    vy_drift: ~astropy.units.Quantity, optional
        The drift velocity in y-direction units convertible to m/s.

    vz_drift: ~astropy.units.Quantity, optional
        The drift velocity in z-direction units convertible to m/s.

    vTh: ~astropy.units.Quantity, optional
        Thermal velocity (most probable) in m/s. This is used for
        optimization purposes to avoid re-calculating `vTh`, for example
        when integrating over velocity-space.

    units: str, optional
        Selects whether to run function with units and unit checks (when
        equal to "units") or to run as unitless (when equal to "unitless").
        The unitless version is substantially faster for intensive
        computations.

    Returns
    -------
    f : ~astropy.units.Quantity
        Probability density in Velocity^-1, normalized so that:
        :math:`\iiint_{0}^{\infty} f(\vec{v}) d\vec{v} = 1`

    Raises
    ------
    TypeError
        The parameter arguments are not Quantities and
        cannot be converted into Quantities.

    ~astropy.units.UnitConversionError
        If the parameters is not in appropriate units.

    ValueError
        If the temperature is negative, or the particle mass or charge state
        cannot be found.

    Notes
    -----
    In three dimensions, the Kappa velocity distribution function describing
    the distribution of particles with speed :math:`v` in a plasma with
    temperature :math:`T` and suprathermal parameter :math:`\kappa` is given by:

    .. math::

       f = A_\kappa \left(1 + \frac{(\vec{v} -
       \vec{V_{drift}})^2}{\kappa v_{Th},\kappa^2}\right)^{-(\kappa + 1)}

    where :math:`v_{Th},\kappa` is the kappa thermal speed
    and :math:`A_\kappa = \frac{1}{2 \pi (\kappa v_{Th},\kappa^2)^{3/2}}
    \frac{\Gamma(\kappa + 1)}{\Gamma(\kappa - 1/2) \Gamma(3/2)}` is the
    normalization constant.

    As :math:`\kappa` approaches infinity, the kappa distribution function
    converges to the Maxwellian distribution function.

    See also
    --------
    kappa_velocity_1D
    kappa_thermal_speed

    Example
    -------
    >>> from plasmapy.physics import kappa_velocity_3D
    >>> from astropy import units as u
    >>> v=1 * u.m / u.s
    >>> kappa_velocity_3D(vx=v,
    ... vy=v,
    ... vz=v,
    ... T=30000 * u.K,
    ... kappa=4,
    ... particle='e',
    ... vx_drift=0 * u.m / u.s,
    ... vy_drift=0 * u.m / u.s,
    ... vz_drift=0 * u.m / u.s)
    <Quantity 3.7833988e-19 s3 / m3>
    """
    # must have kappa > 3/2 for distribution function to be valid
    if kappa <= 3 / 2:
        raise ValueError(f"Must have kappa > 3/2, instead of {kappa}.")
    if units == "units":
        # unit checks and conversions
        # checking velocity units
        vx = vx.to(u.m / u.s)
        vy = vy.to(u.m / u.s)
        vz = vz.to(u.m / u.s)
        # Catching case where drift velocities have default values, they
        # need to be assigned units
        vx_drift = _v_drift_units(vx_drift)
        vy_drift = _v_drift_units(vy_drift)
        vz_drift = _v_drift_units(vz_drift)
        # convert temperature to Kelvins
        T = T.to(u.K, equivalencies=u.temperature_energy())
        if np.isnan(vTh):
            # get thermal velocity and thermal velocity squared
            vTh = parameters.kappa_thermal_speed(T,
                                                 kappa,
                                                 particle=particle)
        elif not np.isnan(vTh):
            # check units of thermal velocity
            vTh = vTh.to(u.m / u.s)
    elif np.isnan(vTh) and units == "unitless":
        # assuming unitless temperature is in Kelvins
        vTh = parameters.kappa_thermal_speed(T * u.K,
                                             kappa,
                                             particle=particle).si.value
    # getting square of thermal velocity
    vThSq = vTh ** 2
    # Get square of relative particle velocity
    vSq = ((vx - vx_drift) ** 2 + (vy - vy_drift) ** 2 + (vz - vz_drift) ** 2)
    # calculating distribution function
    expTerm = (1 + vSq / (kappa * vThSq)) ** (-(kappa + 1))
    coeff1 = 1 / (2 * np.pi * (kappa * vThSq) ** (3 / 2))
    coeff2 = gamma(kappa + 1) / (gamma(kappa - 1 / 2) * gamma(3 / 2))
    distFunc = coeff1 * coeff2 * expTerm
    if units == "units":
        return distFunc.to((u.s / u.m)**3)
    elif units == "unitless":
        return distFunc
