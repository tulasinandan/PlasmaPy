import numpy as np
import pytest
import astropy.units as u
from ...constants import c, h
from ..quantum import deBroglie_wavelength


def test_deBroglie_wavelength():

    dbwavelength1 = deBroglie_wavelength(2e7*u.cm/u.s, 'e')
    assert np.isclose(dbwavelength1.value, 3.628845222852886e-11)
    assert dbwavelength1.unit == u.m

    dbwavelength2 = deBroglie_wavelength(0*u.m/u.s, 'e')
    assert dbwavelength2 == np.inf*u.m

    V_array = np.array([2e5, 0])*u.m/u.s
    dbwavelength_arr = deBroglie_wavelength(V_array, 'e')

    assert np.isclose(dbwavelength_arr.value[0], 3.628845222852886e-11)
    assert dbwavelength_arr.value[1] == np.inf
    assert dbwavelength_arr.unit == u.m

    V_array = np.array([2e5, 2e5])*u.m/u.s
    dbwavelength_arr = deBroglie_wavelength(V_array, 'e')

    assert np.isclose(dbwavelength_arr.value[0], 3.628845222852886e-11)
    assert np.isclose(dbwavelength_arr.value[1], 3.628845222852886e-11)
    assert dbwavelength_arr.unit == u.m

    assert deBroglie_wavelength(-5e5*u.m/u.s, 'p') == \
        deBroglie_wavelength(5e5*u.m/u.s, 'p')

    assert deBroglie_wavelength(-5e5*u.m/u.s, 'e+') == \
        deBroglie_wavelength(5e5*u.m/u.s, 'e')

    assert deBroglie_wavelength(1*u.m/u.s, 5*u.kg) == \
        deBroglie_wavelength(100*u.cm/u.s, 5000*u.g)

    with pytest.raises(ValueError):
        deBroglie_wavelength(c*1.000000001, 'e')

    with pytest.raises(UserWarning):
        deBroglie_wavelength(0.79450719277, 'Be-7 1+')

    with pytest.raises(u.UnitConversionError):
        deBroglie_wavelength(8*u.m/u.s, 5*u.m)

    with pytest.raises(ValueError):
        deBroglie_wavelength(8*u.m/u.s, 'sddsf')
