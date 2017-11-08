"""Tests for functions that uses Distribution functions."""

import numpy as np
import pytest
from astropy import units as u
import scipy.integrate as spint

from ...constants import (m_p, m_e, c, mu0, k_B, e, eps0, pi)
from ..distribution import (Maxwellian_1D,
                            Maxwellian_speed_1D,
                            Maxwellian_velocity_3D,
                            Maxwellian_speed_3D)
from ..parameters import thermal_speed

# test class for Maxwellian_1D (velocity) function:
class Test_Maxwellian_1D(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T_e = 30000*u.K
        self.v = 1e5 * u.m/u.s
        self.V_drift = 1000000*u.m/u.s
        self.V_drift2 = 0 * u.m/u.s
        self.V_drift3 = 1e5 * u.m/u.s
        self.start = -5000
        self.stop = - self.start
        self.dv = 10000 * u.m/u.s
        self.v_vect = np.arange(self.start, 
                                self.stop, 
                                dtype='float64') * self.dv
        self.particle = "e"
        self.vTh = thermal_speed(self.T_e,
                                 particle=self.particle,
                                 method="most_probable")
        self.distFuncTrue = 5.851627151617136e-07
    def test_max_noDrift(self):
        """
        Checks maximum value of distribution function is in expected place,
        when there is no drift applied.
        """
        max_index = Maxwellian_1D(self.v_vect, 
                                  T=self.T_e, 
                                  particle=self.particle, 
                                  V_drift=0*u.m/u.s
                                  ).argmax()
        assert np.isclose(self.v_vect[max_index].value, 0.0)
    def test_max_drift(self):
        """
        Checks maximum value of distribution function is in expected place,
        when there is drift applied.
        """
        max_index = Maxwellian_1D(self.v_vect,
                                  T=self.T_e,
                                  particle=self.particle,
                                  V_drift=self.V_drift
                                  ).argmax()
        assert np.isclose(self.v_vect[max_index].value, self.V_drift.value)
    def test_norm(self):
        """
        Tests whether distribution function is normalized, and integrates to 1.
        """
        # integral of the distribution over v_vect
        integral = (Maxwellian_1D(self.v_vect,
                                  T=30000*u.K,
                                  particle=self.particle)).sum()*self.dv
        assert np.isclose(integral, 1.0)
    def test_std(self):
        """
        Tests standard deviation of function?
        """
        std = (Maxwellian_1D(self.v_vect,
                             T=self.T_e,
                             particle=self.particle)*self.v_vect**2*self.dv).sum()
        std = np.sqrt(std)
        T_distri = (std**2/k_B*m_e).to(u.K)
        assert np.isclose(T_distri.value, self.T_e.value)
    def test_valErr(self):
        """
        Tests whether ValueError is raised when invalid particle name
        string is passed.
        """
        with pytest.raises(ValueError):
            Maxwellian_1D(1*u.m/u.s,
                          T=1*u.K,
                          particle='XXX')
    def test_units_no_vTh(self):
        """
        Tests distribution function with units, but not passing vTh.
        """
        distFunc = Maxwellian_1D(v=self.v,
                                 T=self.T_e, 
                                 particle=self.particle,
                                 units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_units_vTh(self):
        """
        Tests distribution function with units and passing vTh.
        """
        distFunc = Maxwellian_1D(v=self.v,
                                 T=self.T_e, 
                                 vTh=self.vTh,
                                 particle=self.particle,
                                 units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_no_vTh(self):
        """
        Tests distribution function without units, and not passing vTh.
        """
        # converting T to SI then stripping units
        T_e = self.T_e.to(u.K, equivalencies=u.temperature_energy())
        T_e = T_e.si.value
        distFunc = Maxwellian_1D(v=self.v.si.value,
                                 T=T_e,
                                 particle=self.particle,
                                 units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_vTh(self):
        """
        Tests distribution function without units, and with passing vTh.
        """
        # converting T to SI then stripping units
        T_e = self.T_e.to(u.K, equivalencies=u.temperature_energy())
        T_e = T_e.si.value
        distFunc = Maxwellian_1D(v=self.v.si.value,
                                 T=T_e, 
                                 vTh=self.vTh.si.value,
                                 particle=self.particle,
                                 units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_zero_drift_units(self):
        """
        Testing inputting drift equal to 0 with units. These should just
        get passed and not have extra units applied to them.
        """
        distFunc = Maxwellian_1D(v=self.v,
                                 T=self.T_e,
                                 particle=self.particle,
                                 V_drift=self.V_drift2,
                                 units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_value_drift_units(self):
        """
        Testing vdrifts with values
        """
        testVal = ((self.vTh**2 * np.pi) ** (-1 / 2)).si.value
        distFunc = Maxwellian_1D(v=self.v,
                                 T=self.T_e,
                                 particle=self.particle,
                                 V_drift=self.V_drift3,
                                 units="units")
        errStr = (f"Distribution function should be {testVal} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          testVal,
                          rtol=1e-8,
                          atol=0.0), errStr


# test class for Maxwellian_speed_1D function
class Test_Maxwellian_speed_1D(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 1.0 * u.eV
        self.particle = 'H'
        # get thermal velocity and thermal velocity squared
        self.vTh = thermal_speed(self.T,
                                 particle=self.particle,
                                 method="most_probable")
        self.v = 1e5 * u.m/u.s
        self.V_drift = 0 * u.m/u.s
        self.V_drift2 = 1e5 * u.m/u.s
        self.distFuncTrue = 1.8057567503860518e-25
    def test_norm(self):
        """
        Tests whether distribution function is normalized, and integrates to 1.
        """
        # setting up integration from 0 to 10*vTh
        xData1D = np.arange(0, 10.01, 0.01) * self.vTh
        yData1D = Maxwellian_speed_1D(v=xData1D,
                                      T=self.T,
                                      particle=self.particle)
        # integrating, this should be close to 1
        integ = spint.trapz(y=yData1D, x=xData1D)
        exceptStr = "Integral of distribution function should be 1."
        assert np.isclose(integ.value, 1), exceptStr
    def test_units_no_vTh(self):
        """
        Tests distribution function with units, but not passing vTh.
        """
        distFunc = Maxwellian_speed_1D(v=self.v,
                                       T=self.T, 
                                       particle=self.particle,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_units_vTh(self):
        """
        Tests distribution function with units and passing vTh.
        """
        distFunc = Maxwellian_speed_1D(v=self.v,
                                       T=self.T,
                                       vTh=self.vTh,
                                       particle=self.particle,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_no_vTh(self):
        """
        Tests distribution function without units, and not passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_speed_1D(v=self.v.si.value,
                                       T=T, 
                                       particle=self.particle,
                                       units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_vTh(self):
        """
        Tests distribution function without units, and with passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_speed_1D(v=self.v.si.value,
                                       T=T,
                                       vTh=self.vTh.si.value,
                                       particle=self.particle,
                                       units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_zero_drift_units(self):
        """
        Testing inputting drift equal to 0 with units. These should just
        get passed and not have extra units applied to them.
        """
        distFunc = Maxwellian_speed_1D(v=self.v,
                                       T=self.T,
                                       particle=self.particle,
                                       V_drift=self.V_drift,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_value_drift_units(self):
        """
        Testing vdrifts with values
        """
        distFunc = Maxwellian_speed_1D(v=self.v,
                                       T=self.T,
                                       particle=self.particle,
                                       V_drift=self.V_drift2,
                                       units="units")
        errStr = (f"Distribution function should be 0.0 "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          0.0,
                          rtol=1e-8,
                          atol=0.0), errStr
    
    
# test class for Maxwellian_velocity_3D function
class Test_Maxwellian_velocity_3D(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 1.0 * u.eV
        self.particle = 'H'
        # get thermal velocity and thermal velocity squared
        self.vTh = thermal_speed(self.T,
                                 particle=self.particle,
                                 method="most_probable")
        self.vx = 1e5 * u.m/u.s
        self.vy = 1e5 * u.m/u.s
        self.vz = 1e5 * u.m/u.s
        self.Vx_drift = 0 * u.m/u.s
        self.Vy_drift = 0 * u.m/u.s
        self.Vz_drift = 0 * u.m/u.s
        self.Vx_drift2 = 1e5 * u.m/u.s
        self.Vy_drift2 = 1e5 * u.m/u.s
        self.Vz_drift2 = 1e5 * u.m/u.s
        self.distFuncTrue = 2.7654607627522045e-37
    def test_norm(self):
        """
        Tests whether distribution function is normalized, and integrates to 1.
        """
        # converting vTh to unitless
        vTh = self.vTh.si.value
        # setting up integration from -10*vTh to 10*vTh, which is close to Inf
        infApprox = (10 * vTh)
        # integrating, this should be close to 1
        integ = spint.tplquad(Maxwellian_velocity_3D,
                              -infApprox,
                              infApprox,
                              lambda z: -infApprox,
                              lambda z: infApprox,
                              lambda z, y: -infApprox,
                              lambda z, y: infApprox,
                              args=(self.T,
                                    self.particle,
                                    0,
                                    0,
                                    0,
                                    vTh,
                                    "unitless"),
                              epsabs=1e0,
                              epsrel=1e0,
                              )
        # value returned from tplquad is (integral, error), we just need 
        # the 1st
        integVal = integ[0]
        exceptStr = ("Integral of distribution function should be 1 "
                     f"and not {integVal}.")
        assert np.isclose(integVal,
                          1,
                          rtol=1e-3,
                          atol=0.0), exceptStr
    def test_units_no_vTh(self):
        """
        Tests distribution function with units, but not passing vTh.
        """
        distFunc = Maxwellian_velocity_3D(vx=self.vx,
                                          vy=self.vy,
                                          vz=self.vz,
                                          T=self.T, 
                                          particle=self.particle,
                                          units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_units_vTh(self):
        """
        Tests distribution function with units and passing vTh.
        """
        distFunc = Maxwellian_velocity_3D(vx=self.vx,
                                          vy=self.vy,
                                          vz=self.vz,
                                          T=self.T,
                                          vTh=self.vTh,
                                          particle=self.particle,
                                          units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_no_vTh(self):
        """
        Tests distribution function without units, and not passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_velocity_3D(vx=self.vx.si.value,
                                          vy=self.vy.si.value,
                                          vz=self.vz.si.value,
                                          T=T, 
                                          particle=self.particle,
                                          units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_vTh(self):
        """
        Tests distribution function without units, and with passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_velocity_3D(vx=self.vx.si.value,
                                          vy=self.vy.si.value,
                                          vz=self.vz.si.value,
                                          T=T,
                                          vTh=self.vTh.si.value,
                                          particle=self.particle,
                                          units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_zero_drift_units(self):
        """
        Testing inputting drift equal to 0 with units. These should just
        get passed and not have extra units applied to them.
        """
        distFunc = Maxwellian_velocity_3D(vx=self.vx,
                                          vy=self.vy,
                                          vz=self.vz,
                                          T=self.T,
                                          particle=self.particle,
                                          Vx_drift=self.Vx_drift,
                                          Vy_drift=self.Vy_drift,
                                          Vz_drift=self.Vz_drift,
                                          units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_value_drift_units(self):
        """
        Testing vdrifts with values
        """
        testVal = ((3 * self.vTh**2 * np.pi) ** (-3 / 2)).si.value
        distFunc = Maxwellian_velocity_3D(vx=self.vx,
                                          vy=self.vy,
                                          vz=self.vz,
                                          T=self.T,
                                          particle=self.particle,
                                          Vx_drift=self.Vx_drift2,
                                          Vy_drift=self.Vy_drift2,
                                          Vz_drift=self.Vz_drift2,
                                          units="units")
        errStr = (f"Distribution function should be {testVal} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          testVal,
                          rtol=1e-8,
                          atol=0.0), errStr


# test class for Maxwellian_speed_3D function
class Test_Maxwellian_speed_3D(object):
    def setup_method(self):
        """initializing parameters for tests """
        self.T = 1.0 * u.eV
        self.particle = 'H'
        # get thermal velocity and thermal velocity squared
        self.vTh = thermal_speed(self.T,
                                 particle=self.particle,
                                 method="most_probable")
        self.vx = 1e5 * u.m/u.s
        self.vy = 1e5 * u.m/u.s
        self.vz = 1e5 * u.m/u.s
        self.Vx_drift = 0 * u.m/u.s
        self.Vy_drift = 0 * u.m/u.s
        self.Vz_drift = 0 * u.m/u.s
        self.Vx_drift2 = 1e5 * u.m/u.s
        self.Vy_drift2 = 1e5 * u.m/u.s
        self.Vz_drift2 = 1e5 * u.m/u.s
        self.distFuncTrue = 5.888134761477178e-75
    def test_norm(self):
        """
        Tests whether distribution function is normalized, and integrates to 1.
        """
        # converting vTh to unitless
        vTh = self.vTh.si.value
        # setting up integration from 0 to 10*vTh, which is close to Inf
        infApprox = (10 * vTh)
        # integral should be close to 1
        integ = spint.tplquad(Maxwellian_speed_3D,
                              0,
                              infApprox,
                              lambda z: 0,
                              lambda z: infApprox,
                              lambda z, y: 0,
                              lambda z, y: infApprox,
                              args=(self.T,
                                    self.particle,
                                    0,
                                    0,
                                    0,
                                    vTh,
                                    "unitless"),
                              epsabs=1e0,
                              epsrel=1e0,
                              )
        # value returned from tplquad is (integral, error), we just need 
        # the 1st
        integVal = integ[0]
        exceptStr = ("Integral of distribution function should be 1 "
                     f"and not {integVal}")
        assert np.isclose(integVal,
                          1,
                          rtol=1e-3,
                          atol=0.0), exceptStr
    def test_units_no_vTh(self):
        """
        Tests distribution function with units, but not passing vTh.
        """
        distFunc = Maxwellian_speed_3D(vx=self.vx,
                                       vy=self.vy,
                                       vz=self.vz,
                                       T=self.T, 
                                       particle=self.particle,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_units_vTh(self):
        """
        Tests distribution function with units and passing vTh.
        """
        distFunc = Maxwellian_speed_3D(vx=self.vx,
                                       vy=self.vy,
                                       vz=self.vz,
                                       T=self.T,
                                       vTh=self.vTh,
                                       particle=self.particle,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_no_vTh(self):
        """
        Tests distribution function without units, and not passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_speed_3D(vx=self.vx.si.value,
                                       vy=self.vy.si.value,
                                       vz=self.vz.si.value,
                                       T=T, 
                                       particle=self.particle,
                                       units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_unitless_vTh(self):
        """
        Tests distribution function without units, and with passing vTh.
        """
        # converting T to SI then stripping units
        T = self.T.to(u.K, equivalencies=u.temperature_energy())
        T = T.si.value
        distFunc = Maxwellian_speed_3D(vx=self.vx.si.value,
                                       vy=self.vy.si.value,
                                       vz=self.vz.si.value,
                                       T=T,
                                       vTh=self.vTh.si.value,
                                       particle=self.particle,
                                       units="unitless")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_zero_drift_units(self):
        """
        Testing inputting drift equal to 0 with units. These should just
        get passed and not have extra units applied to them.
        """
        distFunc = Maxwellian_speed_3D(vx=self.vx,
                                       vy=self.vy,
                                       vz=self.vz,
                                       T=self.T,
                                       particle=self.particle,
                                       Vx_drift=self.Vx_drift,
                                       Vy_drift=self.Vy_drift,
                                       Vz_drift=self.Vz_drift,
                                       units="units")
        errStr = (f"Distribution function should be {self.distFuncTrue} "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          self.distFuncTrue,
                          rtol=1e-8,
                          atol=0.0), errStr
    def test_value_drift_units(self):
        """
        Testing vdrifts with values
        """
        distFunc = Maxwellian_speed_3D(vx=self.vx,
                                       vy=self.vy,
                                       vz=self.vz,
                                       T=self.T,
                                       particle=self.particle,
                                       Vx_drift=self.Vx_drift2,
                                       Vy_drift=self.Vy_drift2,
                                       Vz_drift=self.Vz_drift2,
                                       units="units")
        errStr = (f"Distribution function should be 0.0 "
                  f"and not {distFunc}.")
        assert np.isclose(distFunc.value, 
                          0.0,
                          rtol=1e-8,
                          atol=0.0), errStr