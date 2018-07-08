import numpy as np
import pytest
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose

from plasmapy.physics.transport import (Coulomb_logarithm,
                                        impact_parameter_perp,
                                        impact_parameter,
                                        collision_frequency,
                                        mean_free_path,
                                        mobility,
                                        Knudsen_number,
                                        coupling_parameter)
from plasmapy.physics.transport.collisions import (Spitzer_resistivity,
                                                   fundamental_electron_collision_freq,
                                                   fundamental_ion_collision_freq)
from plasmapy.utils import exceptions
from plasmapy.utils.pytest_helpers import assert_can_handle_nparray
from plasmapy.constants import m_p, m_e, c


class Test_Coulomb_logarithm:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.temperature1 = 10 * 11604 * u.K
        self.T_arr = np.array([1, 2]) * u.eV
        self.density1 = 1e20 * u.cm ** -3
        self.n_arr = np.array([1e20, 2e20]) * u.cm ** -3
        self.temperature2 = 1 * 11604 * u.K
        self.density2 = 1e23 * u.cm ** -3
        self.z_mean = 2.5
        self.particles = ('e', 'p')
        self.gms1 = 3.4014290066940966
        self.gms1_negative = -3.4310536971592493
        self.gms2 = 3.6349941014645157
        self.gms2_negative = -1.379394033464292
        self.gms3 = 3.4014290066940966
        self.gms3_negative = 2
        self.gms3_non_scalar = (2, 2)
        self.gms4 = 3.401983996820073
        self.gms4_negative = 0.0005230791851781715
        self.gms5 = 3.7196690506837693
        self.gms5_negative = 0.03126832674323108
        self.gms6 = 3.635342040477818
        self.gms6_negative = 0.030720859361047514

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    @pytest.mark.parametrize("kwargs", [{"method": "classical"},
                                        {"method": "GMS-1"},
                                        {"method": "GMS-2", "z_mean": 1.0},
                                        {"method": "GMS-3"},
                                        {"method": "GMS-4"},
                                        {"method": "GMS-5", "z_mean": 1.0},
                                        {"method": "GMS-6", "z_mean": 1.0}, ])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(Coulomb_logarithm, insert_some_nans, insert_all_nans, kwargs)

    def test_unknown_method(self):
        """Test that function will raise ValueError on non-existent method"""
        with pytest.raises(ValueError):
            Coulomb_logarithm(self.T_arr[0],
                              self.n_arr[0],
                              self.particles,
                              method="welcome our new microsoft overlords")

    def test_handle_invalid_V(self):
        """Test that V default, V = None, and V = np.nan all give the same result"""
        methodVal_0 = Coulomb_logarithm(self.T_arr[0],
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled,
                                        V=np.nan * u.m / u.s)
        methodVal_1 = Coulomb_logarithm(self.T_arr[0],
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled,
                                        V=None)
        methodVal_2 = Coulomb_logarithm(self.T_arr[0],
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled)
        assert_quantity_allclose(methodVal_0, methodVal_1)
        assert_quantity_allclose(methodVal_0, methodVal_2)

    def test_handle_zero_V(self):
        """Test that V == 0 returns a PhysicsError"""
        with pytest.raises(exceptions.PhysicsError):
            Coulomb_logarithm(self.T_arr[0],
                              self.n_arr[0],
                              self.particles,
                              z_mean=1 * u.dimensionless_unscaled,
                              V=0 * u.m / u.s)

    def test_handle_V_arraysizes(self):
        """Test that different sized V input array gets handled by _boilerplate"""
        methodVal_0 = Coulomb_logarithm(self.T_arr[0],
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled,
                                        V=np.array([np.nan, 3e7]) * u.m / u.s)
        methodVal_1 = Coulomb_logarithm(self.T_arr[1],
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled,
                                        V=np.array([1e7, np.nan]) * u.m / u.s)
        methodVal_2 = Coulomb_logarithm(self.T_arr,
                                        self.n_arr[0],
                                        self.particles,
                                        z_mean=1 * u.dimensionless_unscaled,
                                        V=np.array([np.nan, np.nan]) * u.m / u.s)
        assert_quantity_allclose(methodVal_0[0], methodVal_2[0])
        assert_quantity_allclose(methodVal_1[1], methodVal_2[1])

    def test_symmetry(self):
        lnLambda = Coulomb_logarithm(self.temperature1, self.density2, self.particles)
        lnLambdaRev = Coulomb_logarithm(self.temperature1, self.density2, self.particles[::-1])
        assert lnLambda == lnLambdaRev

    def test_Chen_Q_machine(self):
        """
        Tests whether Coulomb logarithm gives value consistent with
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 Q-machine example.
        """
        T = 0.2 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e15 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 9.1 + np.log(2)
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Q-machine value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr

    def test_Chen_lab(self):
        """
        Tests whether Coulomb logarithm gives value consistent with
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 lab plasma example.
        """
        T = 2 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e17 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 10.2 + np.log(2)
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Lab plasma value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr

    def test_Chen_torus(self):
        """
        Tests whether Coulomb logarithm gives value consistent with
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 torus example.
        """
        T = 100 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e19 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 13.7 + np.log(2)
        lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Torus value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr

    def test_Chen_fusion(self):
        """
        Tests whether Coulomb logarithm gives value consistent with
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 fusion reactor example.
        """
        T = 1e4 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e21 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 16 + np.log(2)
        with pytest.warns(exceptions.RelativityWarning):
            lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Fusion reactor value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr

    def test_Chen_laser(self):
        """
        Tests whether Coulomb logarithm gives value consistent with
        Chen's Introduction to Plasma Physics and Controlled Fusion
        section 5.6.2 laser plasma example.
        """
        T = 1e3 * u.eV
        T = T.to(u.K, equivalencies=u.temperature_energy())
        n = 1e27 * u.m ** -3
        # factor of np.log(2) corrects for different definitions of thermal
        # velocity. Chen uses v**2 = k * T / m  whereas we use
        # v ** 2 = 2 * k * T / m
        lnLambdaChen = 6.8 + np.log(2)
        with pytest.warns(exceptions.RelativityWarning):
            lnLambda = Coulomb_logarithm(T, n, ('e', 'p'))
        testTrue = np.isclose(lnLambda,
                              lnLambdaChen,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Laser plasma value of Coulomb logarithm should be "
                  f"{lnLambdaChen} and not {lnLambda}.")
        assert testTrue, errStr

    def test_GMS1(self):
        """
        Test for first version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=np.nan * u.dimensionless_unscaled,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-1")
        testTrue = np.isclose(methodVal,
                              self.gms1,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-1 should be "
                  f"{self.gms1} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS1_negative(self):
        """
        Test for first version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks for when
        a negative (invalid) Coulomb logarithm is returned.
        """
        with pytest.warns(exceptions.CouplingWarning, match="relies on weak coupling"):
            Coulomb_logarithm(self.temperature2,
                              self.density2,
                              self.particles,
                              z_mean=np.nan * u.dimensionless_unscaled,
                              V=np.nan * u.m / u.s,
                              method="GMS-1")

    def test_GMS2(self):
        """
        Test for second version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-2")
        testTrue = np.isclose(methodVal,
                              self.gms2,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-2 should be "
                  f"{self.gms2} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS2_negative(self):
        """
        Test for second version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks for when
        a negative (invalid) Coulomb logarithm is returned.
        """
        with pytest.warns(exceptions.CouplingWarning, match="relies on weak coupling"):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-2")

    def test_GMS3(self):
        """
        Test for third version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-3")
        testTrue = np.isclose(methodVal,
                              self.gms3,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-3 should be "
                  f"{self.gms3} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS3_negative(self):
        """
        Test for third version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks whether
        a positive value is returned whereas the classical Coulomb
        logarithm would return a negative value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-3")
        testTrue = np.isclose(methodVal,
                              self.gms3_negative,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-3 should be "
                  f"{self.gms3_negative} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS3_non_scalar_density(self):
        """
        Test for third version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks whether
        passing in a collection of density values returns a
        collection of Coulomb logarithm values.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(10 * 1160 * u.K,
                                          (1e23 * u.cm ** -3, 1e20 * u.cm ** -3),
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-3")
        testTrue = np.isclose(methodVal,
                              self.gms3_non_scalar,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-3 should be "
                  f"{self.gms3_non_scalar} and not {methodVal}.")
        assert testTrue.all(), errStr

    def test_GMS4(self):
        """
        Test for fourth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-4")
        testTrue = np.isclose(methodVal,
                              self.gms4,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-4 should be "
                  f"{self.gms4} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS4_negative(self):
        """
        Test for fourth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks whether
        a positive value is returned whereas the classical Coulomb
        logarithm would return a negative value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-4")
        testTrue = np.isclose(methodVal,
                              self.gms4_negative,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-4 should be "
                  f"{self.gms4_negative} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS5(self):
        """
        Test for fifth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-5")
        testTrue = np.isclose(methodVal,
                              self.gms5,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-5 should be "
                  f"{self.gms5} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS5_negative(self):
        """
        Test for fifth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks whether
        a positive value is returned whereas the classical Coulomb
        logarithm would return a negative value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-5")
        testTrue = np.isclose(methodVal,
                              self.gms5_negative,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-5 should be "
                  f"{self.gms5_negative} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS6(self):
        """
        Test for sixth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature1,
                                          self.density1,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-6")
        testTrue = np.isclose(methodVal,
                              self.gms6,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-6 should be "
                  f"{self.gms6} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS6_negative(self):
        """
        Test for sixth version of Coulomb logarithm from Gericke,
        Murillo, and Schlanges PRE (2002). This checks whether
        a positive value is returned whereas the classical Coulomb
        logarithm would return a negative value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          z_mean=self.z_mean,
                                          V=np.nan * u.m / u.s,
                                          method="GMS-6")
        testTrue = np.isclose(methodVal,
                              self.gms6_negative,
                              rtol=1e-15,
                              atol=0.0)
        errStr = (f"Coulomb logarithm for GMS-6 should be "
                  f"{self.gms6_negative} and not {methodVal}.")
        assert testTrue, errStr

    def test_GMS2_zmean_error(self):
        """
        Tests whether GMS-2 raises z_mean error when a z_mean is not
        provided.
        """
        with pytest.raises(ValueError):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          method="GMS-2")

    def test_GMS5_zmean_error(self):
        """
        Tests whether GMS-5 raises z_mean error when a z_mean is not
        provided.
        """
        with pytest.raises(ValueError):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          method="GMS-5")

    def test_GMS6_zmean_error(self):
        """
        Tests whether GMS-6 raises z_mean error when a z_mean is not
        provided.
        """
        with pytest.raises(ValueError):
            methodVal = Coulomb_logarithm(self.temperature2,
                                          self.density2,
                                          self.particles,
                                          method="GMS-6")

    def test_relativity_warn(self):
        """Tests whether relativity warning is raised at high velocity."""
        with pytest.warns(exceptions.RelativityWarning):
            Coulomb_logarithm(1e5 * u.K, 1 * u.m ** -3, ('e', 'p'), V=0.9 * c)

    def test_relativity_error(self):
        """Tests whether relativity error is raised at light speed."""
        with pytest.raises(exceptions.RelativityError):
            Coulomb_logarithm(1e5 * u.K, 1 * u.m ** -3, ('e', 'p'), V=1.1 * c)

    def test_unit_conversion_error(self):
        """
        Tests whether unit conversion error is raised when arguments
        are given with incorrect units.
        """
        with pytest.raises(u.UnitConversionError):
            Coulomb_logarithm(1e5 * u.g, 1 * u.m ** -3,
                              ('e', 'p'), V=29979245 * u.m / u.s)

    def test_single_particle_error(self):
        """
        Tests whether an error is raised if only a single particle is given.
        """
        with pytest.raises(ValueError):
            Coulomb_logarithm(1 * u.K, 5 * u.m ** -3, 'e')

    def test_invalid_particle_error(self):
        """
        Tests whether an error is raised when an invalid particle name
        is given.
        """
        with pytest.raises(exceptions.InvalidParticleError):
            Coulomb_logarithm(1 * u.K, 5 * u.m ** -3, ('e', 'g'))

    n_e = np.array([1e9, 1e9, 1e24]) * u.cm ** -3
    T = np.array([1e2, 1e7, 1e8]) * u.K
    Lambda = np.array([5.97, 21.66, 6.69])
    particles = ('e', 'p')


class Test_impact_parameter_perp:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.particles = ('e', 'p')
        self.V = 1e4 * u.km / u.s
        self.True1 = 7.200146594293746e-10

    def test_symmetry(self):
        result = impact_parameter_perp(self.T, self.particles)
        resultRev = impact_parameter_perp(self.T, self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = impact_parameter_perp(self.T,
                                          self.particles,
                                          V=np.nan * u.m / u.s)
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = ("Distance of closest approach for 90 degree Coulomb "
                  f"collision, impact_parameter_perp, should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 + 1e-15
        methodVal = impact_parameter_perp(self.T,
                                          self.particles,
                                          V=np.nan * u.m / u.s)
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"impact_parameter_perp value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans,
                             insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(impact_parameter_perp, insert_some_nans,
                                  insert_all_nans, kwargs)

    assert np.isclose(Coulomb_logarithm(1 * u.eV, 5 * u.m ** -3, ('e', 'e')),
                      Coulomb_logarithm(11604.5220 * u.K,
                                        5 * u.m ** -3,
                                        ('e', 'e')))


class Test_impact_parameter:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.T_arr = np.array([1, 2]) * u.eV
        self.n_e = 1e17 * u.cm ** -3
        self.n_e_arr = np.array([1e17, 2e17]) * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = np.array([7.200146594293746e-10, 2.3507660003984624e-08])

    def test_symmetry(self):
        result = impact_parameter(self.T, self.n_e, self.particles)
        resultRev = impact_parameter(self.T, self.n_e,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = impact_parameter(self.T,
                                     self.n_e,
                                     self.particles,
                                     z_mean=np.nan * u.dimensionless_unscaled,
                                     V=np.nan * u.m / u.s,
                                     method="classical")
        bmin, bmax = methodVal
        methodVal = bmin.si.value, bmax.si.value
        testTrue = np.allclose(self.True1,
                               methodVal,
                               rtol=1e-1,
                               atol=0.0)
        errStr = (f"Impact parameters should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = impact_parameter(self.T,
                                     self.n_e,
                                     self.particles,
                                     z_mean=np.nan * u.dimensionless_unscaled,
                                     V=np.nan * u.m / u.s,
                                     method="classical")
        bmin, bmax = methodVal
        methodVal = bmin.si.value, bmax.si.value
        testTrue = not np.allclose(methodVal,
                                   fail1,
                                   rtol=1e-16,
                                   atol=0.0)
        errStr = (f"Impact parameter value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    def test_bad_method(self):
        """Testing failure when invalid method is passed."""
        with pytest.raises(ValueError):
            impact_parameter(self.T,
                             self.n_e,
                             self.particles,
                             z_mean=np.nan * u.dimensionless_unscaled,
                             V=np.nan * u.m / u.s,
                             method="meow")

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    @pytest.mark.parametrize("kwargs", [{"method": "classical"},
                                        {"method": "GMS-1"},
                                        {"method": "GMS-2", "z_mean": 1.0},
                                        {"method": "GMS-3"},
                                        {"method": "GMS-4"},
                                        {"method": "GMS-5", "z_mean": 1.0},
                                        {"method": "GMS-6", "z_mean": 1.0}, ])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(impact_parameter, insert_some_nans, insert_all_nans, kwargs)

    def test_extend_scalar_bmin(self):
        """
        Test to verify that if T is scalar and n is vector, bmin will be extended
        to the same length as bmax
        """
        (bmin, bmax) = impact_parameter(1 * u.eV,
                                        self.n_e_arr,
                                        self.particles)
        assert(len(bmin) == len(bmax))


class Test_collision_frequency:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.electrons = ('e', 'e')
        self.protons = ('p', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 1.3468281539854646e12
        self.True_electrons = 1904702641552.1638
        self.True_protons = 44450104815.91857
        self.True_zmean = 1346828153985.4646

    def test_symmetry(self):
        result = collision_frequency(self.T, self.n, self.particles)
        resultRev = collision_frequency(self.T, self.n, self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = collision_frequency(self.T,
                                            self.n,
                                            self.particles,
                                            z_mean=np.nan * u.dimensionless_unscaled,
                                            V=np.nan * u.m / u.s,
                                            method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Collision frequency should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = collision_frequency(self.T,
                                            self.n,
                                            self.particles,
                                            z_mean=np.nan * u.dimensionless_unscaled,
                                            V=np.nan * u.m / u.s,
                                            method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Collision frequency value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    @pytest.mark.parametrize("kwargs", [{"particles": ("e", "e")},
                                        {"particles": ("e", "p")},
                                        {"particles": ("p", "p")}, ])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(collision_frequency, insert_some_nans, insert_all_nans, kwargs)

    def test_electrons(self):
        """
        Testing collision frequency between electrons.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = collision_frequency(self.T,
                                            self.n,
                                            self.electrons,
                                            z_mean=np.nan * u.dimensionless_unscaled,
                                            V=np.nan * u.m / u.s,
                                            method="classical")
        testTrue = np.isclose(self.True_electrons,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Collision frequency should be {self.True_electrons} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_protons(self):
        """
        Testing collision frequency between protons (ions).
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = collision_frequency(self.T,
                                            self.n,
                                            self.protons,
                                            z_mean=np.nan * u.dimensionless_unscaled,
                                            V=np.nan * u.m / u.s,
                                            method="classical")
        testTrue = np.isclose(self.True_protons,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Collision frequency should be {self.True_protons} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_zmean(self):
        """
        Test collisional frequency function when given arbitrary z_mean.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = collision_frequency(self.T,
                                            self.n,
                                            self.particles,
                                            z_mean=self.z_mean,
                                            V=np.nan * u.m / u.s,
                                            method="classical")
        testTrue = np.isclose(self.True_zmean,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Collision frequency should be {self.True_zmean} and "
                  f"not {methodVal}.")
        assert testTrue, errStr


class Test_fundamental_electron_collision_freq():
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T_arr = np.array([1, 2]) * u.eV
        self.n_arr = np.array([1e20, 2e20]) * u.cm ** -3
        self.ion_particle = 'p'
        self.coulomb_log = 10

    # TODO: array coulomb log
    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(fundamental_electron_collision_freq, insert_some_nans,
                                  insert_all_nans, kwargs)


class Test_fundamental_ion_collision_freq():
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T_arr = np.array([1, 2]) * u.eV
        self.n_arr = np.array([1e20, 2e20]) * u.cm ** -3
        self.ion_particle = 'p'
        self.coulomb_log = 10

    # TODO: array coulomb log
    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(fundamental_ion_collision_freq, insert_some_nans,
                                  insert_all_nans, kwargs)


class Test_mean_free_path:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 4.4047571877932046e-07

    def test_symmetry(self):
        result = mean_free_path(self.T, self.n_e, self.particles)
        resultRev = mean_free_path(self.T, self.n_e,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = mean_free_path(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Mean free path should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = mean_free_path(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Mean free path value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(mean_free_path, insert_some_nans, insert_all_nans, kwargs)


class Test_Spitzer_resistivity:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n = 1e12 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 1.2665402649805445e-3
        self.True_zmean = 0.00020264644239688712

    def test_symmetry(self):
        result = Spitzer_resistivity(self.T, self.n, self.particles)
        resultRev = Spitzer_resistivity(self.T, self.n,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = Spitzer_resistivity(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan * u.dimensionless_unscaled,
                                        V=np.nan * u.m / u.s,
                                        method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Spitzer resistivity should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = Spitzer_resistivity(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=np.nan * u.dimensionless_unscaled,
                                        V=np.nan * u.m / u.s,
                                        method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Spitzer resistivity value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    def test_zmean(self):
        """Testing Spitzer when z_mean is passed."""
        methodVal = Spitzer_resistivity(self.T,
                                        self.n,
                                        self.particles,
                                        z_mean=self.z_mean,
                                        V=np.nan * u.m / u.s,
                                        method="classical")
        testTrue = np.isclose(self.True_zmean,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Spitzer resistivity should be {self.True_zmean} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    # TODO vector z_mean
    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(Spitzer_resistivity, insert_some_nans, insert_all_nans, kwargs)


class Test_mobility:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 0.13066090887074902
        self.True_zmean = 0.32665227217687254

    def test_symmetry(self):
        result = mobility(self.T, self.n_e, self.particles)
        resultRev = mobility(self.T, self.n_e,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = mobility(self.T,
                                 self.n_e,
                                 self.particles,
                                 z_mean=np.nan * u.dimensionless_unscaled,
                                 V=np.nan * u.m / u.s,
                                 method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Mobility should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = mobility(self.T,
                                 self.n_e,
                                 self.particles,
                                 z_mean=np.nan * u.dimensionless_unscaled,
                                 V=np.nan * u.m / u.s,
                                 method="classical")
        testTrue = not np.isclose(methodVal.si.value,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Mobility value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    def test_zmean(self):
        """Testing mobility when z_mean is passed."""
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = mobility(self.T,
                                 self.n_e,
                                 self.particles,
                                 z_mean=self.z_mean,
                                 V=np.nan * u.m / u.s,
                                 method="classical")
        testTrue = np.isclose(self.True_zmean,
                              methodVal.si.value,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Mobility should be {self.True_zmean} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    # TODO vector z_mean
    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(mobility, insert_some_nans, insert_all_nans, kwargs)


class Test_Knudsen_number:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.length = 1 * u.nm
        self.T = 11604 * u.K
        self.n_e = 1e17 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 440.4757187793204

    def test_symmetry(self):
        result = Knudsen_number(self.length, self.T, self.n_e, self.particles)
        resultRev = Knudsen_number(self.length, self.T, self.n_e,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Knudsen_number(self.length,
                                       self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Knudsen number should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        with pytest.warns(exceptions.PhysicsWarning, match="strong coupling effects"):
            methodVal = Knudsen_number(self.length,
                                       self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = not np.isclose(methodVal,
                                  fail1,
                                  rtol=0.0,
                                  atol=1e-16)
        errStr = (f"Knudsen number value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    def test_handle_nparrays(self, insert_some_nans, insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(Knudsen_number, insert_some_nans, insert_all_nans, kwargs)


class Test_coupling_parameter:
    @classmethod
    def setup_class(self):
        """initializing parameters for tests """
        self.T = 11604 * u.K
        self.n_e = 1e21 * u.cm ** -3
        self.particles = ('e', 'p')
        self.z_mean = 2.5
        self.V = 1e4 * u.km / u.s
        self.True1 = 2.3213156755481195
        self.True_zmean = 10.689750083758698
        self.True_quantum = 0.3334662805238162

    def test_symmetry(self):
        result = coupling_parameter(self.T, self.n_e, self.particles)
        resultRev = coupling_parameter(self.T, self.n_e,  self.particles[::-1])
        assert result == resultRev

    def test_known1(self):
        """
        Test for known value.
        """
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = np.isclose(self.True1,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Coupling parameter should be {self.True1} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    def test_fail1(self):
        """
        Tests if test_known1() would fail if we slightly adjusted the
        value comparison by some quantity close to numerical error.
        """
        fail1 = self.True1 * (1 + 1e-15)
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=np.nan * u.dimensionless_unscaled,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = not np.isclose(methodVal,
                                  fail1,
                                  rtol=1e-16,
                                  atol=0.0)
        errStr = (f"Coupling parameter value test gives {methodVal} and "
                  f"should not be equal to {fail1}.")
        assert testTrue, errStr

    def test_zmean(self):
        """
        Test value obtained when arbitrary z_mean is passed
        """
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       z_mean=self.z_mean,
                                       V=np.nan * u.m / u.s,
                                       method="classical")
        testTrue = np.isclose(self.True_zmean,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Coupling parameter should be {self.True_zmean} and "
                  f"not {methodVal}.")
        assert testTrue, errStr

    # TODO vector z_mean
    @pytest.mark.parametrize("insert_some_nans", [[], ["V"]])
    @pytest.mark.parametrize("insert_all_nans", [[], ["V"]])
    # @pytest.mark.parametrize("kwargs", [{"method": "classical"},
    #                                     {"method": "quantum"},])   # TODO quantum issues
    def test_handle_nparrays(self, insert_some_nans,
                             insert_all_nans, kwargs={}):
        """Test for ability to handle numpy array quantities"""
        assert_can_handle_nparray(coupling_parameter, insert_some_nans,
                                  insert_all_nans, kwargs)

    def test_quantum(self):
        """
        Testing quantum method for coupling parameter.
        """
        methodVal = coupling_parameter(self.T,
                                       self.n_e,
                                       self.particles,
                                       method="quantum")
        testTrue = np.isclose(self.True_quantum,
                              methodVal,
                              rtol=1e-1,
                              atol=0.0)
        errStr = (f"Coupling parameter should be {self.True_quantum} and "
                  f"not {methodVal}.")
        assert testTrue, errStr
