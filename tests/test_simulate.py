import unittest

import numpy as np

from tankslosh import (EMMParams, SloshMode, cylinder_free_surface, simulate_coupled,
                       simulate_prescribed, time_grid, trapezoid_pulse, vector_input)

G = 9.80665


def one_mode(zeta=0.0, f=1.0, m=10.0, m1=4.0, axial=False):
    return EMMParams(liquid_mass=m, fixed_mass=m - m1, fixed_height=0.1,
                     lateral_modes=[SloshMode(m1, f, zeta, 0.25)],
                     axial_modes=[SloshMode(2.0, 3.0, zeta)] if axial else [])


class TestPrescribed(unittest.TestCase):
    def test_step_matches_analytic(self):
        # zero IC, constant f: x(t) = -(a/w^2)(1 - cos wt); FOH is exact for this input
        p = one_mode()
        w = p.lateral_modes[0].omega
        t = time_grid(5.0, 0.01)
        a = 0.3
        r = simulate_prescribed(p, t, vector_input(t, x=a), initial="rest")
        exact = -(a / w ** 2) * (1 - np.cos(w * t))
        np.testing.assert_allclose(r.lateral_disp[1:, 0, 0], exact[1:], rtol=1e-9, atol=1e-14)

    def test_static_deflection_and_force(self):
        p = one_mode(zeta=0.7)
        t = time_grid(20.0, 0.01)
        a = 0.5
        r = simulate_prescribed(p, t, vector_input(t, x=a), initial="rest")
        w = p.lateral_modes[0].omega
        self.assertAlmostEqual(r.lateral_disp[-1, 0, 0], -a / w ** 2, places=9)
        self.assertAlmostEqual(r.interface_force[-1, 0], -p.liquid_mass * a, places=7)

    def test_free_decay_frequency(self):
        p = one_mode(zeta=0.0, f=0.7)
        t = time_grid(60.0, 0.005)
        u = vector_input(t, y=trapezoid_pulse(t, 0.0, 0.2, 1.0))
        y = simulate_prescribed(p, t, u, initial="rest").lateral_disp[:, 0, 1]
        sel = t > 0.5
        crossings = t[sel][np.where(np.diff(np.sign(y[sel])) != 0)[0]]
        f_est = 0.5 / np.mean(np.diff(crossings))
        self.assertAlmostEqual(f_est, 0.7, delta=0.7 * 1e-3)

    def test_force_equals_momentum_rate(self):
        # F = -m f - m * d2(x_cg)/dt2 (liquid momentum balance)
        p = one_mode(zeta=0.05)
        t = time_grid(10.0, 0.001)
        f = trapezoid_pulse(t, 1.0, 2.0, 0.2, rise=0.3)
        r = simulate_prescribed(p, t, vector_input(t, x=f))
        xcg_dd = np.gradient(np.gradient(r.liquid_cg[:, 0], t), t)
        expected = -p.liquid_mass * (f + xcg_dd)
        err = np.max(np.abs(r.interface_force[5:-5, 0] - expected[5:-5]))
        self.assertLess(err, 1e-3 * np.max(np.abs(expected)))

    def test_static_free_surface_moment_and_cg(self):
        # Steady lateral accel a with axial g: the liquid tilts as a rigid body.
        # Exact: x_cg = -a R^2 / (4 g h), M_y(bottom) = -m a (h/2 + R^2 / (4 h)).
        R, h, a = 0.2, 0.3, 0.4
        p = cylinder_free_surface(R, h, G, n_modes=8, zeta=0.0)
        t = time_grid(1.0, 0.01)
        r = simulate_prescribed(p, t, vector_input(t, x=a, z=G), initial="static")
        m = p.liquid_mass
        # 8-mode truncation error ~ 4e-5
        self.assertAlmostEqual(r.liquid_cg[-1, 0] / (-a * R ** 2 / (4 * G * h)), 1.0, delta=1e-4)
        self.assertAlmostEqual(r.interface_moment[-1, 1] / (-m * a * (h / 2 + R ** 2 / (4 * h))),
                               1.0, delta=1e-4)
        self.assertAlmostEqual(r.interface_force[-1, 2], -m * G, places=6)
        # y-axis by symmetry: M_x(+a_y) = -M_y(+a_x)
        r2 = simulate_prescribed(p, t, vector_input(t, y=a, z=G), initial="static")
        self.assertAlmostEqual(r2.interface_moment[-1, 0], -r.interface_moment[-1, 1], places=9)


    def test_moment_about_mount_point(self):
        # M_P = M_O - P x F for any reference point P
        p = one_mode(zeta=0.03, axial=True)
        t = time_grid(5.0, 0.002)
        u = vector_input(t, x=trapezoid_pulse(t, 0.5, 1.0, 0.3, rise=0.1),
                         y=trapezoid_pulse(t, 1.0, 1.0, -0.2, rise=0.1),
                         z=0.01 + trapezoid_pulse(t, 2.0, 1.0, 0.4, rise=0.1))
        P = np.array([0.2, -0.05, 0.3])
        r0 = simulate_prescribed(p, t, u)
        rP = simulate_prescribed(p, t, u, ref_point=P)
        expected = r0.interface_moment - np.cross(P, r0.interface_force)
        np.testing.assert_allclose(rP.interface_moment, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(rP.interface_force, r0.interface_force)
        self.assertEqual(rP.ref_point, tuple(P))


class TestCoupled(unittest.TestCase):
    def test_momentum_conservation(self):
        p = one_mode(zeta=0.02, axial=True)
        M_dry = 25.0
        t = time_grid(15.0, 0.002)
        pulse = trapezoid_pulse(t, 0.5, 1.0, 3.0, rise=0.1)
        u = vector_input(t, x=pulse, z=0.5 * pulse)
        r = simulate_coupled(p, t, u, dry_mass=M_dry, initial="rest")

        def cumtrapz(y):
            return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])

        impulse = cumtrapz(pulse)[-1]

        for axis, modes, scale in ((0, p.lateral_modes, 1.0), (2, p.axial_modes, 0.5)):
            v_struct = cumtrapz(r.specific_force[:, axis])
            disp = r.lateral_disp[:, :, 0] if axis == 0 else r.axial_disp
            v_modes = np.gradient(disp, t, axis=0) @ np.array([md.mass for md in modes])
            P = (M_dry + p.liquid_mass) * v_struct + v_modes
            self.assertAlmostEqual(P[-1] / (scale * impulse), 1.0, places=4)

    def test_static_coupled_accel(self):
        p = one_mode(zeta=0.0)
        t = time_grid(5.0, 0.01)
        F = 2.0
        r = simulate_coupled(p, t, vector_input(t, x=F), dry_mass=40.0, initial="static")
        np.testing.assert_allclose(r.specific_force[:, 0], F / 50.0, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
