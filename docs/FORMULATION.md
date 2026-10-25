*TankSlosh · case top-hat-95pct-2mm · branch top-hat-95%-2mm*

# Slosh of a liquid under a ribbed rolling diaphragm

The mathematical formulation behind the model. It covers the boundary-value problems, the constitutive laws, the reductions from three dimensions to a few ordinary differential equations, and every approximation, with its justification and the error it introduces. The case is the 2 mm ribbed top hat, stress-free at 95 % fill, in a 0.4 m × 0.6 m tank.

**Chain of reductions:** 3D liquid + rubber shell, nonlinear → axisymmetric statics: constrained minimum → linearize, split into Fourier modes m = 0, 1 → potential flow → added mass → generalized eigenproblem → uncoupled oscillators (spring-mass) → exact time integration

## 1. Physical setting and scale separation

The tank occupies a rigid axisymmetric domain $\Omega_T\subset\mathbb{R}^3$, with axis $\boldsymbol{e}_z$, radius $R=0.2$ m and overall height $H=0.6$ m. The straight wall runs over $b\le z\le H-b$, closed by 2:1 elliptical heads of depth $b=R/2$. The total volume is $V_T=67.02$ L. An elastomeric diaphragm $\mathcal M$ splits it into a liquid region $\Omega_L(t)$ (water, density $\rho$) and a gas region $\Omega_G(t)$ (N<sub>2</sub>). The tank moves rigidly in translation with acceleration $\boldsymbol{a}(t)$; there is no rotation.

Four separations of scale decide the model.

| Effect | Ratio | Size | Consequence |
|---|---|---|---|
| Liquid compressibility | $(f/f_{ac})^2$, with $f_{ac}\approx c/2H$ | ≈ (10 Hz / 1.2 kHz)² ≈ 10⁻⁴ | $\nabla\!\cdot\boldsymbol{v}=0$: the liquid is incompressible |
| Gas inertia | $\rho_{gas}/\rho$ | ≲ 10⁻² | The gas is massless, so its pressure $p_g(t)$ is uniform |
| Gas stiffness | gas volume fixed by $V_G=V_T-V_L$ | exact | $p_g$ only enters as a constraint multiplier; the absolute pressure drops out |
| Viscosity | $\delta_{Stokes}/L=\sqrt{\nu/\omega}/L$ | ≈ 10⁻³ | The flow is inviscid and irrotational; damping is added separately per mode as $\zeta_n$ |

So the liquid has a velocity potential $\Phi$ with $\Delta\Phi=0$. All motions are small perturbations of a static equilibrium, so the problem is *linear about a nonlinear static state*. The static state itself is strongly nonlinear: large rotations and rolling folds.

## 2. Geometry of the top hat

The molded, stress-free mid-surface is a surface of revolution with meridian $\boldsymbol{X}_0(S)=(R_0(S),Z_0(S))$, where $S\in[0,L]$ is arc length from the clamp. In order along the meridian:

1. The clamp at $(R,\,z_b)$, with $z_b=0.2$ m.

2. A 10 mm flare inward.

3. A straight sidewall at $r=R-\delta$, up to the head seam $z=H-b$.

4. A dome following the upper head, offset inward by $\delta$: $r=(R-\delta)\cos\psi$, $z=H-b+(b-\delta)\sin\psi$.

5. The axis, with $R_0(L)=0$.

The gap $\delta$ is the root of a scalar equation: the gas trapped between the hat and the upper wall must be $5\%$ of the tank when the hat is undeformed.

$$
(1-f_0)\,V_T \;=\; V_T - V_{\le z_b} \;+\; V_d[\boldsymbol{X}_0;\delta], \qquad f_0=0.95 \;\;\Rightarrow\;\; \delta = 6.37\ \text{mm}, \quad L=0.534\ \text{m}.
$$

Here $V_d$ is the signed volume between the diaphragm and the clamp plane, positive below it:

$$
V_d[\boldsymbol{X}] = -\pi\!\int_0^L r(S)^2\, z'(S)\,\mathrm{d} S. \tag{2.1}
$$

The liquid volume is $V_L = V_{\le z_b}-V_d$, so the constraint "fill fraction $f$" is the affine condition $V_d[\boldsymbol{X}]=V_d[\boldsymbol{X}_0]-(f-f_0)V_T$.

## 3. Kinematics of a shell of revolution

An axisymmetric deformed state is a meridian $\boldsymbol{X}(S)=(r(S),z(S))$. The principal stretches are

$$
\lambda_1=\sqrt{r'^2+z'^2}, \qquad \lambda_2=\frac{r}{R_0}, \tag{3.1}
$$

meridional and hoop. Let $\theta$ be the tangent angle measured from the inward radial direction, $(r',z')=\lambda_1(-\cos\theta,\sin\theta)$. The bending measures are the rotation per molded length and the hoop curvature:

$$
\kappa_1 = \frac{\mathrm{d}\theta}{\mathrm{d} S},\qquad \kappa_2 = -\frac{\sin\theta}{r},\qquad \Delta\kappa_\alpha=\kappa_\alpha-\kappa_\alpha^{0}. \tag{3.2}
$$

For a lateral motion the field is no longer axisymmetric, and the stretches come from the surface metric. With $\boldsymbol{a}=\partial_S\boldsymbol{X}$ and $\boldsymbol{b}=R_0^{-1}\partial_\vartheta\boldsymbol{X}$ (the molded orthonormal directions), the in-plane right Cauchy–Green tensor is

$$
\mathsf C=\begin{pmatrix}\boldsymbol{a}\cdot\boldsymbol{a} & \boldsymbol{a}\cdot\boldsymbol{b}\\ \boldsymbol{a}\cdot\boldsymbol{b} & \boldsymbol{b}\cdot\boldsymbol{b}\end{pmatrix},\qquad \lambda_1^2\lambda_2^2=\det\mathsf C,\quad \lambda_1^2+\lambda_2^2=\operatorname{tr}\mathsf C, \tag{3.3}
$$

The twist $\kappa_{12}=R_0^{-1}\,\partial_\vartheta\boldsymbol{t}\cdot\boldsymbol{n}$ joins the two curvatures. In the axisymmetric case $\mathsf C$ is diagonal and $\kappa_{12}=0$, which recovers (3.1)–(3.2).

## 4. Material laws

### Rubber web: incompressible neo-Hookean

The strain energy per volume is $W=\tfrac{\mu}{2}(I_1-3)$ with $J=1$. Plane stress through the thickness fixes $\lambda_3=(\lambda_1\lambda_2)^{-1}$. Integrating over the molded thickness $t$ gives the membrane energy per molded area:

$$
w_m(\lambda_1,\lambda_2)=\frac{\mu t}{2}\Big(\lambda_1^2+\lambda_2^2+\frac{1}{\lambda_1^2\lambda_2^2}-3\Big), \qquad T_1=\frac{1}{\lambda_2}\frac{\partial w_m}{\partial\lambda_1}=\mu t\Big(\frac{\lambda_1}{\lambda_2}-\frac{1}{\lambda_1^3\lambda_2^3}\Big). \tag{4.1}
$$

$T_1$ is the meridional tension per current length; $T_2$ follows by symmetry. For small strain this is linear isotropic elasticity with $E=3\mu$ and $\nu=\tfrac12$. The baseline uses $\mu=1.8$ MPa (assumed, about Shore 70A).

### Bending: Kirchhoff–Love

$$
w_b=\tfrac12\Big[D_s\,\Delta\kappa_1^2 + D_\vartheta\,\Delta\kappa_2^2 + 2\nu D_\vartheta\,\Delta\kappa_1\Delta\kappa_2 + 2(1-\nu)D_\vartheta\,\kappa_{12}^2\Big],\qquad D=\frac{Et^3}{12(1-\nu^2)}=\frac{\mu t^3}{3}=4.8\times10^{-3}\ \text{N·m}. \tag{4.2}
$$

The curvatures are measured per molded length, with the molded thickness. Bending matters only where the strains are a few percent, which here means everywhere. That makes the approximation consistent.

### Ribs: smeared (homogenized) meridional stiffeners

There are $n=36$ ribs of height $h=2$ mm and width $w=3$ mm on the gas face. They are smeared around the circumference with area fraction

$$
\phi(R_0)=\min\!\Big(1,\ \frac{n\,w}{2\pi R_0}\Big), \qquad \phi\approx0.086\ \text{on the sidewall},\quad \phi\to1\ \text{near the apex}. \tag{4.3}
$$

The ribs add three things.

- **Stretching along the meridian.** They act as incompressible fibres in uniaxial tension: $w_f=\tfrac{\mu}{2}\phi h\big(\lambda_1^2+2\lambda_1^{-1}-3\big)$.

- **Meridional bending.** They form a T-section with the web. Its second moment per unit width, about the composite neutral axis $\bar y$, gives $$ I(\phi)=\tfrac{t^3}{12}+t\bar y^2+\phi\Big[\tfrac{h^3}{12}+h\big(\tfrac{t+h}{2}-\bar y\big)^2\Big],\qquad \bar y=\frac{\phi h(t+h)/2}{t+\phi h},\qquad D_s=\frac{12\,I(\phi)}{t^3}D. \tag{4.4}$$ On the sidewall $D_s\approx2.04D$. At the apex, where the ribs merge, $D_s=8D=D\,(\tfrac{t+h}{t})^3$.

- **Hoop bending.** This is the web's, blended toward solid where the ribs merge: $D_\vartheta=(1-\phi)D+\phi D_s$.

The areal density is $\rho_s(t+\phi h)$, giving a total diaphragm mass of 1.344 kg.

> **Homogenization limit**

> Smearing is valid when deformation wavelengths are long compared with the rib spacing, about $2\pi R/n\approx35$ mm on the sidewall. The tight rolling fold (radius about 5–10 mm) violates this locally, so the fold's stiffness is the least certain ingredient.

## 5. The static problem

The equilibrium shape at fill $f$ is a constrained minimum of the stored energy:

$$
\min_{\boldsymbol{X}}\ \Pi[\boldsymbol{X}]=\int_{\mathcal M_0}\big(w_m+w_f+w_b\big)\,\mathrm{d} A_0 \quad\text{subject to}\quad
\begin{cases} V_d[\boldsymbol{X}]=V_d^\ast(f) & \text{(incompressible liquid)}\\[2pt] g\big(\boldsymbol{X}(S)\big)\ge0 & \text{(tank wall, frictionless)}\\[2pt] \operatorname{dist}\big(\boldsymbol{X}(S),\boldsymbol{X}(S')\big)\ge t\ \ \text{for } |S-S'|\gg t & \text{(self-contact)}\\[2pt] \boldsymbol{X}(0)=(R,z_b),\ \theta(0)=\theta_0,\ r(L)=0 & \text{(clamp, axis)}\end{cases} \tag{5.1}
$$

The wall function is $g=1-(r/R)^2-\big((b-z)_+/b\big)^2-\big((z-H+b)_+/b\big)^2$, which is $C^1$ across the head seams. The first-order (KKT) conditions identify the physical reactions:

$$
\delta\Pi=\Delta p\;\delta V_d+\int \nu_w\,\delta g\,\mathrm{d} S+\sum \nu_{c}\,\delta d_c,\qquad \nu_w,\nu_c\ge0,\quad \nu_w g=0,\quad \nu_c(d_c-t)=0. \tag{5.2}
$$

- The volume multiplier is the pressure difference, $\Delta p=p_L-p_g$ (in orbit).

- $\nu_w$ is the wall contact pressure, $\nu_c$ the fold-on-fold contact force.

- The absolute gas pressure appears nowhere. Only the multiplier $\Delta p$ does.

In strong form, (5.2) away from contact is the classical equilibrium of a shell of revolution, with $Q$ the transverse shear from the bending moments $M_\alpha=\partial w_b/\partial\kappa_\alpha$:

$$
\frac{\mathrm{d}}{\mathrm{d} s}(rT_1)-T_2\frac{\mathrm{d} r}{\mathrm{d} s}-rQ\,\kappa_1^{c}=0,\qquad T_1\kappa_1^{c}+T_2\kappa_2^{c}+\frac{1}{r}\frac{\mathrm{d}}{\mathrm{d} s}(rQ)=\Delta p, \tag{5.3}
$$

Here the $\kappa^c$ are current curvatures. The code never forms (5.3) explicitly. It minimizes the discrete form of (5.1), which is equivalent and stays robust through folds, contact and loss of convexity.

> **Path following**

> The fill $f$ is a load parameter. Solutions are continued from $f_0=0.95$ (where $\boldsymbol{X}=\boldsymbol{X}_0$ and $\Pi=0$ exactly) down to 0.05 in steps $\Delta f\le0.01$, each warm-started from the previous one. That follows one branch of a nonconvex problem through the formation of the rolling fold.

## 6. Linearized dynamics

Write $\boldsymbol{X}=\bar{\boldsymbol{X}}+\boldsymbol{u}$, with $\bar{\boldsymbol{X}}$ the static state and multiplier $\overline{\Delta p}$. The liquid is incompressible, so the motion conserves volume. The relevant potential is the Lagrangian of (5.1), and its second variation gives the stiffness:

$$
\tfrac12\,\boldsymbol{u}^{\mathsf T}\mathsf K\boldsymbol{u}=\tfrac12\,\delta^2\big(\Pi-\overline{\Delta p}\,V_d\big)[\boldsymbol{u},\boldsymbol{u}]. \tag{6.1}
$$

The term $-\overline{\Delta p}\,\delta^2V_d$ is the *follower-pressure stiffness*, the change in the work of the pressure as the surface reorients and changes area. Here $\overline{\Delta p}\approx0$, so it is small.

### Fourier separation

The static state is axisymmetric, so $\mathsf K$ and the mass operator commute with rotations about $\boldsymbol{e}_z$. Displacements therefore separate into azimuthal harmonics:

$$
\boldsymbol{u}(S,\vartheta)=\sum_{m\ge0}\Big[u_r^m(S)\cos m\vartheta\,\boldsymbol{e}_r+u_\vartheta^m(S)\sin m\vartheta\,\boldsymbol{e}_\vartheta+u_z^m(S)\cos m\vartheta\,\boldsymbol{e}_z\Big]+(\text{sine family}). \tag{6.2}
$$

A lateral acceleration $a_x$ forces through $x=r\cos\vartheta$, so only $m=1$ is excited. An axial acceleration excites only $m=0$.

- **$m=1$:** $\delta V=\oint u_n\,\mathrm{d} A$ vanishes identically at first order, so the volume constraint is automatic. Regularity on the axis needs $u_\vartheta=-u_r$ and $u_z=0$ at $r=0$.

- **$m=0$:** the constraint $\int u_n\,r\,\mathrm{d} s=0$ must be imposed. The motion is a volume-preserving "breathing" of the diaphragm.

### Constraints in the linear problem

- **Clamp:** $\boldsymbol{u}=0$.

- **Wall contact:** nodes in wall contact satisfy $\boldsymbol{n}_w\!\cdot\boldsymbol{u}=0$, sliding without friction. This assumes the contact persists.

- **Self-contact:** each touching pair satisfies $\boldsymbol{n}\cdot(\boldsymbol{u}_i-\boldsymbol{u}_{\text{seg}})=0$.

All of these are linear, so $\boldsymbol{u}=\mathsf Z\boldsymbol{q}$, with $\mathsf Z$ an orthonormal basis of their null space.

## 7. Liquid: potential flow and added mass

Split the potential into rigid translation and a relative part, $\Phi=\boldsymbol{U}\!\cdot\boldsymbol{x}+\varphi$, where $\boldsymbol{U}(t)$ is the tank velocity. Then

$$
\Delta\varphi=0\ \text{in }\Omega_L,\qquad \partial_n\varphi=0\ \text{on the wall},\qquad \partial_n\varphi=\dot{\boldsymbol{u}}\cdot\boldsymbol{n}\ \text{on the wetted diaphragm}\ \mathcal M_w. \tag{7.1}
$$

For harmonic $m$, set $\varphi=\phi(r,z)\cos m\vartheta$. This reduces (7.1) to a two-dimensional problem on the meridian section $\mathcal D$. In weak form it is: find $\phi$ with $\phi=0$ on the axis (for $m\ge1$) such that for all $\psi$

$$
\int_{\mathcal D}\Big(\nabla\phi\cdot\nabla\psi+\frac{m^2}{r^2}\phi\psi\Big)r\,\mathrm{d} r\,\mathrm{d} z=\int_{\mathcal M_w}\dot u_n\,\psi\,r\,\mathrm{d} s. \tag{7.2}
$$

The Euler–Lagrange equation is $\tfrac1r\partial_r(r\partial_r\phi)+\partial_{zz}\phi-\tfrac{m^2}{r^2}\phi=0$. Linear triangles give $\mathsf A\boldsymbol{\phi}=\mathsf G\dot{\boldsymbol{u}}$. The kinetic energy of the relative flow is a quadratic form in the diaphragm velocity, which defines the *added mass*:

$$
T_\varphi=\tfrac12\rho\!\int_{\Omega_L}|\nabla\varphi|^2=\tfrac12\,\dot{\boldsymbol{u}}^{\mathsf T}\,\underbrace{\rho\,c_m\,\mathsf G^{\mathsf T}\mathsf A^{-1}\mathsf G}_{\mathsf M_a}\,\dot{\boldsymbol{u}},\qquad c_1=\pi,\ c_0=2\pi. \tag{7.3}
$$

For $m=0$ the Neumann problem is solvable only when the net flux is zero, which is exactly the volume constraint. The free constant is fixed at one node.

### Coupling to the tank, and the equations of motion

The cross term in $T=\tfrac12\rho\!\int|\boldsymbol{U}+\nabla\varphi|^2$ is, by Green's identity, $\boldsymbol{U}\!\cdot\rho\oint\boldsymbol{x}\,(\dot{\boldsymbol{u}}\cdot\boldsymbol{n})\,\mathrm{d} A$. That is the rate of change of the liquid's first moment. For $m=1$ it defines the participation vector $\Gamma$ (plus the rubber's own momentum):

$$
\Gamma^{\mathsf T}\dot{\boldsymbol{u}}=\rho\pi\!\int_{\mathcal M_w} r^2\,\dot u_n\,\mathrm{d} s+\textstyle\sum_i \tfrac{m_i}{2}(\dot u_{r,i}-\dot u_{\vartheta,i}) \;=\; \tfrac{\mathrm{d}}{\mathrm{d} t}\big(m\,x_{cg}\big). \tag{7.4}
$$

With the Lagrangian $\mathcal L=\tfrac12 mU^2+U\,\Gamma^{\mathsf T}\dot{\boldsymbol{u}}+\tfrac12\dot{\boldsymbol{u}}^{\mathsf T}\mathsf M\dot{\boldsymbol{u}}-\tfrac12\boldsymbol{u}^{\mathsf T}\mathsf K\boldsymbol{u}$, where $\mathsf M=\mathsf M_a+\mathsf M_s$ includes the rubber's lumped masses, Lagrange's equations give the forced system and the force on the tank:

$$
\boxed{\ \mathsf M\ddot{\boldsymbol{u}}+\mathsf K\boldsymbol{u}=-\Gamma\,a(t),\qquad F_x=-m\,a-\Gamma^{\mathsf T}\ddot{\boldsymbol{u}},\qquad M_y=-m\bar z\,a-H^{\mathsf T}\ddot{\boldsymbol{u}}+m\,a_z\,x_{cg}\ } \tag{7.5}
$$

In the moment, $H^{\mathsf T}\dot{\boldsymbol{u}}=\rho\pi\int_{\mathcal D}\big[z(\phi_r+\phi/r)-r\phi_z\big]r\,\mathrm{d} A$ (plus the rubber's share) is the relative angular momentum. The last term is the moment of the axial load, which acts at the shifted centre of gravity.

## 8. Modal reduction to spring-mass

Solve the generalized symmetric eigenproblem on the constrained subspace, $\mathsf K\boldsymbol{\varphi}_n=\omega_n^2\mathsf M\boldsymbol{\varphi}_n$ with $\boldsymbol{\varphi}_n^{\mathsf T}\mathsf M\boldsymbol{\varphi}_n=1$. Let $q_n$ be the modal coordinates, $\Gamma_n=\boldsymbol{\varphi}_n^{\mathsf T}\Gamma$, and $x_n=q_n/\Gamma_n$. Then (7.5) becomes independent oscillators, each driven by the tank acceleration, with damping added per mode:

$$
\ddot x_n+2\zeta_n\omega_n\dot x_n+\omega_n^2x_n=-a(t),\qquad F_x=-m_0\,a-\sum_n m_n\,(a+\ddot x_n), \tag{8.1}
$$

with

$$
m_n=\Gamma_n^2,\qquad h_n=\frac{\boldsymbol{\varphi}_n^{\mathsf T}H}{\Gamma_n},\qquad m_0=m-\sum_n m_n,\qquad h_0=\frac{m\bar z-\sum_n m_n h_n}{m_0}. \tag{8.2}
$$

> **Proposition**

> (8.1)–(8.2) reproduce the force, the moment and the CG motion of (7.5) exactly when all modes are kept. Moreover $\sum_{\text{all }n}m_n=\Gamma^{\mathsf T}\mathsf M^{-1}\Gamma\le m$: by Cauchy–Schwarz in the $\mathsf M$-inner product, the slosh masses can never exceed the contents.

- **Truncation.** Among the stable modes with $m_n\ge0.002\,m$, the twelve lateral and eleven axial with the largest $m_n$ below $f_c=50$ Hz are kept (padded with the lowest modes above $f_c$, so every fill has the same count), and the rest are lumped into $m_0$. At every fill of the baseline this leaves less than about 2 % of the contents in modes below $f_c$. The truncated modes lie above $f_c$ and move quasi-statically with the tank for excitation well below $f_c$.

- **Stability.** If $\min\operatorname{spec}(\mathsf Z^{\mathsf T}\mathsf K\mathsf Z)<0$, the axisymmetric static state is a saddle for $m=1$: the real diaphragm would buckle sideways. Such modes are reported and excluded.

## 9. Time integration

The spring-mass system is linear and time-invariant, $\dot{\boldsymbol{z}}=\mathsf A\boldsymbol{z}+\mathsf B\boldsymbol{v}(t)$. The input $\boldsymbol{v}$ is either a prescribed acceleration, or a thrust acting on a rigid dry mass, in which case $a=(F+\sum k_nx_n+c_n\dot x_n)/(M_{dry}+m_0)$. If the input is linear between samples (first-order hold), the update is exact:

$$
\boldsymbol{z}_{k+1}=e^{\mathsf A\Delta t}\boldsymbol{z}_k+\Gamma_0\boldsymbol{v}_k+\Gamma_1(\boldsymbol{v}_{k+1}-\boldsymbol{v}_k),\qquad \exp\!\begin{pmatrix}\mathsf A\Delta t&\mathsf B\Delta t&0\\0&0&\mathsf I\\0&0&0\end{pmatrix}=\begin{pmatrix}e^{\mathsf A\Delta t}&\Gamma_0&\Gamma_1\\ \cdot&\cdot&\cdot\end{pmatrix}. \tag{9.1}
$$

The time step therefore only has to resolve the input. It is not a stability or accuracy constraint.

## 10. Gravity (the 1-g test)

With gravity $-g\boldsymbol{e}_z$ the potential gains the liquid's and the rubber's gravitational energy:

$$
\Pi_g=\Pi-\rho g\,Z_d[\boldsymbol{X}]+g\sum_i m_i z_i,\qquad Z_d=-\pi\!\int r^2 z\,z'\,\mathrm{d} S=\int_{\text{cup}}z\,\mathrm{d} V. \tag{10.1}
$$

The multiplier becomes the pressure difference at $z=0$, so $p_L(z)-p_g=\Delta p-\rho gz$. The second variation $-\rho g\,\delta^2Z_d$ is the hydrostatic stiffness. For a flat surface it reduces to $\rho g\int w_n^2\,\mathrm{d} A$, the restoring term of free-surface sloshing. A stress-free, vanishingly thin membrane must therefore reproduce free-surface sloshing with the contact line pinned at the clamp. It does, to 0.003 % (§13).

## 11. Discretization and numerics

| Ingredient | Discretization | Exactness and error |
|---|---|---|
| Meridian | $N=150$ straight segments (piecewise linear $\boldsymbol{X}$), nodes evenly spaced in $S$ | Energies use segment midpoints and nodal curvatures; error $O(N^{-2})$ |
| $V_d$, $Z_d$ | Exact for straight segments (frustum and two-point Gauss) | Exact on the discrete geometry |
| Static minimization | SQP (SLSQP) on $2N$ unknowns with exact gradients; inequality constraints for wall and self-contact | Accepted when volume error $<10^{-8}$, no penetration and small KKT residual |
| Azimuthal integration (m = 1) | 8-point trapezoid rule in $\vartheta$ | Exact: the second-variation integrands are trigonometric polynomials of degree ≤ 6 |
| Stiffness $\mathsf K$ | Element Hessians by central differences of local energies, Richardson-extrapolated over $h$ and $2h$ | $O(h^4)$; symmetric to round-off |
| Liquid | P1 triangles (Delaunay mesh, $h\approx5$ mm, about 4000 nodes); quadrature away from $r=0$; sparse LU | Converged to 0.05 % on free-surface benchmarks |
| Eigenproblem | Dense symmetric generalized eigenproblem on the null space of the constraints | Machine precision |

## 12. Approximations and their errors

| Approximation | Justification | Effect for the top hat |
|---|---|---|
| Incompressible, inviscid liquid | $(f/f_{ac})^2\sim10^{-4}$; boundary layers $\sim10^{-3}L$ | Negligible; damping supplied separately |
| Massless, uniform gas | $\rho_g/\rho\ll1$; gas volume fixed | Negligible; gas pressure drops out |
| Linear perturbation about statics | Small excitation (mm/s² in orbit) | Valid while slosh displacements ≪ fold radius (5–10 mm) |
| Axisymmetric static state | Exact symmetry of tank and diaphragm | **Main limitation.** Several fills are unstable to $m=1$ (§8); real folds may tilt or lobe. Only rings (axisymmetric wrinkles) are representable |
| Neo-Hookean, molded-thickness bending | Strains −5 % … +3 % here | Well inside the small-strain regime; the modulus μ is the real uncertainty (±30 %) |
| Smeared ribs | Wavelength ≫ 35 mm rib spacing | Violated in the tight fold, so fold stiffness is uncertain; rib size is estimated |
| Frictionless contact; contact persists in the linear problem | Lubricated by liquid; small amplitude | Friction would stiffen the fold and add damping |
| Squeeze-film rule (gaps < 2 mm next to the wall, and touching fold layers, are not liquid) | Thin gaps block normal motion | Affects which nodes are wetted near folds |
| Modal truncation | Truncated modes are stiff | Modes outside the 12 lateral + 11 axial kept below 50 Hz are lumped into $m_0$; valid for excitation well below 50 Hz |
| Damping $\zeta_n=0.02$ | Placeholder | Sets ring-down and resonance height, not frequencies |

## 13. Verification

Each operator is tested against a closed-form solution or an independent result. These tests are in `tests/`, 59 in total.

| Operator | Benchmark | Agreement |
|---|---|---|
| Membrane energy + volume | Inflated neo-Hookean sphere, $\Delta p=\frac{2\mu t}{R_0}(\lambda^{-1}-\lambda^{-7})$ | 10⁻⁵ |
| Bending + statics | Clamped plate, $w_0=pa^4/64D$ | 0.05 % |
| Frictionless contact | Uniform $T_1$ along the straight wall | 10⁻⁸ |
| Lateral (m = 1) stiffness | Pre-tensioned membrane $j_{1n}$; clamped plate $\lambda^2=21.26,\,60.82$ | 0.07 % |
| Axial (m = 0) stiffness | Membrane $j_{0n}$; clamped plate $\lambda^2=10.216,\,39.77$ | 0.07 % |
| Liquid FE, m = 1 | Cylinder free surface: $\omega_n$, $m_n$, $h_n$ (Abramson); half-full sphere $\omega^2R/g=1.5601$ | 0.01 % |
| Liquid FE, m = 0 | Axisymmetric free surface ($J_1$ roots); zero axial participation | 0.05 % |
| Gravity coupling | Slack membrane over half-full sphere = pinned free surface, any mesh | 0.003 % |
| Ribs | Fully ribbed plate = solid plate of thickness $t+h$ | 0.5 % |
| Self-consistency | 3D lateral energy = axisymmetric energy; rigid translation costs no energy; $\Delta p=\mathrm{d}\Pi/\mathrm{d} V$ | round-off |
| Published reference | Utsumi (2022) spherical tank, free-surface case | 0–2 % |

## 14. What the top hat does, and why

### Statics: an almost free mechanism

Below 95 % fill the crown descends almost flat, and the top of the sidewall rolls into a U-shaped fold. Consider a fold of fixed shape travelling along a *straight* cylindrical wall. Its energy does not depend on its position, $\partial\Pi/\partial z_{fold}=0$, so translation along the wall is a symmetry. Pushing the fold down sweeps volume at no energy cost, so by (5.2)

$$
\Delta p=\frac{\partial\Pi}{\partial V}\approx0 .
$$

The computed result is $|\Delta p|<0.3$ kPa at every fill, with strains between −5 % and +3 %. This is why rolling diaphragms expel liquid with almost no pressure difference. The flat direction also explains the $m=1$ instabilities: a nearly free mechanism leaves little margin against sideways modes.

### Dynamics: two scaling laws

On a deep liquid, a surface wave of wavenumber $k$ carries added mass $\rho/k$ per area, because the potential decays as $e^{-k\zeta}$. If the restoring force is plate bending, the dispersion relation is

$$
\omega^2\approx\frac{D_{\mathrm{eff}}\,k^5}{\rho}\quad(\text{orbit}),\qquad \omega^2\approx gk\tanh kh+\frac{D_{\mathrm{eff}}\,k^5}{\rho}\quad(\text{1 g}). \tag{14.1}
$$

- **Orbit.** For the lowest $m=1$ crown mode, $k$ lies between $1.84/R$ (free edge) and $3.83/R$ (pinned edge), so $k\approx9$–$19\ \text{m}^{-1}$. With $D_{eff}\approx D_s\approx10^{-2}$ N·m, this gives $f\approx0.1$–$0.8$ Hz. The computation gives a lowest mode of 0.75–1.3 Hz at 80–90 % fill. Bending alone gives the right order: the orbital diaphragm is a weak plate floating on the liquid.

- **1 g.** Gravity dominates: $gk\approx90$–$190\ \text{s}^{-2}$ against about $10\ \text{s}^{-2}$ from bending. So $f\approx1.5$ Hz (free edge) to $2.2$ Hz (pinned edge), and the computed main mode, 1.6–1.9 Hz at most fills, lies inside that bracket. The slack hat barely changes classical sloshing; it mainly pins the contact line.

### Summary of the computed spring-mass parameters

| Fill | Orbit: lateral modes (slosh mass) | Orbit: axial | 1 g: lateral modes | 1 g: axial |
|---|---|---|---|---|
| 10 % | 1.06 Hz (27 %), 4.20 Hz (17 %) | 3.4 Hz (16 %) | 1.00 Hz (31 %), 1.63 Hz (16 %) | 2.5 Hz (6 %) |
| 30 % | 4.1–4.5 Hz (≈ 38 %) | 5.7 Hz (5 %) | 1.86 Hz (22 %), 6.7 Hz (21 %) | 11.3 Hz (2 %) |
| 50 % | 4.47 Hz (23 %)* | 5.3 Hz (2 %) | 1.91 Hz (14 %), 4.69 Hz (15 %)* | 8.6 Hz (2 %) |
| 70 % | 1.99 Hz (22 %), 6.09 Hz (37 %) | 5.4 Hz (22 %) | 2.30 Hz (25 %), 5.99 Hz (34 %) | 6.0 Hz (23 %) |
| 90 % | 0.75 Hz (11 %), 3.15 Hz (50 %) | 3.9 Hz (42 %) | 1.58 Hz (16 %), 4.25 Hz (32 %) | 6.0 Hz (23 %) |

* Static state unstable to $m=1$ at this fill; the unstable mode is left out. The full tables are in `examples/output/baseline_emm.json` and `ground_test_modes.csv`. Mesh convergence, 100 → 200 segments: 0.3 % at 85 % fill. At low fill the dominant mass converges, but its split between closely spaced modes does not.

> **In one paragraph**

> The diaphragm's shape at each fill is found by minimizing rubber energy at fixed liquid volume, with frictionless contact against the wall and against itself. The pressure difference is the Lagrange multiplier. Small motions about that shape split exactly into azimuthal harmonics. The $m=1$ (lateral) and $m=0$ (axial) problems couple the shell's second variation to a Laplace problem for the liquid, which is condensed to an added mass. The resulting symmetric eigenproblem is rewritten exactly as independent spring-mass oscillators driven by the tank acceleration. For this rolling top hat the static stiffness is nearly zero, a free mechanism. The slosh is then governed by rib-reinforced bending in orbit (sub-hertz to a few hertz), and by gravity with a pinned contact line at 1 g (about 1.6–1.9 Hz).
