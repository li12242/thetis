# 2D diffusion-reacton demo
# =========================
#
# This demo shows how the PETSc
# `diffusion-reaction equation demo <https://gitlab.com/petsc/petsc/-/blob/main/src/ts/tutorials/advection-diffusion-reaction/ex5.c>`__
# can be implemented in Thetis.
#
# In this example, the diffusion and reaction of two tracer fields
# :math:`a` and :math:`b` are governed by the Gray-Scott model:
#
# :math:`\frac{\partial a}{\partial t} - D_1\nabla a + ab^2 + \gamma a = \gamma`
#
# :math:`\frac{\partial b}{\partial t} - D_2\nabla b - ab^2 + (\gamma + \kappa)a = 0`
#
# for diffusion coefficients :math:`D_1,D_2>0` and reactivity
# coefficients :math:`\gamma,\kappa>0`.
#
# Ideally, we would like to solve the PDE system using a mixed
# formulation, which allows for a strong coupling between the
# two components. However, doing so would involve significant
# upheaval to Thetis' solver structure. Instead, we solve the
# two equations separately using a fully implicit method, taking
# a sufficiently small timestep so that the nonlinear
# interactions are not completely lost. ::

from thetis import *

# A doubly periodic square domain is used in this demo. As in the
# other tracer demos, an arbitrary positive bathymetry is used. ::

mesh2d = PeriodicRectangleMesh(65, 65, 2.5, 2.5,
                               direction='both', quadrilateral=True)
P1_2d = FunctionSpace(mesh2d, "CG", 1)
bathymetry2d = Function(P1_2d)
bathymetry2d.assign(1.0)
solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
options = solver_obj.options
options.solve_tracer = True
options.tracer_only = True

# We have two tracer quantities, which we simply call A and B.
# Note that the first equation is linear in A, whereas the
# second is nonlinear in B. The equations will be solved in
# the order in which the labels, names and filenames are
# provided, so we should think carefully about whether to
# solve for A or B first. Let's just choose A first for now. ::

labels = ['a_2d', 'b_2d']
names = ['Tracer A', 'Tracer B']
filenames = ['TracerA2d', 'TracerB2d']
options.fields_to_export = labels

# The setup for tracer problems involing reactions is more involved
# than without because of the two-way couplings and nonlinearities.
# Each equation of the system needs to be decomposed into its source
# terms (which are independent of the prognostic variable), linear
# reaction terms (which are linear in the prognostic variable) and
# quadratic rection terms (which are quadratic in the prognostic
# variable). Observe that the first equation is linear in tracer
# A, but the second equation is nonlinear in tracer B. The reactivity
# terms are handled using dictionaries of coefficients. ::

gamma = Constant(0.024)
kappa = Constant(0.06)
sources = {
    'a_2d': gamma,
    'b_2d': None,
}
lin_react_coeffs = {
    'a_2d': {'const': gamma, 'b_2d_sq': Constant(1.0)},
    'b_2d': {'const': gamma + kappa},
}
quad_react_coeffs = {
    'a_2d': {},
    'b_2d': {'a_2d': Constant(-1.0)},
}
diffusion_coeffs = {
    'a_2d': Constant(8.0e-05),
    'b_2d': Constant(4.0e-05),
}
for label, name, filename in zip(labels, names, filenames):
    options.add_tracer_2d(
        label, name, filename, sources=sources.get(label),
        linear_reaction_coefficients=lin_react_coeffs.get(label),
        quadratic_reaction_coefficients=quad_react_coeffs.get(label),
        horizontal_diffusivity=diffusion_coeffs.get(label),
        source=sources.get(label))

# As mentioned above, we use a fully implicit time integration
# scheme. ::

options.timestepper_type = 'CrankNicolson'
options.timestepper_options.implicitness_theta = 1.0
options.simulation_export_time = 10.0

# For purposes of efficiency, we use first order Lagrange elements
# rather than the default discontinuous Galerkin discretisation.
# SUPG stabilisation is not applied because the problem is not
# advection-dominated. (In fact, it has no advection at all!) ::

options.tracer_element_family = 'cg'
options.use_supg_tracer = False

# Initial conditions for each tracer must be assigned
# separately. ::

x, y = SpatialCoordinate(mesh2d)
b_init = interpolate(conditional(
    And(And(x >= 1.0, x <= 1.5), And(y >= 1, y <= 1.5)),
    0.25*(sin(4.0*pi*x)*sin(4.0*pi*y))**2, 0.0), P1_2d)
a_init = interpolate(1.0 - 2.0*b_init, P1_2d)
solver_obj.assign_initial_conditions(a=a_init, b=b_init)

# A small timestep is required to capture the initial dynamics,
# but a larger one suffices for most of the simulation. Note
# that the timestep only ramps up to unity here, whereas it is
# :math:`\mathcal O(10)` in the original PETSc demo. ::

T = 0
for i in range(5):
    options.timestep = 0.0001*10**i
    T += options.timestep
    solver_obj.dt = options.timestep
    options.simulation_end_time = 9*T
    solver_obj.timestepper.timesteppers.a_2d.set_dt(options.timestep)
    solver_obj.timestepper.timesteppers.b_2d.set_dt(options.timestep)
    solver_obj.iterate()
    solver_obj.export_initial_state = False
options.simulation_end_time = 2000.0
solver_obj.iterate()

# This tutorial can be dowloaded as a Python script `here <demo_2d_diffusion_reaction.py>`__.
