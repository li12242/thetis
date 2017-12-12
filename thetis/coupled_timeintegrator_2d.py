"""
Time integrators for solving coupled 2D-3D system of equations.
"""
from __future__ import absolute_import
from .utility import *
from . import timeintegrator
from .log import *
from . import rungekutta
from . import implicitexplicit
from abc import ABCMeta, abstractproperty


class CoupledTimeIntegrator2D(timeintegrator.TimeIntegratorBase):
    """
    Base class of mode-split time integrators that use 2D, 3D and implicit 3D
    time integrators.
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def swe_integrator(self):
        """time integrator for the shallow water equations"""
        pass

    @abstractproperty
    def tracer_integrator(self):
        """time integrator for the tracer equation"""
        pass

    def __init__(self, solver):
        """
        :arg solver: :class:`.FlowSolver` object
        """
        self.solver = solver
        self.options = solver.options
        self.fields = solver.fields
        self.timesteppers = AttrDict()
        print_output('Coupled time integrator: {:}'.format(self.__class__.__name__))
        print_output('  2D time integrator: {:}'.format(self.integrator_2d.__name__))
        self._initialized = False

        self._create_integrators()

    def _create_swe_integrator(self):
        """
        Create time integrator for 2D system
        """
        solver = self.solver
        momentum_source_2d = solver.fields.split_residual_2d
        if self.options.momentum_source_2d is not None:
            momentum_source_2d = solver.fields.split_residual_2d + self.options.momentum_source_2d
        fields = {
            'coriolis': self.options.coriolis_frequency,
            'momentum_source': momentum_source_2d,
            'volume_source': self.options.volume_source_2d,
            'atmospheric_pressure': self.options.atmospheric_pressure,
        }

        if issubclass(self.swe_integrator, (rungekutta.ERKSemiImplicitGeneric)):
            self.timesteppers.swe2d = self.swe_integrator(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe,
                semi_implicit=True,
                theta=self.options.timestepper_options.implicitness_theta_2d)
        else:
            self.timesteppers.swe2d = self.swe_integrator(
                solver.eq_sw, self.fields.solution_2d,
                fields, solver.dt,
                bnd_conditions=solver.bnd_functions['shallow_water'],
                solver_parameters=self.options.timestepper_options.solver_parameters_2d_swe)

    def _create_tracer_integrator(self):
        """
        Create time integrator for salinity equation
        """
        solver = self.solver

        if self.solver.options.solve_tracer:
            fields = {'elev_2d': self.fields.elev_2d,
                      'uv_2d': self.fields.uv_2d,
                      'diffusivity_h': self.solver.tot_h_diff.get_sum(),
                      'source': self.options.tracer_source_2d,
                      'lax_friedrichs_tracer_scaling_factor': self.options.lax_friedrichs_tracer_scaling_factor,
                      }
            self.timesteppers.tracer = self.tracer_integrator(
                solver.eq_tracer, solver.fields.tracer_2d, fields, solver.dt,
                bnd_conditions=solver.bnd_functions['tracer'],
                solver_parameters=self.options.timestepper_options.solver_parameters_tracer)

    def _create_integrators(self):
        """
        Creates all time integrators with the correct arguments
        """
        self._create_swe_integrator()
        self._create_tracer_integrator()
        self.cfl_coeff_2d = min(self.timesteppers.swe2d.cfl_coeff, self.timesteppers.tracer.cfl_coeff)

    def set_dt(self, dt):
        """
        Set time step for the coupled time integrator

        :arg float dt: Time step.
        """
        for stepper in sorted(self.timesteppers):
            self.timesteppers[stepper].set_dt(dt)

    def initialize(self):
        """
        Assign initial conditions to all necessary fields

        Initial conditions are read from :attr:`fields` dictionary.
        """
        self.timesteppers.swe2d.initialize(self.fields.solution_2d)
        if self.options.solve_tracer:
            self.timesteppers.tracer.initialize(self.fields.tracer_2d)

        self._initialized = True

    def advance(self, t, update_forcings=None):
        self.timesteppers.swe2d.advance(t, update_forcing=update_forcings)
        self.timesteppers.tracer.advance(t, update_forcing=update_forcings)


class CoupledCrankNicolson2D(CoupledTimeIntegrator2D):
    swe_integrator = timeintegrator.CrankNicolson
    tracer_integrator = timeintegrator.CrankNicolson
