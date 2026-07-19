import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
import pandas as pd
import matplotlib.patches as mpatches
import os

class GenericMotor:

    """A generic motor class which can bind, unbind, activate and step."""

    def __init__(self,
                 motor_id: int,
                 stiffness: float,
                 rest_length: float,
                 step_size: float,
                 binding_rate: float,
                 func_stepping_rate: Callable[[float], float],
                 func_unbinding_rate: Callable[[float], float],
                 direction: float = 1.0):


        # Identity
        self.id = motor_id
        self.motor_type = "GenericMotor"
        self.direction = direction  # +1 for plus-end directed, -1 for minus-end directed

        # Physical Properties
        self.k = stiffness  # pN/nm
        self.L0 = rest_length  # nm
        self.step_size = step_size  # nm

        # Kinetics
        self.step_rate_function = func_stepping_rate
        self.unbind_rate_function = func_unbinding_rate
        self.binding_rate = binding_rate

        # States
        self.is_bound = False
        self.head_pos = 0.0  # nm. Position of motor head along filament, attached to MT
        self.tail_pos = 0.0  # nm. Position of motor tail relative to cargo center, attached to cargo

    def __repr__(self):
        state = "Bound" if self.is_bound else "Unbound"
        return f"<{self.motor_type} {self.id}: {state} at x={self.head_pos:.1f}>"

    def get_force(self, cargo_position: float) -> float:

        """
        Calculate the force exerted by the motor based on its extension.
        Positive force implies pulling to the right.
        """

        if not self.is_bound:
            return 0.0 # No force if unbound

        # tail_absolute_pos = cargo_position + self.tail_pos
        # extension = self.head_pos - tail_absolute_pos - self.L0

        natural_head_pos = (cargo_position + self.tail_pos) + (self.direction * self.L0)
        extension = self.head_pos - natural_head_pos
        force = self.k * extension

        return force

    def calculate_rates(self, cargo_position: float, system_state: Optional[dict] = None) -> dict:

        """
        Returns a dictionary of allowable events and their rates.
        """

        rates = {}

        if self.is_bound:
            # If bound, we can Step or Unbind
            force = self.get_force(cargo_position)
            hindering_load = force * self.direction  # Load seen by the motor, stepping direction considered

            rates['step'] = self.step_rate_function(hindering_load)
            rates['unbind'] = self.unbind_rate_function(force)
            # Binding rate is 0 because we are already bound

        else:
            # If unbound, we can ONLY Bind
            # We assume binding rate is constant (or could be func of position)
            rates['bind'] = self.binding_rate

        return rates

    def bind(self, cargo_position: float):
        """
        Transitions the motor from Unbound -> Bound.
        """
        if self.is_bound:
            return  # Safety check. If already bound, do nothing.

        self.is_bound = True

        # Standard assumption: It binds in a relaxed state (zero stretch).
        self.head_pos = cargo_position + self.tail_pos + (self.direction * self.L0)

    def unbind(self, force: float = 0.0):
        """
        Transitions the motor from Bound -> Unbound.
        """
        self.is_bound = False
        self.head_pos = None  # It doesn't have a position on the MT anymore
        
        if hasattr(self, 'in_catch_regime'):
            self.in_catch_regime = False

    def step(self):
        """
        Moves the motor head forward by one step size.
        """
        if self.is_bound:
            self.head_pos += self.step_size * self.direction
class Dynein(GenericMotor):

    """Dynein motor subclass with hindered activation."""

    def __init__(self,
                 motor_id: int,
                 stiffness: float,
                 rest_length: float,
                 step_size: float,
                 binding_rate: float,
                 func_stepping_rate: Callable[[float], float],
                 func_unbinding_rate: Callable[[float], float],
                 stall_force: float,
                 k_activation_0: float,
                 beta_hindrance: float,
                 inhibition_trigger_force: float):

        # Initialize base GenericMotor
        super().__init__(motor_id, stiffness, rest_length, step_size, binding_rate,
                         func_stepping_rate, func_unbinding_rate, direction = -1.0)

        self.motor_type = "Dynein"
        self.stall_force = stall_force  # pN
        self.k_activation_0 = k_activation_0
        self.beta = beta_hindrance
        self.inhibition_trigger_force = inhibition_trigger_force  # pN

        # State: False = Active(D), True = Inhibited(D*)
        self.is_inhibited = False
        self.in_catch_regime = False

    def __repr__(self):
        base = super().__repr__()
        status = "INHIBITED" if self.is_inhibited else "ACTIVE"
        return f"{base} [{status}]"

    def calculate_rates(self, cargo_position: float, system_state: Optional[dict] = None) -> dict:
        """
        Extends base method to include activation/inhibition transitions.
        system_state = {'n_inactive': int, 'n_active': int}
        """
        rates = {}

        # Case 1: Bound
        if self.is_bound:
            force = self.get_force(cargo_position)
            hindering_load = force * self.direction

            rates['step'] = self.step_rate_function(hindering_load)
            rates['unbind'] = self.unbind_rate_function(force)
            return rates

        # Case 2: Unbound

        # (D -> D_MT)
        if not self.is_inhibited:
            rates['bind'] = self.binding_rate
            return rates

        # (D* -> D)
        if self.is_inhibited:
            if system_state is None:
                # Fallback if no state provided (should not happen in proper sim)
                raise ValueError("System state required for Dynein activation rate calculation.")

            # Get n_inactive from system state
            n_inactive = system_state.get('n_inactive')

            # # k = k0 / (1 + beta * n * (n - 1))
            # denominator = 1.0 + self.beta * n_inactive * (n_inactive - 1)

            # k = k0 / (1 + beta * (n - 1))
            denominator = 1.0 + self.beta * (n_inactive - 1)
            activation_rate = self.k_activation_0 / denominator

            rates['activate'] = activation_rate
            return rates

    def unbind(self, force: float = 0.0):
        """
        Transitions the motor from Bound -> Unbound.
        If F > F_stall, go to Inhibited state(D*), else Active(D).
        """

        self.is_bound = False
        self.head_pos = None
        self.in_catch_regime = False

        if abs(force) > self.inhibition_trigger_force:
            self.is_inhibited = True  # Go to D*
        else:
            self.is_inhibited = False  # Go to D

    def activate(self):
        """
        Transitions the motor from Inhibited(D*) -> Active(D).
        """
        self.is_inhibited = False


class DyneinActive(GenericMotor):

    """Dynein motor subclass without hindered activation."""

    def __init__(self,
                 motor_id: int,
                 stiffness: float,
                 rest_length: float,
                 step_size: float,
                 binding_rate: float,
                 func_stepping_rate: Callable[[float], float],
                 func_unbinding_rate: Callable[[float], float],
                 stall_force: float,
                 k_activation_0: float,
                 beta_hindrance: float,
                 inhibition_trigger_force: float):

        # Initialize base GenericMotor
        super().__init__(motor_id, stiffness, rest_length, step_size, binding_rate,
                         func_stepping_rate, func_unbinding_rate, direction = -1.0)

        self.motor_type = "Dynein"
        self.stall_force = stall_force  # pN
        self.k_activation_0 = k_activation_0
        self.beta = beta_hindrance
        self.inhibition_trigger_force = inhibition_trigger_force  # pN

        # State: False = Active(D), True = Inhibited(D*)
        self.is_inhibited = False
        self.in_catch_regime = False

    def __repr__(self):
        base = super().__repr__()
        status = "INHIBITED" if self.is_inhibited else "ACTIVE"
        return f"{base} [{status}]"

    def calculate_rates(self, cargo_position: float, system_state: Optional[dict] = None) -> dict:
        """
        Extends base method to include activation/inhibition transitions.
        system_state = {'n_inactive': int, 'n_active': int}
        """
        rates = {}

        # Case 1: Bound
        if self.is_bound:
            force = self.get_force(cargo_position)
            hindering_load = force * self.direction

            rates['step'] = self.step_rate_function(hindering_load)
            rates['unbind'] = self.unbind_rate_function(force)
            return rates

        # Case 2: Unbound

        # (D -> D_MT)
        if not self.is_inhibited:
            rates['bind'] = self.binding_rate
            return rates

        # (D* -> D)
        if self.is_inhibited:
            if system_state is None:
                # Fallback if no state provided (should not happen in proper sim)
                raise ValueError("System state required for Dynein activation rate calculation.")

            # Get n_inactive from system state
            n_inactive = system_state.get('n_inactive')

            # k = k0 / (1 + beta * n * (n - 1))
            denominator = 1.0 + self.beta * n_inactive * (n_inactive - 1)
            activation_rate = self.k_activation_0 / denominator

            rates['activate'] = activation_rate
            return rates

    def unbind(self, force: float = 0.0):
        """
        Transitions the motor from Bound -> Unbound.
        If F > F_stall, go to Inhibited state(D*), else Active(D).
        """

        self.is_bound = False
        self.head_pos = None
        self.in_catch_regime = False

        if abs(force) > self.inhibition_trigger_force:
            self.is_inhibited = False
        else:
            self.is_inhibited = False

    def activate(self):
        """
        Transitions the motor from Inhibited(D*) -> Active(D).
        """
        self.is_inhibited = False


class Kinesin(GenericMotor):
    """
    Kinesin motor subclass.
    Behaves like a standard motor: Binds -> Steps -> Unbinds.
    Unlike Dynein, it does NOT have an inhibited state or activation step.
    """

    def __init__(self,
                 motor_id: int,
                 stiffness: float,
                 rest_length: float,
                 step_size: float,
                 binding_rate: float,
                 func_stepping_rate: Callable[[float], float],
                 func_unbinding_rate: Callable[[float], float],
                 stall_force: float):

        # Initialize Base
        super().__init__(motor_id, stiffness, rest_length, step_size,
                         binding_rate, func_stepping_rate, func_unbinding_rate, direction = 1.0)

        self.motor_type = "Kinesin"
        self.stall_force = stall_force

    def calculate_rates(self, cargo_position: float, system_state: Optional[dict] = None) -> dict:
        """
        Calculates rates for Kinesin.
        Ignores 'system_state' (no cooperativity).
        Stops stepping if Force > Stall Force.
        """
        rates = {}

        if self.is_bound:
            force = self.get_force(cargo_position)
            hindering_load = force * self.direction

            rates['step'] = self.step_rate_function(hindering_load)
            rates['unbind'] = self.unbind_rate_function(force)
            return rates

        else:
            # Unbound Kinesin is always ready to bind
            rates['bind'] = self.binding_rate
            return rates
def create_stepping_function(v0: float, f_stall: float, step_size: float, w: float = 1.0):
    """
    Creates a force-dependent stepping rate function. Force input is the hindering load seen by the motor in its stepping direction.
    """

    # Pre-calculate unloaded rate
    base_rate = v0 / step_size

    def stepping_rate(force: float) -> float:
        # 1. Assisting Load (F < 0) -> Clamp to max speed
        if force < 0:
            return base_rate

        # 2. Super-Stall Load (F > F_stall) -> Stop stepping
        if force >= f_stall:
            return 0.0

        # 3. Normal Regime (0 <= F < F_stall)
        velocity = v0 * (1.0 - force / f_stall) ** w
        return velocity / step_size

    return stepping_rate
def create_kinesin_unbinding(epsilon_0: float, f_detachment: float):
    """
    Creates a standard 'Slip Bond' unbinding function (Bell Model).
    Rate increases exponentially with force.
    """
    def unbinding_rate(force: float) -> float:
        return epsilon_0 * np.exp(abs(force) / f_detachment)

    return unbinding_rate

def create_dynein_unbinding(epsilon_0: float, f_detachment: float,
                            f_inhibition_trigger_force: float, epsilon_flat: float):
    """
    Regime 1 (f < f_inhibition_trigger_force): Slip Bond (Exponential increase).
    Regime 2 (f >= f_inhibition_trigger_force): Catch/Ideal Bond (Drops to constant plateau).
    """
    def unbinding_rate(force: float) -> float:
        f_mag = abs(force)

        # Regime 1: Normal stepping load (Slip Bond)
        if f_mag < f_inhibition_trigger_force:
            return epsilon_0 * np.exp(f_mag / f_detachment)

        # Regime 2: Super-stall load (Plateau)
        # The rate drops to a stable value (epsilon_flat)
        else:
            return epsilon_flat

    return unbinding_rate

def create_dynein_unbinding_tfbd(epsilon_0: float, f_inhibition_trigger_force: float, f_detachment: float, alpha_val: float, f_0: float):
    """
    Creates a 'Threshold Force Bond Dissocation Model based unbinding function for Dynein.
    Regime 1 (f < f_inhibition_trigger_force): Slip Bond (Exponential increase)."
    Regime 2 (f >= f_inhibition_trigger_force): TFBD Model (Decreasing unbinding rate).
    """

    def unbinding_rate(force: float) -> float:
        f_mag = abs(force)

        # Regime 1: Normal stepping load (Slip Bond)
        if f_mag < f_inhibition_trigger_force:
            return epsilon_0 * np.exp(f_mag / f_detachment)

        else:
            D_e = alpha_val * (1.0 - np.exp(-(f_mag - f_inhibition_trigger_force)/f_0)) #Detachment Energy
            eps_catch = epsilon_0 * np.exp(-D_e + f_mag/f_detachment)
            return eps_catch

    return unbinding_rate

class Cargo:
    """
    Simulation Manager with detailed history logging.
    """
    def __init__(self, trap_stiffness=0.0, trap_center=0.0):
        self.motors = []
        self.x = 0.0
        self.time = 0.0

        # Optical Trap
        self.k_trap = trap_stiffness
        self.x_trap = trap_center
        
        self.cumulative_catch_events = 0
        self.cumulative_catch_unbinds = 0

        # History
        self.history = {
            'time': [],
            'cargo_pos': [],
            'n_kinesin_bound': [],
            'n_dynein_bound': [],   # L (Attached)
            'n_dynein_active': [],  # j (Detached, Active)
            'n_dynein_inactive': [], # n_in (Detached, Inhibited)
            'catch_bond_events': [],
            'catch_unbind_events': [],
            'total_kinesin_force': [],
            'total_dynein_force': []
        }

        # Dictionary to track individual motor heads: {motor_id: [pos_t0, pos_t1...]}
        self.history_motor_pos = {}

    def add_motor(self, motor):
        self.motors.append(motor)
        # Initialize a history list for this specific motor
        self.history_motor_pos[motor.id] = []

    def update_position(self):
        numerator = 0.0
        denominator = 0.0

        # Sum forces from BOUND motors
        for m in self.motors:
            if m.is_bound:
                numerator += m.k * (m.head_pos - m.tail_pos - (m.L0 * m.direction))
                denominator += m.k

        # Sum force from Trap
        if self.k_trap > 0:
            numerator += self.k_trap * self.x_trap
            denominator += self.k_trap

        if denominator > 0:
            self.x = numerator / denominator

    def get_system_state(self) -> dict:
        """Counts Active/Inactive Dyneins for cooperativity calculation."""
        n_inactive = 0
        n_active = 0
        for m in self.motors:
            if m.motor_type == "Dynein":
                # Check inhibited flag (only exists on Dynein objects)
                if m.is_inhibited:
                    n_inactive += 1
                elif not m.is_bound:
                    # Unbound and Not Inhibited = Active (j)
                    n_active += 1
        return {'n_inactive': n_inactive, 'n_active': n_active}

    def record_state(self):
        """Snapshots the current system state to history lists."""
        self.history['time'].append(self.time)
        self.history['cargo_pos'].append(self.x)

        # Counters
        k_bound = 0
        d_bound = 0
        d_active = 0   # Unbound active
        d_inactive = 0 # Unbound inactive
        total_k_force = 0.0
        total_d_force = 0.0

        for m in self.motors:
            # 1. Record Individual Head Position
            # If unbound, we store NaN (so plots show gaps instead of lines to 0)
            pos = m.head_pos if m.is_bound else np.nan
            self.history_motor_pos[m.id].append(pos)

            # 2. Update Counts
            if m.motor_type == "Kinesin":
                if m.is_bound: 
                    k_bound += 1
                    total_k_force += abs(m.get_force(self.x))

            elif m.motor_type == "Dynein":
                if m.is_bound:
                    d_bound += 1
                    f_mag = abs(m.get_force(self.x))
                    total_d_force += f_mag
                    
                    if f_mag > m.inhibition_trigger_force:
                        if not m.in_catch_regime:
                            m.in_catch_regime = True
                            self.cumulative_catch_events += 1
                    # Removed the 'else: m.in_catch_regime = False' block.
                    # Flag now only resets when the motor completely unbinds.
                        
                elif m.is_inhibited:
                    d_inactive += 1
                else:
                    d_active += 1

        self.history['n_kinesin_bound'].append(k_bound)
        self.history['n_dynein_bound'].append(d_bound)
        self.history['n_dynein_active'].append(d_active)
        self.history['n_dynein_inactive'].append(d_inactive)
        self.history['catch_bond_events'].append(self.cumulative_catch_events)
        self.history['catch_unbind_events'].append(self.cumulative_catch_unbinds)
        self.history['total_kinesin_force'].append(total_k_force)
        self.history['total_dynein_force'].append(total_d_force)

    def step_gillespie(self):
        # 1. Physics & State
        self.update_position()
        sys_state = self.get_system_state()

        # 2. Record Data (Before the event happens)
        self.record_state()

        # 3. Calculate Rates
        all_rates = []
        all_events = []
        total_rate = 0.0

        for motor in self.motors:
            r_dict = motor.calculate_rates(self.x, sys_state)
            for event_type, rate in r_dict.items():
                if rate > 0:
                    all_rates.append(rate)
                    all_events.append((motor, event_type))
                    total_rate += rate

        if total_rate == 0:
            print("No events can occur. Advancing time by small delta.")
            self.time += 0.01
            return

        # 4. Determine Time Step
        r1 = np.random.random()
        tau = (1.0 / total_rate) * np.log(1.0 / r1)
        self.time += tau

        # 5. Select Event
        r2 = np.random.random() * total_rate
        cumulative_rate = 0.0
        selected_event = None
        for i, rate in enumerate(all_rates):
            cumulative_rate += rate
            if cumulative_rate > r2:
                selected_event = all_events[i]
                break

        # 6. Execute Event
        if selected_event:
            motor_obj, event_type = selected_event
            if event_type == 'step':
                motor_obj.step()
            elif event_type == 'bind':
                motor_obj.bind(self.x)
            elif event_type == 'unbind':
                # Pass force for Dynein catch-bond logic
                force = motor_obj.get_force(self.x)
                if motor_obj.motor_type == "Dynein" and abs(force) > motor_obj.inhibition_trigger_force:
                    self.cumulative_catch_unbinds += 1
                motor_obj.unbind(force)
            elif event_type == 'activate':
                motor_obj.activate()