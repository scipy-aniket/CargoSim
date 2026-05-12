params = {
    # Motor counts
    'n_dynein': 6,
    'n_kinesin': 1,

    # Motor type (DyneinActive or Dynein)
    'dynein_motor_type': 'Dynein',  # Options: 'DyneinActive' or 'Dynein'

    # Simulation settings
    't_max': 20.0,  # seconds
    'base_seed': 100,  # Base random seed

    # Trap settings
    'trap_stiffness': 0.0,  # pN/nm
    'trap_center': 0.0,  # nm

    # Dynein parameters
    'f_stall_d': 1.1,  # pN
    'f_inhibition_trigger_d': 1.3,  # pN
    'k_stiffness_dynein': 0.3,  # pN/nm
    'rest_length_dynein': 70.0,  # nm
    'v0_d': 1000.0,  # nm/s
    'stepsize_dynein': 8.0,  # nm
    'w_dynein': 0.5,  # Force-velocity exponent
    'eps_d': 0.67,  # 1/s - Zero load unbind rate
    'eps_d_flat': 2.0,  # 1/s - Flat rate at super-stall
    'f_d_detach': 0.67,  # pN
    'act_k0': 2.5,  # 1/s - Activation rate k0
    'beta': 1.1,  # Hindrance parameter
    'pi_d': 2.5,  # 1/s - Binding rate

    # TFBD parameters for Dynein
    'alpha_tfbd': 58.08,  # pN*nm - Detachment energy scaling
    'f_0_tfbd': 29.0,  # pN - Characteristic force

    # Kinesin parameters
    'f_stall_k': 6.0,  # pN
    'k_stiffness_kinesin': 0.3,  # pN/nm
    'rest_length_kinesin': 60.0,  # nm
    'v0_k': 1000.0,  # nm/s
    'stepsize_kinesin': 8.0,  # nm
    'w_kinesin': 2.0,  # Force-velocity exponent
    'eps_k': 1,  # 1/s - Zero load unbind rate
    'f_k_detach': 4,  # pN
    'pi_k': 5,  # 1/s - Binding rate
}
