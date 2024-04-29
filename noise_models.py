from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
import qiskit_aer.noise as noise
import numpy as np

# Constructing depolarizing and pauli error (bit-flip) noise model
def qe_noise_model(q1_gate_err,q2_gate_err, reset_error, meas_error, gate1_error):
    # Error probabilities
    prob_1 = q1_gate_err  # 1-qubit gate
    prob_2 = q1_gate_err # 2-qubit gate
    p_reset = reset_error
    p_meas = meas_error
    p_gate1 = gate1_error
    
    # quantum errors objects - depolarizing noise and pauli error (bit-flip)
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)
    error_3 = noise.pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_4 = noise.pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_5 = noise.pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_6 = noise.pauli_error([('X',p_gate1), ('I', 1 - p_gate1)]).tensor(noise.pauli_error([('X',p_gate1), ('I', 1 - p_gate1)]))

    # Add errors to noise model
    qe_noise_model = noise.NoiseModel()
    qe_noise_model.add_all_qubit_quantum_error(error_1, ["id", "rx", "u1"])
    qe_noise_model.add_all_qubit_quantum_error(error_2, ['cz'])
    qe_noise_model.add_all_qubit_quantum_error(error_3, "reset")
    qe_noise_model.add_all_qubit_quantum_error(error_4, "measure")
    qe_noise_model.add_all_qubit_quantum_error(error_5, ["id", "rx", "u1"])
    qe_noise_model.add_all_qubit_quantum_error(error_6, ["cz"])

    return qe_noise_model

# Constructing readout error noise model
def ro_noise_model(p0given1, p1given0):

    ro_error = ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 - p0given1]])
                            
    ro_noise_model = noise.NoiseModel()
    ro_noise_model.add_all_qubit_readout_error(ro_error)

    return ro_noise_model


# Constructing thermal noise model
def th_noise_model(num_qubits, T1s, T2s, time_u1, time_u2, time_u3, time_cx, time_reset, time_measure):
    
    # number of qubits
    num_qubits = num_qubits
    
    # Truncate random T2s <= T1s
    T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(num_qubits)])

    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
    errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
    errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
    errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
             thermal_relaxation_error(t1b, t2b, time_cx))
              for t1a, t2a in zip(T1s, T2s)]
               for t1b, t2b in zip(T1s, T2s)]

    # Add errors to noise model
    th_noise_model = NoiseModel()
    for j in range(num_qubits):
        th_noise_model.add_quantum_error(errors_reset[j], "reset", [j])
        th_noise_model.add_quantum_error(errors_measure[j], "measure", [j])
        th_noise_model.add_quantum_error(errors_u1[j], "u1", [j])
        th_noise_model.add_quantum_error(errors_u2[j], "u2", [j])
        th_noise_model.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(num_qubits):
        th_noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    return th_noise_model