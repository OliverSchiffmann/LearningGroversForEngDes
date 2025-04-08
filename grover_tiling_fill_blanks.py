# readMe:
# This script solves a simplified tiling problem using Grover's algorithm.
# The problem: Place two 1x1 tiles onto an 8x8 grid.
# Constraints:
# 1. Both tiles must be placed against the 'eastern wall' (x-coordinate = 7, represented as binary '111').
# 2. The two tiles cannot occupy the same position.
# This script uses Qiskit to simulate the quantum circuit locally.
# EXERCISE: Find and fix the 10 blanks marked with '# --- BLANK X ---'.

import numpy as np

# --- BLANK 1 ---
# pandas seems to be missing...

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer.primitives import (
    Sampler as AerSampler,
)

# --- Helper Functions ---


def format_binary_position(binary_str):
    """Converts a 3-bit binary string into an integer coordinate."""
    return int(binary_str, 2)


def dict_to_array_for_excel(results_dict):
    """Converts results dictionary to list of lists for Excel."""
    array_result = []
    for binary_key, frequency in results_dict.items():
        x1_bin = binary_key[0:3]
        y1_bin = binary_key[3:6]
        x2_bin = binary_key[6:9]
        y2_bin = binary_key[9:12]
        pos1_str = (
            f"({format_binary_position(x1_bin)}, {format_binary_position(y1_bin)})"
        )
        pos2_str = (
            f"({format_binary_position(x2_bin)}, {format_binary_position(y2_bin)})"
        )
        array_result.append([pos1_str, pos2_str, frequency])
    return array_result


def results_to_excel(array, folder, file_name):
    """Writes the processed results array to an Excel file."""
    # --- BLANK 2 ---
    # This function needs the pandas library (often imported as 'pd') to work.
    # Let's assume 'pd' exists for now, but the import might be missing (See BLANK 1).
    try:
        # The DataFrame creation might fail if pandas wasn't imported correctly.
        df = pd.DataFrame(
            array,
            columns=["Tile 1 Position (x,y)", "Tile 2 Position (x,y)", "Frequency"],
        )
        file_path = f"{folder}/{file_name}"
        df.to_excel(file_path, index=False, header=True)
        print(f"Results successfully saved to: {file_path}")
    except NameError:  # Catching the specific error if 'pd' is not defined
        print(
            "Error saving results: Looks like the 'pandas' library (pd) wasn't imported correctly. See BLANK 1."
        )
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
        print("Ensure the output directory exists and you have write permissions.")
        print("Directory specified:", folder)


def convert_int_key_to_binary(quasi_dists_dict, num_measurement_qubits=12):
    """Converts integer keys from Qiskit results to binary strings."""
    binary_results = {}
    binary_format = f"0{num_measurement_qubits}b"
    for key, value in quasi_dists_dict.items():
        binary_key = format(key, binary_format)
        # --- BLANK 3 ---
        # Qiskit often reverses bit order on measurement. Is this line doing the right thing?
        reversed_binary_key = binary_key  # Hint: How do you reverse a string in Python?
        binary_results[reversed_binary_key] = value
    return binary_results


def quasi_prob_to_frequency(binary_results_dict, num_shots):
    """Converts quasi-probabilities into estimated frequencies."""
    frequency_results = {
        key: round(value * num_shots) for key, value in binary_results_dict.items()
    }
    frequency_results = {k: v for k, v in frequency_results.items() if v > 0}
    return frequency_results


# --- Quantum Circuit Components ---


def initialize_state(qc, x1_q, y1_q, x2_q, y2_q, v_q):
    """Prepares the initial state for Grover's algorithm."""
    print("Initializing state...")
    # Apply Hadamard to create superposition
    for q in x1_q:
        qc.h(q)
    # --- BLANK 4 ---
    # Are we applying superposition correctly to *all* position qubits? Check y1_q.
    for q in y1_q:
        # Something seems to be missing here...
        pass  # Placeholder to avoid syntax error, remove when fixed
    for q in x2_q:
        qc.h(q)
    for q in y2_q:
        qc.h(q)

    # Initialize the oracle qubit 'v' to |-> state.
    qc.x(v_q[0])
    qc.h(v_q[0])
    print("Initialization complete.")


def build_oracle(qc, x1_q, y1_q, x2_q, y2_q, c_q, v_q):
    """Constructs the Oracle for Grover's algorithm."""
    print("Building Oracle...")
    # --- Compute ---
    # Condition 1: Check if x1 is '111' (decimal 7)
    # --- BLANK 5 ---
    # Is the condition for x1=7 being checked correctly?
    # A line is missing here...

    # Condition 1: Check if x2 is '111' (decimal 7)
    qc.mcx([x2_q[0], x2_q[1], x2_q[2]], c_q[1])
    qc.barrier()

    # Condition 2 Helper: Check if y1 == y2 bit by bit -> store in c_q[2,3,4]
    qc.cx(y1_q[0], c_q[2])
    qc.cx(y2_q[0], c_q[2])
    qc.x(c_q[2])  # c_q[2]=1 if y1[0]==y2[0]
    qc.cx(y1_q[1], c_q[3])
    qc.cx(y2_q[1], c_q[3])
    qc.x(c_q[3])  # c_q[3]=1 if y1[1]==y2[1]
    qc.cx(y1_q[2], c_q[4])
    qc.cx(y2_q[2], c_q[4])
    qc.x(c_q[4])  # c_q[4]=1 if y1[2]==y2[2]
    qc.barrier()

    # Combine equality checks: Store 'y1 == y2' in c_q[5].
    qc.mcx([c_q[2], c_q[3], c_q[4]], c_q[5])
    # We need y1 != y2 for a valid solution. Flip the result.
    qc.x(c_q[5])  # Now c_q[5] = 1 if y1 != y2

    # --- Combine all conditions --- Mark solution by flipping v_q[0]
    qc.barrier()
    qc.mcx([c_q[0], c_q[1], c_q[5]], v_q[0])
    qc.barrier()

    # --- Uncompute --- Reverse operations to reset ancilla qubits
    qc.x(c_q[5])
    qc.mcx([c_q[2], c_q[3], c_q[4]], c_q[5])
    qc.barrier()
    qc.x(c_q[4])
    qc.cx(y2_q[2], c_q[4])
    qc.cx(y1_q[2], c_q[4])
    qc.x(c_q[3])
    qc.cx(y2_q[1], c_q[3])
    qc.cx(y1_q[1], c_q[3])
    qc.x(c_q[2])
    qc.cx(y2_q[0], c_q[2])
    qc.cx(y1_q[0], c_q[2])
    qc.barrier()
    qc.mcx([x2_q[0], x2_q[1], x2_q[2]], c_q[1])  # Uncompute x2 check
    # --- BLANK 6 ---
    # Make sure the uncompute step for x1 matches the (missing) compute step from BLANK 5.
    # Is this uncompute needed if compute was missing?

    print("Oracle build complete.")


def build_diffuser(nqubits):
    """Constructs the Grover Diffuser operator."""
    print("Building Diffuser...")
    diff_qc = QuantumCircuit(nqubits, name="Diffuser")
    diff_qc.h(range(nqubits))
    diff_qc.x(range(nqubits))
    diff_qc.h(nqubits - 1)
    diff_qc.mcx(list(range(nqubits - 1)), nqubits - 1)
    diff_qc.h(nqubits - 1)
    diff_qc.x(range(nqubits))
    diff_qc.h(range(nqubits))
    print("Diffuser build complete.")
    return diff_qc.to_gate()


# --- Main Execution ---

if __name__ == "__main__":
    # --- Configuration ---
    num_iterations = 9
    num_shots = 1000
    output_folder = "."
    # --- BLANK 7 ---
    # What should the output file be called? Needs a proper filename extension.
    output_filename = "grover_tiling_output"  # Hint: What kind of file are we creating?

    # --- Quantum Register Setup ---
    x1_q = QuantumRegister(3, "x1")
    y1_q = QuantumRegister(
        3, "y1"
    )  # Size should match encoding (0-7 requires 3 qubits)
    x2_q = QuantumRegister(3, "x2")
    y2_q = QuantumRegister(3, "y2")
    c_q = QuantumRegister(6, "c")
    v_q = QuantumRegister(1, "v")
    # --- BLANK 8 ---
    # How many classical bits do we need to store the measurement results of all position qubits?
    num_position_qubits = 12  # 3 for x1 + 3 for y1 + 3 for x2 + 3 for y2
    cbits = ClassicalRegister(11, "cbits")  # Hint: Does 11 match num_position_qubits?

    # Create the main Quantum Circuit
    qc = QuantumCircuit(x1_q, y1_q, x2_q, y2_q, c_q, v_q, cbits)

    # --- Build the Grover Circuit ---
    initialize_state(qc, x1_q, y1_q, x2_q, y2_q, v_q)
    qc.barrier()

    diffuser_gate = build_diffuser(num_position_qubits)
    position_qubits = list(range(num_position_qubits))

    print(f"Applying {num_iterations} Grover iterations...")
    for i in range(num_iterations):
        # --- BLANK 9 ---
        # It's useful to know which iteration step we are on. Can we add a print statement here?
        # print(f"Iteration {i+1}: ...") # What should be printed?

        build_oracle(qc, x1_q, y1_q, x2_q, y2_q, c_q, v_q)
        qc.barrier()
        print(f"Iteration {i+1}: Applying Diffuser...")
        qc.append(diffuser_gate, position_qubits)
        qc.barrier()
    print("Grover iterations complete.")

    # --- Measure the position qubits ---
    print("Measuring position qubits...")
    # --- BLANK 10 ---
    # We need to measure the final state of the tile positions.
    # Which quantum register(s) hold the x1, y1, x2, and y2 coordinates?
    # The current line only measures x1.
    # Also ensure the classical bits match the number of qubits measured.
    qc.measure(
        x1_q, cbits[0:3]
    )  # Hint: Are we measuring ALL the position coordinates (x1,y1,x2,y2)?

    # --- Simulation ---
    print(f"\nRunning simulation with {num_shots} shots...")
    local_sampler = AerSampler()
    job = local_sampler.run(circuits=qc, shots=num_shots)
    result = job.result()
    print("Simulation complete.")

    # --- Process Results ---
    print("\nProcessing results...")
    if not result.quasi_dists:
        print("Error: No quasi-distribution found in results.")
    else:
        quasi_dists_int = result.quasi_dists[0]
        results_binary_prob = convert_int_key_to_binary(
            quasi_dists_int, num_position_qubits
        )
        results_binary_freq = quasi_prob_to_frequency(results_binary_prob, num_shots)
        results_for_excel = dict_to_array_for_excel(results_binary_freq)
        results_for_excel.sort(key=lambda x: x[2], reverse=True)

        print("\nTop 10 Results (Tile1 Pos, Tile2 Pos, Frequency):")
        for row in results_for_excel[:10]:
            print(row)

        # Save to Excel
        # Make sure the variable 'output_filename' (Blank 7) is correct before calling this.
        results_to_excel(results_for_excel, output_folder, output_filename)

        print("\nProcessing finished.")
