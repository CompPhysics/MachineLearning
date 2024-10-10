import Quanthon as qt
#Initializing a Single Qubit

#Initialize a single qubit by creating an instance of the Qubits class.

qubit = qt.Qubits(1)
# Apply a Hadamard gate on the first qubit
qubit.H(0)

# Apply a Pauli-X gate on the first qubit
qubit.X(0)

# Apply a Pauli-Y gate on the first qubit
qubit.Y(0)

# Apply a Pauli-Z gate on the first qubit
qubit.Z(0)

result = qubit.measure(n_shots=10)
