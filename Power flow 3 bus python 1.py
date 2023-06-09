import numpy as np

# Define the admittance matrix (Ybus)
Ybus = np.array([[2.41 - 1.79j, -0.5 + 0.3j, -0.5 + 0.15j],
                 [-0.5 + 0.3j, 1.47 - 1.05j, -0.97 + 0.75j],
                 [-0.5 + 0.15j, -0.97 + 0.75j, 1.47 - 0.9j]], dtype=np.complex128)

# Define the power injection vector (PQbus)
PQbus = np.array([[-1.0],
                  [0.5],
                  [0.5]], dtype=np.complex128)

# Initialize the initial guess for voltage magnitudes and angles
V_values = np.ones((3, 1), dtype=np.complex128)
theta_values = np.zeros((3, 1), dtype=np.complex128)

# Define the maximum number of iterations and convergence tolerance
max_iterations = 10
tolerance = 1e-6

# Newton-Raphson Load Flow iterations
for iteration in range(max_iterations):
    # Compute the power flow equations
    P = np.real(np.multiply(np.conj(V_values), np.sum(Ybus * np.exp(1j * theta_values), axis=1, keepdims=True)))
    Q = np.imag(np.multiply(np.conj(V_values), np.sum(Ybus * np.exp(1j * theta_values), axis=1, keepdims=True)))

    # Construct the Jacobian matrix
    
    # Construct the Jacobian matrix
    J11 = np.real(np.sum(Ybus * np.exp(1j * theta_values), axis=1, keepdims=True))
    J12 = -np.imag(np.sum(Ybus * np.conj(1j * theta_values), axis=1, keepdims=True))
    J21 = np.imag(np.sum(Ybus * np.conj(1j * theta_values), axis=1, keepdims=True))
    J22 = np.real(np.sum(Ybus * np.exp(1j * theta_values), axis=1, keepdims=True))
    J = np.block([[J11, J12], [J21, J22]])

    # Construct the mismatch vector
    mismatch = np.concatenate((P - PQbus.real, Q - PQbus.imag))

    # Solve the linear system of equations using least squares
    delta, _, _, _ = np.linalg.lstsq(J, -mismatch, rcond=None)


    # Update the voltage magnitudes and angles
    V_values[:3] = np.add(V_values[:3], delta[:1]) 
    theta_values[:3] += delta[:1].reshape((1,1))

    


   

    # Check convergence
    if np.max(np.abs(delta)) < tolerance:
        print("Convergence achieved.")
        break

# Print the final results
print("Final Voltage Magnitudes:", np.abs(V_values))
print("Final Voltage Angles:", np.angle(V_values))
