import time

import cloudpickle
import dill as pickle
from sympy import Symbol, Matrix, symbols, pi, cos, sin, simplify, integrate, Ne, Eq, diag, Array, zeros, lambdify

pickle.settings['recurse'] = True


def derive():
    """Derive the forward dynamics of the soft robot"""
    num_dof = 3  # bend, shear, axial for 1 segment

    print("Initializing...")
    t = time.time()

    # Define symbolic physical parameters
    th0 = Symbol("θ₀", real=True)  # initial angle of the robot
    rho = Symbol("ϱ", real=True, nonnegative=True)  # volumetric mass density [kg/m^3]
    l = Symbol("l", real=True, nonnegative=True)  # length of segment [m]
    r = Symbol("r", real=True, nonnegative=True)  # radius of segment [m]
    gy = Symbol("gᵧ", real=True)  # gravity along y-axis
    bend_offset = Symbol("ε", real=True, nonzero=True)  # bending offset
    q_offset = Matrix([bend_offset, 0, 0])  # offset of the configuration

    q1, q2, q3 = symbols("q₁, q₂, q₃", real=True, nonzero=True)
    q1dot, q2dot, q3dot = symbols("qd₁, qd₂, qd₃", real=True)

    q_orig = Matrix([q1, q2, q3])
    q = q_orig + q_offset
    qdot = Matrix([q1dot, q2dot, q3dot])

    # construct the symbolic matrices
    g = Matrix([0, gy])  # gravity vector

    a1, a2, a3 = symbols("a₁, a₂, a₃", real=True)
    tau = Matrix([a1, a2, a3])  # actuation

    # symbol for the point coordinate
    s = symbols("s", real=True, nonnegative=True)

    # elastic and shear modulus
    # mu as G is already taken
    E, mu = symbols("E, µ", real=True, nonnegative=True)

    # dissipative matrix from the parameters
    d11, d22, d33 = symbols("d₁₁, d₂₂, d₃₃", nonnegative=True)
    D = diag(*[d11, d22, d33])

    # initialize
    # kappa, sigma_x, sigma_y = symbols("κ, σₓ, σᵧ", real=True, nonzero=True)
    kappa, sigma_x, sigma_y = q
    A = pi * r ** 2
    I = A ** 2 / (4 * pi)
    th = th0 + s * kappa
    R = Matrix([[cos(th), -sin(th)], [sin(th), cos(th)]])
    dp_ds = R @ Matrix([sigma_x, sigma_y])
    p = simplify(integrate(dp_ds, (s, 0.0, s))).subs(Ne(q1, -bend_offset), True).subs(Eq(q1, -bend_offset), False)

    Jp = simplify(p.jacobian(q_orig))
    Jo = simplify(Matrix([[th]]).jacobian(q_orig))

    dB_ds = simplify(rho * A * Jp.T @ Jp + rho * I * Jo.T @ Jo)
    B = simplify(integrate(dB_ds, (s, 0, l))).subs(Ne(q1, -bend_offset), True).subs(Eq(q1, -bend_offset), False)

    print("Computed B")

    # compute the Christoffel symbols
    Ch_flat = []
    for i in range(num_dof):
        for j in range(num_dof):
            for k in range(num_dof):
                # Ch[i, j, k] = simplify(0.5 * (B[i, j].diff(q[k]) + B[i, k].diff(q[j]) - B[j, k].diff(q[i])))
                Ch_ijk = 0.5 * (
                        B[i, j].diff(q_orig[k]) + B[i, k].diff(q_orig[j]) - B[j, k].diff(q_orig[i])
                )
                Ch_ijk = simplify(Ch_ijk)
                Ch_flat.append(Ch_ijk)
    Ch = Array(Ch_flat, (num_dof, num_dof, num_dof))

    # compute the coriolis and centrifugal force matrix
    C = zeros(num_dof, num_dof)
    for i in range(num_dof):
        for j in range(num_dof):
            for k in range(num_dof):
                C[i, j] = C[i, j] + Ch[i, j, k] * qdot[k]
    C = simplify(C)

    print("Computed C")

    dU_ds = simplify(rho * A * g.T @ p)
    U = simplify(integrate(dU_ds, (s, 0, l))).subs(Ne(q1, -bend_offset), True).subs(Eq(q1, -bend_offset), False)

    # compute the gravity force vector
    G = simplify(-U.jacobian(q_orig).transpose())

    print("Computed G")

    # Use directly all three strains
    # In this way, the strain basis is the Identity matrix
    q_rest = Matrix([0., 0., 1.])  # rest strain

    # map the configuration to the strains
    q = q_rest + q

    # stiffness matrix of shape (3, 3)
    S = diag(*[I * E, (4 / 3) * A * mu, A * E])

    # we define the elastic matrix of shape (n_xi, n_xi) as K(xi) = K @ xi where K is equal to
    K = S
    K = K @ (q - q_rest - q_offset)

    a11 = simplify(B[2, 2] * B[1, 1] - B[1, 2] ** 2)
    a12 = simplify(B[0, 2] * B[1, 2] - B[2, 2] * B[0, 1])
    a13 = simplify(B[0, 1] * B[1, 2] - B[0, 2] * B[1, 1])

    a22 = simplify(B[2, 2] * B[0, 0] - B[0, 2] ** 2)
    a23 = simplify(B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])

    a33 = simplify(B[0, 0] * B[1, 1] - B[0, 1] ** 2)

    det = simplify((B[0, 0] * a11) + (B[0, 1] * a12) + (B[0, 2] * a13))

    Minv = Matrix([[a11, a12, a13],
                      [a12, a22, a23],
                      [a13, a23, a33]]) / det

    Minv = simplify(Minv)

    print("Inverted M")

    coriolis_term = C * qdot
    coriolis_term = simplify(coriolis_term)

    right_term = (tau - coriolis_term - G - K - D * qdot)
    right_term = simplify(right_term)

    qddot = Minv * right_term
    F = Matrix([qdot, qddot])

    print("Computed F")

    # Hamiltonian
    T = 0.5 * qdot.T @ B @ qdot
    H = T + U

    print("Computed H")

    # Jacobian of forward dynamics
    state = Matrix([q1, q2, q3, q1dot, q2dot, q3dot])
    F_s = F.jacobian(state)
    F_a = F.jacobian(tau)

    print("Computed D")

    F_lambda = lambdify(
        [tuple([th0, rho, l, r, gy, E, mu, d11, d22, d33, bend_offset, q1, q2, q3, q1dot, q2dot, q3dot, a1, a2, a3])],
        F,
        'numpy',
        cse=True
    )

    H_lambda = lambdify(
        [tuple([th0, rho, l, r, gy, E, mu, d11, d22, d33, bend_offset, q1, q2, q3, q1dot, q2dot, q3dot])],
        H,
        'numpy',
        cse=True
    )

    D_lambda = lambdify(
        [tuple([th0, rho, l, r, gy, E, mu, d11, d22, d33, bend_offset, q1, q2, q3, q1dot, q2dot, q3dot, a1, a2, a3])],
        (F_s, F_a),
        'numpy',
        cse=True
    )



    with open("./env/soft_reacher/dynamics.p", "wb") as outf:
        cloudpickle.dump({'H': H_lambda, 'F': F_lambda, 'D': D_lambda}, outf)

    print("Done")


def check():
    inertial_params = [0, 1070, 1e-1, 2e-2, -9.81, 1e3, 5e2, 1e-5, 1e-2, 1e-2]
    eps = 0.
    q = [0.1, 0, 0]
    qdot = [0, 0, 0]
    s = q + qdot
    a = [0, 0, 0]

    with open("./env/soft_reacher/dynamics.p", "rb") as inf:
        funcs = pickle.load(inf)

    H, F, D = funcs['H'], funcs['F'], funcs['D']
    print(H([*inertial_params, eps, *s]))
    print(F([*inertial_params, eps, *s, *a]))
    print(D([*inertial_params, eps, *s, *a]))


if __name__ == "__main__":
    derive()
    check()
