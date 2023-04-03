import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Problem 1


def function(t: float, w: float):
    return t - w*w


def eulers(original_w, start_of_t, end_of_t, num_of_iterations):
    next_w = 0

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # this gets the next approximation
        next_w = w + h * function(t, w)

        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w

    return next_w

# Problem 2


def rk_helper(w, h, t):

    k1 = h * function(t, w)
    k2 = h * function(t + (h/2), w + (k1/2))
    k3 = h * function(t + (h/2), w + k2/2)
    k4 = h * function(t + h, w + k3)

    return w + (k1 + 2*k2 + 2*k3 + k4)/6


def runge_kutta(original_w, start_of_t, end_of_t, num_of_iterations):
    next_w = 0

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # this gets the next approximation
        next_w = rk_helper(w, h, t)

        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w

    return next_w


def gauss():
    A = np.array([[2, -1, 1],
                  [1, 3, 1],
                  [-1, 5, 4]], dtype=np.double)
    b = np.array([6, 0, -3], dtype=np.double)
    n = len(b)

    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)

    # Perform elimination
    for i in range(n):

        # Divide pivot row by pivot element
        pivot = Ab[i, i]
        if (pivot != 0):
            Ab[i, :] = Ab[i, :] / pivot

        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j, i]
            Ab[j, :] -= factor * Ab[i, :]  # operation 2 of row operations

    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j, i]
            Ab[j, :] -= factor * Ab[i, :]

    # Extract solution vector x
    x = Ab[:, n]

    return x


def lu_fact():
    Ab = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2],
                 [-1, 2, 3, -1]], dtype=np.double)
    Adet = np.linalg.det(Ab)
    print("%.5f" % Adet)

    n = len(Ab)

    # Perform elimination
    for i in range(n - 1, 0, -1):

        # Divide pivot row by pivot element
        pivot = Ab[i, i]
        if (pivot != 0):
            Ab[i, :] = Ab[i, :] / pivot

        # Eliminate entries below pivot
        for j in range(n - 2, i, -1):
            factor = Ab[j, i]
            Ab[j, :] -= factor * Ab[i, :]  # operation 2 of row operations

    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j, i]
            Ab[j, :] -= factor * Ab[i, :]

    print(Ab)


def diagDom():

    A = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [
                 0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])

    length = len(A)

    # the question just say diagonally dominant, not strictly so...
    for i in range(length):
        diag = A[i][i]
        others = np.sum(A) - diag

        if (diag < others):
            return False

    return True


def posDef():
    A = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])

    # check if symmetric
    At = np.transpose(A)
    if (np.array_equal(A, At) == False):
        return False

    # check for positive eigen values
    Aeig = np.linalg.eigvals(A)

    for val in Aeig:
        if (val < 0):
            return False

    return True


if __name__ == "__main__":
    print("%.5f" % eulers(1, 0, 2, 10))
    print()
    print("%.5f" % runge_kutta(1, 0, 2, 10))
    print()
    print(gauss())
    print()
    lu_fact()
    print()
    print(diagDom())
    print()
    print(posDef())
    print()
