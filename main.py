import numpy as np
import timeit
import random


def compute_fourier_term(k, f, N):
    A_k = 0
    B_k = 0
    for i in range(N):
        A_k += f[i] * np.cos(-2 * np.pi * k * i / N)
        B_k += f[i] * np.sin(-2 * np.pi * k * i / N)
    return A_k, B_k


def compute_fourier_coefficient(k, f, N):
    A_k, B_k = compute_fourier_term(k, f, N)
    return A_k + 1j * B_k


def compute_dft(f, N):
    coefficients = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        C_k = compute_fourier_coefficient(k, f, N)
        coefficients[k] = C_k
    return coefficients


n = 102
N = 8
N1 = 64
f = [0, 1, 1, 0, 0, 1, 1, 0]
f1 = [random.choice([0, 1]) for _ in range(N1)]

# кількість операцій додавання та множення
dft_add = N * (N - 1)
dft_multi = N ** 2
fft_add = int(N * np.log2(N))
fft_multi = int(N / 2 * np.log2(N))
dft_add1 = N1 * (N1 - 1)
dft_multi1 = N1 ** 2
fft_add1 = int(N1 * np.log2(N1))
fft_multi1 = int(N1 / 2 * np.log2(N1))

print("\nЧас виконання DFT: ", timeit.timeit(lambda: compute_dft(f, N), number=1), " секунд")
print("Час виконання FFT: ", timeit.timeit(lambda: np.fft.fft(f), number=1), " секунд")

print(
    f"\nКількість операцій DFT: {dft_add + dft_multi}; Кількість операцій додавання {dft_add}; Кількість операцій множення {dft_multi}")
print(
    f"\nКількість операцій FFT: {fft_add + fft_multi}; Кількість операцій додавання {fft_add}; Кількість операцій множення {fft_multi}")

print("\nЧас виконання DFT (тестові дані): ", timeit.timeit(lambda: compute_dft(f1, N1), number=1), " секунд")
print("Час виконання FFT (тестові дані): ", timeit.timeit(lambda: np.fft.fft(f1), number=1), " секунд")

print(
    f"\nКількість операцій DFT(тестові дані): {dft_add1 + dft_multi1}; Кількість операцій додавання {dft_add1}; Кількість операцій множення {dft_multi1}")
print(
    f"\nКількість операцій FFT(тестові дані): {fft_add1 + fft_multi1}; Кількість операцій додавання {fft_add1}; Кількість операцій множення {fft_multi1}")
