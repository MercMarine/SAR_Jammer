"""

Author - MercMarine
GitHub - https://github.com/MercMarine

radar_utils.py - Утилиты для расчёта параметров РЛС и помех

"""

import numpy as np


def calculate_jamming_amplitude(P_j_watts, G_j_dB, G_r_dB, freq_mhz, R_j_km, L_dB=0, ref_amplitude=1.0,
                                ref_range_km=100.0):
    G_j = 10 ** (G_j_dB / 10)
    G_r = 10 ** (G_r_dB / 10)
    L = 10 ** (L_dB / 10)
    lambda_m = 300.0 / freq_mhz
    R_j_m = R_j_km * 1000.0

    # Мощность помехи на входе приёмника

    P_rj = (P_j_watts * G_j * G_r * lambda_m ** 2) / ((4 * np.pi) ** 3 * R_j_m ** 2 * L)

    # Нормализация относительно опорной дальности для удобства GUI

    P_ref = (P_j_watts * G_j * G_r * lambda_m ** 2) / ((4 * np.pi) ** 3 * (ref_range_km * 1000) ** 2 * L)
    scale_factor = np.sqrt(P_rj / P_ref) * ref_amplitude

    # Защита от деления на ноль и отрицательных значений

    return max(scale_factor, 1e-6)