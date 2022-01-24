from math import exp, log


def formula1(
    v: float,
    x: float,
    alpha: float
) -> float:
    return v + alpha * x


def formula1_i(
    v: float,
    v_original: float,
    alpha: float
) -> float:
    return (v - v_original) / alpha


def formula2(
    v: float,
    x: float,
    alpha: float
) -> float:
    return v * (1 + alpha * x)


def formula2_i(
    v: float,
    v_original: float,
    alpha: float
) -> float:
    return (v / v_original - 1) / alpha


def formula3(
    v: float,
    x: float,
    alpha: float
) -> float:
    return v * exp(alpha * x)


def formula3_i(
    v: float,
    v_original: float,
    alpha: float
) -> float:
    return log(v / v_original) / alpha