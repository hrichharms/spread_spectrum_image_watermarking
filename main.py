from math import exp, sqrt

from typing import Callable

from numpy import ndarray, float64, dot
from numpy.random import randn


def formula1(
    v: float64,
    x: float64,
    alpha: float
):
    return v + alpha * x


def formula2(
    v: float64,
    x: float64,
    alpha: float
):
    return v * (1 + alpha * x)


def formula3(
    v: float64,
    x: float64,
    alpha: float
):
    return v * exp(alpha * x)


def similarity(a: ndarray, b: ndarray):
    return dot(a, b) / sqrt(dot(a, b))


def generate_watermark(n: int):
    return randn(n)


def insert_watermark(
    image: ndarray,
    watermark: ndarray,
    alpha: float,
    insertion_formula: Callable = formula2
):
    pass


if __name__ == "__main__":

    from PIL import Image
