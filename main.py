from math import exp, sqrt
from heapq import nlargest
from itertools import product

from scipy.fft import dctn, idctn
from numpy import array, ndarray, dot, uint8
from numpy.random import randn

from typing import Callable


def formula1(
    v: float,
    x: float,
    alpha: float
):
    return v + alpha * x


def formula2(
    v: float,
    x: float,
    alpha: float
):
    return v * (1 + alpha * x)


def formula3(
    v: float,
    x: float,
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
    insertion_function: Callable = formula2,
) -> ndarray:

    # ensure image dtype is set to float
    if image.dtype != float:
        image = image.astype(float)

    # compute transform domain of image
    transform_domain = dctn(image)

    # locate most significant coefficients
    coefficients = nlargest(
        len(watermark),
        list(product(
            range(image.shape[0]),
            range(image.shape[1]),
            range(image.shape[2])
        ))[1:],
        transform_domain.__getitem__)

    # insert watermark using specified insertion function
    for (j, k, l), xi in zip(coefficients, watermark):
        transform_domain[j, k, l] = insertion_function(transform_domain[j, k, l], xi, alpha)

    # return reconstructed and quantized spatial domain
    return idctn(transform_domain).astype(uint8)


if __name__ == "__main__":

    from sys import argv
    from cv2 import imread, imwrite

    watermark = generate_watermark(4096)

    img = imread(argv[1])

    watermarked = insert_watermark(img, watermark, 0.02)

    imwrite(argv[2], watermarked)
