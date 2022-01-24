from math import exp, sqrt, log
from heapq import nlargest
from itertools import product

from scipy.fft import dctn, idctn
from numpy import array, ndarray, dot, append, uint8
from numpy.random import randn

from typing import Callable


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


def similarity(a: ndarray, b: ndarray):
    print(dot(b, a), dot(b, b))
    return dot(b, a) / sqrt(dot(b, b))


def generate_watermark(n: int):
    return randn(n)


def insert_watermark(
    image: ndarray,
    watermark: ndarray,
    alpha: float,
    insertion_function: Callable[[float, float, float], float] = formula1,
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


def extract_watermark(
    original_image: ndarray,
    distorted_image: ndarray,
    watermark_length: int,
    alpha: float,
    insertion_function_i: Callable[[float, float, float], float] = formula1_i
) -> ndarray:

    # compute transform domain of both images
    original_transform_domain = dctn(original_image)
    distorted_transform_domain = dctn(distorted_image)

    # locate most significant coefficients in original image
    coefficients = nlargest(
        watermark_length,
        list(product(
            range(original_image.shape[0]),
            range(original_image.shape[1]),
            range(original_image.shape[2])
        ))[1:],
        original_transform_domain.__getitem__)
    
    # retreive watermark from most significant coefficients of distorted image
    watermark = array([], dtype=float)
    for i in coefficients:
        watermark = append(watermark, insertion_function_i(distorted_transform_domain[i], original_transform_domain[i], alpha))

    # return extracted watermark
    return watermark


if __name__ == "__main__":

    from sys import argv
    from cv2 import imread, imwrite

    n = 4096
    alpha = 0.1

    # generate watermark using standard gaussian distribution
    watermark = generate_watermark(n)

    # read specified input image
    img = imread(argv[1])

    # watermark image
    watermarked = insert_watermark(img, watermark, alpha)

    # open comparison image
    comparison_img = imread(argv[2])

    # extract watermark from comparison image
    comparison_watermark = extract_watermark(img, comparison_img, n, alpha)

    # calculate and output similarity between extracted watermark and original watermark
    print(similarity(watermark, comparison_watermark))

    # # write watermarked image to specified filename
    # imwrite(argv[2], watermarked)

    # # open quantized image
    # quantized = imread(argv[2])

    # # extract watermark from quantized image
    # extracted_watermark = extract_watermark(img, quantized, n, alpha, formula1_i)

    # # calculate and output similarity of extracted watermark from inserted watermark
    # print(similarity(watermark, extracted_watermark))
