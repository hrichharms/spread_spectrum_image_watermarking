from math import sqrt
from heapq import nlargest
from itertools import product

from scipy.fft import dctn, idctn
from numpy import array, ndarray, dot, append, uint8
from numpy.random import randn

from typing import Callable


from insertion_functions import formula1, formula1_i


def similarity(a: ndarray, b: ndarray) -> float:
    """
    Estimation of similarity between two equal length arrays.
    """
    return dot(b, a) / sqrt(dot(b, b))


def generate_watermark(n: int) -> ndarray:
    """
    Generates standard gaussian distributed ndarray of length n.
    """
    return randn(n)


def insert_watermark(
    image: ndarray,
    watermark: ndarray,
    alpha: float,
    insertion_function: Callable[[float, float, float], float] = formula1,
) -> ndarray:
    """
    Inserts watermark into highest magnitude DCT coefficients of image.
    """

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
    """
    Extracts watermark of specified length from DCT coefficients of distorted image.
    """

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
        watermark = append(
            watermark,
            insertion_function_i(
                distorted_transform_domain[i],
                original_transform_domain[i],
                alpha
            )
        )

    # return extracted watermark
    return watermark


def main(
    inserting: bool,
    filename_1: str,
    filename_2: str,
    watermark_filename: str,
    n: int,
    alpha: float
) -> None:

    from cv2 import imread, imwrite
    from json import dump, load

    if inserting:

        # generate watermark
        watermark = generate_watermark(n)

        # write watermark to watermark filename
        with open(watermark_filename, "w") as watermark_file:
            dump(watermark.tolist(), watermark_file)

        # write watermarked image to output filename
        imwrite(filename_2, insert_watermark(imread(filename_1), watermark, alpha))

    else:

        # read watermark from watermark filename
        with open(watermark_filename) as watermark_file:
            watermark_1 = load(watermark_file)

        # extract watermark from second image
        watermark_2 = extract_watermark(imread(filename_1), imread(filename_2), n, alpha)

        # calculate and output similarity between stored and extracted watermarks
        print(f"Similarity: {similarity(watermark_1, watermark_2):.2f}")


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(
        description="Insert or compare spread spectrum watermarks into high magnitude DCT coefficients."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("-i", action="store_true", help="insertion mode")
    mode_group.add_argument("-c", action="store_true", help="comparison mode")
    parser.add_argument(
        "image_1",
        metavar="IMAGE_1",
        type=str,
        help="original image for insertion or comparison, depending upon selected mode"
    )
    parser.add_argument(
        "image_2",
        metavar="IMAGE_2",
        type=str,
        help="watermarked output filename or second image for comparison, depending upon selected mode"
    )
    parser.add_argument(
        "-w",
        metavar="WATERMARK_FILENAME",
        type=str,
        help="watermark filename for storage during insertion mode or retreival during comparison mode",
        default="watermark.json",
        required=False
    )
    parser.add_argument(
        "-n",
        metavar="WATERMARK_LEN",
        type=int,
        help="watermark length",
        default=4096,
        required=False
    )
    parser.add_argument(
        "-a",
        metavar="ALPHA",
        type=float,
        help="watermark scaling factor",
        default=0.1,
        required=False
    )
    args = parser.parse_args()

    main(args.i == True, args.image_1, args.image_2, args.w, args.n, args.a)
