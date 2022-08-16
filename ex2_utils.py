import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    ans = []
    i = 1
    j = 1
    minArr = in_signal if in_signal.size < k_size.size else k_size
    maxArr = k_size if in_signal.size < k_size.size else in_signal

    for k in range(1, in_signal.size + k_size.size):
        # change the boundaries for both of the lists by each case
        if k > maxArr.size:
            a = minArr[j:k]
            b = maxArr[i:k]
            i += 1
            j += 1
        elif k > minArr.size:
            a = minArr
            b = maxArr[i:k]
            i += 1
        else:
            a = minArr[:k]
            b = maxArr[:k]
        b = b[::-1]
        ans.append(np.dot(a, b))

    return np.array(ans)


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    ans = np.zeros_like(in_image, dtype=float)
    kernel = np.flip(kernel)
    # check if the kernel is 1D or 2D
    if kernel.ndim == 2:
        center = (int(kernel.shape[0] / 2), int(kernel.shape[1] / 2))
    else:
        center = (int(kernel.shape[0] / 2), 0)
    image = cv2.copyMakeBorder(in_image, top=center[0], bottom=center[0], left=center[1], right=center[1],
                               borderType=cv2.BORDER_REPLICATE)
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            # multiply the current part of the image with the kernel
            ans[i, j] = (image[i: i + kernel.shape[0], j: j + kernel.shape[1]] * kernel).sum()
    # if the image is scaling 0-255 so I round the pixels
    if np.amax(ans) > 1:
        ans = np.round(ans)
    return ans


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[1, 0, -1]])
    x = cv2.filter2D(in_image, -1, kernel)
    y = cv2.filter2D(in_image, -1, kernel.T)
    dire = np.arctan2(y, x).astype(np.float64)
    mag = np.sqrt(x ** 2 + y ** 2).astype(np.float64)

    return dire, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    # create a Gaussian kernel
    ax = np.linspace(-(k_size - 1) / 2, (k_size - 1) / 2, k_size)
    gaussian = np.exp(-0.5 * np.square(ax))
    gaussianKernel = np.outer(gaussian, gaussian)
    gaussianKernel = gaussianKernel / np.sum(gaussianKernel)
    # convolve the image with the kernel
    return conv2D(in_image, gaussianKernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, 1)
    kernel = np.dot(kernel, kernel.T)
    in_image = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    return in_image


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: Edge matrix
    """
    # create the kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    imLaplacian = cv2.filter2D(img, -1, kernel)
    imgEdge = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try:
                # take all the neighbours of the pixel
                neighboursA = [imLaplacian[i, j - 1], imLaplacian[i - 1, j - 1], imLaplacian[i - 1, j],
                               imLaplacian[i - 1, j + 1]]
                neighboursB = [imLaplacian[i, j + 1], imLaplacian[i + 1, j + 1], imLaplacian[i + 1, j],
                               imLaplacian[i + 1, j - 1]]
                # check if there is an edge by checking different cases
                if imLaplacian[i, j] == 0:
                    for a, b in zip(neighboursA, neighboursB):
                        if a > 0 and b < 0 or a < 0 and b > 0:
                            imgEdge[i, j] = 1
                if imLaplacian[i, j] < 0:
                    if max(neighboursA) > 0 or max(neighboursB) > 0:
                        imgEdge[i, j] = 1
                if imLaplacian[i, j] > 0:
                    if max(neighboursA) < 0 or max(neighboursB) < 0:
                        imgEdge[i, j] = 1
            # if I get out of boundaries I catch the error
            except IndexError as e:
                pass

    return imgEdge


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    return img


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    # the thresholds that give me the best results is 88.
    print("the thresholds that give me the best results is 88.")
    rowsSize = img.shape[0]
    columnsSize = img.shape[1]
    imageEdge = cv2.Canny((img * 255).astype(np.uint8), img.shape[0], img.shape[1])
    circlesFound = []
    for Radius in range(min_radius, max_radius + 1):
        circleProb = np.zeros(imageEdge.shape)
        for i in range(0, rowsSize, 2):
            for j in range(0, columnsSize, 2):
                if imageEdge[i, j] == 255:
                    for angel in range(180):
                        a = int(j - math.cos(angel * math.pi / 90) * Radius)
                        b = int(i - math.sin(angel * math.pi / 90) * Radius)
                        if 0 <= b < rowsSize and 0 <= a < columnsSize:
                            circleProb[b, a] += 4
        if np.max(circleProb) > 88:
            circleProb[circleProb < 88] = 0
            for i in range(1, rowsSize - 1, 2):
                for j in range(1, columnsSize - 1, 2):
                    if circleProb[i, j] >= 88:
                        if all((j - xc) ** 2 + (i - yc) ** 2 > rc ** 2 for xc, yc, rc in circlesFound):
                            circlesFound.append((j, i, Radius))
                            circleProb[i - Radius:i + Radius, j - Radius: j + Radius] = 0
    return circlesFound


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    ans = cv2.bilateralFilter(in_image, k_size, sigma_space,sigma_color)
    imgbi = np.zeros_like(in_image)
    k_size = int(k_size / 2)
    image = cv2.copyMakeBorder(in_image, top=k_size, bottom=k_size, left=k_size, right=k_size,
                               borderType=cv2.BORDER_REFLECT_101).astype(int)
    for y in range(k_size, image.shape[0] - k_size):
        for x in range(k_size, image.shape[1] - k_size):
            pivot_v = image[y, x]
            neighbor_hood = image[
                            y - k_size:y + k_size + 1,
                            x - k_size:x + k_size + 1
                            ]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_space))
            gaus = cv2.getGaussianKernel(2 * k_size + 1, sigma=sigma_color)
            gaus = gaus.dot(gaus.T)
            combo = gaus * diff_gau
            result = (combo * neighbor_hood / combo.sum()).sum()
            imgbi[y - k_size, x - k_size] = round(result)


    return ans.astype(int), imgbi.astype(int)


def myID():
    return 319097036
