import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_homography_matrix(source, destination):
    """ Calculates the entries of the Homography matrix between two sets of matching points.

    Args
    ----
        - `source`: Source points where each point is int (x, y) format.
        - `destination`: Destination points where each point is int (x, y) format.

    Returns
    ----
        - A numpy array of shape (3, 3) representing the Homography matrix.

    Raises
    ----
        - `source` and `destination` is lew than four points.
        - `source` and `destination` is of different size.
    """
    assert len(source) >= 4, "must provide more than 4 source points"
    assert len(destination) >= 4, "must provide more than 4 destination points"
    assert len(source) == len(destination), "source and destination must be of equal length"
    A = []
    b = []
    for i in range(len(source)):
        s_x, s_y = source[i]
        d_x, d_y = destination[i]
        A.append([s_x, s_y, 1, 0, 0, 0, (-d_x)*(s_x), (-d_x)*(s_y)])
        A.append([0, 0, 0, s_x, s_y, 1, (-d_y)*(s_x), (-d_y)*(s_y)])
        b += [d_x, d_y]
    A = np.array(A)
    h = np.linalg.lstsq(A, b)[0]
    h = np.concatenate((h, [1]), axis=-1)
    return np.reshape(h, (3, 3))


if __name__ == "__main__":
    source_points = np.array([
        [200, 541],
        [1216, 413],
        [1373, 993],
        [267, 1208]
    ])
    destination_points = np.array([
        [0, 0],
        [300, 0],
        [300, 300],
        [0, 300],
    ])
    source_image = cv2.imread("sample.jpeg")
    t_source_image = source_image.copy()

    # draw markings on the source image
    for i, pts in enumerate(source_points):
        cv2.putText(source_image, str(i+1), (pts[0] + 15, pts[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 215, 255), 5)
        cv2.circle(source_image, pts, 10, (0, 215, 255), 10)

    h = get_homography_matrix(source_points, destination_points)
    destination_image = cv2.warpPerspective(t_source_image, h, (300, 300))

    figure = plt.figure(figsize=(12, 6))

    subplot1 = figure.add_subplot(1, 2, 1)
    subplot1.title.set_text("Source Image")
    subplot1.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))

    subplot2 = figure.add_subplot(1, 2, 2)
    subplot2.title.set_text("Destination Image")
    subplot2.imshow(cv2.cvtColor(destination_image, cv2.COLOR_BGR2RGB))

    # plt.show()
    plt.savefig("output.png")
