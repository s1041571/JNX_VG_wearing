import cv2


### Jack Add
def plot_fense_dot_line(pts, img, color, line_thickness):
    for j in range(len(pts) - 1):
        cv2.circle(img , pts[j], 5, color=color)
        cv2.line(img , pt1=pts[j], pt2=pts[j + 1], color=color, thickness=line_thickness)

    cv2.circle(img , pts[-1], 5, color=color)
    cv2.line(img , pt1=pts[0], pt2=pts[-1], color=color, thickness=line_thickness)


def plot_all_fense(fences, img, color, line_thickness):
    for fence in fences:
        plot_fense_dot_line(fence, img, color, line_thickness)


# https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


