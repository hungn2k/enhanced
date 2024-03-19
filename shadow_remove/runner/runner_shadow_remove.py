import cv2 as cv
from src.shadow_remove import ShadowRemove
from settings import SHADOW_REMOVE


def enhanced(image, use_filter=True, filter_name=SHADOW_REMOVE["DEFAULT_FILTER_NAME"]):
    """
    :param image: low light input image.
    :param use_filter: use filter (default: True). 
    :param filter_name: filter name in case :use_filter=True. 
        There are three values: cr, fbs, guided_filter (default: fbs)
    :return -> enhanced image (Array).
    """
    try:
        shadow_remove = ShadowRemove()
        shadow_remove.origin_image = image
        shadow_remove.start(use_cont_regula=use_filter, filter_type=filter_name)
        shadow_remove.run()
        return shadow_remove.shadow_remove_image
    except Exception as e:
        print(e)


def main():
    try:
        image_path = "../images/1.jpg"
        origin_image = cv.imread(image_path)
        enhanced_image = enhanced(origin_image, filter_name="fbs")
        cv.imshow("enhanced image", enhanced_image)
        cv.waitKey(0)
    except Exception as e:
        print(e)


if __name__ in "__main__":
    main()
