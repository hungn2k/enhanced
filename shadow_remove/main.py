import cv2 as cv
from src.shadow_remove import ShadowRemove
from settings import SHADOW_REMOVE
import time
import os
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
        sample_dir = './sample/19-iamges_ed/'
        result_dir = './res/delight_images-Constraint/'
        
        total_time = 0
        processed_images = 0
        # image_path = "./sample/frame_1.jpg"
        # origin_image = cv.imread(image_path)
        # enhanced_image = enhanced(origin_image, filter_name="fbs")

        # # cv.imshow("enhanced image", enhanced_image)
        # # cv.waitKey(0)
        
        for filename in os.listdir(sample_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(sample_dir, filename)
                origin_image = cv.imread(image_path)
                
                start_time = time.time()

                enhanced_image = enhanced(origin_image, filter_name="fbs")
                # g_blur = cv.GaussianBlur(enhanced_image, (9,9), 0)
                # enhanced_image = cv.addWeighted(enhanced_image, 2.5, g_blur, -1.5, -20)
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time
                print(os.path.join(result_dir, filename))
                cv.imwrite(os.path.join(result_dir, filename), enhanced_image)
                processed_images += 1
        average_time = total_time / processed_images
        print(f"Thời gian trung bình thực hiện tăng sáng cho mỗi ảnh: {average_time} giây")
        
    except Exception as e:
        print(e)

if __name__ in "__main__":

    main()
