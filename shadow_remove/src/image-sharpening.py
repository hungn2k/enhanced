# import cv2

# in_img = cv2.imread('../sample/frame_11.png')

# g_blur = cv2.GaussianBlur(in_img, (9,9), 0)

# intensity1 = cv2.addWeighted(in_img,1.5, g_blur, -0.5, -10)
# intensity2 = cv2.addWeighted(in_img,2.5, g_blur, -1.5, -20)
# intensity3 = cv2.addWeighted(in_img,3.5, g_blur, -2.5, -30)

# img_re3 = cv2.resize(intensity3, (0,0), fx=0.9, fy=0.9)
# img_re2 = cv2.resize(intensity2, (0,0), fx=0.9, fy=0.9)
# img_re1 = cv2.resize(intensity1, (0,0), fx=0.9, fy=0.9)
# img_re0 = cv2.resize(in_img, (0,0), fx=0.9, fy=0.9)

# cv2.imshow('Intensity Level 2', img_re3)
# cv2.imshow('Intensity Level 3', img_re2)
# cv2.imshow('Intensity Level 1', img_re1)
# cv2.imshow('original', img_re0)

# cv2.imwrite('intensity1.jpg', intensity1)
# cv2.imwrite('intensity2.jpg', intensity2)
# cv2.imwrite('intensity3.jpg', intensity3)

# cv2.waitKey(0)

import os
import cv2

input_folder = '../sample/19-iamges_ed/'
output_folder = '../res/outout_19/'

file_list = os.listdir(input_folder)

for file_name in file_list:
    file_path = os.path.join(input_folder, file_name)
    

    in_img = cv2.imread(file_path)
    
    # Làm sắc nét
    g_blur = cv2.GaussianBlur(in_img, (9,9), 0)
    # intensity1 = cv2.addWeighted(in_img, 1.5, g_blur, -0.5, -10)
    # intensity2 = cv2.addWeighted(in_img, 2.5, g_blur, -1.5, -20)
    intensity3 = cv2.addWeighted(in_img, 3.5, g_blur, -2.5, -30)


    img_re3 = cv2.resize(intensity3, (0,0), fx=0.9, fy=0.9)
    # img_re2 = cv2.resize(intensity2, (0,0), fx=0.9, fy=0.9)
    # img_re1 = cv2.resize(intensity1, (0,0), fx=0.9, fy=0.9)
    # img_re0 = cv2.resize(in_img, (0,0), fx=0.9, fy=0.9)


    # cv2.imshow('Intensity Level 2', img_re3)
    # cv2.imshow('Intensity Level 3', img_re2)
    # cv2.imshow('Intensity Level 1', img_re1)
    # cv2.imshow('original', img_re0)

    # output_file_path1 = os.path.join(output_folder, 'intensity1_' + file_name)
    # output_file_path2 = os.path.join(output_folder, 'intensity2_' + file_name)
    output_file_path3 = os.path.join(output_folder, 'intensity3_' + file_name)


    # cv2.imwrite(output_file_path1, intensity1)
    # cv2.imwrite(output_file_path2, intensity2)
    cv2.imwrite(output_file_path3, intensity3)


    # cv2.waitKey(0)


# cv2.destroyAllWindows()
