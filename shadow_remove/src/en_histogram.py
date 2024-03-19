import os
import cv2


input_folder = '../sample/19-iamges_ed/'
output_folder = '../res/his_delight_out/'


file_list = os.listdir(input_folder)

for file_name in file_list:
    file_path = os.path.join(input_folder, file_name)
    
    in_img = cv2.imread(file_path)
    
    hist_equalized = cv2.equalizeHist(cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY))
    hist_equalized = cv2.cvtColor(hist_equalized, cv2.COLOR_GRAY2BGR)

    output_file_path = os.path.join(output_folder, 'hist_equalized_' + file_name)
    cv2.imwrite(output_file_path, hist_equalized)


