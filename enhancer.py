import os
import cv2
import time
import sys

enhancer_folder = os.path.dirname(os.path.abspath(__file__))

# WCZYTYWANIE --------------------------------------------------------------------
image_path = os.path.join(enhancer_folder, 'test_images', 'bikes_col.png')

save_name = 'bikes'
# --------------------------------------------------------------------------------

resoult_path = os.path.join(enhancer_folder, 'resoult_images')
img = cv2.imread(image_path)
sr = cv2.dnn_superres.DnnSuperResImpl_create()

model_file_name = 'LapSRN_x4.pb'
model_type = "lapsrn"
model_increase = 4

is_running = True

def model_function(is_running):
    start_time = time.time()
    path = os.path.join(enhancer_folder, 'models', model_file_name)
    sr.readModel(path)
    sr.setModel(model_type, model_increase)
    upscaled = sr.upsample(img)

    '''
    while is_running == True:
        sys.stdout.write('\rloading |')
        time.sleep(0.5)
        sys.stdout.write('\rloading /')
        time.sleep(0.5)
        sys.stdout.write('\rloading -')
        time.sleep(0.5)
        sys.stdout.write('\rloading \\')
        time.sleep(0.5)
    sys.stdout.write('\rDone!     ')
    '''

    cv2.imwrite(resoult_path + "/" + model_type + str(model_increase) + "_" + save_name + ".png", upscaled)
    is_running = False
    end_time = time.time()
    print("\nTime elapsed: " + "%.2f" % (end_time - start_time) + " sekund")

model_function(is_running)