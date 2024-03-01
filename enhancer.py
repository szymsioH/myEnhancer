import os
import cv2
import time
import sys
import itertools
import threading

enhancer_folder = os.path.dirname(os.path.abspath(__file__))

# WCZYTYWANIE --------------------------------------------------------------------
image_path = os.path.join(enhancer_folder, 'test_images', 'bikes_col.png')

save_name = 'bikes'
# --------------------------------------------------------------------------------

resoult_path = os.path.join(enhancer_folder, 'resoult_images')
img = cv2.imread(image_path)
sr = cv2.dnn_superres.DnnSuperResImpl_create()

model_file_name = 'LapSRN_x8.pb'
model_type = "lapsrn"
model_increase = 8

is_running = False

def model_function():
    global is_running
    start_time = time.time()
    path = os.path.join(enhancer_folder, 'models', model_file_name)
    sr.readModel(path)
    sr.setModel(model_type, model_increase)
    upscaled = sr.upsample(img)

    cv2.imwrite(resoult_path + "/" + model_type + str(model_increase) + "_" + save_name + ".png", upscaled)
    is_running = True
    end_time = time.time()
    print("\nTime elapsed: " + "%.2f" % (end_time - start_time) + " sekund")

def animate():
    global is_running
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if is_running:
            break
        sys.stdout.write('\rloading ' + c)
        time.sleep(0.3)
        sys.stdout.flush()
    sys.stdout.write('\rDone!     ')


if __name__ == '__main__':
    t1 = threading.Thread(target=model_function)
    t2 = threading.Thread(target=animate)
    t2.start()
    t1.start()
    t2.join()
    t1.join()
