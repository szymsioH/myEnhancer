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

model_file_name = '.'
model_type = "."
model_increase = 0
input_choice = "."
is_running = False
if_again = "Y"
exit_flag = False


def input_enhance():
    global input_choice, model_increase, model_file_name, if_again, exit_flag
    input_choice = input("Wybrane powiększenie: \n")
    if model_type == "lapsrn":
        if input_choice == "2" or input_choice == "x2":
            model_increase = 2
            model_file_name = 'LapSRN_x2.pb'
        elif input_choice == "4" or input_choice == "x4":
            model_increase = 4
            model_file_name = 'LapSRN_x4.pb'
        elif input_choice == "8" or input_choice == "x8":
            model_increase = 8
            model_file_name = 'LapSRN_x8.pb'
        elif input_choice == "exit":
            exit_flag = True
        else:
            print("Podano nieprawidłową wartość!")
            text_menu()
            if_again = "N"
    else:
        if input_choice == "2" or input_choice == "x2":
            model_increase = 2
            model_file_name = str(model_type.upper() + '_x2.pb')
        elif input_choice == "3" or input_choice == "x3":
            model_increase = 3
            model_file_name = str(model_type.upper() + '_x3.pb')
        elif input_choice == "4" or input_choice == "x4":
            model_increase = 4
            model_file_name = str(model_type.upper() + '_x4.pb')
        elif input_choice == "exit":
            exit_flag = True
        else:
            print("Podano nieprawidłową wartość!")
            text_menu()
            if_again = "N"


def text_menu():
    global model_type, input_choice, if_again, exit_flag
    print("Wybierz model: \n 1. LapSRN - wybierz 1 \n 2. FSRCNN - wybierz 2 \n 3. ESPCN - wybierz 3 \n 4. EDSR - wybierz 4 \n exit - type exit \n")
    input_choice = input("Wybrany model: ")
    print("Wybierz przybliżenie:")
    if input_choice == "1" or input_choice == "lapsrn":
        print(" x2 - wybierz 2 \n x4 - wybierz 4 \n x8 - wybierz 8 \n exit - type exit")
        model_type = "lapsrn"
        input_enhance()
    elif input_choice == "2":
        print(" x2 - wybierz 2 \n x3 - wybierz 3 \n x4 - wybierz 4 \n exit - type exit")
        model_type = "fsrcnn"
        input_enhance()
    elif input_choice == "3":
        print(" x2 - wybierz 2 \n x3 - wybierz 3 \n x4 - wybierz 4 \n exit - type exit")
        model_type = "espcn"
        input_enhance()
    elif input_choice == "4":
        print(" x2 - wybierz 2 \n x3 - wybierz 3 \n x4 - wybierz 4 \n exit - type exit")
        model_type = "edsr"
        input_enhance()
    elif input_choice == "exit":
        exit_flag = True
    else:
        print("Podano nieprawidłową wartość! \n")
        text_menu()
        if_again = "N"


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

#def start_generating():
#    global is_running, model_increase, input_choice, model_type, model_file_name, if_again


if __name__ == '__main__':
    while if_again == "Y":
        text_menu()
        t1 = threading.Thread(target=model_function)
        t2 = threading.Thread(target=animate)
        t2.start()
        t1.start()
        t2.join()
        t1.join()
        if exit_flag:
            break
        if_again = input("Again? (Y/N)\n ")


