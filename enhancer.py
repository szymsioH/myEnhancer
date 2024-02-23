import os
import cv2

enhancer_folder = os.path.dirname(os.path.abspath(__file__))

# WCZYTYWANIE --------------------------------------------------------------------
image_path = os.path.join(enhancer_folder, 'test_images', 'bikes_col.png')

save_name = 'bikes'
save_folder = 'bikes'
# --------------------------------------------------------------------------------

resoult_path = os.path.join(enhancer_folder, 'resoult_images', save_folder)

img = cv2.imread(image_path)

sr = cv2.dnn_superres.DnnSuperResImpl_create()

# EDSR x4:

# 2
path = os.path.join(enhancer_folder, 'models', 'EDSR_x2.pb')
sr.readModel(path)

sr.setModel("edsr", 2)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + "/edsr2_" + save_name + ".png", upscaled)

# x4
path = os.path.join(enhancer_folder, 'models', 'EDSR_x4.pb')
sr.readModel(path)

sr.setModel("edsr", 4)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + "/edsr4_" + save_name + ".png", upscaled)

# ESPCN x4:

# 2
path = os.path.join(enhancer_folder, 'models', 'ESPCN_x2.pb')
sr.readModel(path)

sr.setModel("espcn", 2)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/espcn2_' + save_name + '.png', upscaled)

# 4
path = os.path.join(enhancer_folder, 'models', 'ESPCN_x4.pb')
sr.readModel(path)

sr.setModel("espcn", 4)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/espcn4_' + save_name + '.png', upscaled)

# FSRCNN x4:

# 2
path = os.path.join(enhancer_folder, 'models', 'FSRCNN_x2.pb')
sr.readModel(path)

sr.setModel("fsrcnn", 2)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/fsrcnn2_' + save_name + '.png', upscaled)

# 4
path = os.path.join(enhancer_folder, 'models', 'FSRCNN_x4.pb')
sr.readModel(path)

sr.setModel("fsrcnn", 4)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/fsrcnn4_' + save_name + '.png', upscaled)

# LapSRN x4:

# 2
path = os.path.join(enhancer_folder, 'models', 'LapSRN_x2.pb')
sr.readModel(path)

sr.setModel("lapsrn", 2)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/lapsrn2_' + save_name + '.png', upscaled)

# 4
path = os.path.join(enhancer_folder, 'models', 'LapSRN_x4.pb')
sr.readModel(path)

sr.setModel("lapsrn", 4)

upscaled = sr.upsample(img)
cv2.imwrite(resoult_path + '/lapsrn4_' + save_name + '.png', upscaled)
