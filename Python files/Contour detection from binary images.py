from skimage import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import slideio

# actual image read
slide = slideio.open_slide(r"/content/drive/My Drive/monusac/TCGA-5P-A9K0-01Z-00-DX1_1.svs", 'SVS')
scene = slide.get_scene(0)
image = scene.read_block(slices=(0,scene.num_z_slices))


img1 = io.imread(r"/content/drive/My Drive/monusac/Macrophage/4_mask.tif")
# Find Canny edges 
img1 = img1.astype(np.uint8)
edged = cv2.Canny(img1, 30, 220) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

print("Number of Contours found = " + str(len(contours))) 

# Draw all contours 
# -1 signifies drawing all contours 
#img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB) 
image = image.astype(np.uint8)
con = cv2.drawContours(image, contours, -1, (0, 0, 255), 7)

plt.figure(figsize=(7,7))
plt.imshow(con)
plt.axis("off")
plt.title("Detected Macrophage: no of cells present " + str(len(contours)));
