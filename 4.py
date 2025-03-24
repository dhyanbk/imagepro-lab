import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def enhance_contrast_color(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def segment_image_color(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(gray_image)
    return (gray_image > threshold).astype(np.uint8) * 255


image_path = "flower.jpg"  
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found. Check the file path.")


enhanced_image = enhance_contrast_color(image)
segmented_image = segment_image_color(enhanced_image)


titles = ['Original Image', 'Enhanced Image', 'Segmented Image']
images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB),
    segmented_image  
]

plt.figure(figsize=(15, 5))
for i, img in enumerate(images):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img if i < 2 else img, cmap='gray' if i == 2 else None)
    plt.title(titles[i])
    plt.axis('off')

plt.show()
