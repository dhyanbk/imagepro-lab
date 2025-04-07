import numpy as np
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found")
    exit()

binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
struct_elem = np.ones((3, 3), np.uint8)

encoded_image = cv2.erode(binary_image, struct_elem)
dilated_image = cv2.dilate(encoded_image, struct_elem)

encoded_diff = cv2.absdiff(encoded_image, binary_image)
dilated_diff = cv2.absdiff(dilated_image, binary_image)

titles = ['Original Binary', 'Eroded', 'Dilated', 'Erosion Diff', 'Dilation Diff']

images = [binary_image, encoded_image, dilated_image, encoded_diff, dilated_diff]

plt.figure(figsize=(16, 6))
for i, img in enumerate(images):
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()