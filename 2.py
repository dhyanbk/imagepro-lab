import cv2
import numpy as np

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image")
        exit()
    return image

def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    return translated_image

def rotate_image(image, angle, scale=1.0):
    rows, cols = image.shape[:2]
    center = (cols / 2, rows / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def scale_image(image, scale_x, scale_y):
    new_width = int(image.shape[1] * scale_x)
    new_height = int(image.shape[0] * scale_y)
    scaled_image = cv2.resize(image, (new_width, new_height))
    return scaled_image

def main():
    image_path = input("Enter the image path: ")
    image = load_image(image_path)

    tx = int(input("Enter translation in x direction: "))
    ty = int(input("Enter translation in y direction: "))
    angle = float(input("Enter the rotation angle: "))
    scale_x = float(input("Enter the scale in x direction: "))
    scale_y = float(input("Enter the scale in y direction: "))

    translated_image = translate_image(image, tx, ty)
    rotated_image = rotate_image(translated_image, angle)
    scaled_image = scale_image(rotated_image, scale_x, scale_y)

    cv2.imshow("Original", image)
    cv2.imshow("Translated", translated_image)
    cv2.imshow("Rotated", rotated_image)
    cv2.imshow("Scaled", scaled_image)

    save_option = input("Do you want to save the transformed images? (yes/no): ")
    if save_option.lower() == "yes":
        save_path = input("Enter the save path (folder): ")
        cv2.imwrite(f"{save_path}/translated_image.jpg", translated_image)
        cv2.imwrite(f"{save_path}/rotated_image.jpg", rotated_image)
        cv2.imwrite(f"{save_path}/scaled_image.jpg", scaled_image)
        print("Images saved successfully")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
