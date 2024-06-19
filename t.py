import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = './data/c.png'  
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Unable to open image file: {img_path}")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

color_ranges = {
    'Brown': ([10, 100, 20], [20, 255, 200]),
    'Red': ([0, 50, 50], [10, 255, 255]),
    'Orange': ([10, 100, 20], [25, 255, 255]),
    'Yellow': ([25, 50, 50], [35, 255, 255]),
    'Green': ([35, 50, 50], [85, 255, 255]),
    'Blue': ([85, 50, 50], [130, 255, 255]),
    'Purple': ([130, 50, 50], [160, 255, 255]),
    'Pink': ([160, 50, 50], [180, 255, 255]),
    'Black': ([0, 0, 0], [180, 255, 30]),
    'Gray': ([0, 0, 50], [180, 50, 200])
}

result_img = img.copy()

for color, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # Further increased threshold
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 20 and h > 20:  # Ignore rectangles that are too small
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()