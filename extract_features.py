import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

dataset_path = "wikiart_dataset"
output_csv = "wikiart_features_all_styles.csv"
IMG_SIZE = (128, 128)

headers = (
    [f"color_{i}" for i in range(512)] +
    [f"hog_{i}" for i in range(128)] +
    [f"lbp_{i}" for i in range(10)] +
    ["edge_density", "label"]
)

if not os.path.exists(dataset_path):
    for item in os.listdir("."):
        if "wikiart" in item.lower() and os.path.isdir(item):
            dataset_path = item
            break

print(f"Папка датасету: {dataset_path}")

styles = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
styles.sort()
print(f"Знайдено стилів: {len(styles)}")
print(f"Стилі: {', '.join(styles)}\n")

with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    total_count = 0

    for style in styles:
        style_dir = os.path.join(dataset_path, style)
        image_files = [f for f in os.listdir(style_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in tqdm(image_files, desc=f"Стиль: {style}"):
            img_path = os.path.join(style_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)

            # HSV гістограма (512)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            color_feat = (hist / (np.sum(hist) + 1e-7)).flatten()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # HOG (128)
            hog_feat = hog(
                gray,
                orientations=8,
                pixels_per_cell=(32, 32),
                cells_per_block=(1, 1),
                visualize=False
            )

            # LBP (10)
            lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
            lbp_feat, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

            # Canny крайова щільність (1)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (IMG_SIZE[0] * IMG_SIZE[1])

            row = np.hstack([color_feat, hog_feat, lbp_feat, edge_density]).tolist()

            row.append(style)
            writer.writerow(row)
            total_count += 1

print(f"\nОброблено зображень: {total_count}")

if IN_COLAB:
    files.download(output_csv)
else:
    print(f"Збережено у файл: {os.path.abspath(output_csv)}")