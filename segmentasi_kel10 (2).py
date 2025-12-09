import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Konfigurasi Berkas Input ---
OUTPUT_DIR = "output_citra_derau"

# Daftar citra yang harus dibaca dari folder output_citra_derau
FILES_TO_LOAD = [
    "citra_potret_asli.png",
    "citra_lanskap_asli.png",
    "citra_salt_and_pepper.png",
    "citra_gaussian.png",
]

# --- Fungsi Segmentasi (Roberts, Prewitt, Sobel, Frei-Chen) ---

def segmentasi_roberts(img):
    """Segmentasi menggunakan Operator Roberts."""
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalisasi ke rentang 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def segmentasi_prewitt(img):
    """Segmentasi menggunakan Operator Prewitt."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]], dtype=np.float32)
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

def segmentasi_sobel(img):
    """Segmentasi menggunakan Operator Sobel."""
    # Gradien X dan Y menggunakan Sobel
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude menggunakan fungsi OpenCV untuk visualisasi yang baik
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return magnitude

def segmentasi_frei_chen(img):
    """Segmentasi menggunakan Operator Frei-Chen."""
    # Kernel Frei-Chen
    kernel_x = np.array([[-1, 0, 1], [-np.sqrt(2), 0, np.sqrt(2)], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -np.sqrt(2), -1], [ 0,  0,  0], [ 1,  np.sqrt(2),  1]], dtype=np.float32)
    
    grad_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)

metode_segmentasi = {
    "Roberts": segmentasi_roberts,
    "Prewitt": segmentasi_prewitt,
    "Sobel": segmentasi_sobel,
    "Frei-Chen": segmentasi_frei_chen,
}

# --- 3. Eksekusi Pembacaan dan Segmentasi ---

print("Memulai proses segmentasi (Poin 2)...")
citra_list = {}

# 1. Membaca 4 Citra dari Folder Output
for filename in FILES_TO_LOAD:
    filepath = os.path.join(OUTPUT_DIR, filename)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"\nERROR: Gagal memuat '{filepath}'. Pastikan program 1_buat_citra_derau.py sudah dijalankan.")
        exit()
    
    citra_list[f"Citra: {filename}"] = img

print("4 Citra input berhasil dibaca.")

# 2. Melakukan Segmentasi
results = {}
for img_name, img in citra_list.items():
    print(f"\n--- Memproses {img_name} ({img.shape[1]}x{img.shape[0]}) ---")
    results[img_name] = {"Original": img}
    
    for method_name, func in metode_segmentasi.items():
        print(f"  > Menerapkan {method_name}...")
        hasil_segmentasi = func(img)
        results[img_name][method_name] = hasil_segmentasi

# 3. Tampilkan Hasil (Visualisasi)
# 
for img_name, res in results.items():
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle(f"Hasil Segmentasi Citra: {img_name}", fontsize=14)
    
    axes[0].imshow(res["Original"], cmap='gray')
    axes[0].set_title("Citra Asli")
    axes[0].axis('off')
    
    i = 1
    for method_name, segmented_img in res.items():
        if method_name != "Original":
            axes[i].imshow(segmented_img, cmap='gray')
            axes[i].set_title(method_name)
            axes[i].axis('off')
            i += 1
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\nProses segmentasi selesai. Visualisasi hasil telah ditampilkan.")