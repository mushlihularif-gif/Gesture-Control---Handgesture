import cv2
from ultralytics import YOLO
import pyautogui
import time

# 1. Inisialisasi Model YOLO dengan file bisindo.pt
# Pastikan nama file sesuai dengan yang Anda unduh
model = YOLO(r"D:\Informatics Engineering\Informatika Semester 4\Kecerdasan Buatan\Setelah UTS\bisindo.pt")

# 2. Akses Kamera
cap = cv2.VideoCapture(0)

# --- OPTIMASI KUALITAS KAMERA ---
# Diturunkan ke 1280x720 untuk mengurangi beban CPU dan mencegah lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = "Praktikum AI - Gesture Control BISINDO"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(window_name, 720, 405)

# Pengaturan jeda waktu agar satu gesture tidak mengeksekusi aksi berkali-kali
last_action_time = 0
delay = 0.5  # Jeda 0.5 detik antar aksi (Eksperimen Parameter)

print("Memulai program... Tekan 'q' pada jendela video untuk keluar.")

while True:
    success, frame = cap.read()
    if not success:
        print("Kamera tidak terbaca atau belum diizinkan!")
        break

    # Efek cermin agar pergerakan lebih intuitif
    frame = cv2.flip(frame, 1)

    # 3. Proses Deteksi menggunakan model BISINDO
    # conf=0.5: Diturunkan dari 0.8 agar deteksi lebih responsif dan cepat
    results = model.predict(frame, conf=0.5, verbose=False)

    gesture_text = "Menunggu gesture..."

    # 4. Membaca Hasil Deteksi
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Ekstraksi koordinat kotak (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # Ekstraksi nama kelas (akan menghasilkan huruf A, B, C, dst.)
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id] 

            # 5. Logika Eksekusi Aksi berdasarkan Huruf
            current_time = time.time()
            action_executed = False
            
            # Cek apakah sudah melewati masa jeda (cooldown)
            if current_time - last_action_time > delay:
                # Pemetaan Huruf ke Perintah Keyboard
                if class_name == 'A':
                    pyautogui.press("right")
                    action = "NEXT SLIDE (A)"
                    action_executed = True
                elif class_name == 'B':
                    pyautogui.press("left")
                    action = "PREV SLIDE (B)"
                    action_executed = True
                elif class_name == 'C':
                    pyautogui.press("volumeup")
                    action = "VOLUME NAIK (C)"
                    action_executed = True
                elif class_name == 'D':
                    pyautogui.press("volumedown")
                    action = "VOLUME TURUN (D)"
                    action_executed = True
                # TASK 1: Ekspansi 5 Gestur
                elif class_name == 'E':
                    pyautogui.press("space")
                    action = "PLAY/PAUSE (E)"
                    action_executed = True
                elif class_name == 'F':
                    pyautogui.press("m")
                    action = "MUTE/UNMUTE (F)"
                    action_executed = True
                elif class_name == 'G':
                    pyautogui.press("up")
                    action = "SCROLL UP (G)"
                    action_executed = True
                elif class_name == 'H':
                    pyautogui.press("down")
                    action = "SCROLL DOWN (H)"
                    action_executed = True
                elif class_name == 'I':
                    pyautogui.press("esc")
                    action = "ESCAPE (I)"
                    action_executed = True
                else:
                    action = f"TIDAK ADA AKSI UNTUK '{class_name}'"

                if action_executed:
                    last_action_time = current_time
                    gesture_text = f"Aksi Terakhir: {action}"

            # TASK 3: Kustomisasi Visual
            # Hijau jika berhasil eksekusi (atau siap eksekusi), Oranye saat cooldown
            if current_time - last_action_time > delay or action_executed:
                box_color = (0, 255, 0) # Hijau dalam BGR
            else:
                box_color = (0, 165, 255) # Oranye dalam BGR

            # Gambar Bounding Box dan nama hurufnya di layar
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"Deteksi Huruf: {class_name}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    # Tampilkan status aksi di pojok kiri atas
    cv2.putText(frame, gesture_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan video
    cv2.imshow(window_name, frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan memori dan tutup kamera
cap.release()
cv2.destroyAllWindows()