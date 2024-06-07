import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicializa la captura de video desde la cámara
cam = cv2.VideoCapture(0)

# Inicializa MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Obtiene el tamaño de la pantalla
screen_size = pyautogui.size()
screen_w, screen_h = screen_size.width, screen_size.height

while True:
    ret, frame = cam.read()
    
    # Verifica si la captura de video es exitosa
    if not ret:
        print("No se pudo acceder a la cámara")
        break
    
    # Voltea la imagen horizontalmente para eliminar el efecto de espejo
    frame = cv2.flip(frame, 1)
    
    # Convierte la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesa la imagen para detectar la malla facial
    output = face_mesh.process(rgb_frame)
    
    # Obtiene el tamaño del frame
    frame_h, frame_w, _ = frame.shape
    
    # Esta parte se utiliza para detectar una cara y obtener las coordenadas
    landmarks_points = output.multi_face_landmarks
    if landmarks_points:
        landmarks = landmarks_points[0].landmark
        
        # Landmarks para los ojos
        left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        left_eye_coords = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in left_eye_indices]
        right_eye_coords = [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in right_eye_indices]
        
        # Calcula el centro del ojo izquierdo y derecho
        left_eye_center = np.mean(left_eye_coords, axis=0)
        right_eye_center = np.mean(right_eye_coords, axis=0)
        
        # Coordenadas de la pupila
        left_pupil = (landmarks[468].x * frame_w, landmarks[468].y * frame_h)
        right_pupil = (landmarks[473].x * frame_w, landmarks[473].y * frame_h)
        
        # Dibuja los ojos y las pupilas
        for coord in left_eye_coords:
            cv2.circle(frame, tuple(np.array(coord, int)), 1, (0, 255, 0), -1)
        for coord in right_eye_coords:
            cv2.circle(frame, tuple(np.array(coord, int)), 1, (0, 255, 0), -1)
        cv2.circle(frame, tuple(np.array(left_pupil, int)), 3, (255, 0, 0), -1)
        cv2.circle(frame, tuple(np.array(right_pupil, int)), 3, (255, 0, 0), -1)
        
        # Mapear el movimiento de la pupila a la pantalla
        screen_x = screen_w / frame_w * left_pupil[0]
        screen_y = screen_h / frame_h * left_pupil[1]
        pyautogui.moveTo(screen_x, screen_y)
        
        # Detecta parpadeo (diferencia en y-coordinates de landmarks específicos)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    # Muestra la imagen
    cv2.imshow('Eye Controlled Mouse', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

# Libera la captura de video y cierra las ventanas
cam.release()
cv2.destroyAllWindows()
