import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe
mp_manos = mp.solutions.hands
manos = mp_manos.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_dibujo = mp.solutions.drawing_utils

# Cámara
camara = cv2.VideoCapture(0)

# Lienzo para dibujar
lienzo = None

# Punto anterior
x_anterior = 0
y_anterior = 0

while True:
    ret, frame = camara.read()

    # Voltear imagen
    frame = cv2.flip(frame, 1)

    # Crear lienzo
    if lienzo is None:
        lienzo = np.zeros_like(frame)

    # Convertir colores
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar manos
    resultado = manos.process(rgb)

    if resultado.multi_hand_landmarks:

        for mano in resultado.multi_hand_landmarks:

            # Dibujar puntos de la mano
            mp_dibujo.draw_landmarks(
                frame,
                mano,
                mp_manos.HAND_CONNECTIONS
            )

            # Dedo índice
            indice = mano.landmark[8]

            alto, ancho, _ = frame.shape

            x = int(indice.x * ancho)
            y = int(indice.y * alto)

            # Dibujar línea
            if x_anterior == 0 and y_anterior == 0:
                x_anterior = x
                y_anterior = y

            cv2.line(
                lienzo,
                (x_anterior, y_anterior),
                (x, y),
                (0, 0, 255),
                5
            )

            x_anterior = x
            y_anterior = y

    else:
        x_anterior = 0
        y_anterior = 0

    # Combinar cámara y dibujo
    frame = cv2.add(frame, lienzo)

    # Texto
    cv2.putText(
        frame,
        "PINTA EN EL AIRE",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(frame)

    tecla = cv2.waitKey(1)

    # ESC para salir
    if tecla == 27:
        break

    # C para borrar
    if tecla == ord('c'):
        lienzo = np.zeros_like(frame)

camara.release()
cv2.destroyAllWindows()

