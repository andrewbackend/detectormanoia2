
import cv2
import mediapipe as mp

# INICIAR MEDIAPIPE

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ABRIR CAMARA

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # EFECTO ESPEJO
    frame = cv2.flip(frame, 1)

    # CONVERTIR A RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # DETECTAR MANOS
    results = hands.process(rgb)

    dedos = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # DIBUJAR MANO
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=3, circle_radius=3),
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )

            # PUNTOS
            puntos = hand_landmarks.landmark

            # CONTAR DEDOS

            # Pulgar
            if puntos[4].x < puntos[3].x:
                dedos += 1

            # Índice
            if puntos[8].y < puntos[6].y:
                dedos += 1

            # Medio
            if puntos[12].y < puntos[10].y:
                dedos += 1

            # Anular
            if puntos[16].y < puntos[14].y:
                dedos += 1

            # Meñique
            if puntos[20].y < puntos[18].y:
                dedos += 1

    # TEXTO CONTADOR
    cv2.putText(
        frame,
        f"Dedos: {dedos}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        3
    )

    # FOOTER
    cv2.putText(
        frame,
        "IA Project | Andrew Torres Ramirez",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    # MOSTRAR VENTANA
    cv2.imshow("Detector de Mano IA", frame)

    # SALIR CON ESC
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

