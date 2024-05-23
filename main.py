import cv2
import mediapipe as mp

# Инициализация MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Функция для выделения ягодиц
def highlight_buttocks(image, left_hip, right_hip):
    # Вычисляем центр ягодиц
    butt_center_x = int((left_hip.x + right_hip.x) / 2 * image.shape[1])
    butt_center_y = int((left_hip.y + right_hip.y) / 2 * image.shape[0])

    # Рисуем круг в центре ягодиц
    cv2.circle(image, (butt_center_x, butt_center_y), 10, (0, 0, 255), -1)

    # Рисуем квадраты вокруг бедер
    cv2.rectangle(image, (int(left_hip.x * image.shape[1] - 10), int(left_hip.y * image.shape[0] - 10)),
                  (int(left_hip.x * image.shape[1] + 10), int(left_hip.y * image.shape[0] + 10)), (0, 255, 0), 2)
    cv2.rectangle(image, (int(right_hip.x * image.shape[1] - 10), int(right_hip.y * image.shape[0] - 10)),
                  (int(right_hip.x * image.shape[1] + 10), int(right_hip.y * image.shape[0] + 10)), (0, 255, 0), 2)

    return butt_center_x, butt_center_y

# Инициализация видеозахвата
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры. Завершение.")
            break

        # Преобразование изображения в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Обработка изображения
        results = pose.process(image)

        # Обратно преобразуем изображение в BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Рисуем аннотации на изображении
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Получаем координаты бедер
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            # Выделяем область ягодиц и центр
            butt_center_x, butt_center_y = highlight_buttocks(image, left_hip, right_hip)

            # Отправляем координаты на робот для укола
            # send_to_robot(butt_center_x, butt_center_y)

        # Отображаем результат
        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
