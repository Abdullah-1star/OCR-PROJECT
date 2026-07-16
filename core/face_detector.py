import cv2
import numpy as np


class FaceDetector:
    def __init__(self, proto_path, model_path, confidence=0.6, padding=0.35):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.confidence = confidence
        self.padding = padding

    def detect(self, id_card):
        h, w = id_card.shape[:2]
        blob = cv2.dnn.blobFromImage(id_card, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        best_idx = np.argmax(detections[0, 0, :, 2])
        best_score = detections[0, 0, best_idx, 2]

        if best_score < self.confidence:
            return None

        box = detections[0, 0, best_idx, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        pad_x = int((x2 - x1) * self.padding)
        pad_y = int((y2 - y1) * self.padding)
        x1, y1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
        x2, y2 = min(x2 + pad_x, w), min(y2 + pad_y, h)

        face = id_card[y1:y2, x1:x2]
        return face
