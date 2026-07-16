from deepface import DeepFace


class FaceVerifier:
    def __init__(self, model_name="ArcFace", distance_metric="cosine", threshold=None):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold

    def verify(self, live_frame, card_face):
        result = DeepFace.verify(
            live_frame,
            card_face,
            model_name=self.model_name,
            distance_metric=self.distance_metric,
            detector_backend="opencv",
            align=True,
            enforce_detection=False,
            threshold=self.threshold,
        )
        return result["verified"], result["distance"]