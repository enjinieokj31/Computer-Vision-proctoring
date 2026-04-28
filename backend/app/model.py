import onnxruntime as ort
import numpy as np
import cv2
import time

class YOLOModel:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = 640

    def preprocess(self, image):
        self.orig_h, self.orig_w = image.shape[:2]

        img = cv2.resize(image, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image):
        inp = self.preprocess(image)

        start = time.time()
        outputs = self.session.run(None, {self.input_name: inp})
        latency = round((time.time() - start) * 1000, 2)

        boxes = self.postprocess(outputs[0])

        return boxes, latency

    def postprocess(self, output):
        preds = output[0].T   # (8400,5)

        boxes = []
        scores = []

        scale_x = self.orig_w / self.input_size
        scale_y = self.orig_h / self.input_size

        for row in preds:
            xc, yc, w, h, conf = row

            if conf < 0.45:
                continue

            x1 = int((xc - w / 2) * scale_x)
            y1 = int((yc - h / 2) * scale_y)
            x2 = int((xc + w / 2) * scale_x)
            y2 = int((yc + h / 2) * scale_y)

            boxes.append([x1, y1, x2 - x1, y2 - y1])   # x,y,w,h
            scores.append(float(conf))

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            score_threshold=0.45,
            nms_threshold=0.35
        )

        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]

                detections.append({
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "conf": scores[i]
                })

        return detections