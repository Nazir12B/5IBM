import cv2
import numpy as np
import os
import torch

class ObjectDetection:
    def __init__(self, model_name='yolov5s'):
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.target_classes = ["laptop", "mouse","person"]  # Liste des classes cibles
    
    def access_camera(self):
        cap = cv2.VideoCapture(0)  # Utilise la webcam par défaut (0)

        if not cap.isOpened():
            print("Erreur d'ouverture de la webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Détection d'objets avec YOLOv5
            results = self.model(frame)
            # Filtrer les résultats pour ne garder que les classes spécifiées
            results = self.filter_results(results)
            # Annoter les résultats sur les images avec code couleur
            frame = self.annotate_frame(frame, results)
            # Afficher les FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Afficher les images
            cv2.imshow('Web Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_color(self, class_id):
        # Palette de couleurs
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (0, 128, 128), (128, 0, 128)
        ]
        return palette[class_id % len(palette)]

    def filter_results(self, results):
        filtered_results = []
        for bbox in results.xyxy[0].numpy():
            cls = int(bbox[5])
            if self.model.names[cls] in self.target_classes:
                filtered_results.append(bbox)
        results.xyxy[0] = torch.tensor(filtered_results)
        return results

    def annotate_frame(self, frame, results):
        for bbox in results.xyxy[0].numpy():
            x1, y1, x2, y2, conf, cls = bbox
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            color = self.get_color(int(cls))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return frame

def main():
    detector = ObjectDetection()
    detector.access_camera()

if __name__ == "__main__":
    main()
