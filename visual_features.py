# Visual Feature Outputs - Weapons + Objects + textOCR

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
import detectron2
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
from ultralytics import YOLO 

from object_level_features import ObjectFeatures
import cv2

from face_detection_er import *

class VisualFeatures:

    def __init__(self, PERSONAL_TOKEN = '1497c2be9d2447dcb6a21c94b382515a'):

        ## connects to clarifai api for weapon detection and text ocr
        channel = ClarifaiChannel.get_grpc_channel()
        self.stub = service_pb2_grpc.V2Stub(channel)

        ## metadata for authentification details 
        ## and other required objects passed for the api request
        self.metadata = (('authorization', 'Key ' + PERSONAL_TOKEN),)
        self.userDataObject = resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')

        ## custom object detection module
        self.obj_detection = ObjectFeatures()
        ## ADDED yolo v8 model for object detection
        self.yolo_model = YOLO("yolov8n.pt")
        #custom face detection module
        self.emo_recognition = FaceFeatures()

    def weapon_detection(self, frame):
        # frame is a numpy array
        weapons_dict = {}

        ## convert the numpy frame to png 
        success, encoded_image = cv2.imencode('.png', frame)
        file_bytes = encoded_image.tobytes()

        ## send png image to clarifai's api
        post_model_outputs_response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=self.userDataObject,
                model_id='weapon-detection',
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image = resources_pb2.Image(base64=file_bytes)
                        )
                    )
                ]
            ),
            metadata=self.metadata
        )
        print('post_model_outputs_response',post_model_outputs_response)
        ## iterate through the detected regions and filter by confidence threshold 0.8
        output = post_model_outputs_response.outputs[0]
        threshold = 0.6
        

        ## stores detected weapon types and cofidence scores in dictionary e.g. "handgun": [0.92]
        for regions in output.data.regions:
        # print(regions.data.concepts)
            if regions.value > threshold:
                if regions.data.concepts[0].name in weapons_dict:
                    weapons_dict[regions.data.concepts[0].name].append(regions.value)
                else:
                    weapons_dict[regions.data.concepts[0].name] = [regions.value]
        
        return weapons_dict

    ## uses clarifai again to detect text in images 
    def text_ocr(self, frame):
        # text ocr ClarifAI stored in a list
        text_ocr_list = []

        # convert images into bytes
        success, encoded_image = cv2.imencode('.png', frame)
        file_bytes = encoded_image.tobytes()

        # send to clarifai's API model
        post_model_outputs_response = self.stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=self.userDataObject,
                model_id='ocr-scene-english-paddleocr',
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=file_bytes)
                        )
                    )
                ]
            ),
            metadata=self.metadata
        )

        # extracts recognized text from detected regions e.g. ["WARNING:", "Authorized Personnel Only"]
        output = post_model_outputs_response.outputs[0]
        for regions in output.data.regions:
            text_ocr_list.append(regions.data.text.raw)
            
        ## return the list of detected strings
        return text_ocr_list

    # detecting objects using custom model
    def object_detection(self, frame):
        results = self.yolo_model(frame)
        detections = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, class_id = box.tolist()
                label = self.yolo_model.names[int(class_id)]  # Get class label

                detections.append({
                    "object": label,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 2)
                })

        # print("detections",detections)
        return detections
        ## old custom object detection method 
        #return self.obj_detection.get_predictions(frame)
    
    ## detects emotinos from faces using FaceFeatures module 
    def emotion_recognition(self,frame):
        # detects faces in the image
        faces = self.emo_recognition.get_face_output(frame)

        #detects cropped faces
        cropped_faces = self.emo_recognition.crop_faces(frame,faces)

        #runs emotion classification on the faces
        # output e.g. {"emotion": "happy", "confidence": 0.85},
        outputs = self.emo_recognition.get_emotion(cropped_faces)
        # print("outputs",outputs)
        return outputs






    



        
