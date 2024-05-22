import cv2 as cv
import math
import time
import argparse

class AgeGenderDetector:
    def __init__(self, face_model, face_proto, age_model, age_proto, gender_model, gender_proto):
        # Initialize face, age, and gender detection networks
        self.faceNet = cv.dnn.readNet(face_model, face_proto)
        self.ageNet = cv.dnn.readNetFromCaffe(age_proto, age_model)
        self.genderNet = cv.dnn.readNetFromCaffe(gender_proto, gender_model)
        # Mean values used for preprocessing input images
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        # Labels for age and gender classification
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male', 'Female']



    def get_face_box(self, frame, conf_threshold=0.7):
        # Detect faces in the input frame
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes


    def predict_age_gender(self, face):
        # Predict age and gender from the input face
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()
        gender = self.genderList[genderPreds[0].argmax()]

        self.ageNet.setInput(blob)
        agePreds = self.ageNet.forward()
        age = self.ageList[agePreds[0].argmax()]

        return gender, genderPreds[0].max(), age, agePreds[0].max()



def main(args):
    # Initialize the detector object
    detector = AgeGenderDetector("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt",
                                 "age_net.caffemodel", "age_deploy.prototxt",
                                 "gender_net.caffemodel", "gender_deploy.prototxt")

    # Open video capture
    cap = cv.VideoCapture(args.i if args.i else 0)
    padding = 20
    while cv.waitKey(1) < 0:
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        # Get faces from the frame
        frameFace, bboxes = detector.get_face_box(frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue

        for bbox in bboxes:
            # Extract face region
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            # Predict age and gender
            gender, gender_confidence, age, age_confidence = detector.predict_age_gender(face)

            print("Gender : {}, confidence = {:.3f}".format(gender, gender_confidence))
            print("Age : {}, confidence = {:.3f}".format(age, age_confidence))

            # Draw labels on the frame
            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (bbox[0] - 5, bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
            name = args.i
            cv.imwrite('./detected/' + name, frameFace)
        print("Time : {:.3f}".format(time.time() - t))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    args = parser.parse_args()
    main(args)

