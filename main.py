from PyQt5 import QtWidgets, QtCore, QtGui
import sys
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import face_recognition
import numpy as np
import imutils
import time
import glob
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

class MaskAndFaceDetectorThread(QtCore.QThread):
    maskDetected = QtCore.pyqtSignal("PyQt_PyObject")
    faceRecognized = QtCore.pyqtSignal(tuple)
    faceCount = QtCore.pyqtSignal(int)
    fps = QtCore.pyqtSignal(float)

    def __init__(self, identificate=True):
        QtCore.QThread.__init__(self)

        self.known_face_encodings = list()
        self.known_face_names = list()
        self.identificate = identificate
        self.is_running = False
        for path in glob.glob("faces/*"):
            try:
                image = face_recognition.load_image_file(path)
                face_encoding = face_recognition.face_encodings(image)[0]
                self.known_face_names.append(os.path.basename(path).split(".")[0])
                self.known_face_encodings.append(face_encoding)
            except Exception as e:
                print(e)
            
    def run(self):
        self.is_running = True
        cam = cv2.VideoCapture(0)
        for _ in range(20):
            cam.read()
        process_this_frame = True

        while self.is_running:
            start = time.time()
            ret, frame = cam.read()
            
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            self.mask_is_on = False
            for (box, pred) in zip(locs, preds):
                (mask, withoutMask) = pred
                self.mask_is_on = mask > withoutMask
                self.maskDetected.emit(self.mask_is_on)
            if not self.identificate:
                end = time.time()
                self.fps.emit(1 / (end - start))
                if not len(locs):
                    self.maskDetected.emit(False)
                continue
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame)

                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                try:
                    if len(face_encodings):
                        face_encoding = face_encodings[0]
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = ""

                        # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = self.known_face_names[first_match_index]
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]

                        self.faceRecognized.emit((name, self.mask_is_on))
                except Exception as e:
                    print(e)
                    
            process_this_frame = not process_this_frame
            end = time.time()
            self.faceCount.emit(len(face_encodings) or len(locs))
            if len(face_encodings):
                self.fps.emit(1 / (end - start))


class DetectionWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.centralWidget = QtWidgets.QWidget()
        self.gridLayout = QtWidgets.QGridLayout()

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)

        self.video_thread = VideoThread()
        self.video_thread.frameChanged.connect(self.change_pixmap)
        self.video_thread.start() 

        self.mask_status_label = QtWidgets.QLabel()
        self.mask_status_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.mask_status_label.setFont(font)

        self.status_bar = QtWidgets.QProgressBar()

        self.gridLayout.addWidget(self.status_bar, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.video_label, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.mask_status_label, 2, 0, 1, 1)

        self.centralWidget.setLayout(self.gridLayout)
        self.setCentralWidget(self.centralWidget)

        self.mask_is_on = False

    def change_pixmap(self, image):
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.mask_and_face_detector_thread.is_running = False
        self.mask_and_face_detector_thread.exit()
        event.accept()


class VideoThread(QtCore.QThread):
    frameChanged = QtCore.pyqtSignal(QtGui.QImage)
    is_running = False

    def run(self):
        self.is_running = True
        cam = cv2.VideoCapture(0)
        for _ in range(20):
            cam.read()
        while self.is_running:
            ret, frame = cam.read()
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbImage = cv2.flip(rgbImage, 1)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            self.frameChanged.emit(p)


class MaskAndFaceRecognitionWindow(DetectionWindow):
    def __init__(self, parent=None):
        DetectionWindow.__init__(self, parent=parent)

        self.person_name = None
        self.no_face_count = 0
        self.fps = 0

        self.mask_and_face_detector_thread = MaskAndFaceDetectorThread()
        self.mask_and_face_detector_thread.maskDetected.connect(self.change_mask_status)
        self.mask_and_face_detector_thread.faceRecognized.connect(self.face_recognized)
        self.mask_and_face_detector_thread.faceCount.connect(self.face_count)
        self.mask_and_face_detector_thread.fps.connect(self.set_fps)
        self.mask_and_face_detector_thread.start()

    def change_mask_status(self, mask):
        if mask:
            self.mask_status_label.setText("Спустите маску для идентификации")
            self.mask_status_label.setStyleSheet("color: green")
            self.status_bar.setValue(33.3)
            self.mask_is_on = True
        else:
            if not self.mask_is_on:
                self.mask_status_label.setText("Наденьте маску")
                self.mask_status_label.setStyleSheet("color: red")
                self.status_bar.setValue(0)

    def face_recognized(self, name_mask):
        name, mask = name_mask
        if name:
            self.person_name = name
        if self.mask_is_on and not mask:
            self.status_bar.setValue(66.6)
            self.mask_status_label.setText("Поднесите руку к термометру")

    def face_count(self, count):
        if not count:
            self.no_face_count += 1
            if self.no_face_count >= self.fps:
                self.status_bar.setValue(0)
                self.no_face_count = 0
                self.mask_is_on = False
                self.person_name = ""
                self.mask_status_label.setText("Наденьте маску")
                print(1)
        else:
            self.no_face_count = 0

    def set_fps(self, fps):
        if fps:
            self.fps = (fps + self.fps) / 2
        else:
            self.fps = fps


class MaskRecognitionWindow(DetectionWindow):
    def __init__(self, parent=None):
        DetectionWindow.__init__(self, parent=parent)

        self.mask_and_face_detector_thread = MaskAndFaceDetectorThread(identificate=False)
        self.mask_and_face_detector_thread.detected.connect(self.change_mask_status)
        self.mask_and_face_detector_thread.frameChanged.connect(self.change_pixmap)
        self.mask_and_face_detector_thread.start()

        self.step = 50

    def change_mask_status(self, mask):
        if mask:
            self.mask_status_label.setText("Поднесите руку к термометру")
            self.mask_status_label.setStyleSheet("color: green")
            self.status_bar.setValue(50)
            self.mask_is_on = True
        else:
            self.mask_status_label.setText("Наденьте маску")
            self.mask_status_label.setStyleSheet("color: red")
            self.status_bar.setValue(0)


class Main(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.resize(590, 400)

        self.centralWidget = QtWidgets.QWidget()
        self.gridLayout = QtWidgets.QGridLayout()

        self.ccs_label = QtWidgets.QLabel()
        self.ccs_label.setText("CCS - Covid Control System")
        font = QtGui.QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.ccs_label.setFont(font)
        self.ccs_label.setAlignment(QtCore.Qt.AlignCenter)

        self.mask_and_face_button = QtWidgets.QPushButton()
        self.mask_and_face_button.setText("Распознавание маски и лица")
        self.mask_and_face_button.clicked.connect(self.start_mask_and_face_window)
        self.mask_and_face_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        font.setPointSize(16)
        font.setBold(False)
        self.mask_and_face_button.setFont(font)

        self.mask_button = QtWidgets.QPushButton()
        self.mask_button.setText("Распознавание маски")
        self.mask_button.clicked.connect(self.start_mask_window)
        self.mask_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        font.setPointSize(16)
        font.setBold(False)
        self.mask_button.setFont(font)

        self.gridLayout.addWidget(self.ccs_label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.mask_and_face_button, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.mask_button, 2, 0, 1, 1)

        self.centralWidget.setLayout(self.gridLayout)
        self.setCentralWidget(self.centralWidget)

    def start_mask_and_face_window(self):
        w = MaskAndFaceRecognitionWindow(self)
        w.show()

    def start_mask_window(self):
        w = MaskRecognitionWindow(self)
        w.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    # main.setWindowState(QtCore.Qt.WindowFullScreen)
    # main.showFullScreen()
    main.show()
    app.exec()
