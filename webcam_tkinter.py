import tkinter as tk
from tkinter import *
from tkinter import ttk
import threading
import cv2, os, time
from PIL import Image, ImageTk
import tensorflow as tf
print("webcam imports complete")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels_video.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph_video.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

prediction_global = [[0, 0, 0, 0, 0]]

avg_time = [0, 0, 0, 0, 0]

class CamView(threading.Thread):
    root = None
    frame1 = None

    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.root = frame
        self.label1 = tk.Label(self.root, text = "CAM VIEW")
        self.label1.grid(row=0, column=0)

        frame1 = tk.Frame(self.root)
        frame1.grid(row=1, column=1)
        self.frame1 = frame1
        self.label2 = ttk.Label(frame1, text='Angry')
        self.label2.grid(row=1, column=0, sticky=W, padx=10)
        self.label3 = ttk.Label(frame1, text='Disgust')
        self.label3.grid(row=2, column=0, sticky=W, padx=10)
        self.label4 = ttk.Label(frame1, text='Fear')
        self.label4.grid(row=3, column=0, sticky=W, padx=10)
        self.label5 = ttk.Label(frame1, text='Joy')
        self.label5.grid(row=4, column=0, sticky=W, padx=10)
        self.label6 = ttk.Label(frame1, text='Sadness')
        self.label6.grid(row=5, column=0, sticky=W, padx=10)

        self.scale1 = ttk.Scale(frame1, from_=0, to=100, orient=HORIZONTAL)
        self.scale1.grid(row=1, column=1, sticky=W, padx=10)
        self.scale2 = ttk.Scale(frame1, from_=0, to=100, orient=HORIZONTAL)
        self.scale2.grid(row=2, column=1, sticky=W, padx=10)
        self.scale3 = ttk.Scale(frame1, from_=0, to=100, orient=HORIZONTAL)
        self.scale3.grid(row=3, column=1, sticky=W, padx=10)
        self.scale4 = ttk.Scale(frame1, from_=0, to=100, orient=HORIZONTAL)
        self.scale4.grid(row=4, column=1, sticky=W, padx=10)
        self.scale5 = ttk.Scale(frame1, from_=0, to=100, orient=HORIZONTAL)
        self.scale5.grid(row=5, column=1, sticky=W, padx=10)

        self.entry1 = ttk.Entry(frame1, text='Angry_cam', width=10)
        self.entry1.grid(row=1, column=2, sticky=W, padx=10)
        self.entry2 = ttk.Entry(frame1, text='Disgust_cam', width=10)
        self.entry2.grid(row=2, column=2, sticky=W, padx=10)
        self.entry3 = ttk.Entry(frame1, text='Fear_cam', width=10)
        self.entry3.grid(row=3, column=2, sticky=W, padx=10)
        self.entry4 = ttk.Entry(frame1, text='Joy_cam', width=10)
        self.entry4.grid(row=4, column=2, sticky=W, padx=10)
        self.entry5 = ttk.Entry(frame1, text='Sadness_cam', width=10)
        self.entry5.grid(row=5, column=2, sticky=W, padx=10)

        self.label7 = ttk.Label(frame1, text='Avg Time')
        self.label7.grid(row=6, column=1, sticky=W, padx=10)
        self.entry6 = ttk.Entry(frame1, text='Avg Time', width=10)
        self.entry6.grid(row=6, column=2, sticky=W, padx=10)

    def setValues(self):
        global prediction_global
        self.entry1.delete(0, END)
        self.entry2.delete(0, END)
        self.entry3.delete(0, END)
        self.entry4.delete(0, END)
        self.entry5.delete(0, END)
        self.entry1.insert(0, str(round(prediction_global[0][0]*100, 2)) + " %")
        self.entry2.insert(0, str(round(prediction_global[0][2]*100, 2)) + " %")
        self.entry3.insert(0, str(round(prediction_global[0][4]*100, 2)) + " %")
        self.entry4.insert(0, str(round(prediction_global[0][3]*100, 2)) + " %")
        self.entry5.insert(0, str(round(prediction_global[0][1]*100, 2)) + " %")

        self.entry6.delete(0, END)
        self.entry6.insert(0, str(sum(avg_time)//5) + str(" ms"))

        self.scale1.set(prediction_global[0][0]*100)
        self.scale2.set(prediction_global[0][2]*100)
        self.scale3.set(prediction_global[0][4]*100)
        self.scale4.set(prediction_global[0][3]*100)
        self.scale5.set(prediction_global[0][1]*100)

    def run(self):
        while True:
            time.sleep(0.10)
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )

            b,g,r = cv2.split(frame)
            frame = cv2.merge((r,g,b))
            # Draw a rectangle around the faces
            try:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                frame_cropped = frame[y:y+h, x:x+w]
                frame_cropped = cv2.resize(frame_cropped, (0,0), fx=0.6, fy=0.6)
                frame_cropped = cv2.flip(frame_cropped, 1)

                frame_cropped_gray = gray[y:y+h, x:x+w]

                cv2.imwrite("1.jpg", frame_cropped_gray)
                # print("saved")
                img_cropped = Image.fromarray(frame_cropped)
                photoimg_cropped = ImageTk.PhotoImage(img_cropped)
                panel1 = tk.Label(self.frame1, image = photoimg_cropped)
                panel1.grid(row=0, column=0, columnspan=3)
                self.setValues()
            except:
                print("Error in CampView")

            frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
            frame = cv2.flip(frame, 1)

            img = Image.fromarray(frame)
            photoimg = ImageTk.PhotoImage(img)

            panel = tk.Label(self.root, image = photoimg)
            panel.grid(row=1, column=0)


class Predictor(CamView):
    # image_data = None
    def __init__(self):
        threading.Thread.__init__(self)
        # self.image_data = image

    def run(self):
        while True:
            try:
                # Read in the image_data
                image_data = tf.gfile.FastGFile("1.jpg", 'rb').read()
                s = time.time()*1000
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                global avg_time
                avg_time = avg_time[1:]
                avg_time.append(time.time()*1000 - s)
                # print("predictions : ", predictions)
                global prediction_global
                prediction_global = predictions
            except:
                print("Error in Predictor")

root = tk.Tk()

thread1 = CamView(root)
thread1.start()
Thread = Predictor()
Thread.start()

root.mainloop()