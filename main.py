from tkinter import ttk
from tkinter import *
import tkinter as tk
import webbrowser
import speech_recognition as sr
from pygame import mixer
import time
import audioFunctions
import threading
import os

import json
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wv


root = tk.Tk()
root.title('Emotion Detector')
root.iconbitmap('cool.ico')

# for audio
frame1 = ttk.Frame(root)
frame1.grid(row=0, column=0)

# for watson
frame2 = ttk.Frame(root)
frame2.grid(row=1, column=0)

# for video
frame3 = ttk.Frame(root)
frame3.grid(row=2, column=0)

btn1 = tk.StringVar()
audiofile = tk.StringVar()


# style = ttk.Style()
# style.theme_use('winnative')

choices = {'Audio 1':'a0.wav', 'Audio 2':'a01.wav', 'Audio 3':'a02.wav', 'Audio 4':'a03.wav', 'Audio 5':'a04.wav',
           'Audio 6':'a05.wav', 'Audio 7':'a06.wav', 'Audio 8':'a07.wav', 'Audio 9':'a08.wav', 'Audio 10':'a09.wav',
           'Audio 11':'a10.wav', 'Audio 12':'a11.wav'}


def callback():
    if btn1.get() == "input":
        audioFunctions.payload = text2.get("1.0",END)
        audioFunctions.invokeToneConversation()
        time.sleep(0.01)
        while audioFunctions.done == False:
            continue
    elif btn1.get() == "speech":
        audioFunctions.payload = text1.get("1.0",END)
        audioFunctions.invokeToneConversation()
        time.sleep(0.01)
        while audioFunctions.done == False:
            continue
    else:
        print("Invalid Input-function_name:callback")

    dict = audioFunctions.tones
    entry1.delete(0, END)
    entry2.delete(0, END)
    entry3.delete(0, END)
    entry4.delete(0, END)
    entry5.delete(0, END)
    entry1.insert(0, str(round(dict['Anger'], 2)) + " %")
    entry2.insert(0, str(round(dict['Disgust'], 2)) + " %")
    entry3.insert(0, str(round(dict['Fear'], 2)) + " %")
    entry4.insert(0, str(round(dict['Joy'], 2)) + " %")
    entry5.insert(0, str(round(dict['Sadness'], 2)) + " %")
    scale1.set(round(dict['Anger'], 2))
    scale2.set(round(dict['Disgust'], 2))
    scale3.set(round(dict['Fear'], 2))
    scale4.set(round(dict['Joy'], 2))
    scale5.set(round(dict['Sadness'], 2))

r = sr.Recognizer()
def speechSelect():
    file = 'audio/' + choices[audiofile.get()]
    print(file)
    with sr.WavFile(file) as source:
        audio = r.record(source)
        mixer.init()
        mixer.music.load(file)
        mixer.music.play()
        audioFunctions.audioforSpeechToText = audio
        GoogleThread = audioFunctions.GoogleTextToSpeech()
        GoogleThread.start()
        with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())
                f.close()
    ThreadAudioAnalysis = SoundAnalysis()
    ThreadAudioAnalysis.start()


def speechAnalysis():
    mixer.init()
    mixer.music.load('chime1.mp3')
    mixer.music.play()

    r = sr.Recognizer()
    r.pause_threshold = 0.7
    r.energy_threshold = 400
    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
            audioFunctions.audioforSpeechToText = audio
            GoogleThread = audioFunctions.GoogleTextToSpeech()
            GoogleThread.start()
            with open("microphone-results.wav", "wb") as f:
                f.write(audio.get_wav_data())
                f.close()
            ThreadAudioAnalysis = SoundAnalysis()
            ThreadAudioAnalysis.start()
            mixer.music.load('chime2.mp3')
            mixer.music.play()
        except:
            print('Exception in speechAnalysis Class')


class UpdateText(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            try:
                file = open("Transcript.txt", "r")
                text = file.read()
                file.close()
                text1.delete('1.0', END)
                text1.insert(END, text)
                time.sleep(0.5)
            except:
                pass


class SoundAnalysis(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        dict = soundAnalysis()
        print("dict in class SoundAnalysis : ", dict)
        entry6.delete(0, END)
        entry7.delete(0, END)
        entry8.delete(0, END)
        entry9.delete(0, END)
        entry10.delete(0, END)
        entry6.insert(0, str(round(dict['angry']*100, 2)) + " %")
        entry7.insert(0, str(round(dict['disgust']*100, 2)) + " %")
        entry8.insert(0, str(round(dict['fear']*100, 2)) + " %")
        entry9.insert(0, str(round(dict['joy']*100, 2)) + " %")
        entry10.insert(0, str(round(dict['sad']*100, 2)) + " %")
        scale6.set(round(dict['angry']*100, 2))
        scale7.set(round(dict['disgust']*100, 2))
        scale8.set(round(dict['fear']*100, 2))
        scale9.set(round(dict['joy']*100, 2))
        scale10.set(round(dict['sad']*100, 2))


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("retrained_labels_audio.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph_audio.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

def soundAnalysis():
    in_file = 'microphone-results.wav'
    '''in_file is a wave file, output is a dictionary with angry , joy , fear, disgust '''
    s = np.zeros([360000],np.int)
    fs,data = wv.read(in_file)
    data = np.array(data)
    s[0:data.size] = data[0:data.size]
    s = s.reshape([600,600])
    plt.imsave(in_file.split('.')[0]+".png",s)
    im = Image.open(in_file.split('.')[0]+".png")
    rgb_im = im.convert('RGB')
    rgb_im.save(in_file.split('.')[0] + ".jpeg")

    emotions = {'angry':0,'joy':0,'fear':0,'disgust':0}

    image_path = in_file.split('.')[0]+".jpeg"

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
    print(predictions)
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        try:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            emotions[human_string] = score
        except:
            pass

    os.remove(in_file.split('.')[0]+".png")
    os.remove(in_file.split('.')[0] + ".jpeg")
    return emotions


class WebCamInitializer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        os.system("python webcam_tkinter.py")


# ........ Watson .........

label1 = ttk.Label(frame2, text='TEXT TONE ANALYSER')
label1.grid(row=0, columnspan=2)

text1 = tk.Text(frame2, height=7, width=25)
text1.grid(row=1, column=0, sticky=W)
text2 = tk.Text(frame2, height=7, width=25)
text2.grid(row=1, column=1, sticky=W)

RadioButton1 = ttk.Radiobutton(frame2, text='Speech to Text', value='speech', variable = btn1)
RadioButton1.grid(row=2, column=0, sticky=W)

RadioButton2 = ttk.Radiobutton(frame2, text='Custom Input', value='input', variable = btn1)
RadioButton2.grid(row=2, column=1, sticky=W)

frame4 = ttk.Frame(frame2)
frame4.grid(row=1, column=2)

label2 = ttk.Label(frame4, text='Angry')
label2.grid(row=0, column=0, sticky=W, padx=10)
label3 = ttk.Label(frame4, text='Disgust')
label3.grid(row=1, column=0, sticky=W, padx=10)
label4 = ttk.Label(frame4, text='Fear')
label4.grid(row=2, column=0, sticky=W, padx=10)
label5 = ttk.Label(frame4, text='Joy')
label5.grid(row=3, column=0, sticky=W, padx=10)
label6 = ttk.Label(frame4, text='Sadness')
label6.grid(row=4, column=0, sticky=W, padx=10)

scale1 = ttk.Scale(frame4, from_=0, to=100, orient=HORIZONTAL)
scale1.grid(row=0, column=1, sticky=W, padx=10)
scale2 = ttk.Scale(frame4, from_=0, to=100, orient=HORIZONTAL)
scale2.grid(row=1, column=1, sticky=W, padx=10)
scale3 = ttk.Scale(frame4, from_=0, to=100, orient=HORIZONTAL)
scale3.grid(row=2, column=1, sticky=W, padx=10)
scale4 = ttk.Scale(frame4, from_=0, to=100, orient=HORIZONTAL)
scale4.grid(row=3, column=1, sticky=W, padx=10)
scale5 = ttk.Scale(frame4, from_=0, to=100, orient=HORIZONTAL)
scale5.grid(row=4, column=1, sticky=W, padx=10)

entry1 = ttk.Entry(frame4, text='Angry', width=10)
entry1.grid(row=0, column=2, sticky=W, padx=10)
entry2 = ttk.Entry(frame4, text='Disgust', width=10)
entry2.grid(row=1, column=2, sticky=W, padx=10)
entry3 = ttk.Entry(frame4, text='Fear', width=10)
entry3.grid(row=2, column=2, sticky=W, padx=10)
entry4 = ttk.Entry(frame4, text='Joy', width=10)
entry4.grid(row=3, column=2, sticky=W, padx=10)
entry5 = ttk.Entry(frame4, text='Sadness', width=10)
entry5.grid(row=4, column=2, sticky=W, padx=10)

MyButton3 = ttk.Button(frame2, text='Analyse Text', width=12, command=callback)
MyButton3.grid(row=0, column=2, sticky=E)


# ........ Audio .........
label7 = ttk.Label(frame1, text='SPEECH ANALYSER')
label7.grid(row=0, columnspan=2)

frame5 = ttk.Frame(frame1)
frame5.grid(row=1, column=0)

label8 = ttk.Label(frame5, text='Select Audio File')
label8.grid(row=0, column=0)

audiofile.set('Audio 1')

popupMenu = tk.OptionMenu(frame5, audiofile, *choices)
popupMenu.grid(row=1, column=0)

frame6 = ttk.Frame(frame1)
frame6.grid(row=1, column=1)

label9 = ttk.Label(frame6, text='Angry')
label9.grid(row=0, column=0, sticky=W, padx=10)
label10 = ttk.Label(frame6, text='Disgust')
label10.grid(row=1, column=0, sticky=W, padx=10)
label11 = ttk.Label(frame6, text='Fear')
label11.grid(row=2, column=0, sticky=W, padx=10)
label12 = ttk.Label(frame6, text='Joy')
label12.grid(row=3, column=0, sticky=W, padx=10)
label13 = ttk.Label(frame6, text='Sadness')
label13.grid(row=4, column=0, sticky=W, padx=10)

scale6 = ttk.Scale(frame6, from_=0, to=100, orient=HORIZONTAL)
scale6.grid(row=0, column=1, sticky=W, padx=10)
scale7 = ttk.Scale(frame6, from_=0, to=100, orient=HORIZONTAL)
scale7.grid(row=1, column=1, sticky=W, padx=10)
scale8 = ttk.Scale(frame6, from_=0, to=100, orient=HORIZONTAL)
scale8.grid(row=2, column=1, sticky=W, padx=10)
scale9 = ttk.Scale(frame6, from_=0, to=100, orient=HORIZONTAL)
scale9.grid(row=3, column=1, sticky=W, padx=10)
scale10 = ttk.Scale(frame6, from_=0, to=100, orient=HORIZONTAL)
scale10.grid(row=4, column=1, sticky=W, padx=10)

entry6 = ttk.Entry(frame6, text="Angry1", width=10)
entry6.grid(row=0, column=2, sticky=W, padx=10)
entry7 = ttk.Entry(frame6, text="Disgust1", width=10)
entry7.grid(row=1, column=2, sticky=W, padx=10)
entry8 = ttk.Entry(frame6, text="Fear1", width=10)
entry8.grid(row=2, column=2, sticky=W, padx=10)
entry9 = ttk.Entry(frame6, text="Joy1", width=10)
entry9.grid(row=3, column=2, sticky=W, padx=10)
entry10 = ttk.Entry(frame6, text="Sadness1", width=10)
entry10.grid(row=4, column=2, sticky=W, padx=10)

MyButton4 = ttk.Button(frame5, text='Select', width=10, command=speechSelect)
MyButton4.grid(row=1, column=1, sticky=E)

photo = PhotoImage(file='microphone.png').subsample(35, 35)

label14 = ttk.Label(frame5, text='Click And Speak')
label14.grid(row=2, column=0, sticky=W, padx=10)

MyButton5 = ttk.Button(frame5, image=photo, command=speechAnalysis)
MyButton5.grid(row=2, column=1)

btn1.set('input')

# All Threads
ThreadReadFile = UpdateText()
ThreadReadFile.start()
'''
ThreadStartCam = WebCamInitializer()
ThreadStartCam.start()
'''
root.mainloop()