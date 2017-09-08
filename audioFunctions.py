import json, os, threading
from watson_developer_cloud import ConversationV1
from watson_developer_cloud import ToneAnalyzerV3
import numpy as np
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
from PIL import Image
import speech_recognition as sr
r = sr.Recognizer()

tones = {'Anger':0.0, 'Disgust':0.0, 'Fear':0.0, 'Joy':0.0, 'Sadness':0.0}
tone_analyzer = ToneAnalyzerV3(
    username='1d3684c8-af39-4908-be61-fc45f1f579d9',
    password='t5ZBkiLZaz8S',
    version='2016-02-11')

workspace_id = os.environ.get('WORKSPACE_ID') or 'YOUR WORKSPACE ID'

maintainToneHistoryInContext = True

payload = "who authorised the unlimited expense account"
done = False

def invokeToneConversation():
    global done
    done = False
    tone = tone_analyzer.tone(text=payload)
    # print(tone["document_tone"]["tone_categories"])
    tones["Anger"] = tone["document_tone"]["tone_categories"][0]["tones"][0]["score"] *100
    tones["Disgust"] = tone["document_tone"]["tone_categories"][0]["tones"][1]["score"] *100
    tones["Fear"] = tone["document_tone"]["tone_categories"][0]["tones"][2]["score"] *100
    tones["Joy"] = tone["document_tone"]["tone_categories"][0]["tones"][3]["score"] *100
    tones["Sadness"] = tone["document_tone"]["tone_categories"][0]["tones"][4]["score"] *100
    print(payload)
    done = True
    for k,v in tones.items():
        print(k, " : ", round(v, 2), "%")


audioforSpeechToText = None
class GoogleTextToSpeech(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        try:
            global converted
            converted = False
            print("Inside")
            print(audioforSpeechToText)
            text = r.recognize_google(audioforSpeechToText)
            print("Transcription: " + text)
            file = open("Transcript.txt", "w")
            file.write(text)
            file.close()
        except:
            print("Could not understand audio")