import pyttsx3

engine = pyttsx3.init()
engine.say("Hello, Raspberry Pi speaking from Bluetooth speaker.")
engine.runAndWait()
