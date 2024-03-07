import tkinter as tk
from tkinter import ttk, scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import speech_recognition as sr

# Load the necessary data
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(sender="user", message=None):
    if message is not None and message.strip() != "":
        if sender == "user":
            user_message = f"You: {message}\n"
            text_box.configure(state='normal')
            text_box.insert(tk.END, user_message, 'user_message')
            entry_box.delete(0, tk.END)  # Clear entry box
            bot_response = get_bot_response(message)
            send_message("bot", bot_response)  # Specify sender as "bot"
        else:
            bot_response = f"Bot: {message}\n"
            text_box.insert(tk.END, bot_response, 'bot_response')
        
        text_box.see(tk.END)

def process_input():
    message = entry_box.get()
    if message.strip() != "":
        send_message("user", message)

def listen_and_send():
    query = listen()
    if query is not None:
        send_message("user", query)

def get_bot_response(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = r.listen(source, timeout=5)  # Set a timeout of 5 seconds
    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f"User: {query}")
        send_message("bot", query)  # Send recognized speech as a bot message
        return query
    except sr.WaitTimeoutError:
        print("Listening timed out")
    except Exception as e:
        print(e)
        print("Couldn't recognize the audio")
        
# Create GUI
root = tk.Tk()
root.title("ChatBot")
root.configure(bg="#E6F3FF")

# Add comforting messages as a heading
comforting_messages = [
    "You're not alone. I'm here to listen.",
    "It's okay not to be okay.",
    "You're stronger than you think.",
    "Take a deep breath. Everything will be okay.",
    "You're loved and valued."
]
random_comforting_message = random.choice(comforting_messages)
comforting_message_heading = tk.Label(root, text=random_comforting_message, font=('Arial', 16, 'bold'), bg="#E6F3FF", padx=10, pady=10)
comforting_message_heading.pack()

# Create and configure text box for messages
text_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=('Arial', 12))
text_box.pack(expand=True, fill='both', padx=10, pady=10)

# Create entry box for typing messages
entry_box = ttk.Entry(root, width=60, font=('Arial', 14))
entry_box.pack(pady=10, padx=10)

# Create a frame to hold the buttons
button_frame = tk.Frame(root, bg="#E6F3FF")
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Create listen button
listen_button = ttk.Button(button_frame, text="Listen", command=listen_and_send, width=20)
listen_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Create send button
send_button = ttk.Button(button_frame, text="Send", command=process_input, width=20)
send_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Start GUI
root.mainloop()
