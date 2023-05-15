import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time
import datetime
import pandas as pd
import speech_recognition as sr
from googletrans import Translator
import sounddevice as sd

# nltk.download('punkt')
# nltk.download('wordnet')

# Load the text file and preprocess the data
with open('buttbot.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()

sent_tokens = nltk.sent_tokenize(dataset)
word_tokens = nltk.word_tokenize(dataset)
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(tokens):
    return [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]

corpus = [" ".join(preprocess(nltk.word_tokenize(sentence))) for sentence in sent_tokens]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    # Preprocess user input
    user_input = " ".join(preprocess(nltk.word_tokenize(user_input)))

    # Vectorize user input
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and corpus
    similarities = cosine_similarity(user_vector, X)

    # Get index of most similar sentence
    idx = similarities.argmax()

    # Return corresponding sentence from corpus
    return sent_tokens[idx]

def preprocess_audio(sp_language, supported_languages):
    
    def chosen(language):
        if language in supported_languages.keys():
            selected_language_code = supported_languages[language]
            # print(selected_language_code)
            return selected_language_code

    sp_chosen_language = chosen(sp_language)
    # print(sp_chosen_language)

    def audio_prep():
        
        # Initialize recognizer class
        r = sr.Recognizer()

        # Configure the recognizer to use the selected language
        r.energy_threshold = 4000
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.0
        r.phrase_threshold = 0.3
        r.non_speaking_duration = 0.8
        r.operation_timeout = None

        # Record the speech
        with sr.Microphone(device_index=1) as source:
                r.adjust_for_ambient_noise(source)
                status_text = st.empty()
                status_text.info("Recording in progress...")
                audio = r.listen(source)
                status_text.empty()
                status_text.info("Here you go...")

        # Transcribe the speech
        try:
            text = r.recognize_google(audio, language=sp_chosen_language)
        except sr.UnknownValueError:
            st.warning("No speech detected.")
            text = ""
        return text
    return audio_prep()


# Create a Streamlit app with the updated chatbot function
def main():
    # Define supported languages
    supported_languages = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Japanese": "ja",
        "Korean": "ko",
        "Chinese": "zh-CN",
        "Yoruba": "yo"
    }

    st.title("Butt, the Bot")
    st.write("Hello there! My name is Butt, and I'm a simple Bot!\n\n You can ask me some simple data science or data science related terminologies, nothing more! \n\n You can also check out my chat history with other visitors. Cheers, my gee!")
    # Get the user's question
    

    def mode():
        question = ''
        mode_of_input = st.selectbox('How would you like to input your queries?', (None, 'Text', 'Speech'))

        # global question
        if mode_of_input == 'Text':
            question = st.text_input("You:")
        elif mode_of_input == 'Speech':
            # Add a dropdown menu to select the language
            sp_language = st.selectbox("Select your speaking language", list(supported_languages.keys()))
            if sp_language:
                if st.button('Start recording'):
                    question = preprocess_audio(sp_language, supported_languages)
                    if question != '':
                        st.write("Transcription: ", question)
        return question
    
    question = mode()

    # Create a button to submit the question
    if st.button("Submit"):
        with st.spinner('Generating response...'):
            time.sleep(2)
        response = chatbot_response(question)
        
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()