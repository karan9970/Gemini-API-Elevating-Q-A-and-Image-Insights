# Import necessary libraries and modules
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import pyttsx3

# Configure the GenerativeAI API key
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get responses for Q&A
model_qa = genai.GenerativeModel("gemini-pro")
chat_qa = model_qa.start_chat(history=[])

def get_gemini_response_qa(question):
    response_qa = chat_qa.send_message(question, stream=True)
    return response_qa

# Function to load OpenAI model and get responses for Image
def get_gemini_response_image(input, image): 
    model_image = genai.GenerativeModel('gemini-pro-vision')
    if input != "":
        response_image = model_image.generate_content([input, image])
    else:
        response_image = model_image.generate_content(image)
    return response_image.text

# Function to convert text to speech using pyttsx3
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Initialize the Streamlit app
st.set_page_config(page_title="Q&A and Image Demo")
st.header("Test model 1")

# Q&A Section
st.subheader("Q&A Section")
input_qa = st.text_input("Input (Q&A):", key="input_qa")
submit_qa = st.button("Ask the Question")

if submit_qa and input_qa:
    response_qa = get_gemini_response_qa(input_qa)
    st.subheader("Q&A Response:")
    for chunk_qa in response_qa:
        st.write(chunk_qa.text)

# Image Section
st.subheader("Image Section")
input_image = st.text_input("Input Prompt (Image):", key="input_image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit_image = st.button("Tell me about the image")

if submit_image:
    response_image = get_gemini_response_image(input_image, image)
    st.subheader("Image Response:")
    st.write(response_image)

    # Convert the response to speech
    text_to_speech(response_image)