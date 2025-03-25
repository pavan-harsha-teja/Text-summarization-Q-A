from django.shortcuts import render, redirect , get_object_or_404
from django.contrib import messages
from userapp.models import *
import random
import urllib.parse, urllib.request, ssl
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
import urllib.request
import urllib.parse
from django.contrib.auth import logout
from django.core.mail import send_mail
import os
import random
from django.conf import settings
from userapp.models import *
from django.core.files.storage import default_storage



def generate_otp(length=4):
    otp = "".join(random.choices("0123456789", k=length))
    return otp


def user_logout(request):
    logout(request)
    messages.info(request, "Logout Successfully ")
    return redirect("user_login")

EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')

def index(request):
    # Retrieve all feedback entries, ordered by the most recent submission
    feedbacks = Feedback.objects.all().order_by('-submitted_at')
    return render(request, 'index.html', {'feedbacks': feedbacks})


def about(request):
    return render(request,'about.html')



def admin(request):
    return render(request,'admin.html')



def contact(request):
    return render(request,'contact.html')



def user_register(request):
    if request.method == "POST":
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password') 
        phone_number = request.POST.get('phone_number')
        age = request.POST.get('age')
        address = request.POST.get('address')
        photo = request.FILES.get('photo')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('user_register') 
        user = User(
            full_name=full_name,
            email=email,
            password=password, 
            phone_number=phone_number,
            age=age,
            address=address,
            photo=photo
        )
        otp = generate_otp()
        user.otp = otp
        user.save()
        subject = "OTP Verification for Account Activation"
        message = f"Hello {full_name},\n\nYour OTP for account activation is: {otp}\n\nIf you did not request this OTP, please ignore this email."
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [email]
        request.session["id_for_otp_verification_user"] = user.pk
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
        messages.success(request, "Otp is sent your mail and phonenumber !")
        return redirect("user_otp")
    return render(request,"user-register.html")



def user_otp(request):
    otp_user_id = request.session.get("id_for_otp_verification_user")
    if not otp_user_id:
        messages.error(request, "No OTP session found. Please try again.")
        return redirect("user_register")
    if request.method == "POST":
        entered_otp = "".join(
            [
                request.POST["first"],
                request.POST["second"],
                request.POST["third"],
                request.POST["fourth"],
            ]
        )
        try:
            user = User.objects.get(id=otp_user_id)
        except User.DoesNotExist:
            messages.error(request, "User not found. Please try again.")
            return redirect("user_register")
        if user.otp == entered_otp:
            user.otp_status = "Verified"
            user.save()
            messages.success(request, "OTP verification successful!")
            return redirect("user_login")
        else:
            messages.error(request, "Incorrect OTP. Please try again.")
            return redirect("user_otp")
    return render(request,"user-otp.html")



def user_login(request):
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        try:
            user = User.objects.get(email=email)
            if user.password != password:
                messages.error(request, "Incorrect password.")
                return redirect("user_login")
            if user.status == "Accepted":
                if user.otp_status == "Verified":
                    request.session["user_id_after_login"] = user.pk
                    messages.success(request, "Login successful!")
                    return redirect("user_dashboard")
                else:
                    new_otp = generate_otp()
                    user.otp = new_otp
                    user.otp_status = "Not Verified"
                    user.save()
                    subject = "New OTP for Verification"
                    message = f"Your new OTP for verification is: {new_otp}"
                    from_email = settings.EMAIL_HOST_USER
                    recipient_list = [user.email]
                    send_mail(
                        subject, message, from_email, recipient_list, fail_silently=False
                    )
                    messages.warning(
                        request,
                        "OTP not verified. A new OTP has been sent to your email and phone.",
                    )
                    request.session["id_for_otp_verification_user"] = user.pk
                    return redirect("user_otp")
            else:
                messages.success(request, "Your Account is Not Accepted by Admin Yet")
                return redirect("user_login")
        except User.DoesNotExist:
            messages.error(request, "No User Found.")
            return redirect("user_login")
    return render(request,"user-login.html")




def user_dashboard(request):
    return render(request,"user_dashboard.html")



import PyPDF2


def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with default_storage.open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
    return extracted_text

def pdf(request):
    if request.method == "POST" and request.FILES.get("pdf_file"):
        pdf_file = request.FILES["pdf_file"]
        file_path = default_storage.save("pdfs/" + pdf_file.name, pdf_file)
        full_path = os.path.join(settings.MEDIA_URL, file_path)

        print("PDF file selected:", pdf_file)
        print("Saved file path:", full_path)

        
        request.session["pdf_path"] = file_path  

        context = {"pdf_path": full_path}
        return render(request, "pdf.html", context)
    return render(request, "pdf.html")











import requests
import json
import re
def summarize_pdf(request):
    pdf_path = request.session.get("pdf_path")
    if not pdf_path:
        return redirect("pdf_upload")

    extracted_text = extract_text_from_pdf(pdf_path)
    summary = None
    selected_language = "English"  # default language
    languages = ["English", "Hindi", "Tamil", "Telugu", "Bengali", "Marathi", "Gujarati", "Kannada", "Malayalam"]

    if request.method == "POST":
        selected_language = request.POST.get("language", "English")
        try:
            headers = {
                "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            }

            # Truncate text to fit context window.
            truncated_text = extracted_text[:2000] + '...' if len(extracted_text) > 2000 else extracted_text

            # Build the prompt based on the language.
            if selected_language.lower() == "english":
                prompt = f"""Analyze this text and return ONLY a JSON object with:
                {{
                    "summary": "concise 3-5 sentence summary",
                    "key_points": [3-5 most important points],
                    "readability_score": 1-100
                }}
                Text: {truncated_text}"""
                system_content = "You are a document analysis expert. Return ONLY valid JSON."
            else:
                # For other languages, instruct the API to produce a summary in that language.
                prompt = f"Please provide a concise summary of the following text in {selected_language}: {truncated_text}"
                system_content = "Return a concise summary in the requested language."

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ]
                },
                headers=headers
            )

            if response.status_code == 200:
                if selected_language.lower() == "english":
                    # Extract JSON from the API response.
                    json_str = re.search(r'\{.*\}', response.json()['choices'][0]['message']['content'], re.DOTALL).group()
                    result = json.loads(json_str)
                    summary = {
                        "text": result["summary"],
                        "key_points": result["key_points"],
                        "score": result["readability_score"]
                    }
                else:
                    # For non-English, we expect a plain text summary.
                    summary = response.json()['choices'][0]['message']['content']
            else:
                summary = "Error generating summary. Please try again later."

        except Exception as e:
            summary = f"Summarization failed: {str(e)}"

    context = {
        "pdf_path": pdf_path,
        "extracted_text": extracted_text,
        "summary": summary,
        "selected_language": selected_language,
        "languages": languages,
    }
    return render(request, "summarization.html", context)




import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from django.http import JsonResponse

def qa_pdf(request):
    print("qa_pdf view called")
    pdf_path = request.session.get("pdf_path")
    if not pdf_path:
        print("No pdf_path found in session, redirecting to pdf_upload")
        return redirect("pdf_upload")
    
    print("pdf_path found:", pdf_path)
    extracted_text = extract_text_from_pdf(pdf_path)
    print("Extracted text length:", len(extracted_text))
    
    answer = None
    error = None

    if request.method == "POST":
        print("POST request received")
        question_text = ""
        
        # Check if an audio file was uploaded
        if request.FILES.get("audio"):
            print("Audio file found in POST")
            try:
                audio_file = request.FILES["audio"]
                audio_data = audio_file.read()
                audio_io = BytesIO(audio_data)
                audio_segment = AudioSegment.from_file(audio_io)
                wav_io = BytesIO()
                audio_segment.export(wav_io, format="wav")
                wav_io.seek(0)
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_io) as source:
                    audio_rec = recognizer.record(source)
                    question_text = recognizer.recognize_google(audio_rec)
                    print("Recognized question text from audio:", question_text)
            except Exception as e:
                error = f"Audio processing error: {str(e)}"
                print("Error during audio processing:", error)
        else:
            question_text = request.POST.get("question", "")
            print("Question text from POST:", question_text)

        if question_text:
            try:
                context_str = f"PDF Content: {extracted_text[:3000]}\n\nQuestion: {question_text}"
                print("Context for API:", context_str)
                
                headers = {
                    "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                print("Sending request to Perplexity API")
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system", 
                                "content": (
                                    "You are a PDF analysis assistant. Strictly follow these rules:\n"
                                    "1. Answer ONLY using information from the provided PDF content\n"
                                    "2. If the question cannot be answered using the PDF, respond: "
                                    "\"I can't answer that based on the document.\"\n"
                                    "3. Keep answers concise and under 100 words"
                                )
                            },
                            {
                                "role": "user",
                                "content": context_str
                            }
                        ],
                        "temperature": 0.2
                    },
                    headers=headers
                )
                print("API response status code:", response.status_code)
                if response.status_code == 200:
                    result = response.json()
                    raw_answer = result['choices'][0]['message']['content']
                    answer = re.sub(r'\n+', ' ', raw_answer).strip()
                    print("Cleaned answer:", answer)
                    if "can't answer" in answer.lower():
                        answer = "I can't answer that based on the document."
                        print("Answer updated to fallback response")
                else:
                    error = "Error generating answer. Please try again."
                    print("API response error:", error)
            except Exception as e:
                error = f"API Error: {str(e)}"
                print("Exception in API call:", error)

    context = {
        "pdf_path": pdf_path,
        "extracted_text": extracted_text,
        "answer": answer,
        "error": error
    }
    print("Returning context to template:", context)
    return render(request, "qa.html", context)

import os
import nltk

# Set NLTK data path (choose one appropriate for your system)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.environ['NLTK_DATA'] = nltk_data_path

# Download required datasets
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')



nltk.download('punkt')
nltk.download('stopwords')


import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def sentence_similarity(sent1, sent2):
    
    words1 = [word for word in word_tokenize(sent1.lower()) if word.isalnum()]
    words2 = [word for word in word_tokenize(sent2.lower()) if word.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    words1 = [word for word in words1 if word not in stop_words]
    words2 = [word for word in words2 if word not in stop_words]
    if not words1 or not words2:
        return 0
    
    common = set(words1) & set(words2)
    return float(len(common)) / (np.log(len(words1) + 1) + np.log(len(words2) + 1))

def build_similarity_matrix(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    return sim_matrix

def textrank_summarize(text, num_sentences=2):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text  
    sim_matrix = build_similarity_matrix(sentences)
    
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)  
    
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    selected_indices = sorted([i for (_, s, i) in ranked_sentences[:num_sentences]])
    summary = " ".join([sentences[i] for i in selected_indices])
    return summary

def text(request):
    if request.method == "POST":
        user_text = request.POST.get("text", "")
        print("Entered text:", user_text)
        summary = textrank_summarize(user_text, num_sentences=3)
        print("Summary:", summary)
        messages.success(request, "Text Summarized Successfully")
        return render(request, "text.html", {"summary": summary})
    return render(request, "text.html")




from django.utils.datastructures import MultiValueDictKeyError

def user_profile(request):
    user_id  = request.session.get('user_id_after_login')
    print(user_id)
    user = User.objects.get(pk= user_id)
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        try:
            profile = request.FILES['profile']
            user.photo = profile
        except MultiValueDictKeyError:
            profile = user.photo
        password = request.POST.get('password')
        location = request.POST.get('location')
        user.user_name = name
        user.email = email
        user.phone_number = phone
        user.password = password
        user.address = location
        user.save()
        messages.success(request , 'updated succesfully!')
        return redirect('user_profile')
    return render(request,'user-profile.html',{'user':user})


import re
from datetime import datetime
from django.shortcuts import render, redirect
def tlb(request):
    arranged_text = None
    error = None
    print("tlb view called.")
    
    if request.method == "POST":
        timeline_text = request.POST.get("text", "")
        print("Timeline text:", timeline_text)
        
        if timeline_text:
            try:
                # Call Perplexity API to process the text
                headers = {
                    "Authorization": f"Bearer {settings.PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                }
                print("Sending request to Perplexity API")
                
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    json={
                        "model": "sonar",
                        "messages": [
                            {"role": "system", "content": "Rearrange the given text into a well-structured chronological order. Ensure the output contains:\n"
                                                        "- Consistent numbering format (1., 2., 3.)\n"
                                                        "- Each entry starts with a bolded year range (e.g., **606 to 647 CE**)\n"
                                                        "- No extra symbols or unexpected characters\n"
                                                        "- A clean, readable format"},
                            {"role": "user", "content": timeline_text}
                        ],
                        "temperature": 0.2
                    },
                    headers=headers
                )
                
                print("API response status code:", response.status_code)
                
                if response.status_code == 200:
                    result = response.json()
                    raw_answer = result['choices'][0]['message']['content']
                    cleaned_answer = re.sub(r'\n+', '\n', raw_answer).strip()
                    
                    # Convert response into bullet points
                    arranged_text = "\n".join(["• " + line for line in cleaned_answer.split("\n")])
                    print("Arranged text:", arranged_text)
                
                else:
                    error = "Error generating answer. Please try again."
                    print("API response error:", error)
            
            except Exception as e:
                error = f"Error processing request: {str(e)}"
                print("Exception:", error)
        
        else:
            error = "No timeline text provided."
            print("Timeline text is empty.")
    
    context = {"arranged_text": arranged_text, "error": error}
    return render(request, "tlb.html", context)













from io import TextIOWrapper
import pickle
import joblib
from transformers import AlbertTokenizer
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Feedback, User


print("Loading tokenizer...")
loaded_tokenizer = AlbertTokenizer.from_pretrained('amazone review/albert_tokenizer')


print("Loading label encoder...")
label_encoder = joblib.load('amazone review/label_encoder.joblib')


print("Loading model architecture...")
model_architecture_path = 'amazone review/bylstm_model_architecture.json'
with open(model_architecture_path, 'r') as json_file:
    loaded_model_json = json_file.read()


print("Loading model weights...")
model_weights_path = 'amazone review/bylstm_model_weights.h5'
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(model_weights_path)

max_len = 256


def predict_sentiment(text):
    print(f"Predicting sentiment for text: {text}")

    
    print(f"Tokenizing and padding input text...")
    sequences = [loaded_tokenizer.encode(text, max_length=max_len, truncation=True, padding='max_length')]
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    
    print(f"Making prediction using the model...")
    predictions = loaded_model.predict(padded_sequences)
    print(f"Model prediction output: {predictions}")

    
    predicted_label = label_encoder.inverse_transform(predictions.argmax(axis=1))[0]
    print(f"Predicted sentiment label: {predicted_label}")
    
    
    return predicted_label


def feedback(request):
    user_id = request.session.get('user_id_after_login')
    print(f"User ID from session: {user_id}")

    if request.method == 'POST':
        print(f"Processing POST request...")
        
        user = get_object_or_404(User, pk=user_id)

      
        user_name = request.POST.get('user_name')
        user_email = request.POST.get('user_email')
        rating = request.POST.get('rating')
        additional_comments = request.POST.get('additional_comments')

        print(f"Feedback form data: user_name={user_name}, user_email={user_email}, rating={rating}, additional_comments={additional_comments}")

        
        print(f"Performing sentiment analysis...")
        sentiment = predict_sentiment(additional_comments)  
        print(f"Predicted sentiment: {sentiment}")

        
        
        if sentiment == 1:
            sentiment_label = "neutral"
        elif sentiment == 2:
            sentiment_label = "positive"
        else:
            sentiment_label = "negative"
        
        print(f"Mapped sentiment label: {sentiment_label}")

        
        print(f"Saving feedback to the database...")
        feedback = Feedback.objects.create(
            user=user,
            user_name=user_name,
            user_email=user_email,
            rating=rating,
            additional_comments=additional_comments,
            sentiment=sentiment_label  
        )
        print(f"Feedback saved: {feedback}")

        
        messages.success(request, "Your feedback has been submitted successfully.")
        return redirect('feedback')

    print("Returning feedback page...")
    return render(request, "user-feedback.html")
