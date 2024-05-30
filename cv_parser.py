import pymupdf
import re
import nltk
import os
import openai
import time
from dotenv import load_dotenv
from openai import OpenAI

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))
filepath = 'job_description.txt'

#Prepare the bert based NER model tonextract the skills from job description
model_name="dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_text_from_pdf(pdf_path):
    text=''
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            text+= page.get_text()+ '\n'
    return text

def preprocess_text(text):
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    # Remove newlines not followed by a capital letter or a number
    text = text.replace(' | ', ',')
    text = re.sub(r'\n(?![A-Z0-9])', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text

def get_name_title(text):
    #lines=text.split('\n')
    lines = [line.strip() for line in text.split('\n')]
    name = lines[0]
    title = lines[1].split(',')
    return name, title
    
def get_contact_info(text):
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone = re.search(r'\b\d(?:\s?\d){9}\b', text)
    return email.group(0) if email else None, phone.group(0) if phone else None

def get_skills(text, required_skills, stop_words):
    found_skills = set()
    #tokenize the text and remove stop words
    words = word_tokenize(text)
    words=[word.lower() for word in words]
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    #generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_words, 2, 3)))
    for word in filtered_words:
        if word in required_skills:
            found_skills.add(word)

    for ngram in bigrams_trigrams:
        if ngram in required_skills:
            found_skills.add(ngram)
    return found_skills


def get_job_description(filepath):
    with open(filepath, 'r') as file:
        content=file.read()
        return content

'''using the bert based model to do the NER task, 
parsing the necessaire skills according to the job description'''

def get_required_skills(text): 
  sentences = sent_tokenize(text)
  results=[]
  required_skills=[]
  for sentence in sentences:
    results.extend(nlp(sentence))
  
  for result in results:
    if not result['word'].startswith('##'):
      required_skills.append(result['word'])
  return required_skills

# Function to generate a cover letter using GPT-3.5-turbo
def generate_cover_letter(name, title, email, phone, skills):
    # Construct the prompt
    skills_str = ', '.join(skills)
    prompt = f"""
    Generate a professional cover letter for the following candidate:
    Name: {name}\nTitle: {title[0]}\nEmail: {email}\nPhone: {phone}\nSkills: {skills_str}
    """
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Choose the appropriate model
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=128,
                temperature=0.7
            )
            
            # Extract the cover letter from the response
            cover_letter = response.choices[0].message["content"].strip()
            return cover_letter
        except openai.RateLimitError as e :
            print(f"Rate limit exceeded. Attempt {attempt + 1} of {max_retries}. Retrying in {2 ** attempt} seconds...")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return "Unable to generate cover letter due to rate limits."


resume_raw = extract_text_from_pdf('/Users/xulitong/Desktop/YongXie.pdf')
resume = preprocess_text(resume_raw)
email, phone = get_contact_info(resume)
name, title = get_name_title(resume)
#print(f'name: {name}\ntitle: {title}\nphone: {phone}\nemail: {email}')
job_descrip = get_job_description(filepath)
required_skills = get_required_skills (job_descrip)
found_skills = get_skills(text=resume, required_skills=required_skills, stop_words=stop_words)

#print(f'required skills: {requirements}\nfound_skills: {found_skills}')


coverletter = generate_cover_letter(
        name=name,
        title=title,
        email=email,
        phone=phone,
        skills=found_skills,
    )
print(coverletter)