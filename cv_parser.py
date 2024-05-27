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

load_dotenv()

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))
filepath = 'job_description.txt'


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

def get_skills(text, requirements, stop_words):
    found_skills = set()
    #tokenize the text and remove stop words
    words = word_tokenize(text)
    words=[word.lower() for word in words]
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    
    #generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_words, 2, 3)))
    for word in filtered_words:
        if word in requirements:
            found_skills.add(word)

    for ngram in bigrams_trigrams:
        if ngram in requirements:
            found_skills.add(ngram)
    return found_skills

def get_requirements():
    input_string = input('Please enter the skills needed for the job split by commas: ')
    requirements = [word.strip() for word in input_string.split(',')]
    return requirements

def get_job_description(filepath):
    with open(filepath, 'r') as file:
        content=file.read()
        return content


# Function to generate a cover letter using GPT-3.5-turbo
def generate_cover_letter(name, title, email, phone, skills):
    # Construct the prompt
    skills_str = ', '.join(skills)
    prompt = f"""
    Generate a professional cover letter for the following candidate:
    Name: {name}\nTitle: {title}\nEmail: {email}\nPhone: {phone}\nSkills: {skills_str}
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


raw_text = extract_text_from_pdf('/Users/xulitong/Desktop/YongXie.pdf')
text = preprocess_text(raw_text)
email, phone = get_contact_info(text)
name, title = get_name_title(text)
#print(f'name: {name}\ntitle: {title}\nphone: {phone}\nemail: {email}')

requirements = get_requirements()
found_skills = get_skills(text=text, requirements=requirements, stop_words=stop_words)
#job_description = get_job_description(filepath)

#print(f'required skills: {requirements}\nfound_skills: {found_skills}')


coverletter = generate_cover_letter(
        name=name,
        title=title,
        email=email,
        phone=phone,
        skills=found_skills,
    )
print(coverletter)