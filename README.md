# Resume Analyzer and Cover Letter Generator
This Python script is designed to extract information from resume provided in PDF format, analyze the text to identify required skills, and generate a professional cover letter tailored to the job requirements. The script utilizes various libraries and an API key for OpenAI's GPT-3.5 model.

## Requirements
- Python 3
- Required Python libraries (pymupdf, re, nltk, os, openai, time, dotenv)
- OpenAI API key

## Installation
1. Clone the repository or download the script.
2. Install the required Python libraries using pip: ```pip install pymupdf nltk python-dotenv openai```
3. Set up a .env file in the project directory and provide your OpenAI API key: ```OPENAI_API_KEY=your_api_key_here``

## Usage
1. Ensure that the resume is available in PDF format.
2. Update the filepath variable in the script to point to the location of the PDF file.
3. Run the script using Python: ```python3 cv_parser.py```
4. Follow the prompts to enter the skills needed for the job and other necessary information.
5. The script will generate a cover letter based on the provided information and the identified skills.

## Functionality
1. PDF Text Extraction and Preprocessing
- The script utilizes the pymupdf library to extract text from the provided PDF file.
- Preprocessing steps include replacing non-breaking spaces, removing unnecessary characters, and ensuring proper formatting.

2. Information Extraction
- The script extracts candidate name, job title, contact information (email and phone) from the resume text.
- It prompts the user to enter the skills required for the job.

3. Skill Identification
- Using Natural Language Processing (NLP) techniques, the script identifies relevant skills from the job description text based on the provided requirements.
- Stop words are removed, and bigrams/trigrams are generated to capture multi-word skills (e.g., "artificial intelligence").

4. Cover Letter Generation
- The script utilizes OpenAI's GPT-3.5 model to generate a professional cover letter.
- It constructs a prompt based on the candidate's information and the identified skills, then sends it to the API for completion.
- If rate limits are reached, the script retries with an exponential backoff strategy.

## Credits
- This script was created by Litong XU.
- OpenAI's GPT-3.5 model powers the cover letter generation functionality.


