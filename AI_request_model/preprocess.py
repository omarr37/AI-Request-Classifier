import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # إزالة الرموز والعلامات غير الضرورية
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9أ-ي\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def apply_stemming(text):
    # تطبيق الـ Stemming على النص الإنجليزي فقط (بسيط)
    tokens = nltk.word_tokenize(text)
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

def apply_lemmatization(text):
    # تطبيق Lemmatization على النص الإنجليزي فقط
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

def preprocess_message(text):
    cleaned = clean_text(text)
    lemmatized = apply_lemmatization(cleaned)
    stemmed = apply_stemming(lemmatized)
    return stemmed
