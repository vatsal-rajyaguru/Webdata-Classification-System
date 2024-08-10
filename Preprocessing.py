import os
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup

#directories should be inputed here (replace with your own directory paths), for both train and test sets
dir1=r'C:\Users\Richa\Downloads\A1\dataset\train\course'
dir2=r'C:\Users\Richa\Downloads\A1\dataset\train\faculty'
dir3=r'C:\Users\Richa\Downloads\A1\dataset\train\student'
dir4=r'C:\Users\Richa\Downloads\A1\dataset\test\course'
dir5=r'C:\Users\Richa\Downloads\A1\dataset\test\faculty'
dir6=r'C:\Users\Richa\Downloads\A1\dataset\test\student'

directories = [dir1,dir2,dir3,dir4,dir5,dir6]

#initializing stop words and stemming
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


#iterate through each directory to begin preprocessing
for directory in directories:
    # counter to keep track of how many files have been processed
    file_count = 0

    for filename in os.listdir(directory):
        if os.access(os.path.join(directory, filename), os.R_OK):

            with open(os.path.join(directory, filename), "r") as file:

                #using beautiful soup to extract text from webpages
                soup = BeautifulSoup(file, "html.parser")

                text = soup.getText()

                #preprocessing the words extracted

                text= text.lower()
                #remove special characters
                text = re.sub(r"[^a-zA-Z]+", " ", text)
                #tokenize the words
                words = word_tokenize(text)
                #for stop words
                words = [word for word in words if word not in stop_words]
                #for word stemming (might remove this if accuracy in models drop due to words being incorrectly stemmed)
                words = [stemmer.stem(word) for word in words]

                text = " ".join(words)

                #store the text into a new text file which are outputted into the directory they originate
                with open(os.path.join(directory, str(file_count) + ".txt"), "w") as text_file:
                    text_file.write(text)

            #increment the counter
            file_count += 1
        else:
            print("Error: cannot open", os.path.join(directory, filename))