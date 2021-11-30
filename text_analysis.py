from bs4 import BeautifulSoup
import re

def __strip_html(string):
    soup = BeautifulSoup(string, "html.parser")
    return soup.get_text()

def __remove_between_square_brackets(text):
  return re.sub('\[[^]]*\]', '', text)

def __remove_special_characters(text):
  pattern=r'[^a-zA-z0-9\s]'
  text=re.sub(pattern,'',text)
  return text

def denoise_text(text):
  text = __strip_html(text)
  text = __remove_between_square_brackets(text)
  text = __remove_special_characters(text)
  text = text.lower()
  return text
