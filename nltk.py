from nltk import tokenize

text = 'これはサンプル文です。'
token = tokenize.word_tokenize(text)
print (token)