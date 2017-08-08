from gensim.models.keyedvectors import KeyedVectors
from gensim.models import word2vec
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
word="Blusas to and are am I all we his her Femininas 2015 Fashion Tropical Women Blouses Sexy Lace Shirt Sleeveless Worsted Design Solid Pattern Female Party Blouse S-2XL Black"
word=word.lower()
import nltk
words=nltk.word_tokenize(word)
category="fashion"
for word in words:
    try:
        print("{} and {} = {}".format(category,word,model.wv.similarity(category,word)))
    except:
        print("Error",word)