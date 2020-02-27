import pandas as pd
import spacy,pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



def predict_sentiment(ques):
	word_dict=pickle.load(open('word_dict.txt','rb'))
	text_list=[]
	text_list.append(ques)
	nlp = spacy.load('en', disable=['parser','ner','tagger'])
	nlp.vocab.add_flag(lambda s:s.lower() in spacy.lang.en.stop_words.STOP_WORDS,spacy.attrs.IS_STOP)

	docs=nlp.pipe(text_list,n_threads=2)

	word_sequences=[]

	for doc in docs:
		word_seq=[]
		for token in doc:
			if (token.text in word_dict) and (token.pos_ is not "PUNCT"):
				word_seq.append(word_dict[token.text])
			if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
				word_seq.append(0)
		word_sequences.append(word_seq)

	word_sequences=pad_sequences(word_sequences, maxlen=55, padding='post')
	model=load_model('model0.h5')
	pred_prob=np.squeeze(model.predict(word_sequences,verbose=1))
	print(pred_prob)
	if pred_prob>.35:
		print("Insincere question.")
	else:
		print("Genuine question.")
	


predict_sentiment("where is taj mahal located in India.")
