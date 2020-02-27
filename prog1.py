import pandas as pd
import spacy,pickle
import numpy as np
from nltk.stem import SnowballStemmer


sb = SnowballStemmer("english")


train = pd.read_csv('train.csv').fillna(' ')
test = pd.read_csv('test.csv').fillna(' ')
train_text = train['question_text']
test_text = test['question_text']
text_list = pd.concat([train_text,test_text])
y = train['target'].values

num_train_data = y.shape[0]

'''
#only 6.2% positive examples.
cnt=0
for i in range(len(y)):
	if  y[i]==0:
		cnt+=1

print(cnt/len(y))
print(1-cnt/len(y))

'''

nlp = spacy.load('en', disable=['parser','ner','tagger'])
nlp.vocab.add_flag(lambda s:s.lower() in spacy.lang.en.stop_words.STOP_WORDS,spacy.attrs.IS_STOP)
word_dict={}
lemma_dict={}
word_index=1

docs=nlp.pipe(text_list,n_threads=2)
word_sequences=[]


for doc in docs:
	word_seq=[]
	for token in doc:
		if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
			word_dict[token.text]=word_index
			word_index+=1
			lemma_dict[token.text]=token.lemma_
		if (token.pos_ is not "PUNCT"):
			word_seq.append(word_dict[token.text])
	word_sequences.append(word_seq)



train_word_sequences=word_sequences[:num_train_data]
test_word_sequences=word_sequences[num_train_data:]




pickle.dump(train_word_sequences,open('train_word_sequences.txt','wb'))
pickle.dump(test_word_sequences,open('test_word_sequences.txt','wb'))
pickle.dump(word_dict,open('word_dict.txt','wb'))
pickle.dump(lemma_dict,open('lemma_dict.txt','wb'))



#word_sequences=pickle.load(open('word_sequences.txt','rb'))
#test_word_sequences=pickle.load(open('test_word_sequences.txt','rb'))

print(len(word_sequences))



#260974 unique words in dictionary.

def load_glove(word_dict,lemma_dict):
	EMBEDDING_FILE='glove.6B.300d.txt'

	def get_coefs(word,*arr):
		return word,np.asarray(arr,dtype='float32')

	embedding_index=dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
	embedding_size=300
	vocab_size=len(word_dict)+1

	embedding_matrix=np.zeros((vocab_size,embedding_size),dtype=np.float32)
	unknown_vector = np.zeros((embedding_size,), dtype=np.float32) - 1.
	print(unknown_vector.shape)

	for key in word_dict:
		word=key
		embedding_vector=embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[word_dict[key]]=embedding_vector
			continue

		word=key.lower()
		embedding_vector=embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[word_dict[key]]=embedding_vector
			continue

		word=key.upper()
		embedding_vector=embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[word_dict[key]]=embedding_vector
			continue

		word = key.capitalize()
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
		    embedding_matrix[word_dict[key]] = embedding_vector
		    continue

		word = sb.stem(key)
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
		    embedding_matrix[word_dict[key]] = embedding_vector
		    continue

		word = lemma_dict[key]
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
		    embedding_matrix[word_dict[key]] = embedding_vector
		    continue

		embedding_matrix[word_dict[key]] = unknown_vector                    
	return embedding_matrix


word_dict=pickle.load(open('word_dict.txt','rb'))
lemma_dict=pickle.load(open('lemma_dict.txt','rb'))
embedding_matrix = load_glove(word_dict, lemma_dict)



pickle.dump(embedding_matrix,open('embedding_matrix.txt','wb'))
print(embedding_matrix.shape)
