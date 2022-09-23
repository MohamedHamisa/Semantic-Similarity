#Download ~800mb spacy model because it is MUCH more accurate at semantic similarity
!python -m spacy download en_core_web_lg

#Import model for similarity calculation
import en_core_web_lg
nlp = en_core_web_lg.load()

words = 'education university'
tokens = nlp(words)

print(tokens[0].text, tokens[1].text, tokens[0].similarity(tokens[1]))

####################################

#Use spacy to find similarities between tags
tag_list =  list(tags['tags_tag_name'])
#3871 contains nan so delete it
del tag_list[3871]

#Get rid of hyphens and turn the split words into an extra tag
corpus = ' '.join(list(tag_list)).replace('-',' ')
words = corpus.split()
corpus = " ".join(sorted(set(words), key=words.index))

#Apply the model on our dataset of tags
tokens = nlp(corpus)

#Convert tags into vectors for our clustering model
word_vectors = []
for i in tokens:
  word_vectors.append(i.vector)
word_vectors = np.array(word_vectors)
