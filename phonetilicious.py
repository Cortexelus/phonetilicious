from nltk.corpus import wordnet as wn
from nltk.corpus import cmudict as cmu
from re import sub, split
from random import randrange
from numpy import *
from nltk import word_tokenize, sent_tokenize, pos_tag	


DEBUG = True

prondict = cmu.dict() 
Phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
num_phonemes = len(Phonemes)
max_synonyms = 100

# TODO
# consonance_only
# assonance_only
# look at first phoneme of each word
# look at first phoneme of each syllable
# look at first phoneme of stressed syllables
# swap out synonyms in same part of speech only
# only swap out Nouns, Adjectives, Verbs, Adverbs
# ignore most common words
# keep punctuation

#alliteration

# HEURISTIC. Efficient. Runs in n*m time. 

# given a sentence (list of Word Objects), returns phonetlicious score
def phonetilicious_score(loWO, flatten=False, normalize=False):
	
	phonevectors = map(lambda WO: WO["phonevector"], loWO)
	
	if flatten:
		# set each word's phoneme count to 0 or 1.
		phonevectors = map(lambda v: map(lambda p: 0 if p == 0 else 1, v),  phonevectors)
	
	# sum up phonevectors for each word
	vector_sum = reduce(lambda v,w: v + w, phonevectors)
	
	score = 0.0

	phone_count = 0
	for n in vector_sum:
		if n > 0:
			score += n * n * n 
			phone_count += n
	
	if normalize:
		score /= phone_count
	return score 
	
"""
Word Object looks like:
,
WO = {	'word': 'truly',
		'pos': 'RB',
		'phonevector': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0])}
"""
# given a phone ph, and list of Word Objects, returns the Word Object with max occurance of ph
def maxphone(ph, loWO):
	return max(loWO, key = lambda x: x['phonevector'][Phonemes.index(ph)])

def algo(text, alliteration_only = False, number_of_examples = 4, flatten = False, normalize = False, debug = False):
	
	DEBUG = debug
	
	out = []
	
	#sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
	sentences = sent_tokenize(text)
	for sentence in sentences: 
		words = extractwords(sentence)
		
		if DEBUG: 
			print words
		
		# for each word, find a list of related Word Objects
		lolorWO = map(lambda w: map(lambda rw: {"word": rw, "phonevector": phonevector(phones(rw), alliteration_only)}, related_words(w)), words) 
		
		if DEBUG: 
			print lolorWO
		
		loS = map(lambda p: map(lambda loWO: maxphone(p, loWO), lolorWO), Phonemes)
		
		if DEBUG:
			print loS
		
		sentences_with_score = map(lambda s: {'sentence': sentencify(s), 'score': phonetilicious_score(s, flatten, normalize)}, loS)
		
		indexes = unique(sentences_with_score, return_index=True)[1]
		sentences_unique = [sentences_with_score[index] for index in sorted(indexes)]
		# unique array, still sorted by score. 
		# http://stackoverflow.com/questions/12926898/numpy-unique-without-sort
		
		best_sentences = sorted(sentences_unique, key=lambda s: s['score'], reverse=True)
		
		if DEBUG:
			for s in best_sentences:
				print s
				
		# trim down number of examples
		best_sentences = best_sentences[:number_of_examples]
				
		if DEBUG:
			print "\nBEST SENTENCE:"
			print best_sentences[0]
			print "\n"
		
		for s in best_sentences:
			out.append(s['sentence'])
		
	return out

# given a list of Word Objects, returns sentence
def sentencify(loWO):
	words = map(lambda WO: WO['word'], loWO)
	return " ".join(words)

# given a word, return list of related words
def related_words(word):
	# synonyms
	syn_sets = wn.synsets(word)
	#syn_sets = hypernyms(word)
	words = []
	# TODO: pick synsets based on part of speech
	for syn_set in syn_sets:
		for w in syn_set.lemma_names:
			if w not in words:
				if w and ("_" not in w) and (w is not word):  # for now, ignore multi-word synonyms
					words.append(justlowerletters(w))
			elif len(words) >= max_synonyms:	
				break
		if len(words) >= max_synonyms:	
			break
	if not words:
		words = [word]
	return words
	
# given a word, return list of phones
def phones(word):
	if word in prondict:
		return prondict[word][0]
	else: # word is not in cmudict, return empty list
		return []
	
# given a list of phones, return phone vector.
def phonevector(lop, alliteration_only):
	vector = [0]*num_phonemes

	for p in lop:
		p = p[:2] # ignore stress markers
		vector[Phonemes.index(p)] += 1 #= 1 to flatten phonemes
		if alliteration_only:
			break
		
	return array(vector)

# given sentence, returns list of formatted words
# TODO: consider using a tokenizer
def extractwords(sentence):
	sentence = justlowerletters(sentence)
	return sentence.split(" ")
	
	
# given string, returns just a-z lowercase letters
def justlowerletters(string):
	return sub(r'\s+',' ',sub(r'[^a-z\s]', '', str.lower(string)))





	
	
