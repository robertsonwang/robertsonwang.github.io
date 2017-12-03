#Class set up:
#TextCleaner() - Contains all the data/methods for cleaning the self.text
#SumrGraph() - Contains all the data/methods for creating a summary using TextRank
#NaiveSumr() - Contains all the data/methods for creating a summary using naive BOW models
#LSASumr() - Contains all the data/methods for creating a summary using Latent Semantic Indexing

from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from collections import Counter
from math import fabs #This takes in an absolute value
import re
from re import split as regex_split, sub as regex_sub, UNICODE as REGEX_UNICODE
from itertools import combinations
import math
import numpy as np
import networkx as nx
import itertools

class TextCleaner():
#This class cleans the text and creates a tf-idf matrix
	def __init__(self, doc_list):
		self.text_dict = {}
		self.doc_list = doc_list
		self.sentence_list = []
		self.tf_idf = None

	def read_docs(self, base_path = 'quarter_example/'):
		text_dict = {}

		for doc in self.doc_list:
			with open(base_path + doc) as f:
				text_dict[doc] = f.read().decode('utf8')
				#Deal with ASCII characters
				#Deal with parenthesis
				text_dict[doc] = text_dict[doc].replace(u"\u2018", "'").replace(u"\u2019", "'")
				#Deal with quotes
				text_dict[doc] = text_dict[doc].replace(u"\u201c", '"').replace(u"\u201d", '"')
				#Deal with byte order mark
				text_dict[doc] = text_dict[doc].replace(u"\ufeff", "")
				#Remove new lines
				text_dict[doc] = text_dict[doc].replace('\n', '')

		self.text_dict = text_dict
		self.base_path = base_path

	def split_sentences(self, text):
	# The regular expression matches all sentence ending punctuation and splits the string at those points.
	# At this point in the code, the list looks like this ["Hello, world", "!" ... ]. The punctuation and all quotation marks
	# are separated from the actual self.text. The first s_iter line turns each group of two items in the list into a tuple,
	# excluding the last item in the list (the last item in the list does not need to have this performed on it). Then,
	# the second s_iter line combines each tuple in the list into a single item and removes any whitespace at the beginning
	# of the line. Now, the s_iter list is formatted correctly but it is missing the last item of the sentences list. The
	# second to last line adds this item to the s_iter list and the last line returns the full list.
		sentences = regex_split(u'(?<![A-Z])([.!?]"?)(?=\s+\"?[A-Z])',text,flags=REGEX_UNICODE)
		s_iter = zip(*[iter(sentences[:-1])] * 2)
		s_iter = [''.join(map(unicode,y)).lstrip() for y in s_iter]
		s_iter.append(sentences[-1])
		
		return s_iter

	def make_tf_idf(self):
		#Create a counter object (term frequency) with words from each document
		tf_idf = {}
		num_regex = re.compile('[0-9]')
		#Check to see if the text_dict object exists
		if self.text_dict:
			text_dict = self.text_dict
		else:
			text_dict = self.read_docs(self.doc_list, self.base_path)

		for doc in text_dict.keys():
			word_list = self.split_words(text_dict[doc])
			word_list = [x for x in word_list if not num_regex.match(x)]
			tf_idf[doc] = dict(Counter(x for x in word_list if x not in stopWords))
			
		#Get the inverse document measure
		word_count_dict = {}
		doc_list = tf_idf.keys()
		n = len(doc_list)
		for j in range(2, n+1):
			doc_combs = [comb for comb in combinations(doc_list, j)]
			for doc_comb in doc_combs:
				doc_set = [set(tf_idf[doc].keys()) for doc in doc_comb]
				comb_words = set()
				for word_set in doc_set:
					if not comb_words:
						comb_words = word_set
					else:
						comb_words = comb_words.intersection(word_set)
				try: 
					word_count_dict[j].append(comb_words)
				except KeyError:
					word_count_dict[j] = list(comb_words)
					
		#Adjust the tf_idf dict to reflect the new measure
		df = 1
		for doc in tf_idf:
			for word in tf_idf[doc].keys():
				for i in range(n,1,-1):
					if word in word_count_dict[i]:
						df = i
						break
				tf_idf[doc][word] = tf_idf[doc][word] * math.log(float(n)/float(df))
		
		self.tf_idf = tf_idf
		return tf_idf

	def split_words(self, text):
		#split a string into array of words
		try:
			text = regex_sub(r'[^\w ]', '', text, flags=REGEX_UNICODE)  # strip special chars
			return [x.strip('.').lower() for x in text.split()]

		except TypeError:
			print "Error while splitting characters"
			return None

class NaiveSumr():
#This class is inspired by the tf-idf algorithm put forth in Gong & Liu, 2001. 
#This also includes other features:
#1. sbs,summation based selection, (Hu, Sun and Lim) - Weight sentences greater if they contain more 
#"representative words", i.e. sentences with more keywords are weighted higher
#2. dbs, density-based selection - Similair to SBS, weight sentences greater that contain more representative words AND if those words are closer
#together in the sentence
#3. keywords - These contain manual dictionary entries for terms that are known to be useful for financial stablity/policy making
	def __init__(self, TextCleaner):
		self.TextCleaner = TextCleaner
		self.summary_dict = {}

	def Summarize(self, doc_key):
		summaries = []
		sentences = self.TextCleaner.split_sentences(self.TextCleaner.text_dict[doc_key])
		sentences = sentences[1:len(sentences)]
		keys = self.keywords(doc_key)

		if len(sentences) <= 5:
			return sentences

		#score setences, and use the top 5 sentences
		ranks = self.score(sentences, keys).most_common(5)
		sum_dict = {sentences.index(a):a for (a,b) in ranks}

		for key in sorted(sum_dict.keys()):
			summaries.append(sum_dict[key])

		self.summary_dict[doc_key] = summaries

		return summaries

	def score(self, sentences, keywords):
		#score sentences based on different features
		ranks = Counter()
		senSize = len(sentences)
		for i, s in enumerate(sentences):
			#i is in the index position in the list and s is the value itself
			sentence = self.TextCleaner.split_words(s)
			sentencePosition = self.sentence_position(i+1, senSize)
			sbsFeature = self.sbs(sentence, keywords)
			dbsFeature = self.dbs(sentence, keywords)

			#weighted average of scores from four categories
			totalScore = (1.0/2.0 * sbsFeature + 1.0/4.0 * dbsFeature + 1.0/4.0 * sentencePosition) / 4.0
			ranks[s] = totalScore

		return ranks

	def sentence_position(self, i, size):
		"""different sentence positions indicate different
		probability of being an important sentence"""

		normalized = i*1.0 / size
		if 0 < normalized <= 0.1:
			return 0.22
		elif 0.1 < normalized <= 0.2:
			return 0.22
		elif 0.2 < normalized <= 0.3:
			return 0.22
		elif 0.3 < normalized <= 0.4:
			return 0.08
		elif 0.4 < normalized <= 0.5:
			return 0.05
		elif 0.5 < normalized <= 0.6:
			return 0.04
		elif 0.6 < normalized <= 0.7:
			return 0.06
		elif 0.7 < normalized <= 0.8:
			return 0.04
		elif 0.8 < normalized <= 0.9:
			return 0.04
		elif 0.9 < normalized <= 1.0:
			return 0.03
		else:
			return 0

	def sbs(self, words, keywords):
	#Return Max(0, \frac{1}{ \frac{|# of Words in Sentence * Keyword Score|}{10}}
		score = 0.0
		if len(words) == 0:
			return 0
		for word in words:
			if word in keywords:
				score += keywords[word]
		return (1.0 / fabs(len(words)) * score)/10.0

	def dbs(self, words, keywords):
		if (len(words) == 0):
			return 0
		summ = 0
		first = []
		second = []

		for i, word in enumerate(words):
			if word in keywords:
				score = keywords[word]
				if first == []:
					first = [i, score]
				else:
					second = first
					first = [i, score]
					dif = first[0] - second[0]
					summ += (first[1]*second[1]) / (dif ** 2)

		# number of intersections
		k = len(set(keywords.keys()).intersection(set(words))) + 1
		return (1/(k*(k+1.0))*summ)

	def keywords(self, doc_key):
	# get the top 10 keywords and their frequency scores
	# ignores blacklisted words in stopWords,
	# counts the number of occurrences of each word. # We should pass this a tf idf matrix trained on the entire quarters corpus!

		keywords = self.TextCleaner.tf_idf[doc_key]
		keywords = {key: value for (key, value) in keywords.items() if value != 0.0}

		lm_words = ['loss', 'losses', 'decline', 'declined', 'declines', 'negative', 'lower', 'higher', 'raised', 'lowered']
		fed_words = ['foreign', 'exchange', 'interest', 'rates', 'rate', 'environment', 'charge', 'cost', 'tax',
		'multicurrency', 'equity', 'markets', 'conditions', 'stable', 'volatile', 'fluctuations', 'conditions',
		'risk', 'risky', 'currency', 'currencies', 'credit', 'market', 'VaR', 'VAR', 'capital'
		'RWA', 'RWAs', 'restructuring', 'federal', 'reserve', 'LIBOR', 'economy', 'economic', 'charge']

		max_key = max(keywords, key=keywords.get)

		for k in fed_words + lm_words:
			keywords[k] = keywords[max_key]

		return keywords

class SumrGraph():
#This class follows the algorithm put forth in Mihalcea and Tarau, 2004
	def __init__(self, TextCleaner):
		self.TextCleaner = TextCleaner
		self.summary_dict = {}
		self.graph = None
		self.text = ""

	def Summarize(self, doc_key):
		self.text = self.TextCleaner.text_dict[doc_key]
		self.build_graph()
		#This is a modified pagerank (Brin and Page, 1998) http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
		#We use sentence similarity as weights ala TextRank (Mihalcea and Tarau, 2005) https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
		calculated_page_rank = nx.pagerank(self.graph, weight='weight', alpha = 0.7)
		sentences = sorted(calculated_page_rank, key=calculated_page_rank.get,
				   reverse=True)
		#Add first sentence of earnings announcement
		self.summary_dict[doc_key] = sentences[0:5]
		
		return sentences[0:5]

	def build_graph(self):
		# Return a networkx graph instance.
		# input nodes: List of hashables that represent the nodes of a graph.

		gr = nx.Graph()  # initialize an undirected graph
		nodes = self.TextCleaner.split_sentences(self.text) #Create nodes using sentences from document
		gr.add_nodes_from(nodes)
		nodePairs = list(itertools.combinations(nodes, 2))

		# add edges to the graph (weighted by Levenshtein distance)
		for pair in nodePairs:
			firstString = pair[0]
			secondString = pair[1]
			levDistance = self.levenshtein_distance(firstString, secondString)
			gr.add_edge(firstString, secondString, weight=levDistance)

		self.graph = gr

	def levenshtein_distance(self, first, second):
		# Return the inverse Levenshtein distance between two strings.
		# We want lower levenshtein distances to imply greater sentence similarity
		# Based on:
		#	 http://rosettacode.org/wiki/Levenshtein_distance#Python

		if len(first) > len(second):
			first, second = second, first
		distances = range(len(first) + 1)
		for index2, char2 in enumerate(second):
			new_distances = [index2 + 1]
			for index1, char1 in enumerate(first):
				if char1 == char2:
					new_distances.append(distances[index1])
				else:
					new_distances.append(1 + min((distances[index1],
												 distances[index1 + 1],
												 new_distances[-1])))
			distances = new_distances
		return 1.0/float(distances[-1])

class LSASumr():
# #This class follows the algorithm put forth in Gong & Liu, 2001:
# #1. Decompose the document D into individual sentences, call this the candidate set S
# #2. Construct the terms by sentence matrix A for document D, rows are terms and columns are sentences
# #3. Perform the SVD on A to obtain the singular value matrix E and the right singular vector matrix V^T
# #4. Select the k'th right singular vector from matrix V^T
# #5. Select the sentence that has the highest loading on the k'th right singular vector
# #6. Repeat until you get a summary of desired length
# #Recall, SVD decomposes A = UEV^T, where E is a diagonal matrix and U and V^T are orthongonal matrices
# #Therefore, AA^T = UEV^TVE^TU^T = UE^2U^T and A^TA = VE^TU^TUEV^T = VE^2V^T.
# #E^2 is a diagonal matrix, therefore it must be the case that V contains the eigenvectors of A^TA
# #and U contains the eigenvectors of AA^T.
# #The elements of A^TA contain the dot products of each term across the document against the individual terms in each sentence,
# #Therefore we can think of A^TA as the "term covariance matrix" so that projecting the matrix onto a lower dimensional space
# #finds the "latent" terms or topics that underlie the document.
	def __init__(self, TextCleaner):
		self.TextCleaner = TextCleaner
		self.summary_dict = {}
		self.term_sentence_matrix = None
		self.word_index = None
		self.sentence_index = None
		self.eigenvalues = None
		self.rightsvd = None
		self.leftsvd = None

 	def make_term_sentence_matrix(self, doc_key):
 		num_regex = re.compile('[0-9]')
 		sentences = self.TextCleaner.split_sentences(self.TextCleaner.text_dict[doc_key])
 		text = self.TextCleaner.text_dict[doc_key]
 		words = self.TextCleaner.split_words(text)

 		#Exclude stop words
 		words = [word for word in words if word not in stopWords]
 		#Exclude numbers
 		words = [word for word in words if not num_regex.match(word)]

 		#Create a unique word dictionary index
 		self.word_index = {k:v for k,v in zip(range(len(words)), words)}
 		#Create a sentence index
 		self.sentence_index = {k:v for k,v in zip(range(len(sentences)), sentences)}

		doc_list = []
		for word in words:
		    word_list = []
		    df = 0
		    #Get term count measure and document frequency measure
		    for sentence in sentences:
		        sentence_words = self.TextCleaner.split_words(sentence)
		        if word in sentence_words:
		        	df += 1
		        	term_frequency = float(len([sentence_word for sentence_word in sentence_words if sentence_word == word]))
		        	sentence_length = float(len(sentence))
		        	#Scale term frequency by length of sentence 
		        	word_list.append(term_frequency/sentence_length)
		        else:
		        	word_list.append(0)
		    #Calculate inverse document frequency measure:
		    idf = math.log(float(len(sentences))/float(df))
		    word_list = [tf * idf for tf in word_list]
		    doc_list.append(word_list)
		ts_matrix = np.matrix(doc_list)

 		self.term_sentence_matrix = ts_matrix

 	def get_singular_vector(self, doc_key = None):
 		if self.term_sentence_matrix is not None:
 			u, s, v = np.linalg.svd(self.term_sentence_matrix)
 		elif doc_key is not None:
 			print "Making term sentence matrix..."
 			self.make_term_sentence_matrix(doc_key)
 		else:
 			print "No document passed"
 			return None

 		self.eigenvalues = s
 		self.rightsvd = v
 		self.leftsvd = u

 	def Summarize(self, doc_key):
 		#Select the k'th vector from the right singular vector
 		k = 5
 		summary = []
 		sentence_indices = []
  		#Selecting the "largest" loading is actually a subtle assumption 
 		#when it comes to semantic analysis. Do negative loadings
 		#mean antonyms? What does this mean from a direction interpretation?
 		for vec in self.rightsvd[0:k]:
 			sentence_index = abs(vec).argmax()
 			if sentence_index in sentence_indices:
 				sentence_index = np.argsort(abs(vec))[0,-2]
 			sentence_indices.append(sentence_index)

 		#Put the sentences in chronological order
 		for sentence in sorted(sentence_indices):
 			summary.append(self.sentence_index[sentence])

 		self.summary_dict[doc_key] = summary
 		return summary
