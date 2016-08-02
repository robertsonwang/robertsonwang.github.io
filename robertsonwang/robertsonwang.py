#Simple game to get our automated test up and running
class lex(object):

	def __init__(self, sentence):
		self.sentence = sentence
		self.words = self.sentence.split()
	
	def scan(self, sentence):
		words = sentence.split()
		direction = ['north', 'south', 'east'];
		verbs = ['go', 'kill', 'eat'];
		stops = ['the', 'in', 'of'];
		nouns = ['bear', 'princess'];
		result = []
		
		for item in words:
			if item in direction:
				result.append(('direction', item))
			elif item in verbs:
				result.append(('verb', item))
			elif item in stops:
				result.append(('stop', item))
			elif item in nouns:
				result.append(('noun', item))
			elif item.isdigit():
				result.append(('number', int(item)))
			else:
				result.append(('error', item))
        
		return result