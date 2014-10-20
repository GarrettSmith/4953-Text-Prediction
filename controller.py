# Garrett Smith
# 3018390
#
# Machine Learning
# ACS-4953
#
# This is the main logic of the project, this includes data downloading, model generation, and 
# text prediction.
# Defaults are all defined in variable below and can be overrriden by being passed to the object
# at instantiation or by changing them below.
#
# Example usage:
# controller = PredictionController(corpus=['webtext'])
# controller.run_test()


# default to floating point division
# Use // for integer division
from __future__ import division

from random import randint
import sys

import nltk
from nltk.tokenize import RegexpTokenizer

# The default copora to use, these can be any copora supported by nltk
# Examples include
# gutenberg
# webtext
DEFAULT_CORPORA = ['brown']

# The default n to create ngrams for
DEFAULT_NGRAMS = [2,3]

# DEFAULT_DICTIONARY = 'words'

# Only alphanum words, no punctuation and words with apostrophies
# DEFAULT_TOKENIZER = RegexpTokenizer(r'\w+\'\w+|\w+')
# Default and much more powerful text tokenizer
DEFAULT_TOKENIZER = nltk.PunktWordTokenizer()

# Used to greatly increase test speeds
SENTENCE_STOPS = ".!?\""

# Download required packages for nltk
def setup():
  nltk.download('punkt')

# The model object containing n-gram and word frequencies as well as the corpora and tokenizer to use
class PredictionModel(object):

  def __init__(self, **kwargs):
    # required setup
    print "Setup required packages"
    setup()

    self.test = kwargs.get('test', False)

    self.tokenizer = kwargs.get('tokenizer', DEFAULT_TOKENIZER)

    self.corpora = []

    self.text = kwargs.get('text', None)

    if self.text is None:
      self.text = {}
      # Use given corpora or the default
      print "Loading corpora"
      self._load_corpora(kwargs.get('corpora', DEFAULT_CORPORA))

    # count n grams
    print "Creating n-grams"
    self.frequency = self._create_ngrams(kwargs.get('ngrams', DEFAULT_NGRAMS))
    # count individual word occurence
    print "Counting individual words"
    self.frequency[1] = self._count_words()

    # Add words from the dictionary as a fallback
    # This greratly slows everything down and was removed
    # self.frequency[1].update(self._add_dictionary(kwargs.get('dictionary', DEFAULT_DICTIONARY)))

  def _load_corpora(self, copora_names):
    for corpus_name in copora_names:
      # Check if corpus is valid and download the corpus if needed
      if nltk.download(corpus_name):
        # Load the corpus
        corpus = getattr(nltk.corpus, corpus_name)
        self.corpora.append(corpus)
        self.text[corpus] = self._itokenized(corpus)

  def _count_words(self):
    counter = nltk.FreqDist()
    for text in self.text.values():
      # counter.update(self._itokenized(corpus))
      counter.update(text)
    return counter

  def _create_ngrams(self, ns):
    ngrams_dict = dict()
    for n in ns:
      for text in self.text.values():
        ngrams = nltk.util.ingrams(text, n)
        split_grams = ((ngram[:-1], ngram[-1]) for ngram in ngrams)
        ngrams_dict[n] = nltk.ConditionalFreqDist(split_grams)
    return ngrams_dict

  def _itokenized(self, corpus):
    return [w.lower() for w in corpus.words()]

  def _add_dictionary(self, dictionary_name):
    if nltk.download(dictionary_name):
      dictionary = getattr(nltk.corpus, dictionary_name)
      return dictionary.words()

# Contains all the actions that can be done using the information in the PredictionModel
# This includes predictions and running tests
class PredictionController(object):

  def __init__(self, **kwargs):
    # Create dataset
    self.model = PredictionModel(**kwargs)

  # Execute tests
  def run_test(self, tokens, random_select=False, threshold=1):
    print "Testing accuracy"

    correct = 0
    total = len(tokens)

    for i, w in enumerate(tokens):
      if i % 1000 == 0:
        sys.stdout.write('\r%d%%' %(i/total*100))
        sys.stdout.flush()

      text = tokens[max(0, i-2):i]
      text = ' '.join(text)
      prediction = self.predict(text, False, 10, threshold)

      if random_select:
        predicted_word = self.weighted_random_prediction(prediction)
      else:
        predicted_word = prediction.max()

      if w == predicted_word:
        correct += 1

    print 

    return correct / total

  # Convert a prediction into an ordered list
  def prediction_list(self, prediction):
    return (t[0] for t in sorted(prediction.iteritems(), key = lambda tup: tup[1], reverse=True))

  # Predict the next word with the given input text
  def predict(self, text, in_progress=False, max_predictions=10, min_occurences=1):

    # Find the last sentence end
    sent_start = max(text.rfind(c) for c in SENTENCE_STOPS) + 1
    text = text[sent_start:]

    text = text.lower()
    words = self.model.tokenizer.tokenize(text)

    if in_progress and len(words) > 0:
      current_word = words.pop()
    else:
      current_word = ''

    # What size of ngram should we use?
    n = min(max(self.model.frequency.keys()), len(words) + 1)
    n = max(n, 1)

    # print "current_word - {}".format(current_word)

    prediction = nltk.FreqDist()

    # until we  find an n that gives us predictions
    while len(prediction) == 0 and n >= 1:
      input_tuple = tuple(words[-n + 1:]) if n > 1 else tuple()
      # print "Input - {}".format(input_tuple)
      n_prediction = self._predict_next(n, input_tuple, current_word, max_predictions, min_occurences)
      prediction.update(n_prediction)
      n -= 1

    return prediction

  def _predict_next(self, n, input_tuple, current_word, max_predictions, min_occurences):
    prediction = nltk.FreqDist()
    freq = self.model.frequency[n]
    if n > 1:
      freq = self.model.frequency[n][input_tuple]
    matches = (w for w in freq.keys() if w.startswith(current_word))
    for w in matches:
      if len(prediction) >= max_predictions:
        break
      elif freq[w] >= min_occurences:
        prediction[w] = freq[w]
    return prediction

  # Randomly select a predicted word with probabilities weighted by frequency
  def weighted_random_prediction(self, prediction):
    total = prediction.N()
    choice = randint(0, total)
    for w in prediction:
      choice -= prediction[w]
      if choice <= 0:
        return w

def test1():
  corpora = ["webtext"]

  results = {}
  random_results = {}
  text = [w.lower() for w in nltk.corpus.webtext.words()]
  l = len(text)
  tenth = l // 10
  for i in xrange(10):

    print("%d/10" %i)
    test = text[tenth*i:tenth*(i+1)]
    train1, train2 = text[:tenth*i], text[tenth*(i+1):]
    c = PredictionController(text={'t1':train1, 't2':train2})
    results[i] = c.run_test(test)
    random_results[i] = c.run_test(test, True)

  print "max results"
  print results
  print "random results"
  print random_results

def test2():
  corpora = ["webtext"]
  thresholds = [2, 10, 100]

  results = {}
  text = [w.lower() for w in nltk.corpus.webtext.words()]
  l = len(text)
  tenth = l // 10

  for t in thresholds:
    results[t] = {}

  for i in xrange(10):
    test = text[tenth*i:tenth*(i+1)]
    train1, train2 = text[:tenth*i], text[tenth*(i+1):]
    c = PredictionController(text={'t1':train1, 't2':train2})
    for t in thresholds:
      print("%d/10" %i)
      results[t][i] = c.run_test(test, 10, t)

  print "max results"
  print results

  averages = {}
  for i in results.keys():
    averages[i] = sum(results[i].values())/10
  print averages

# Execute tests if run directly
if __name__ == '__main__':
  # print "max or weighted"
  # test1()
  # print 
  print "Thresholds"
  test2()

