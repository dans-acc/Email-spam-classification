# Python modules
import os
import re
import exceptions
import string
from collections import Counter

# Natural language toolkit modules
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Math library
import numpy as np

# Machine learning imports
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

# Called in order to clean the email prior to adding it to the list of processed emails
def clean_email(lines, from_line):
	punctuation = string.punctuation
	stopwordz = stopwords.words("english")
	lemmatizer = WordNetLemmatizer()
	email = []
	for index in range(from_line, len(lines)):
		for word in lines[index].split():
			clean_word = "".join(char for char in word if char not in punctuation)
			if not clean_word.isalpha() or clean_word.isdigit() or len(clean_word) <= 2:
				continue
			if clean_word not in stopwordz:
				email.append(lemmatizer.lemmatize(clean_word))
	return email

# Called in order to read the emails from within a specific directory
def read_emails(dir_path, regex, from_line):
	emails, labels = [], []
	for file_name in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file_name)
		if not os.path.join(dir_path, file_name):
			continue
		label = 1 if regex.match(file_name) else 0
		try:
			with open(file_path, "r") as file:
				email = clean_email(file.readlines(), from_line)
				if len(email) > 0:
					emails.append(email)
					labels.append(label)
				file.close()
		except IOError as e:
			print "Failed to read file: {}, err: {}".format(file_name, e.strerror)
	assert len(emails) == len(labels)
	return emails, labels

# Create a dictionary with the most common words
def create_dictionary(emails, num_features):
	words = []
	for email in emails:
		words += email
	dictionary = Counter(words)
	return dictionary.most_common(num_features)

# Create a feature matrix based on the emails and the dictionary entries
def create_feature_matrix(emails, dictionary):
	matrix = np.zeros((len(emails), len(dictionary)), dtype=int)
	for i in range(len(emails)):
		for index, entry in enumerate(dictionary):
			matrix[i][index] = emails[i].count(entry[0])
	return matrix

# Run a classification model
def run_model(name, model_type, train_matrix, train_labels, test_matrix, test_labels, digits = 8):

	# Create, train, and run the model
	model = model_type()
	model.fit(train_matrix, train_labels)
	result = model.predict(test_matrix)

	# Print the results
	print "*" * 100
	print "\n",name,"Dataset"
	print "\nConfusion Matrix:\n"
	print metrics.confusion_matrix(test_labels, result)
	print "\nClassification Report:\n"
	print metrics.classification_report(test_labels, result, digits=digits)

# Suppress np version warnings
np.warnings.filterwarnings('ignore')

# Download the necessary nltk resources required for the algorithm
nltk.download("stopwords")
nltk.download("wordnet")

print "\n"

#---------------------- Ling dataset spam classification --------------------------

# Read the training and test set, whilst classifying and cleaning the email
spam_regex = re.compile(r"^(spmsg).*(\.txt)$")
train_set, train_labels = read_emails("datasets/ling/train-mails", spam_regex, 2)
test_set, test_labels = read_emails("datasets/ling/test-mails", spam_regex, 2)

print str(len(train_set) + len(test_set))
print str(len(train_set))
print str(len(test_set))

# Create a dictionary, and extract the feature matrix for the train and test sets
dictionary = create_dictionary(train_set, 4500)
train_matrix = create_feature_matrix(train_set, dictionary)
test_matrix = create_feature_matrix(test_set, dictionary)

# Run an SVM, and naive bayes classification models
run_model("SVM, with Ling", svm.LinearSVC, train_matrix, train_labels, test_matrix, test_labels)
run_model("Naive Bayes, with Ling", naive_bayes.MultinomialNB, train_matrix, train_labels, test_matrix, test_labels)

#---------------------- Enron dataset spam classification --------------------------

# Read and clean the enron ham emails
spam_regex = re.compile(r"^.*(\.spam\.txt)$")
ham_set, ham_labels = read_emails("datasets/enron/ham", spam_regex, 1)
spam_set, spam_labels = read_emails("datasets/enron/spam", spam_regex, 1)

print str(len(ham_set) + len(spam_set))

# When shuffling the data, we need to preserve the labels
ham = [tuple([ham_labels[i], ham_set[i]]) for i in range(len(ham_set))]
spam = [tuple([spam_labels[i], spam_set[i]]) for i in range(len(spam_set))]
train_split, test_split = model_selection.train_test_split(ham + spam, shuffle=True, test_size=0.40)

# Break down to obtain the train set, and train labels
train_set, train_labels = [], []
for labels, email in train_split:
	train_set.append(email)
	train_labels.append(labels)

# Break down the split
test_set, test_labels = [], []
for labels, email in test_split:
	test_set.append(email)
	test_labels.append(labels)

# Create a dictionary, and both a train and test feature matrix
dictionary = create_dictionary(train_set, 4500)
train_matrix = create_feature_matrix(train_set, dictionary)
test_matrix = create_feature_matrix(test_set, dictionary)

# Run an SVM, and naive bayes classification models
run_model("SVM, with Enron", svm.LinearSVC, train_matrix, train_labels, test_matrix, test_labels)
run_model("Naive Bayes, with Enron", naive_bayes.MultinomialNB, train_matrix, train_labels, test_matrix, test_labels)



