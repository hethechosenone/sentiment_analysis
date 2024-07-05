import math, re

# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        return self.mfc

# A Sentiment Lexicon baseline
class Lexicon:
    def __init__(self, test_texts, neg_words, pos_words):
        self.final_class = []
        self.detrmine_class(test_texts, neg_words, pos_words)

    def detrmine_class(self, test_texts, neg_words, pos_words):
      
        for tweet in test_texts:
            tweet_list = tokenize(tweet)
            positive = 0
            negative = 0            
            for word in tweet_list:
                if word in neg_words:
                    negative+=1
                if word in pos_words:
                    positive+=1
            if negative == positive:
                self.final_class.append('neutral')

            elif positive > negative:
                self.final_class.append('positive')

            elif negative > positive:
                self.final_class.append('negative')
    
    def classify(self):
        return self.final_class

# A Multinomial Naive Bayes Implementation.
class NaiveBayes:
    def __init__(self, train_texts, klasses):
        self.final_dict = {}
        self.result = []
        self.vocab = []
        self.train(train_texts, klasses)

    def train(self, train_texts, klasses):
        count = 0
        for k in klasses:
            if k not in self.final_dict.keys():
                self.final_dict[k] = {}
                self.final_dict[k]["class_count"] = 1
                self.final_dict[k]["tokens"] = {}
                tweet = tokenize(train_texts[count])
                for token in tweet:
                    if token not in self.vocab:
                        self.vocab.append(token)
                    if token not in self.final_dict[k]["tokens"].keys():
                        self.final_dict[k]["tokens"][token] = 1
                    else:
                        self.final_dict[k]["tokens"][token] += 1


            elif k in self.final_dict.keys():
                self.final_dict[k]["class_count"] += 1
                tweet = tokenize(train_texts[count])
                for token in tweet:
                    if token not in self.vocab:
                        self.vocab.append(token)
                    if token not in self.final_dict[k]["tokens"].keys():
                        self.final_dict[k]["tokens"][token] = 1
                    else:
                        self.final_dict[k]["tokens"][token] += 1

            count+=1

        self.vocab_size = 0

        for key in self.final_dict:
            self.final_dict[key]["token_size"] = sum(self.final_dict[key]["tokens"].values())
            self.vocab_size += len(self.final_dict[k]["tokens"])
            self.final_dict[key]["logprior"] = math.log(self.final_dict[key]["class_count"]) - math.log(count)

    def test_naive_bayes(self, test_texts):
        for tweet in test_texts:
            best_prob = float("-inf")
            best_klass = "neutral"
            tweet_list = tokenize(tweet)
            flag = False  
            for key in self.final_dict:
                total_prob = self.final_dict[key]["logprior"]
                for word in tweet_list:
                    if word in self.vocab:
                        flag = True
                        if word in self.final_dict[key]["tokens"].keys():
                            total_prob += (math.log(self.final_dict[key]["tokens"][word] + 1) - math.log(self.final_dict[key]["token_size"] + self.vocab_size))
                        else:
                            total_prob += (math.log(0 + 1) - math.log(self.final_dict[key]["token_size"] + self.vocab_size))

                if flag:
                    if total_prob > best_prob:
                        best_prob = total_prob
                        best_klass = key
            self.result.append(best_klass)


    
    def classify(self):
        return self.result


# A Binarized Multinomial Naive Bayes Implementation.
class BinarizedNaiveBayes:
    def __init__(self, train_texts, klasses):
        self.final_dict = {}
        self.result = []
        self.vocab = []
        self.train(train_texts, klasses)

    def train(self, train_texts, klasses):
        count = 0
        for k in klasses:
            if k not in self.final_dict.keys():
                self.final_dict[k] = {}
                self.final_dict[k]["class_count"] = 1
                self.final_dict[k]["tokens"] = {}
                tweet = tokenize(train_texts[count])
                tweet = list(dict.fromkeys(tweet)) # Removing deplicates so the count of each token in a document is one
                for token in tweet:
                    if token not in self.vocab:
                        self.vocab.append(token)
                    if token not in self.final_dict[k]["tokens"].keys():
                        self.final_dict[k]["tokens"][token] = 1
                    else:
                        self.final_dict[k]["tokens"][token] += 1


            elif k in self.final_dict.keys():
                self.final_dict[k]["class_count"] += 1
                tweet = tokenize(train_texts[count])
                tweet = list(dict.fromkeys(tweet)) # Removing deplicates so the count of each token in a document is one
                for token in tweet:
                    if token not in self.vocab:
                        self.vocab.append(token)
                    if token not in self.final_dict[k]["tokens"].keys():
                        self.final_dict[k]["tokens"][token] = 1
                    else:
                        self.final_dict[k]["tokens"][token] += 1

            count+=1

        self.vocab_size = 0

        for key in self.final_dict:
            self.final_dict[key]["token_size"] = sum(self.final_dict[key]["tokens"].values())
            self.vocab_size += len(self.final_dict[k]["tokens"])
            self.final_dict[key]["logprior"] = math.log(self.final_dict[key]["class_count"]) - math.log(count)

    def test_naive_bayes(self, test_texts):
        for tweet in test_texts:
            best_prob = float("-inf")
            best_klass = "neutral"
            tweet_list = tokenize(tweet)
            flag = False  
            for key in self.final_dict:
                total_prob = self.final_dict[key]["logprior"]
                for word in tweet_list:
                    if word in self.vocab:
                        flag = True
                        if word in self.final_dict[key]["tokens"].keys():
                            total_prob += (math.log(self.final_dict[key]["tokens"][word] + 1) - math.log(self.final_dict[key]["token_size"] + self.vocab_size))
                        else:
                            total_prob += (math.log(0 + 1) - math.log(self.final_dict[key]["token_size"] + self.vocab_size))

                if flag:
                    if total_prob > best_prob:
                        best_prob = total_prob
                        best_klass = key
            self.result.append(best_klass)


    
    def classify(self):
        return self.result

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]
    
    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]

    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        train_counts = count_vectorizer.fit_transform(train_texts)

        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)

        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        test_counts = count_vectorizer.transform(test_texts)
        # Predict the class for each test document
        results = clf.predict(test_counts)

    elif method == 'lexicon':
        pos_words_fname = 'pos-words.txt'
        neg_words_fname = 'neg-words.txt'

        pos_words = [x.strip() for x in open(pos_words_fname,
                                           encoding='utf8')]
        neg_words = [x.strip() for x in open(neg_words_fname,
                                             encoding='utf8')]

        classifier = Lexicon(test_texts, neg_words, pos_words)
        results = classifier.classify()

    elif method == 'nb':
        if len(train_texts) != len(train_klasses):
            print("The Training Class size does not match the Training Document\n")
            quit()
        classifier = NaiveBayes(train_texts, train_klasses)
        classifier.test_naive_bayes(test_texts)
        results = classifier.classify()

    elif method == 'nbbin':
        if len(train_texts) != len(train_klasses):
            print("The Training Class size does not match the Training Document\n")
            quit()
        classifier = BinarizedNaiveBayes(train_texts, train_klasses)
        classifier.test_naive_bayes(test_texts)
        results = classifier.classify()

    for r in results:
        print(r)
