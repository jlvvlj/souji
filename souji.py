def process_tweet(tweet):
    import nltk
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    from nltk.tokenize import TweetTokenizer
    import string
    import re
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def build_frequency_list(tweets, labels):
    import numpy
    labels = numpy.squeeze(labels).tolist()
    freqs = {}
    for each_label, each_tweet in zip(labels, tweets):
        for each_word in process_tweet(each_tweet):
            labeled_word = (each_word, each_label)
            if labeled_word in freqs:
                freqs[labeled_word] += 1
            else:
                freqs[labeled_word] = 1
    return freqs