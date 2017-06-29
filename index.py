import nltk, os
from flask import Flask, jsonify,render_template
from nltk import word_tokenize
from sklearn.externals import joblib

nermap = {"B-PER": "Persoon","I-PER": "Persoon", "B-LOC": "Location","B-ORG": "Organization", "O": "Word"}
app = Flask(__name__)
nltk.data.path.append(app.root_path)
tagger = nltk.data.load('./conll2002_ned_IIS.pickle')
crf = joblib.load(os.path.join(app.root_path, "./Dutch.pkl"))

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


@app.route('/ner/<sen>')
def ner(sen):
    tagged = [tagger.tag(word_tokenize(sen))]
    p = crf.predict([sent2features(s) for s in tagged])
    r = []
    e = tagged[0]
    for i in range(len(e)):
        tag = p[0][i]
        parts = tag.split("-")
        if len(parts)>1:
            what = parts[1]
        else:
            what = parts[0]
        if what != "O":
            r.append({"word": e[i][0], "pos": e[i][1], "entity": what})
    return jsonify(r)


@app.route('/')
def all():
    return render_template('t1.html', content="")


if __name__ == '__main__':
    app.run(host="0.0.0.0")
