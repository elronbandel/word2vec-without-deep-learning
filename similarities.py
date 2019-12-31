from collections import Counter, defaultdict
from operator import itemgetter, attrgetter
from typing import NamedTuple
from math import log2
from dataclasses import make_dataclass
from sparse import SparseMatrix
from timeit import timeit
import sparse

WORDS_THRESHOLD = 100
ATTS_THRESHOLD = 75
CONTEXTS_THRESHOLD = 100

tiny_sample_file = 'wikipedia.tinysample.trees.lemmatized'
sample_file = 'wikipedia.sample.trees.lemmatized'

Node = NamedTuple('Node', [('id', int), ('lemma', str), ('cpostag', str), ('head', int), ('deprel', str)])
NodeIndexes = [0, 2, 3, 6, 7]
NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
isnoun = lambda node: node.cpostag in NOUN_TAGS
CONTENT_TAGS = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
                'VBZ', 'WRB'}
iscontent = lambda node: node.cpostag in CONTENT_TAGS
PREP_TAGS = {'IN'}
isprep = lambda node: node.cpostag in PREP_TAGS

SEN, WIN, DEP = 0, 1, 2

class Indexer:
    def __init__(self, values_iter=None):
        self.indexes = defaultdict()
        self.values = []
        self.counter = Counter()
        if values_iter is not None:
            self.values = list(sorted(values_iter))
            self.indexes.update({(v, i) for i, v in enumerate(self.values)})

    def index(self, value):
        if value in self.indexes:
            i = self.indexes[value]
        else:
            i = len(self.values)
            self.values.append(value)
            self.indexes[value] = i
        self.counter[i] += 1
        return i






def load_data(file):
    trees = [[]]
    wordsc, tags, deprels = Counter(), set(), set()

    for line in open(file, encoding="utf8"):
        if line.strip():
            node = Node._make(itemgetter(*NodeIndexes)(line.split()))
            wordsc[node.lemma] += 1
            tags.add(node.cpostag), deprels.add(node.deprel)
            trees[-1].append(node)
        else:
            trees.append(list())
    words, tags, deprels = Indexer(filter(lambda w: wordsc[w] > CONTEXTS_THRESHOLD, wordsc.keys())), Indexer(tags), Indexer(deprels)
    words.counter = Counter(dict(map(lambda pair: (pair[1], wordsc[pair[0]]), words.indexes.items())))
    global words_contexts
    words_contexts = words
    iscommon = lambda node : words.counter[node.lemma] > WORDS_THRESHOLD
    return [[Node(int(node.id), words.index(node.lemma) if node.lemma in words.indexes else None, tags.index(node.cpostag),
                  int(node.head), deprels.index(node.deprel)) for node in tree] for tree in trees], words, tags, deprels, iscommon

def relax(words_contexts):
    for word in words_contexts.keys():
        words_contexts[word] = Counter(dict(words_contexts[word].most_common(CONTEXTS_THRESHOLD)))
    return words_contexts


def sentence_context_vectors(data, iscommon):
    global sentence_contexts_set
    sentence_contexts_set = set()
    words_contexts = defaultdict(Counter)
    for sentence in data:
        for word in filter(lambda w: w.lemma is not None and iscommon(w), sentence):
            for context in filter(lambda c: iscontent(c) and c.lemma is not None and c != word, sentence):
                    words_contexts[word.lemma][context.lemma] += 1
                    sentence_contexts_set.add(context.lemma)
    return relax(words_contexts)

def window(word, sentence):
    win = []
    i, taken = word.id - 1, 0
    while taken < 2 and i > 0:
        if iscontent(sentence[i]):
            win.append(sentence[i])
            taken += 1
        i -= 1
    i, taken = word.id + 1, 0
    while taken < 2 and i < len(sentence):
        if iscontent(sentence[i]):
            win.append(sentence[i])
            taken += 1
        i += 1
    return win

def window_context_vectors(data, iscommon):
    global window_contexts_set
    window_contexts_set = set()
    words_contexts = defaultdict(Counter)
    for sentence in data:
        for word in filter(lambda w: w.lemma is not None and iscommon(w), sentence):
            for context in filter(lambda c: c.lemma is not None, window(word, sentence)):
                words_contexts[word.lemma][context.lemma] += 1
                window_contexts_set.add(context.lemma)
    return relax(words_contexts)


def noun_child(prep, sentence):
    for word in sentence:
        if isnoun(word) and word.head == prep.id:
            return word


def dependencies(word, sentence, indexer):
    deps = []
    for context in sentence:
        if context.head == word.id and word.lemma is not None:
            if isprep(context):
               context = noun_child(context, sentence)
            if context:
                if context.lemma is not None:
                    deps.append(indexer.index((context.lemma, context.deprel, 1)))
        elif word.head == context.id and word.lemma is not None:
            deprel = word.deprel
            if isprep(context):
                deprel = context.deprel
                context = sentence[context.head - 1]
            if context.lemma is not None:
                deps.append(indexer.index((context.lemma, deprel, 0)))
    return deps

def dependency_context_vectors(data, iscommon):
    words_contexts = defaultdict(Counter)
    indexer = Indexer()
    for sentence in data:
        for word in filter(lambda w: w.lemma is not None and iscommon(w), sentence):
            for connection in dependencies(word, sentence, indexer):
                words_contexts[word.lemma][connection] += 1
    global dependency_contexts
    dependency_contexts = indexer
    return relax(words_contexts)



def calc_pmi(X_Y_C):
    P = SparseMatrix()
    X_C, Y_C = Counter(), Counter()
    x_y_sum = 0
    for x, x_Y_C in X_Y_C.items():
        for y, c in x_Y_C.items():
            X_C[x] += c
            Y_C[y] += c
            x_y_sum += c
    for x, x_Y_C in X_Y_C.items():
        for y, c in x_Y_C.items():
            P[x, y] = log2((c * x_y_sum) / (X_C[x] * Y_C[y]))
    return P


def most_similar(k, words, target, pmi):
    similarity = sparse.cosine(pmi, pmi[words.indexes[target]]).most_common(k + 1)
    return list(map(lambda pair: words.values[pair[0]], similarity))[1:]

index = lambda targets, mapper:  set(map(lambda t: mapper.indexes[t], targets))

context_vectors = {SEN : sentence_context_vectors, WIN : window_context_vectors, DEP : dependency_context_vectors}

if __name__ == '__main__':
    k = 20
    data, words, tags, deprels, iscommon = load_data(sample_file)
    CONTENT_TAGS, NOUN_TAGS, PREP_TAGS = index(CONTENT_TAGS, tags), index(NOUN_TAGS, tags), index(PREP_TAGS, tags)
    targets = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse',
               'fox', 'table', 'bowl', 'guitar', 'piano']
    pmi, similar, pmis = dict(), defaultdict(dict), defaultdict(dict)
    methods = [SEN, WIN, DEP]

    # pmi calculation and evaluation
    for method in methods:
        pmi[method] = calc_pmi(context_vectors[method](data, iscommon))
    with open('top{}contexts.txt'.format(k), 'w+') as output:
        for target in targets:
            output.write(target + '\n')
            top_contexts = dict()
            for method in methods:
                top_contexts[method] = list(map(itemgetter(0), pmi[method][words.indexes[target]].most_common(k + 1)))[1:]
            for i in range(k):
                word, deprel, direction = dependency_contexts.values[top_contexts[DEP][i]]
                output.write('{} {} {}_{}_{}\n'.format(words.values[top_contexts[SEN][i]], words.values[top_contexts[WIN][i]], words.values[word], deprels.values[deprel], direction))
            output.write("*********\n")
    # similarity evaluation and calculations
    for target in targets:
        for method in methods:
            similar[target][method] = most_similar(k, words, target, pmi[method])
    with open('top{}.txt'.format(k), 'w+') as output:
        for target in targets:
            output.write(target + '\n')
            for i in range(k):
                output.write("{} {} {}\n".format(similar[target][SEN][i], similar[target][WIN][i], similar[target][DEP][i]))
            output.write("*********\n")
    with open('counts_words.txt', 'w+') as output:
        for word, count in words_contexts.counter.most_common(50):
            output.write("{} {}\n".format(words.values[word], count))
    with open('counts_contexts_dep.txt', 'w+') as output:
        for context, count in dependency_contexts.counter.most_common(50):
            word, deprel, direction = dependency_contexts.values[context]
            output.write("{}_{}_{} {}\n".format(words.values[word], deprels.values[deprel], direction, count))

    # general statistics evaluations
    statistics = {
        'pmi_word_threshold': WORDS_THRESHOLD,
        'context_threshold': CONTEXTS_THRESHOLD,
        'attributes_threshold': ATTS_THRESHOLD,
        'number_of_pmi_words': len(pmi[SEN]._m),
        'number_of_sentence_context_attributes': len(sentence_contexts_set),
        'number_of_window_context_attributes' : len(window_contexts_set),
        'number_of_dependency_context_attributes' : len(dependency_contexts.values)
    }
    with open('statistics.txt', 'w+') as output:
        for field, val in statistics.items():
            output.write('{} : {}\n'.format(field, val))




