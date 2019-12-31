#precision

def precision(retrieved, relevant):
    return len(set(retrieved).intersection(relevant)) / len(retrieved)


def avarage_precision(retrieved, relevant):
    return sum(map(lambda pair: precision(retrieved[:pair[0] + 1], relevant), filter(lambda pair: pair[1] in relevant, enumerate(retrieved)))) / len(relevant)


def mean_avarage_precision(retrieveds, relevants):
    return sum(avarage_precision(retrieved, set(relevant)) for retrieved, relevant in zip(retrieveds, relevants)) / len(retrieveds)

def load(file):
    return [[tokens.split() for tokens in line.split('|')] for line in open(file)]

if __name__ == '__main__':
    data = load('pdata.txt')
    topical, semantic, sentence, window, dependency = data
    print('SEN-MAP: topical:{}, semantic:{}'.format(mean_avarage_precision(sentence, topical),
                                                    mean_avarage_precision(sentence, semantic)))
    print('WIN-MAP: topical:{}, semantic:{}'.format(mean_avarage_precision(window, topical),
                                                    mean_avarage_precision(window, semantic)))
    print('DEP-MAP: topical:{}, semantic:{}'.format(mean_avarage_precision(dependency, topical),
                                                    mean_avarage_precision(dependency, semantic)))