from tqdm import tqdm

from datasketch import MinHash, LeanMinHash, MinHashLSH
import pickle

import data_fns


def remove_odd_characters(text):
    ''' Removes punctuation and unknown characters. '''
    chars_to_remove = r'"#$%&\()*+/:;<=>@[\\]^_`{|}~.?,!\''

    text = text.replace("-\n", "").replace("\n", " ")
    text = text.translate(str.maketrans('', '', chars_to_remove))
    text = text.encode('ascii', 'ignore').decode()

    return text


def minhash(text, n_gram_size, num_hashes):
    ''' Returns hash object given a text. '''
    text = remove_odd_characters(text)

    m = MinHash(num_perm=num_hashes)
    words = text.split()

    if len(words) > n_gram_size:
        n_grams = list(zip(*[words[i:] for i in range(n_gram_size)]))
        n_grams = [" ".join(list(x)) for x in n_grams]
    # If text is too short, just hash the entire text
    else:
        n_grams = [text]

    m.update_batch([s.encode('utf-8') for s in n_grams])

    return LeanMinHash(m)


def get_hashes_batched(articles, n_gram_size, num_hashes):
    ''' Returns hashed articles. '''
    minhashes = []

    for article in tqdm(articles):
        minhashes.append(minhash(article, n_gram_size, num_hashes))

    return minhashes


def lsh_similar(minhashes, num_hashes, bands, rows, n_gram_size, texts):
    ''' Creates edges between articles given hashes. '''

    lsh = MinHashLSH(num_perm=num_hashes, params=(bands, rows))
    for i, hsh in enumerate(tqdm(minhashes)):
        # Check if duplicate of already seen item
        for j in lsh.query(hsh):
            yield (j, i)
        # Add to the seen items
        lsh.insert(i, hsh)


if __name__ == '__main__':

    corpus = data_fns.open_c4_by_url(pattern="patents.google.com", name="patents")
    # corpus = data_fns.get_super_glue()

    num_hashes = 30
    n_gram_size = 10

    hashes = get_hashes_batched(corpus, n_gram_size, num_hashes)

    similar_all = list(lsh_similar(hashes, num_hashes, 15, 2, n_gram_size, corpus))
    total_count = len(similar_all)
    print("Total Edges: ", total_count)

    with open(f'', 'wb') as f:
        pickle.dump(similar_all, f, protocol=4)

    

