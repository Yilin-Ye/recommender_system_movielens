import numpy as np

def bkdr2hash64(str01):
    mask60 = 0x0fffffffffffffff
    seed = 131
    hash = 0
    for s in str01:
        hash = hash * seed + ord(s)

    return hash & mask60

# Read files
def read_embedding_file(file):
    dic = dict()
    with open(file) as f:
        for line in f:
            tmp = line.split('\t')
            embedding = [float(_) for _ in tmp[1].split(',')]
            dic[tmp[0]] = embedding

    return dic

def get_hash2id(file):
    movie_dict = {}
    user_dict = {}
    with open(file) as f:
        for line in f:
            tmp = line.split(',')
            user_dict[str(bkdr2hash64('User_ID=' + tmp[0]))] = tmp[0]
            movie_dict[str(bkdr2hash64('Item_ID=' + tmp[1]))] = tmp[1]


    return user_dict, movie_dict


def split_user_movie(embedding_file, train_file):
    user_dict, movie_dict = get_hash2id(train_file)
    embedding_dict = read_embedding_file(embedding_file)

    movie_embedding = {}
    user_embedding = {}
    for k,v in embedding_dict.items():
        m_id = movie_dict.get(k, None)
        if m_id is not None:
            movie_embedding[m_id] = v
        u_id = user_dict.get(k, None)

        if u_id is not None:
            user_embedding[u_id] = v

    return movie_embedding, user_embedding

# Item to Item
def col_sim(movie_sim_movie_file, movie_embedding):
    wfile = open(movie_sim_movie_file, 'w')
    i = 0
    for m, vec1 in movie_embedding.items():
        sim_movie_tmp = {}
        for n, vec2 in movie_embedding.items():
            if m == n:
                continue
            sim_movie_tmp[n] = np.dot(np.asarray(vec2), np.asarray(vec1))
            i = i + 1

        sim_movie = sorted(sim_movie_tmp.items(), key=lambda _:_[1], reverse=True)
        sim_movie = [str(_[0]) for _ in sim_movie][:200]
        s = m + '\t' + ','.join(sim_movie) + '\n'
        print(s,"#"*10, ','.join(sim_movie))
        wfile.write(s)


# User to Item
def write_user_movie_embedding(movie_embedding_file, user_embedding_file, movie_embedding, user_embedding):
    wfile01 = open(movie_embedding_file, 'w')
    for k, v in movie_embedding.items():
        wfile01.write(k + '\t' + ','.join([str(_) for _ in v]) + '\n')

    wfile01.close()
    wfile02 = open(user_embedding_file, 'w')
    for k, v in user_embedding.items():
        wfile02.write(k + '\t' + ','.join([str(_) for _ in v]) + '\n')

    wfile02.close()


if __name__ == '__main__':

    # saved_embedding
    embedding_file = 'data/saved_embedding'
    train_file = 'data/raw_data/train_set'
    movie_embedding, user_embedding = split_user_movie(embedding_file, train_file)


    # For user to item
    movie_embedding_file = 'data/movie_embedding_file'
    user_embedding_file = 'data/user_embedding_file'
    write_user_movie_embedding(movie_embedding_file, user_embedding_file, movie_embedding, user_embedding)


    # For item to item
    movie_sim_movie_file = 'data/movie_sim_movie_file'
    col_sim(movie_sim_movie_file, movie_embedding)