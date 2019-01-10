
ESSAY_COUNT = 723

def read_all_essays():
    for i in range(1,ESSAY_COUNT+1):
        yield read_essay(i)

def read_essay(num):
    path = f"data/essays/essay_{num}.txt"
    return open(path,'r').read().lower()

