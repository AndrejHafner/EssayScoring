import os
import random
import time

ESSAY_COUNT = 723

def read_all_essays():
    for i in range(1,ESSAY_COUNT+1):
        yield read_essay(i)

def read_essay(num):
    path = f"data/essays/essay_{num}.txt"
    return open(path,'r').read().lower()

def read_books():
    files = os.listdir("data/books")
    for filename in files:
        try:
            yield open(f"data/books/{filename}","r").read().lower()
        except:
            continue

def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def random_date(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y', prop)

def random_time(start, end, prop):
    return strTimeProp(start, end, '%H:%-M', prop)
