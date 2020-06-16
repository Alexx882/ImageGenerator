import os

def purify():
    '''
    loads the 'sources.txt' files and removes every duplicate from it
    '''

    sfile = open("crawler/sources.txt")
    data = sfile.read().split("\n")

    print("initially {0} urls in the dataset".format(len(data)))

    data = list(dict.fromkeys(data))

    print("after pruning {0} urls in the dataset".format(len(data)))

    sfile.close()

    tfile = open("crawler/sources.txt", "w")
    tfile.write('\n'.join(data))
    tfile.close()

if __name__ == '__main__':
    purify()