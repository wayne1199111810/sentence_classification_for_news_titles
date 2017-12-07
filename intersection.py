w2v_file = 'wrong_w2v_logist'
bow_file = 'wrong_bow_logist'
cnn_file = 'wong_cnn'

def getLines(file_name):
    lines = set()
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            idx = line.find('::')
            lines.add(line[idx+2:])
    return lines

# a = getLines(w2v_file)
# b = getLines(bow_file)
    # c = getLines(cnn_file)
# ab = a.intersection(b)
    # bc = b.union(c)
    # ac = a.union(c)
    # abc = ab.union(c)
# print('a V b: ' + str(float(len(ab))/len(a)))
# print('a V b: ' + str(float(len(ab))/len(b)))

    # print('b V c: ' + str(float(len(bc))/(len(b)+len(c))))
    # print('a V b: ' + str(float(len(ac))/(len(a)+len(c))))
    # print('a V b V c:' + str(float(len(abc))/ (len(a) + len(b) + len(c)) ))

category = ['Business', 'Games', 'Health', 'Science']

label_enc = {
    'Business':0,
    'Games':1,
    'Health':2,
    'Science':3
}

num_instance = 6688

def countCorrectInFirstK(file_name):
    count = dict()
    for i in range(len(category)):
        count[i+1] = 0

    k = 3
    outfile = open(file_name+'_out'+str(k), 'w', encoding = 'utf-8')

    with open(file_name, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            idx = line.find(']')
            array = map(float, line[1:idx].strip())

            idx_1 = line.find(']')
            array = list(map(float,line[1:idx_1].strip().split()))
            idx_2 = line.find('::')
            label = line[idx_2+2:].split()[0]
            value = array[label_enc[label]]
            firstk = len(list(filter(lambda x: x>=value, array)))
            count[firstk] += 1
            line_count += 1
            if firstk == k:
                outfile.write(line)

    outfile.close()

    print(count)
    print(line_count)
    count[1] = num_instance
    print(count)

    for i in range(len(category)-1):
        count[1] -= count[i+2]
    for i in range(1,len(category)+1):
        acc = count[1]/num_instance
        for j in range(2,i+1):
            acc += count[j]/num_instance
        print('accuracy of first ' + str(i) + ' is ' + str(acc))

if __name__ == '__main__':
    print(w2v_file)
    countCorrectInFirstK(w2v_file)
    print(bow_file)
    countCorrectInFirstK(bow_file)
