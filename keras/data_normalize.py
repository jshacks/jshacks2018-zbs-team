import pickle
import random
import tensorflow.keras as tf
import numpy as np
import matplotlib.pyplot as plt



def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_data(filename):

    f = open(filename, 'r');
    data = f.readlines();

    vector = {}
    char_map = {}

    avg = cnt = 0

    skip = {'Shift': 2699, 'Shift.t': 1034, 'minus': 193, 'Shift.a': 98,'Tab': 760, 'Return': 727, 'Shift.h': 459, 'Shift.i': 310, 'Shift.f': 17, 'Shift.b': 40,'Shift.u': 29, 'Shift.p': 56, 'Shift.c': 13, 'Shift.nine': 44, 'Shift.zero': 44,'Shift.o': 50,'Shift.s': 168,'Shift.slash': 18, 'Shift.space': 69, 'Shift.w': 15, 'Shift.l': 12, 'Shift.one': 10, 'Shift.n': 9,'Left': 77, 'Right': 32, 'Down': 3,'Up': 3, 'Remove': 5, 'Shift.e': 14, 'Shift.m': 12,'Shift.y': 8, 'Caps_Lock': 2, 'Caps_Lock.Return': 2, 'Caps_Lock.Tab': 1, 'Caps_Lock.Shift': 1, 'Shift.Caps_Lock.t': 1, 'Caps_Lock.h': 5, 'Caps_Lock.e': 4, 'Caps_Lock.space': 9, 'Caps_Lock.d': 1, 'Caps_Lock.Delete': 6, 'Caps_Lock.Caps_Lock': 2, 'Shift.semicolon': 8, 'Shift.Delete': 8, 'Shift.apostrophe': 31, 'Shift.j': 5, 'Shift.d': 10, 'bracketright': 2, 'Shift.r': 9, 'Shift.seven': 1, 'Shift.g': 4, 'Shift.q': 1, 'grave': 6, 'Caps_Lock.i': 6, 'Caps_Lock.t': 6, 'Caps_Lock.s': 4, 'Caps_Lock.w': 3, 'Caps_Lock.a': 2, 'Caps_Lock.r': 3, 'Caps_Lock.n': 1, 'Caps_Lock.g': 1, 'Caps_Lock.minus': 1, 'Caps_Lock.comma': 1, 'Caps_Lock.o': 1, 'Shift.x': 1, 'Select': 1, 'Control': 1, 'Shift.Return': 2, 'F13': 1, 'F12': 1}

    for i in data[1:]:
        line = ((list(filter(None, i.replace('\n', '').split(' ')))))

        if (line[0] not in vector.keys()):
            vector[line[0]] = []

        if (line[-3] not in skip.keys()):
            vector[line[0]].append(line[-3])
            vector[line[0]].append(float(line[-1])/0.24977727489123377)
            avg += float(line[-1])
            cnt += 1
            word = line[-3]
            var = not any(char.isdigit() for char in word)
            if var:
                if word in char_map.keys():
                 char_map[word] += 1
                else:
                 char_map[word] = 1

    avg = (avg) / cnt

    id_map = {}

    cnt = 0

    for ch in char_map.keys():
        id_map[ch] = cnt/len(char_map.keys());
        cnt += 1

    data_final = {}
    for key in vector.keys():
        for i in range(0, len(vector[key])):
            if (vector[key][i] in id_map.keys()):
                vector[key][i] = id_map[vector[key][i]]
            else:
                try:
                    vector[key][i] = (float(vector[key][i]));
                except ValueError:
                    print("division by zero")
    return  vector,char_map,id_map,avg

def get_random_sequence(data, char_map, id_map, avg, length = 50):
    rnd_key = random.choice(list(data.keys()));
    rnd_index = random.randint(0, len(data[rnd_key]) - length*2);
    if(rnd_index%2 == 1):
        rnd_index+=1
    seq = data[rnd_key][rnd_index : min(rnd_index + 2*length, len(data[rnd_key]))]
    for i in range(0, len(seq)):
        seq[i] = float(seq[i])
    return  seq

def get_random_drunk_sequence(data, char_map, id_map, avg, length = 50):

    rnd_key = random.choice(list(data.keys()));
    rnd_index = random.randint(0, len(data[rnd_key]) - length*2);
    if (rnd_index % 2 == 1):
        rnd_index += 1

    seq = data[rnd_key][rnd_index : min(rnd_index + 2*length, len(data[rnd_key]))]
    del_cnt = random.randint(2, 10)

    seq = seq[0:len(seq) - 4*del_cnt]

    #add delay
    for i in range(0,len(seq)):
        if(isinstance(seq[i],float)):
            seq[i] *= float(float(random.randint(20,75))/25.0)

    #add random delete
    zaibit = {'h': 0, 'e': 1, 's': 3, 'c': 4, 'n': 5, 'l': 6, 'o': 7, 'k': 8, 'i': 9, 't': 10, 'a': 11, 'p': 12, 'd': 13, 'r': 14, 'f': 15, 'u': 16, 'g': 17, 'm': 19, 'v': 22, 'w': 23, 'y': 24, 'b': 25, 'j': 26, 'x': 31, 'q': 33, 'z': 35}
    allow_map = {v: k for k, v in zaibit.items()}

    for x in range (0, del_cnt):
        poz_list = []
        for i in range(0, len(seq)):
            if (seq[i] in allow_map.keys()):
                poz_list.append(i)
        if(poz_list == []):
            return None
        poz = random.choice(poz_list)
        chr = allow_map[seq[poz]]
        seq.insert(poz - 1, id_map['Delete'])
        seq.insert(poz - 1, avg * float(float(random.randint(20, 75)) / 25.0))
        print(get_keys_around(chr))
        seq.insert(poz - 1, zaibit[get_keys_around(chr)])
        seq.insert(poz - 1, avg * float(float(random.randint(20, 75)) / 25.0))
    for x in range(0,len(seq)):
        seq[x] = float(seq[x])
    return  seq

def get_avg_seq(seq):
    cnt = avg = 0
    for i in range(0,len(seq)):
        if(isinstance(seq[i],float)):
            avg += seq[i]
            cnt += 1
    return  avg/cnt

def get_keys_around(key):
    lines = 'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
    line_index, index = [(i, l.find(key)) for i, l in enumerate(lines) if key in l][0]
    lines = lines[line_index-1: line_index+2] if line_index else lines[0: 2]
    rlst = [
        line[index + i] for line in lines for i in [-1, 0, 1]
        if len(line) > index + i and line[index + i] != key and index + i >= 0]
    return random.choice(rlst)

def representation(seq, allow_map):
    for i in range(0,len(seq)):
        if(seq[i] in  allow_map.keys()):
            print(allow_map[seq[i]], end = ' ')
        else:
            print(seq[i], end = ' ')
    print()

def dac_tat_zbs(data,char_map,id_map,avg):
    for i in range(0,100):
        seq = get_random_sequence(data,char_map,id_map,avg, 50)
        print(get_avg_seq(seq), end = '')
        representation(seq, allow_map)
        seq = get_random_drunk_sequence(data, char_map, id_map, avg, 50)
        print(get_avg_seq(seq), "Drunk :" ,end = '')
        representation(seq, allow_map)

def plot(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def define_model():
    model = tf.Sequential()
    model.add(tf.layers.Conv1D(strides=1, kernel_size=3, filters=128, input_shape=(100, 1), activation='relu'))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Conv1D(kernel_size=25, filters=64, activation='relu'))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Conv1D(kernel_size=10, filters=64, activation='relu'))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Conv1D(kernel_size=5, filters=32, activation='relu'))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Flatten())
    model.add(tf.layers.Dropout(0.2))
    model.add(tf.layers.Dense(32))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Dense(16))
    model.add(tf.layers.Activation('relu'))
    model.add(tf.layers.Dense(2))
    model.add(tf.layers.Activation('softmax'))
    return model

def prepare_data(data,char_map,id_map,avg):
    x = []
    y = []

    for i in range(0, 200):
        val = get_random_sequence(data, char_map, id_map, avg, 50)
        if (None not in val):
            x.append(np.array(val))
            y.append([1, 0])
        val = get_random_sequence(data, char_map, id_map, avg, 50)
        if (None not in val):
            x.append(np.array(val))
            y.append([0, 1])

    X = np.array(x)
    Y = np.array(y)
    X = np.expand_dims(X, axis=2)
    return X,Y

if __name__ == '__main__':

    data,char_map,id_map,avg = get_data(r"/home/master/Desktop/untitled1/TimingFeatures-DD.txt");
    print(data.values(),char_map,id_map,avg)

    X, Y = prepare_data(data,char_map,id_map,avg)
    print(X.shape,Y.shape)

    model = define_model()

    lr = 0.01; epoch = 100
    decay_rate = lr / epoch
    model.compile(loss='mean_squared_error', optimizer=tf.optimizers.SGD(lr = lr, momentum = 0.8, decay = decay_rate), metrics=['accuracy'])
    model.summary()

    history = model.fit(X, Y, validation_split=0.25,batch_size=50, epochs=epoch)
    print(history.history.keys())
    model.save("model5");
    print(history.history.keys())
    plot(history)

    #dac_tat_zbs(data,char_map,id_map,avg)
