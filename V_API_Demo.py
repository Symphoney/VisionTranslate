from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import os
import io
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

print(tf.__version__)

try:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "My Project-1379e3a8f3f2.json"
except:
    print("failed to auth with google services")


# for gui
translation_result = ""

def detect_text(path):
    """Detects text in the file."""
    tupColor = (255, 255, 0)
    import cv2
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    imageFinal = cv2.imread(path)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts[1:]:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        first = True
        count = 0
        for vertex in text.bounding_poly.vertices:
            if first:
                cv2.line(imageFinal, (vertex.x, vertex.y), (vertex.x, vertex.y), tupColor, 2)
                sx = vertex.x
                sy = vertex.y
                tempx = sx
                tempy = sy
                first = False
                count = count + 1
            elif count != 4:
                cv2.line(imageFinal, (tempx, tempy), (vertex.x, vertex.y), tupColor, 2)
                tempx = vertex.x
                tempy = vertex.y
                count = count + 1

        cv2.line(imageFinal, (tempx, tempy), (sx, sy), tupColor, 2)

        first = True
        cv2.putText(imageFinal, text.description, (sx, (sy - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        print('bounds: {}'.format(','.join(vertices)))

    """show the output image"""

    cv2.imshow("Detected Text", imageFinal)
    cv2.waitKey(0)

    return texts[0].description

# ------------------------
# Neural Machine Translation


path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
    return zip(*word_pairs)


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path):
    inp_lang, targ_lang = create_dataset(path)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file)

max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 128
units = 512
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)


# Decoder Model
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, predict):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, predict)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

'''
EPOCHS = 20
for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
'''


# Evaluates sentence being fed
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.index_word[predicted_id] + ' '
        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


def translate(sentence):
    result, sentence = evaluate(sentence)

    print('Input: %s' % sentence)
    print('Predicted translation: {}'.format(result))

<<<<<<< HEAD
    global translation_result
    translation_result = result

=======
>>>>>>> b63574f32d9de496475dd380970238ca2d11c037

# restores checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# -------------------



def browse_fileWindows():
    rep = filedialog.askopenfilename(title = "Select file")
    print("~{}".format(rep))
    enc_sentence = detect_text(rep)
    translate(enc_sentence)
<<<<<<< HEAD
    W = tk.Toplevel()
    W.wm_title("Translation")
    S = tk.Scrollbar(W)
    T = tk.Text(W, height=5, width=50)
    S.pack(side=tk.RIGHT, fill=tk.Y)
    T.pack(side=tk.LEFT, fill=tk.Y)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    T.insert(tk.END, translation_result)
=======

>>>>>>> b63574f32d9de496475dd380970238ca2d11c037

def browse_fileMac():
    rep = filedialog.askopenfilename(title = "Select file")
    print("~{}".format(rep))
<<<<<<< HEAD

=======
>>>>>>> b63574f32d9de496475dd380970238ca2d11c037
    enc_sentence = detect_text("/{}".format(rep))
    translate(enc_sentence)

    popup = tk.Tk()
    def on_configure(event):
        # update scrollregion after starting 'mainloop'
        # when all widgets are in canvas
        canvas.configure(scrollregion=canvas.bbox('all'))
    canvas = tk.Canvas(popup)
    canvas.pack(side=tk.LEFT)
    scrollbar = tk.Scrollbar(popup, command=canvas.yview)
    scrollbar.pack(side=tk.LEFT, fill='y')
    canvas.configure(yscrollcommand = scrollbar.set)
    canvas.bind('<Configure>', on_configure)
    frame = tk.Frame(canvas)
    canvas.create_window((0,0), window=frame, anchor='nw')
    l = tk.Label(frame, text=translation_result, font="-size 10", wraplength=70)
    l.pack()

# build the GUI

root = tk.Tk()
root.wm_title("OCR")
style = ttk.Style(root)
style.theme_use("clam")
root.geometry("250x250")
root.geometry("+300+300")

browserButton = tk.Button(master = root, text = 'Browse Windows', width = 14, command=browse_fileWindows)
browserButton.place(x=125, y=125)
browserButton.pack()

browserButton = tk.Button(master = root, text = 'Browse Mac', width = 14, command=browse_fileMac)
browserButton.place(x=125, y=25)
browserButton.pack()

<<<<<<< HEAD

tk.mainloop()


# en_sentence = u"May I borrow this book?"
# sp_sentence = u"¿Puedo tomar prestado este libro?"
# print(preprocess_sentence(enc_sentence))
# print(preprocess_sentence(dec_sentence).encode('utf-8'))
# -------------------
=======
tk.mainloop()
>>>>>>> b63574f32d9de496475dd380970238ca2d11c037
