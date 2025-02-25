import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0
        self.depth = embed_dim // num_heads
        self.wq = Dense(embed_dim)
        self.wk = Dense(embed_dim)
        self.wv = Dense(embed_dim)
        self.dense = Dense(embed_dim)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.dense(concat_attention)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, mask=None):
        attn_output = self.att(inputs, inputs, inputs, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.norm2(out1 + ffn_output)

class QuestionAnsweringModel:
    def __init__(self, vocab_size=30522, embed_dim=128, num_heads=8, ff_dim=512, num_blocks=4, max_len=512):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.max_len = max_len
        self.model = self.build_model()
    
    def build_model(self):
        inputs = keras.Input(shape=(self.max_len,), dtype=tf.int32)
        embedding = Embedding(self.vocab_size, self.embed_dim)(inputs)
        x = embedding
        for _ in range(self.num_blocks):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim)(x)
        start_logits = Dense(1, activation='softmax')(x)
        end_logits = Dense(1, activation='softmax')(x)
        return keras.Model(inputs=inputs, outputs=[start_logits, end_logits])
    
    def compile_model(self):
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train_model(self, train_dataset, epochs=3):
        self.model.fit(train_dataset, epochs=epochs)
    
    def save_model(self, filename="qa_model.keras"):
        self.model.save(filename)

def load_training_data(data_folder="training-data"):
    dataset = []
    for file in os.listdir(data_folder):
        if file.endswith(".json"):
            with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                dataset.extend(data)
    return dataset

# Instantiate and train the model
qa_wrapper = QuestionAnsweringModel()
qa_wrapper.compile_model()
# train_dataset = load_training_data()
# qa_wrapper.train_model(train_dataset)
qa_wrapper.save_model("qa_model.keras")


def question_answer(paragraph, question):
    model = keras.models.load_model("qa_model.keras", compile=False)
    tokenizer = keras.preprocessing.text.Tokenizer()
    max_len = 512
    
    tokenizer.fit_on_texts([paragraph, question])
    encoded_input = tokenizer.texts_to_sequences([question, paragraph])
    
    input_padded = keras.preprocessing.sequence.pad_sequences(encoded_input, maxlen=max_len, padding='post')
    input_tensor = tf.convert_to_tensor(input_padded)
    
    start_logits, end_logits = model.predict(input_tensor)
    start_index = tf.argmax(start_logits, axis=1).numpy()[0]
    end_index = tf.argmax(end_logits, axis=1).numpy()[0]
    
    words = tokenizer.sequences_to_texts([input_padded[start_index:end_index+1]])
    return " ".join(words).strip() if start_index < end_index else "Unable to find a valid answer."
