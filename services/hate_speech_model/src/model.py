import os
import glob
import pandas as pd
import numpy as np
import random
import time
import gc
import torch
import pickle
from typing import Tuple, List
from torch import nn
from torch.utils import data as torch_utils_data
from torch.nn import functional as F
from tqdm.notebook import tqdm_notebook as tqdm
from keras.preprocessing import text as keras_prep_text
from keras.utils.data_utils import pad_sequences
from sklearn.metrics import f1_score


VOCAB_SIZE = 100_000  # total distinct words or features - this limits the words to be embedded
EMBED_DIM = 300
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS


def get_coefs(word, *arr):
    """
    Converts a line from the embedding file to a tuple of (word, 32-bit numpy array)

    :param word: the first element in each line is the word
    :param arr: elements 2-n are the embedding dimensions
    """
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path: str):
    """
    Utility function to load word embeddings.  Each word embedding looks like:
    word 0.3 0.4 0.5 0.6 ...
    This function converts the embeddings to a dictionary of {word: numpy array}
    """
    with open(path, 'r', encoding='UTF-8') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))


def get_word_embeddings(word_index: dict, path: str):
    """
    Maps words found in the text (word_index) to their corresponding word embeddings from the
    pre-trained model loaded from (path).  If any words cannot be found in the pre-trained model,
    they are tracked in unknown_words.
    """
    embedding_index = load_embeddings(path)
    # create an empty matrix of shape (nbr_words, embed_dim)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBED_DIM))
    unknown_words = []

    # map all words from the text to their embeddings, if they exist in the embedding index
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words


def sigmoid(x: np.ndarray):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def threshold_search(y_true, y_proba):
    """Finds the best probability threshold to maximize F1 score"""
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result


class Attention(nn.Module):
    """
    Implements an attention module.
    """

    def __init__(self, feature_dim: int, step_dim: int, bias: bool = True, **kwargs):
        """
        Builds the decoder piece of attention module.  The encoder piece is assumed to have
        been built with either a bi-directional RNN or self attention.

        :param feature_dim: the number of features, or input layer size
        :param step_dim: the max sequence length
        """
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        # initialize weights
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask = None):
        """
        Implements forward pass through attention module

        :param x: the encoded input vector from a bi-directional RNN, or in other words
            the concatenated forward and backward hidden states, h_j, from the equations
            for attention.  For help understanding this,
            see: https://bgg.medium.com/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad
        """
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        # alignment vector that scores how well the inputs at position j match the output at position i
        # this is e_ij = a(s_i-1, h_j),
        #   where s_i-1 is the decoder hidden states (self.weight) and h_j is the jth input label (x)
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)

        # the attention score, a_ij, is just 'a' here, and the next line computs the numerator of a_ij
        a = torch.exp(eij)

        # if masked, multiply the attention score by hidden states of the input sequence
        if mask is not None:
            a = a * mask

        # finalize computation of a_ij
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        # weight the input by multiplying it by the attention score, a_ij * h_j
        weighted_input = x * torch.unsqueeze(a, -1)

        # sum the weighted input to return the context vector, ci
        return torch.sum(weighted_input, 1)


class SpatialDropout(nn.Dropout2d):
    """
    Implements the functionality of Keras' SpatialDropout1D.
    Randomly drop features, i.e. [[1, 1, 1], [2, 1, 2]] -> [[1, 0, 1], [2, 0, 2]]
    Compare this with ordinary dropout that drops by sample, i.e. [[1, 1, 1], [2, 1, 2]] -> [[1, 0, 1], [0, 1, 2]]
    """
    def forward(self, x):
        x = x.unsqueeze(2)  # add a dimension of size 1 at position 2, producing (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # re-order dimensions to (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # re-order dimensions to (N, T, 1, K)
        x = x.squeeze(2)  # remove dimension of size 1 at position 2, producing (N, T, K)
        return x


class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, num_aux_targets: int):
        """Sets up neural network architecture"""
        super(NeuralNet, self).__init__()

        # set up a non-trainable, pre-trained embedding layer from the provided embedding_matrix
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)  # randomly drop this percent of features

        # each bidirectional layer outputs 2 sequences: 1 forward, 1 backward, and concatenates them
        # so stacking 2 enriches the sequence features
        self.lstm1 = nn.LSTM(
            input_size=EMBED_DIM * 2,
            hidden_size=LSTM_UNITS,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=LSTM_UNITS * 2,
            hidden_size=LSTM_UNITS,
            bidirectional=True,
            batch_first=True
        )

        # skip connections...
        # add a product of a dense layer with the hidden layer to the output of the hidden layer
        self.linear1 = nn.Linear(in_features=DENSE_HIDDEN_UNITS, out_features=DENSE_HIDDEN_UNITS, bias=True)
        self.linear2 = nn.Linear(in_features=DENSE_HIDDEN_UNITS, out_features=DENSE_HIDDEN_UNITS, bias=True)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        # auxiliary outputs to be predicted as an alternative to the main output
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

    def forward(self, x):
        """Implements forward pass"""
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        # attenion module can be inserted here, as an attention module's encoder uses a bi-directional RNN,
        #   which was just defined above
        # atten_1 = Attention(LSTM_UNITS * 2, MAX_LEN)(h_lstm1)  # skip connection
        # atten_2 = Attention(LSTM_UNITS * 2, MAX_LEN)(h_lstm2)

        avg_pool = torch.mean(h_lstm2, 1)  # global mean pooling
        max_pool, _ = torch.max(h_lstm2, 1)  # global max pooling

        # concatenate to reshape from (batch_size, MAX_LEN, LSTM_UNITS * 2) to h_conc (BATCH_SIZE, LSTM_UNITS * 4)
        # if using attention, un-comment the next line and comment out the line after
        # h_conc = torch.cat((atten_1, atten_2, max_pool, avg_pool), 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1 = F.relu(self.linear1(h_conc))
        h_conc_linear2 = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


class HateSpeechClassifier:
    def __init__(self, seed: int = 14, max_len: int = 220, num_models: int = 1,
                 learning_rates: Tuple[float] = tuple([0.001]), batch_size: int = 512, num_epochs: int = 1):
        """
        A classifier for hate speech

        :param seed: any integer to ensure consistency in random number generation
        :param max_len: max word embeddings per document
        :param num_models: number of models to train, defaults to 1 but can be higher for ensembling
        :param learning_rates: learning rates for each model
        :param batch_size: number of training samples per batch
        :param num_epochs: number of epochs to train for (number of iterations over training set)
        """
        if num_models != len(learning_rates):
            raise ValueError(f"num_models {num_models} != the number of learning rates {len(learning_rates)}")
        self.seed = seed
        self.vocab_size = VOCAB_SIZE
        self.max_len = max_len
        self.num_models = num_models
        self.learning_rates = learning_rates
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # hardcoded attributes
        self.word_embeddings = {
            'fasttext': 'word_vectors/crawl-300d-2M.vec',
            'glove': 'word_vectors/glove.840B.300d.txt'
        }
        self.target_column = 'target'
        self.identity_columns = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
        ]
        # placeholders
        self.tokenizer = None
        self.max_features = VOCAB_SIZE
        self.models = []
        self.output_dim = 1

    @staticmethod
    def set_seed(seed):
        """Ensures model will run deterministically for any given seed, even with CUDA"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def preprocess(data: pd.Series):
        """
        Cleans the text by removing special characters and returning a pd.Series of string type.
        Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
        """
        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        def clean_special_chars(text: str, punct: str):
            """Replaces the given characters, punct, in the string, text."""
            for p in punct:
                text = text.replace(p, ' ')
            return text

        data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
        return data

    def create_embedding_matrix(self):
        """Combines Glove and Fast Text into 1 embedding matrix."""
        fasttext_embeddings, fasttext_unknown_words = get_word_embeddings(
            self.tokenizer.word_index, self.word_embeddings['fasttext']
        )
        print('Unknown words (fast text): ', len(fasttext_unknown_words))
        glove_embeddings, glove_unknown_words = get_word_embeddings(
            self.tokenizer.word_index, self.word_embeddings['glove']
        )
        print('Unknown words (glove): ', len(glove_unknown_words))
        embedding_matrix = np.concatenate([fasttext_embeddings, glove_embeddings], axis=-1)
        print("Embedding matrix shape: ", embedding_matrix.shape)

        del fasttext_embeddings
        del glove_embeddings
        return embedding_matrix

    def train(self, train_data_path: str, sample_frac: float = 1.0):
        """
        Trains self.num_models models on the training data.

        :param train_data_path: path to the CSV file with the training data
        :param sample_frac: percentage of the training data to use for training - it is helpful to set this
            lower for debugging, like 0.005, and then let it go back to 1.0 for training
        """
        self.set_seed(seed=self.seed)

        # read data, expect ~2Gb RAM for default training data from Jigsaw
        train_df = pd.read_csv(train_data_path).sample(frac=sample_frac, random_state=self.seed)
        print(f"Training with {len(train_df):,} samples.")

        # pre-process data
        x_train = self.preprocess(train_df['comment_text'])
        y_train = np.where(train_df['target'] >= 0.5, 1, 0)  # binarize the target
        y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

        # tokenize the text
        self.tokenizer = keras_prep_text.Tokenizer(num_words=self.vocab_size)
        self.tokenizer.fit_on_texts(list(x_train))
        x_train = self.tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen=self.max_len)

        self.max_features = min(self.vocab_size, len(self.tokenizer.word_index) + 1)
        print(f"Updated Max Features (Vocab Size): {self.max_features:,}")

        # create word embeddings
        embedding_matrix = self.create_embedding_matrix()
        gc.collect()

        # move data to CUDA if available
        x_train_torch = torch.tensor(x_train, dtype=torch.long)  #.cuda()
        y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32)  #.cuda()
        self.output_dim = y_train_torch.shape[-1]
        if torch.cuda.is_available():
            x_train_torch = x_train_torch.cuda()
            y_train_torch = y_train_torch.cuda()

        # convert to tensor datasets
        train_dataset = torch_utils_data.TensorDataset(x_train_torch, y_train_torch)

        # train self.num_models models
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        for model_idx in range(self.num_models):
            print('\nTraining Model ', model_idx)

            # fit each model with a different seed, otherwise they will be identical
            self.set_seed(self.seed + model_idx)

            model = NeuralNet(embedding_matrix, num_aux_targets=y_aux_train.shape[-1])
            if torch.cuda.is_available():
                model.cuda()

            # set model parameters
            param_lrs = [{'params': param, 'lr': self.learning_rates[model_idx]} for param in model.parameters()]
            optimizer = torch.optim.Adam(param_lrs, lr=self.learning_rates[model_idx])

            # decay the learning rate using a schedule of 0.6^epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

            # set up a DataLoader for the training dataset
            train_loader = torch_utils_data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.num_epochs):
                start_time = time.time()

                model.train()  # set model in training mode
                avg_loss = 0.

                for data in tqdm(train_loader, disable=False):
                    # the target is the final column in data
                    x_batch = data[:-1]
                    y_batch = data[-1]

                    # forward pass and calculate loss
                    y_pred = model(*x_batch)
                    loss = loss_fn(y_pred, y_batch)

                    # zero out the gradients for the optimizer, now that the loss has been calculated
                    optimizer.zero_grad()

                    # backpropagate the loss
                    loss.backward()

                    # update the weights using the optimizer
                    optimizer.step()

                    # track the mean loss over all batches in this epoch
                    avg_loss += loss.item() / len(train_loader)

                # increment learning rate schedule
                scheduler.step()

                model.eval()  # set model in eval/inference mode
                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch + 1}/{self.num_epochs} \t loss={avg_loss:.4f} \t time={elapsed_time:.2f}s')

            self.models.append(model)  # model is appended in eval mode a
        print(f"Trained {self.num_models} models.")

    def predict(self, test_data: List[dict]):
        """
        Evaluates trained models using an averaged prediction.

        :param test_data: list of dictionaries, each with keys 'id' and 'comment_text'
        """
        if self.tokenizer is None:
            raise TypeError("No fitted tokenizer was found.  Has the model been trained yet?")

        self.set_seed(seed=self.seed)

        # read data
        test_df = pd.DataFrame(test_data)

        # pre-process data
        x_test = self.preprocess(test_df['comment_text'])

        # tokenize the text
        x_test = self.tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=self.max_len)

        # move data to CUDA if available
        x_test_torch = torch.tensor(x_test, dtype=torch.long)  #.cuda()
        if torch.cuda.is_available():
            print("moving test data to cuda")
            x_test_torch = x_test_torch.cuda()

        # convert to tensor datasets
        test_dataset = torch_utils_data.TensorDataset(x_test_torch)

        # set up a DataLoader for the training dataset
        test_loader = torch_utils_data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        all_test_preds = []
        for model_idx in range(self.num_models):
            # run each batch of the test data through the model
            test_preds = np.zeros((len(test_dataset), self.output_dim))
            for i, x_batch in enumerate(test_loader):
                y_pred = sigmoid(self.models[model_idx](*x_batch).detach().cpu().numpy())
                test_preds[i * self.batch_size:(i + 1) * self.batch_size, :] = y_pred
            all_test_preds.append(test_preds)

        test_df['pred'] = np.mean(all_test_preds, axis=0)[:, 0]
        return test_df.to_dict(orient="records")

    def save_classifier(self, save_path: str):
        """
        Saves trained models and the tokenizer.

        :param save_path: folder path ending in '/', e.g. "models/"
        """
        # models
        if len(self.models) == 0:
            raise TypeError("No fitted model was found.  Has a model been trained yet?")
        for model_idx, model in enumerate(self.models):
            model.eval()  # set to eval/inference mode
            torch.save(model.state_dict(), f"{save_path}model_{model_idx}.pt")
        print(f"Saved {len(self.models)} models.")

        # tokenizer
        with open(f"{save_path}tokenizer.pkl", "wb") as file:
            pickle.dump(self.tokenizer, file)
        print("Saved tokenizer.")

        # save output dim
        with open(f"{save_path}output_dim.txt", "w") as file:
            file.write(str(self.output_dim))
        print("Saved model output dimensions from training set.")

    def load_classifier(self, load_path: str):
        """
        Loads trained models and the tokenizer.

        :param load_path: folder path ending in '/', e.g. "models/"
        """
        # tokenizer
        tok_files = glob.glob(f"{load_path}*.pkl")
        for file in tok_files:
            with open(file, 'rb') as file_handler:
                self.tokenizer = pickle.load(file_handler)
        print("Loaded tokenizer.")

        # output dim
        od_files = glob.glob(f"{load_path}*.txt")
        for file in od_files:
            with open(file, 'r') as file_handler:
                self.output_dim = int(file_handler.readline())
        print("Updated output dimensions.")

        # set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device {device}.")

        # models
        if load_path.split("/")[0] == "src":
            self.word_embeddings = {k: "src/" + v for k, v in self.word_embeddings.items()}
        embedding_matrix = self.create_embedding_matrix()
        files = glob.glob(f"{load_path}*.pt")
        for file in files:
            model = NeuralNet(embedding_matrix, num_aux_targets=self.output_dim-1)
            model.load_state_dict(torch.load(file, map_location=device))
            model.eval()
            model.to(device)
            self.models.append(model)
        print(f"Loaded {len(files)} models.")
