import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from datasets import load_dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    import re
    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]
    return [t.split()[:max_length] for t in text]

def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]
    return {word: ix for ix, word in enumerate(sorted_words)}

# modify build_word_counts for SNLI
# so that it takes into account batch['premise'] and batch['hypothesis']
def build_word_counts(dataloader) -> "dict[str, int]":
    word_counts = {}
    for batch in dataloader:
        for words in batch:
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts

def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]

class CharSeqDataloader():
    def __init__(self, filepath, seq_len, examples_per_epoch):
        text = ""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(filepath, 'r') as f:
            text = list(f.read())
        self.unique_chars = list(set(text))
        self.vocab_size = len(self.unique_chars)
        self.mappings = self.generate_char_mappings(self.unique_chars) # fill in
        self.seq_len =  seq_len # fill in
        self.examples_per_epoch = examples_per_epoch
        self.indices = torch.tensor(self.convert_seq_to_indices(text)).to(self.device)
    
    
    def generate_char_mappings(self, uq):
        char_to_idx = {}
        idx_to_char = {}
        for index, value in enumerate(uq):
            char_to_idx[value] = index
            idx_to_char[index] = value
        return {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}
            

    def convert_seq_to_indices(self, seq):
        # your code here
        result = []
        for i in seq:
            result.append(self.mappings['char_to_idx'][i])
        return result

    def convert_indices_to_seq(self, seq):
        result = []
        #print(seq)
        for i in seq:
            #print(i)
            result.append(self.mappings['idx_to_char'][i])
        return result

    def get_example(self):
        for i in range(self.examples_per_epoch):
            start_index = np.random.randint(0, len(self.indices) - self.seq_len - 1)
            end_index = start_index + self.seq_len
            train = self.indices[start_index:end_index]
            test = self.indices[start_index + 1:end_index + 1]
            yield train, test
            

class CharRNN(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.embedding_size = embedding_size
        self.embedding_layer = nn.Embedding(self.n_chars, self.embedding_size).to(self.device)
        self.wax = nn.Linear(self.embedding_size, self.hidden_size, bias=False).to(self.device)
        self.waa = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device)
        self.wya = nn.Linear(self.hidden_size, self.n_chars).to(self.device)
        
    def rnn_cell(self, i, h):
        # your code here
        h_new = torch.tanh(self.wax(i) + self.waa(h))
        o = self.wya(h_new)
        return o, h_new

    def forward(self, input_seq, hidden = None):
        embedded = self.embedding_layer(input_seq)
        #print(embedded)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(self.device)
        outputs = []
        for i in range(embedded.size(0)):
            #print(embedded.size(0))
            output, hidden = self.rnn_cell(embedded[i], hidden)
            outputs.append(output)
        hidden_last = hidden
        out = torch.stack(outputs, dim=0)
        return out, hidden_last

    def get_loss_function(self):
        # your code here
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        # your code here
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        result = [starting_char]
        hidden = torch.zeros(self.hidden_size).to(self.device)
        input_int = torch.tensor([starting_char]).to(self.device)
        #print(input_char)
        for _ in range(seq_len):
            embedded = self.embedding_layer(input_int)
            output, hidden = self.rnn_cell(embedded[0], hidden)
            #print(output)
            output = output / temp
            #print(output)
            if top_k is not None:
                output = top_k_filtering(output, top_k)
                
            if top_p is not None:
                output = top_p_filtering(output, top_p)
                
            probability = F.softmax(output, dim=0)
            #print(probs)
            distribution = Categorical(probability)
            next_int = distribution.sample()
            #print(next_char)
            result.append(next_int.item())
            
            input_int = torch.tensor([next_int.item()]).to(self.device)

        return result


class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_layer = nn.Embedding(n_chars, embedding_size).to(device)
        self.forget_gate = nn.Linear(embedding_size + hidden_size, hidden_size).to(self.device)
        self.input_gate = nn.Linear(embedding_size + hidden_size, hidden_size).to(self.device)
        self.output_gate = nn.Linear(embedding_size + hidden_size, hidden_size).to(self.device)
        self.cell_state_layer = nn.Linear(embedding_size + hidden_size, hidden_size).to(self.device)
        self.fc_output = nn.Linear(hidden_size, n_chars).to(self.device)
        
    def forward(self, input_seq, hidden = None, cell = None):
        embedded = self.embedding_layer(input_seq)
        if hidden is None:
            hidden = torch.zeros(self.hidden_size).to(self.device)
        if cell is None:
            cell = torch.zeros(self.hidden_size).to(self.device)
        outputs = []
        for i in range(embedded.size(0)):
            #print(embedded.size(0))
            output, hidden, cell = self.lstm_cell(embedded[i], hidden, cell)
            outputs.append(output)
        out_seq = torch.stack(outputs, dim=0)
        hidden_last = hidden
        cell_last = cell
        return out_seq, hidden_last, cell_last

    def lstm_cell(self, i, h, c):
        combined = torch.cat((i, h), 0)
        forget = torch.sigmoid(self.forget_gate(combined))
        input_ = torch.sigmoid(self.input_gate(combined))
        output_ = torch.sigmoid(self.output_gate(combined))
        tanh = torch.tanh(self.cell_state_layer(combined))
        c_new = forget * c + input_ * tanh
        h_new = output_ * torch.tanh(c_new)
        o = self.fc_output(h_new)
        return o, h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)
    
    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_k=None, top_p=None):
        result = [starting_char]
        hidden = torch.zeros(self.hidden_size).to(self.device)
        cell = torch.zeros(self.hidden_size).to(self.device)
        input_int = torch.tensor([starting_char]).to(self.device)
        for _ in range(seq_len):
            embedded = self.embedding_layer(input_int)
            output, hidden, cell = self.lstm_cell(embedded[0], hidden, cell)
            output = output / temp
            if top_k is not None:
                output = top_k_filtering(output, top_k)
            if top_p is not None:
                output = top_p_filtering(output, top_p)
            probability = F.softmax(output, dim=0)
            distribution = Categorical(probability)
            next_int = distribution.sample()
            result.append(next_int.item())
            input_int = torch.tensor([next_int.item()]).to(self.device)
        return result
    
def top_k_filtering(logits, top_k=40):
    values = torch.topk(logits, top_k)[0][..., -1, None]
    mask = logits < values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits[mask] = float('-inf')
    return logits.to(device)

def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_index = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = cumulative_probabilities > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0
    to_remove = mask.scatter(1, index=sorted_index, src=mask)
    logits[to_remove] = float('-inf')
    return logits.to(device)

def train(model, dataset, lr, out_seq_len, num_epochs):

    # code to initialize optimizer, loss function
    optimizer = model.get_optimizer(lr)
    loss_fn = model.get_loss_function() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  
    n = 0
    running_loss = 0
    for epoch in range(num_epochs):
        for in_seq, out_seq in dataset.get_example():
            # main loop code
            in_seq = in_seq.to(device)
            out_seq= out_seq.to(device)
            hidden = None
            output = model(in_seq, hidden)
            output = output[0]
            loss = loss_fn(output, out_seq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += 1

        # print info every X examples
        print(f"Epoch {epoch}. Running loss so far: {(running_loss/n):.8f}")

        print("\n-------------SAMPLE FROM MODEL-------------")

        # code to sample a sequence from your model randomly

        with torch.no_grad():
            random_number = random.randint(0, model.n_chars)
            generated_seq = model.sample_sequence(random_number, out_seq_len)
            print(dataset.convert_indices_to_seq(generated_seq))

        print("\n------------/SAMPLE FROM MODEL/------------")

        n = 0
        running_loss = 0

    
    return model # return model optionally


def run_char_rnn():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10 # one epoch is this # of examples
    out_seq_len = 200
    #data_path = "./data/shakespeare.txt"
    dataset = CharSeqDataloader('/kaggle/input/dataset-a3/shakespeare.txt', seq_len, epoch_size)
    model = CharRNN(len(dataset.unique_chars),embedding_size, hidden_size)
    # code to initialize dataloader, model
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)
    # code to initialize dataloader, model
    

def run_char_lstm():
    hidden_size = 512
    embedding_size = 300
    seq_len = 100
    lr = 0.002
    num_epochs = 100
    epoch_size = 10
    out_seq_len = 200
    #data_path = "./data/shakespeare.txt"
    # code to initialize dataloader, model
    dataset = CharSeqDataloader('/kaggle/input/dataset-a3/shakespeare.txt', seq_len, epoch_size)
    # code to initialize dataloader, model
    #print(len(dataset.unique_chars))
    model = CharLSTM(len(dataset.unique_chars),embedding_size, hidden_size)
    train(model, dataset, lr=lr, 
                out_seq_len=out_seq_len, 
                num_epochs=num_epochs)
    

def fix_padding(batch_premises, batch_hypotheses):
    batch_premises = [torch.tensor(x) for x in batch_premises]
    batch_hypotheses = [torch.tensor(x) for x in batch_hypotheses]
    batch_premises_padded = nn.utils.rnn.pad_sequence(batch_premises, batch_first=True)
    batch_hypotheses_padded = nn.utils.rnn.pad_sequence(batch_hypotheses, batch_first=True)
    batch_premises_reversed = [x.flip(dims=[0]) for x in batch_premises]
    batch_hypotheses_reversed = [x.flip(dims=[0]) for x in batch_hypotheses]
    batch_premises_reversed_padded = nn.utils.rnn.pad_sequence(batch_premises_reversed, batch_first=True)
    batch_hypotheses_reversed_padded = nn.utils.rnn.pad_sequence(batch_hypotheses_reversed, batch_first=True)
    return batch_premises_padded, batch_hypotheses_padded, batch_premises_reversed_padded, batch_hypotheses_reversed_padded


def create_embedding_matrix(word_index, emb_dict, emb_dim):
    result = torch.zeros(len(word_index), emb_dim)
    for key, value in word_index.items():
        if key in emb_dict:
            result[value] = torch.from_numpy(emb_dict[key])
    return result

def pad_to_length(tensor, length):
    if len(tensor) < length:
        tensor = np.pad(tensor, (0, length - len(tensor)), mode='constant')
    return torch.tensor(tensor, dtype=torch.long)

def evaluate(model, dataloader, index_map):
    pass

class UniLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(UniLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)
        # your code here

    def forward(self, a, b):
        a, b, _, _ = fix_padding(a, b)
        a = a.to(self.device)
        b = b.to(self.device)
        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        _, (_, a) = self.lstm(a)
        _, (_, b) = self.lstm(b)
        a = a[-1]
        b = b[-1]
        x = torch.cat((a, b), dim=1)
        x = F.relu(self.int_layer(x))
        x = self.out_layer(x)
        return x


class ShallowBiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, num_classes):
        super(ShallowBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm_forward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_backward = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.int_layer = nn.Linear(hidden_dim * 4, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, a, b):
        # Fix padding and move tensors to the same device as the model
        a, b, a_reversed, b_reversed = fix_padding(a, b)
        a = a.to(self.device)
        b = b.to(self.device)
        a_reversed = a_reversed.to(self.device)
        b_reversed = b_reversed.to(self.device)
        a = self.embedding_layer(a)
        b = self.embedding_layer(b)
        a_reversed = self.embedding_layer(a_reversed)
        b_reversed = self.embedding_layer(b_reversed)
        _, (_, a) = self.lstm_forward(a)
        _, (_, b) = self.lstm_forward(b)
        _, (_, a_reversed) = self.lstm_backward(a_reversed)
        _, (_, b_reversed) = self.lstm_backward(b_reversed)
        a = a[-1]
        b = b[-1]
        a_reversed = a_reversed[-1]
        b_reversed = b_reversed[-1]
        x = torch.cat((a, a_reversed, b, b_reversed), dim=1)
        x = F.relu(self.int_layer(x))
        x = self.out_layer(x)
        return x


def run_snli(model):
    dataset = load_dataset("snli")
    glove = pd.read_csv('./data/glove.6B.100d.txt', sep=" ", quoting=3, header=None, index_col=0)

    glove_embeddings = "" # fill in your code

    train_filtered = dataset['train'].filter(lambda ex: ex['label'] != -1)
    valid_filtered = dataset['validation'].filter(lambda ex: ex['label'] != -1)
    test_filtered =  dataset['test'].filter(lambda ex: ex['label'] != -1)

    # code to make dataloaders

    word_counts = build_word_counts(dataloader_train)
    index_map = build_index_map(word_counts)

    # training code

def run_snli_lstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

def run_snli_bilstm():
    model_class = "" # fill in the classs name of the model (to initialize within run_snli)
    run_snli(model_class)

if __name__ == '__main__':
    run_char_rnn()
    # run_char_lstm()
    # run_snli_lstm()
    # run_snli_bilstm()
