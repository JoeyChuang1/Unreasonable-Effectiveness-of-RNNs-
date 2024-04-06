import torch
import torch.nn as nn

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.embedding = nn.Embedding(n_chars, embedding_size)
        self.forget_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.cell_state_layer = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, n_chars)

    def lstm_cell(self, i, h, c):
        combined = torch.cat((i, h), 1)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_state_layer(combined))
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return self.fc_output(h_new), h_new, c_new

###############33
###############3
#############
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class CharLSTM(nn.Module):
    def __init__(self, n_chars, embedding_size, hidden_size):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_chars = n_chars

        self.embedding = nn.Embedding(n_chars, embedding_size)
        self.forget_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.cell_state_layer = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.fc_output = nn.Linear(hidden_size, n_chars)

    def forward(self, input_seq, hidden=None, cell=None):
        if hidden is None:
            hidden = torch.zeros(self.hidden_size)
        if cell is None:
            cell = torch.zeros(self.hidden_size)

        out_seq = []
        for i in range(input_seq.size(0)):
            output, hidden, cell = self.lstm_cell(input_seq[i], hidden, cell)
            out_seq.append(output)
        out_seq = torch.stack(out_seq)

        return out_seq, hidden, cell

    def lstm_cell(self, i, h, c):
        combined = torch.cat((i, h), 1)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        c_tilde = torch.tanh(self.cell_state_layer(combined))
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return self.fc_output(h_new), h_new, c_new

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sample_sequence(self, starting_char, seq_len, temp=0.5, top_p=None, top_k=None):
        generated_seq = [starting_char.item()]
        hidden = torch.zeros(self.hidden_size)
        cell = torch.zeros(self.hidden_size)

        input_char = torch.tensor([starting_char])
        for _ in range(seq_len):
            embedded = self.embedding(input_char)
            output, hidden, cell = self.lstm_cell(embedded[0], hidden, cell)
            output = output / temp
            if top_k is not None:
                output = top_k_filtering(output, top_k)
            if top_p is not None:
                output = top_p_filtering(output, top_p)
            probs = F.softmax(output, dim=0)
            dist = Categorical(probs)
            next_char = dist.sample()
            generated_seq.append(next_char.item())
            input_char = torch.tensor([next_char.item()])

        return generated_seq
