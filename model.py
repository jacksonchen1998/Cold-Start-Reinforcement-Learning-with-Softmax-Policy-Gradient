import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden = decoder_hidden.unsqueeze(1)
        energy = torch.tanh(self.W(decoder_hidden + encoder_outputs))
        attention_scores = self.V(energy).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context_vector, attention_weights

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + 2 * hidden_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, input_seq, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_seq)
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        lstm_input = torch.cat((embedded, context_vector), dim=2)
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(0))
        return predictions, hidden, cell, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        encoder_outputs, hidden, cell = self.encoder(input_seq)
        decoder_outputs, _, _, _ = self.decoder(target_seq, hidden, cell, encoder_outputs)
        return decoder_outputs

# Define hyperparameters
input_vocab_size = 10000
target_vocab_size = 8000
embedding_dim = 256
hidden_size = 512

# Create encoder and decoder instances
encoder = Encoder(input_vocab_size, embedding_dim, hidden_size)
decoder = Decoder(target_vocab_size, embedding_dim, hidden_size)

# Create the Seq2Seq model
model = Seq2Seq(encoder, decoder)
