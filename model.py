from utils import *
from embedding import embed

class rnn_encoder_decoder(nn.Module):
    def __init__(self, x_cti_size, x_wti_size, y_wti_size):
        super().__init__()

        # architecture
        self.enc = encoder(x_cti_size, x_wti_size)
        self.dec = decoder(y_wti_size)
        self = self.cuda() if CUDA else self

    def forward(self, xc, xw, y0): # for training
        b = y0.size(0) # batch size
        loss = 0
        self.zero_grad()
        mask, lens = maskset(xw)
        self.dec.M = self.enc(b, xc, xw, lens)
        self.dec.hidden = self.enc.hidden
        self.dec.attn.Va = zeros(b, 1, HIDDEN_SIZE)
        yi = LongTensor([SOS_IDX] * b)
        for t in range(y0.size(1)):
            yo = self.dec(yi.unsqueeze(1), mask)
            yi = y0[:, t] # teacher forcing
            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX)
        loss /= y0.size(1) # divide by senquence length
        return loss

    def decode(self, x): # for inference
        pass

class encoder(nn.Module):
    def __init__(self, cti_size, wti_size):
        super().__init__()
        self.hidden = None # encoder hidden states

        # architecture
        self.embed = embed(ENC_EMBED, cti_size, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )

    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, b, xc, xw, lens):
        self.hidden = self.init_state(b)
        x = self.embed(xc, xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h

class decoder(nn.Module):
    def __init__(self, wti_size):
        super().__init__()
        self.M = None # source hidden states
        self.hidden = None # decoder hidden states

        # architecture
        self.embed = embed(DEC_EMBED, 0, wti_size)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE, # input feeding
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, wti_size)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, y1, mask):
        x = self.embed(None, y1)
        x = torch.cat((x, self.attn.Va), 2) # input feeding
        h, _ = self.rnn(x, self.hidden)
        h = self.attn(h, self.M, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)
        return y

class attn(nn.Module):
    def __init__(self):
        super().__init__()

        # architecture
        self.Wa = None # attention weights
        self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.Va = None # attention vector

    def align(self, ht, hs, mask):
        a = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        a = F.softmax(a.masked_fill(mask.unsqueeze(1), -10000), 2)
        return a # attention weights

    def forward(self, ht, hs, mask):
        self.Wa = self.align(ht, hs, mask)
        c = self.Wa.bmm(hs) # context vector [B, 1, L] @ [B, L, H] = [B, 1, H]
        self.Va = torch.tanh(self.Wc(torch.cat((c, ht), 2)))
        return self.Va # attention vector
    
    class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
    
    class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, num_layers=3, bidirectional = True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0)