import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical

def DUP(z, voc):
    out = (torch.arange(len(voc)) == z[-1][:, None]) * -1 # (B, voc_size)
    return out

def EOS(voc, t, target_sentence, end_token):
    out = torch.zeros(len(voc))
    if t < len(target_sentence):
        #  end_token = # TODO: get end_token form voc
        torch.arange(len(voc))
        out[voc == end_token] = -1
    return out

class Updater:
    def __init__(self, model, R, vocabulary, p_drop=0.5, W=10000, J=1, device='cpu') -> None:
        self.p_drop = p_drop
        self.W = W
        self.J = J
        self.R = R
        self.model = model
        self.voc = vocabulary
        self.device = device

    def update(self, x, y):
        L_BBSPG = 0
        rewards = []
        for j in range(1, self.J+1):
            z = torch.tensor([], device=self.device) # (T, B)
            last_R = 0
            for t in range(1, len(x)+1):
                mu = torch.rand((1, ))

                # x: (T, B, D) 
                model_output = self.model(x, y) # what should be input to model? x?
                # model_output: (B, C), C = voc.size
                if mu > self.p_drop:
                    current_R = self.R(z, y, self.voc, t)

                    # delta_r = self.W * (self.R(z, y, self.voc, t) - self.R(z, y, self.voc, t-1) + DUP(z, self.voc) + EOS(self.voc, t, y))
                    # delta_r = self.W * (self.R(z, y, self.voc, t) - self.R(z, y, self.voc, t-1))
                    delta_r = self.W * (current_R - last_R)

                    prob = softmax(torch.log(model_output).cpu() + delta_r, dim=1)
                    zt_idx = Categorical(probs=prob).sample() # (B,)

                    L_BBSPG -= torch.log(model_output[torch.arange(len(zt_idx)), zt_idx]) / self.J # (B,)
                    
                    last_R = torch.mean(current_R)
                else:
                    prob = model_output
                    zt_idx = Categorical(probs=prob).sample() # (B,)

                z = torch.cat([z, zt_idx.cuda()[None]], dim=0) # (T, B) token id

            rewards.append(current_R)

        L_BBSPG = L_BBSPG.mean()
        loss = L_BBSPG.item()
        L_BBSPG.backward()

        return rewards, loss
        