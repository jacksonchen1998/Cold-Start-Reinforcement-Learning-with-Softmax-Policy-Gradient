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
    def __init__(self, model, R, vocabulary, p_drop=0.5, W=10000, J=1) -> None:
        self.p_drop = p_drop
        self.W = W
        self.J = J
        self.R = R
        self.model = model
        self.voc = vocabulary

    def updata(self, x, y):
        L_BBSPG = 0
        rewards = []
        for j in range(1, self.J+1):
            z = torch.tensor([])
            for t in range(1, self.T+1):
                mu = torch.rand((1, ))

                # x: (T, B, D) 
                model_output = self.model(x, y) # what should be input to model? x?
                # model_output: (B, C), C = voc.size
                if mu > self.p_drop:
                    # delta_r = self.W * (self.R(z, y, self.voc, t) - self.R(z, y, self.voc, t-1) + DUP(z, self.voc) + EOS(self.voc, t, y))
                    delta_r = self.W * (self.R(z, y, self.voc, t) - self.R(z, y, self.voc, t-1) + DUP(z, self.voc))

                    prob = softmax(torch.log(model_output) + delta_r, dim=1)
                    zt_idx = Categorical(probs=prob).sample() # (B,)

                    L_BBSPG -= torch.log(model_output[torch.arange(len(zt_idx)), zt_idx]) / self.J # (B,)
                    
                else:
                    prob = model_output
                    zt_idx = Categorical(probs=prob).sample() # (B,)

                z = torch.cat([z, zt_idx[None]], dim=0) # (T, B) token id

            rewards.append(self.R(z, y, self.voc, len(z)))

        L_BBSPG = L_BBSPG.mean()
        loss = L_BBSPG.item()
        L_BBSPG.backward()

        return torch.mean(rewards), loss
        