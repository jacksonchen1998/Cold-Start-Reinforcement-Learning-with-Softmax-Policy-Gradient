import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical

def DUP(z, voc):
    out = torch.zeros_like(voc)
    out[voc == z[-1]] = -1
    return out

def EOS(voc, t, target_sentence, end_token):
    out = torch.zeros_like(voc)
    if t < len(target_sentence):
        # voc == end_token 也許再改
        out[voc == end_token] = -1
    return out

class Updater:
    def __init__(self, model, R: function, vocabulary, p_drop=0.5, W=10000, J=1) -> None:
        self.p_drop = p_drop
        self.W = W
        self.J = J
        self.R = R
        self.model = model
        self.voc = vocabulary

    def updata(self, x, y):
        L_BBSPG = 0
        loss = 0 
        rewards = []
        for j in range(1, self.J+1):
            z = []
            for t in range(1, self.T+1):
                mu = torch.rand((1, ))

                model_output = self.model() # what should be input to model? x? 
                if mu > self.p_drop:
                    # 參數傳入也許再改
                    # eos 應該可以從voc得到 所以應該寫在EOS裡而不虛傳入
                    delta_r = self.W * (self.R(z, self.voc, t, y) - self.R(z, self.voc, t-1, y) + DUP(z, self.voc) + EOS(self.voc, t, y, eos))

                    prob = softmax(torch.log(model_output) + delta_r)
                    zt_idx = Categorical(probs=prob).sample()
                    z_t = self.voc[zt_idx]

                    L_BBSPG -= torch.log(model_output[zt_idx])
                    L_BBSPG /= self.J
                    loss += L_BBSPG.item()
                    L_BBSPG.backward()
                
                else:
                    prob = model_output
                    z_t = self.voc[Categorical(probs=prob).sample()]

                z.append(z_t)

            rewards.append(self.R(z, self.voc, len(z), y))

        return torch.mean(rewards), loss
        