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
        # end_token = # TODO: get end_token form voc
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
            z = []
            for t in range(1, self.T+1):
                mu = torch.rand((1, ))

                # x: (T, B, D) 
                model_output = self.model(x, y) # what should be input to model? x?
                # model_output: (B, C), C = voc.size
                if mu > self.p_drop:
                    # 參數傳入也許再改
                    # eos 應該可以從voc得到 所以應該寫在EOS裡而不虛傳入
                    delta_r = self.W * (self.R(z, y, self.voc, t) - self.R(z, y, self.voc, t-1) + DUP(z, self.voc) + EOS(self.voc, t, y))

                    prob = softmax(torch.log(model_output) + delta_r, dim=1)
                    zt_idx = Categorical(probs=prob).sample()
                    z_t = self.voc[zt_idx] # (B, 1)

                    L_BBSPG -= torch.log(model_output[torch.arange(len(zt_idx)), zt_idx]) / self.J # (B,)
                    
                else:
                    prob = model_output
                    # voc.shape = ?
                    z_t = self.voc[Categorical(probs=prob).sample()]

                z.append(z_t)

            rewards.append(self.R(z, self.voc, len(z), y))

        L_BBSPG = L_BBSPG.mean()
        loss = L_BBSPG.item()
        L_BBSPG.backward()

        return torch.mean(rewards), loss
        