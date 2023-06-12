from model import Attention, Decoder, Encoder, Seq2Seq
import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical
from dataloader import get_vocab, text_transform
from torchtext.data.utils import get_tokenizer

@torch.no_grad()
def inference(model, ipt, voc):
    '''
    Inference pipeline.
    '''
    model.eval()
    ipt = ipt.to(device)
    output = torch.tensor([voc['<BOS>']], device=device)[None]
    print(ipt.shape, output)

    for t in range(1, len(ipt)+1):
        model_output = model(ipt, output)
        # model_output: (B=1, C), C = voc.size
        top = torch.topk(model_output, k=50, dim=1)
        prob = top.values
        zt_idx = Categorical(probs=prob).sample() # (B,)
        word_idx = top.indices[zt_idx]

        output = torch.cat([output, word_idx[None]], dim=0) # (T, B) token id
    
    print(voc.get_itos()[output])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = get_tokenizer('basic_english')
    SRC_vocab, TRG_vocab = get_vocab()

    model = Seq2Seq(
        encoder=Encoder(
            # TODO: paramater of Encoder
            input_dim=len(SRC_vocab),
            emb_dim=512,
            enc_hid_dim=128,
            dec_hid_dim=128,
            dropout=0.2
        ),
        decoder=Decoder(
            # TODO: paramater of Decoder
            output_dim=len(TRG_vocab),
            emb_dim=512,
            enc_hid_dim=128,
            dec_hid_dim=128,
            dropout=0.2,
            attention=Attention(128, 128)
        ),
        device=device
    ).to(device)

    example = '''
    Reinforcement learning is the study of optimal sequential decision-making in an environment [16]. Its recent developments underpin a large variety of applications related to robotics [11, 5] and games [20].
    Policy search in reinforcement learning refers to the search for optimal parameters for a given policy parameterization [5]. Policy search based on policy-gradient [26, 21] has been recently applied to
    structured output prediction for sequence generations.
    '''
    example = torch.tensor(text_transform(example, SRC_vocab))[:, None]
    print(example)
    inference(model, example, TRG_vocab)