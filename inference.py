from model import Attention, Decoder, Encoder, Seq2Seq
import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical
from dataloader import get_vocab, text_transform
from torchtext.data.utils import get_tokenizer
import numpy as np

@torch.no_grad()
def inference(model, ipt, voc, temperature=2):
    '''
    Inference pipeline.
    '''
    model.eval()
    ipt = ipt.to(device)
    output = torch.tensor([voc['<BOS>']], device=device)[None]

    for t in range(1, len(ipt)+1):
        model_output = model(ipt, output)
        # model_output: (B=1, C), C = voc.size
        top = torch.topk(model_output, k=50, dim=1)
        prob = softmax(top.values/temperature, dim=1)
        zt_idx = Categorical(probs=prob).sample() # (B,)
        word_idx = top.indices[torch.arange(len(top.indices)), zt_idx]

        output = torch.cat([output, word_idx[None]], dim=0) # (T, B) token id
    
    return np.array(voc.get_itos())[output.cpu()]

def cat_str_array(array):
    head = array[0]
    space = np.array([' ']*head.shape[-1])
    for tg in array[1:]:
        head = np.core.defchararray.add(head, space)
        head = np.core.defchararray.add(head, tg)
    
    return head

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

    state_dict = torch.load('checkpoint.pth')['model_state_dict']
    model.load_state_dict(state_dict)

    example = '''
    japan 's nec corp. and UNK computer corp. of the united states said wednesday they had agreed to join forces in supercomputer sales .
    '''
    example = torch.tensor(text_transform(example, TRG_vocab))[:, None]
    print(example)
    ouput_array = inference(model, example, TRG_vocab)
    print(cat_str_array(ouput_array))