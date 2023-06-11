from pipeline import Updater
from dataloader import get_vocab, get_train_dataloader
from model import Attention, Decoder, Encoder, Seq2Seq
from tqdm import tqdm, trange
from torchmetrics import MeanMetric
from torchmetrics.functional.text.rouge import rouge_score
import torch
import numpy as np
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

def compute_rouge(z, y, voc, t):
    # z, y: token_ids (T, B)
    # Convert y: token_id to str
    targets = np.array(TRG_vocab.get_itos())[y.cpu()] # [str]
    head = targets[0]
    space = np.array([' ']*head.shape[-1])
    for t in targets[1:]:
        head = np.core.defchararray.add(head, space)
        head = np.core.defchararray.add(head, t)
    targets = head.tolist()

    tile_voc = torch.tile(torch.arange(len(voc), device=device), dims=(1, batch_size, 1))

    # print(z.cpu())
    # head = np.array(voc.get_itos())[z.cpu()]
    # tqdm.write(head)
    # tqdm.write(head.shape)
        # exit()
    return 2
    # if z.shape[0] == 0:
    #     pred = tile_voc
    # else:
    #     tile_z = torch.tile(z[..., None], dims=(1, 1, len(voc)))
    #     # print('222', tile_z.shape, tile_voc.shape)
    #     pred = torch.cat([tile_z, tile_voc], dim=0)
    
    # print(pred.shape)
    # for tokens in pred.T:


    score = rouge_score(pred, target, tokenizer=tokenizer)
    # score = {'rouge1_fmeasure': tensor(0.7500),
    #  'rouge1_precision': tensor(0.7500),
    #  'rouge1_recall': tensor(0.7500),
    #  'rouge2_fmeasure': tensor(0.),
    #  'rouge2_precision': tensor(0.),
    #  'rouge2_recall': tensor(0.),
    #  'rougeL_fmeasure': tensor(0.5000),
    #  'rougeL_precision': tensor(0.5000),
    #  'rougeL_recall': tensor(0.5000),
    #  'rougeLsum_fmeasure': tensor(0.5000),
    #  'rougeLsum_precision': tensor(0.5000),
    #  'rougeLsum_recall': tensor(0.5000)}
    return score['rouge1_fmeasure']

def train(model, train_loader):
    '''
    Training pipeline.
    '''

    for epoch in (overall:=trange(1, epochs+1, position=1, desc='[Overall]')):
        model.train()
        
        running_reward.reset()
        running_loss.reset()

        for X, Y in (bar:=tqdm(train_loader, desc=f'[Train {epoch:3d}] lr={scheduler.get_last_lr()[0]:2.2e}', position=0)):
            optimizer.zero_grad()

            X = X.to(device)
            Y = Y.to(device)

            reward, loss = updater.update(X, Y)

            running_reward.update(torch.tensor(reward))
            running_loss.update(loss)
            
            optimizer.step()

            bar.set_postfix_str(f'reward {running_reward.compute():.4f} | loss {running_loss.compute():5.4f}')
        
        # End of epoch
        scheduler.step()
        writer.add_scalar('loss', running_loss.compute(), epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('reward', running_reward.compute(), epoch)

        if epoch % 5 == 0:
            save_checkpoint(
                epoch, 
                model, 
                optimizer, 
                scheduler.state_dict(), 
                'checkpoint.pth'
            )

def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)
    tqdm.write('Save checkpoint')

def load_ckpt():
    ckpt = torch.load('checkpoint.pth')
    global start_epoch
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    running_loss = MeanMetric(accumulate=True).to(device)
    running_reward = MeanMetric(accumulate=True)

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

    writer = SummaryWriter()

    batch_size = 8
    num_workers = 4
    epochs = 50
    lr = 8e-4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max= epochs,
        eta_min= 1e-5
    )
    J = 1

    updater = Updater(
        model,
        R=compute_rouge,
        vocabulary=SRC_vocab,
        J=J,
        device=device
    )

    tokenizer = get_tokenizer('basic_english')

    train_loader = get_train_dataloader(batch_size)

    train(model, train_loader)