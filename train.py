from pipeline import Updater
from dataloader import 
from model import Attention, Decoder, Encoder, Seq2Seq
from tqdm import tqdm, trange
from torchmetrics import MeanMetric
import torch
from torch.utils.tensorboard import SummaryWriter

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

            running_reward.update(reward)
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
    running_reward = MeanMetric(accumulate=True).to(device)

    model = pass

    writer = SummaryWriter()

    batch_size = 128
    num_workers = 4
    epochs = 50
    lr = 8e-4
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.CosineAnnealingLR(
        optimizer,
        T_max= epochs,
        eta_min= 1e-5
    )
    updater = Updater(
        model,
        optimizer,
        R=,
        vocabulary=,
    )

    train_loader = Dataloader(

    )

    train(model, train_loader)