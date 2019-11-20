import argparse
from train import Trainer

def parse():
    parser = argparse.ArgumentParser(description='Basic LSTM model')
    parser.add_argument('-m', '--mode', help='Training mode or testing mode: value either train or test', required=True)
    parser.add_argument('-tr', '--trainpath', help='Training file location', default='data/train.txt')
    parser.add_argument('-te', '--testpath', help='Test file location', default='data/test.txt')
    parser.add_argument('-de', '--devpath', help='Development file location', default='data/dev.txt')
    parser.add_argument('-pr', '--pretrained_path', default='glove.50d.txt', help='Pretrained word embedding used in training')
    parser.add_argument('-nd', '--num_dimensions', type=int, default=300, help='Number of word embedding dimensions')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch size')
    # parser.add_argument('-hi', '--hidden_dim', type=int, default=700, help='Hidden size in encoder and decoder')
    # parser.add_argument('-dd', '--first_dense_dim', type=int, default=100, help='Hidden size in first dense layer')
    parser.add_argument('-pt', '--patience', type=int, default=10, help='Patience used in early stopping')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.5, help='Dropout probability for lstm and dropout layers')
    parser.add_argument('-cp', '--checkpoint_dir', default='model_checkpoint', help='Checkpoint directory')
    parser.add_argument('-ep', '--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('-tb', '--trainable', default=False, type=bool, help='Trainable param used in embedding')

    args = parser.parse_args()
    return args

def run(args):
    print('----Setting up hyperparams----:')
    print(args)
    print('-------------------------------')
    trainer = Trainer(args)
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.test()

if __name__=="__main__":
    args = parse()
    run(args)