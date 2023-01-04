from utils.config import parse_args
from utils.data_loader import get_data_loader
import torch
import cv2

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_sn import WGAN_SN
import os
import wandb

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #wandb.init(project="Deep-Learning", config=args)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-SN':
        model = WGAN_SN(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    #calculate scores for original images
    train_features, train_labels = next(iter(train_loader))

    # Start model training
    if args.is_train == 'True':
        print("train")
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
