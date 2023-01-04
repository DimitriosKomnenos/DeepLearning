import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import os
from torchvision import utils
import wandb
from PIL import Image as PILImage
from utils.calculate_fid_score import calc_fid_score, calc_IS_score, calc_IS_score_from_grid, calc_fid_score_grid
import torch.nn.utils.spectral_norm as spectral_norm

SAVE_PER_TIMES = 100


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.conv1 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0))
        self.conv2 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1))

        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1))
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)

        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.conv4 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.conv6 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1))
        self.conv7 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.conv8 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0))
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            self.conv1,
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1, inplace=True),

            # State (256x16x16)
            self.conv2,
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.1, inplace=True),

            # State (512x8x8)
            self.conv3,
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1, inplace=True),
            #
            self.conv4,
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.1, inplace=True),
#           
            self.conv5,
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),

            self.conv6,
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.1, inplace=True),

            self.conv7,
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.1, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            self.conv8)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 512 * 4 * 4)


class WGAN_SN(object):
    def __init__(self, args):
        print("WGAN_SN init model.")
        self.G = Generator(args.channels)
        self.D = Discriminator(args.channels)

        wandb.init(project="Deep-Learning", config=args)
        wandb.watch(self.G, log_freq=10)
        wandb.watch(self.D, log_freq=100)
        self.C = args.channels

        # check if cuda is available
        self.check_cuda(args.cuda)

        # WGAN values from paper
        self.learning_rate = 0.00005

        self.batch_size = 64
        self.weight_cliping_limit = 0.01

        # WGAN with gradient clipping uses RMSprop instead of ADAM
        self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), lr=self.learning_rate)
        self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), lr=self.learning_rate)
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.dataset = args.dataset
        self.model = args.model
        self.number_of_images = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        self.t_begin = t.time()
        # self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):

            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit
                for p in self.D.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean(0).view(1)
                d_loss_real.backward(one)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean(0).view(1)
                d_loss_fake.backward(mone)

                d_loss = d_loss_fake - d_loss_real
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()

            # Train generator
            # Compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean().mean(0).view(1)
            g_loss.backward(one)
            g_cost = -g_loss
            self.g_optimizer.step()

            # wandb.log({"Generator iteration": g_iter/self.generator_iters, "g_loss": g_loss.data})
            # wandb.log({"Discriminator iteration": d_iter, "d_loss_fake": d_loss_fake.data, "d_loss_real": d_loss_real.data})
            # print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss.data}')
            # print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake.data}, loss_real: {d_loss_real.data}')

            # Saving model and sampling images every 1000th generator iterations
            if g_iter % SAVE_PER_TIMES == 0:
                print("g iter save is:" + str(g_iter))
                self.save_model()
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(10):
                #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                #
                # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_SN_images/' + self.dataset + '/'):
                    os.makedirs('training_SN_images/' + self.dataset + '/')

                if not os.path.exists('training_SN_images_IS/' + self.dataset + '/'):
                    os.makedirs('training_SN_images_IS/' + self.dataset + '/')

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(800, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()[:64]
                grid = utils.make_grid(samples)
                # utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))

                i = 0
                for im in samples:
                    utils.save_image(im,
                                     'training_SN_images_IS/' + self.dataset + '/img_{}_{}.png'.format(
                                         str(g_iter).zfill(3), i))
                    utils.save_image(im,
                                     'training_SN_images/' + self.dataset + '/img_{}.png'.format(i))
                    i = i + 1

                wandb.log({
                    "Generated Images": [wandb.Image(im) for im in grid]})

                fid = calc_fid_score_grid('training_SN_images/' + self.dataset + '/', self.dataset)

                wandb.log({
                    "FID score for " + self.dataset + "and model " + self.model: fid/4})

                is_score = calc_IS_score('training_SN_images_IS/' + self.dataset + '/', self.dataset)
                wandb.log({
                    "IS score for " + self.dataset + " and model " + self.model: is_score})

                # Testing
                time = t.time() - self.t_begin
                # print("Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                # output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                # self.file.write(output)

                # ===========Wandb logging ============#
                # (1) Log the scalar values
                wandb.log({
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'G cost': g_cost.data,
                    'Loss G': g_loss.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                })

                if (self.dataset == "cifar"):
                    wandb.log({
                        "real_images": [wandb.Image(PILImage.fromarray(im, mode="RGB")) for im in
                                        self.real_images(images, self.number_of_images)]})
                    wandb.log({
                        "generated_images": [wandb.Image(PILImage.fromarray(im, mode="RGB")) for im in
                                             self.generate_img(z, self.number_of_images)]})
                else:
                    wandb.log({
                        "real_images": [wandb.Image(im) for im in self.real_images(images, self.number_of_images)]})
                    wandb.log({
                        "generated_images": [wandb.Image(im) for im in
                                             self.generate_img(z, self.number_of_images)]})
                # for tag, images in info.items():
                # self.logger.image_summary(tag, images, g_iter + 1)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # self.file.close()

        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 32, 32)[:number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between two noise (z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1 * alpha + z2 * (1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
