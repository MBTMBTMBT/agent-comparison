import torch
import torch.nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(256, num_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x


class FlexibleImageEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_size):
        super(FlexibleImageEncoder, self).__init__()
        self.feature_extractor = SimpleCNN(input_channels, 256)
        self.adapt_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(256, output_size),
            # torch.nn.LeakyReLU(inplace=True),
            # torch.nn.Linear(128, 128),
            # torch.nn.Tanh(),
            # torch.nn.Linear(128, output_size),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = torch.tanh(x)
        x = self.fc(x)
        x = torch.tanh(x)
        return x


class FlexibleImageDecoder(torch.nn.Module):
    def __init__(self, n_latent_dims, img_channels, img_size, initial_scale_factor=2, num_hidden_channels=32):
        """
        Initializes the flexible generator model with customizable parameters for image generation.

        :param n_latent_dims: Dimension of the latent space (input vector), which determines the complexity of the input feature representation.
        :param img_channels: Number of channels in the output image (e.g., 3 for RGB images), defining the color space of the generated image.
        :param img_size: Tuple of two ints for the size of the output image, setting the spatial dimensions of the generated image.
        :param initial_scale_factor: Factor to determine the initial tensor size before upsampling, affecting the number of upsampling stages.
        :param num_hidden_channels: Number of channels in the hidden layers, allowing for control over the model's capacity and the complexity of features it can learn.
        """
        super(FlexibleImageDecoder, self).__init__()
        self.num_hidden_channels = num_hidden_channels
        # Adjust for img_size being a tuple (height, width)
        self.init_height = img_size[0] // initial_scale_factor
        self.init_width = img_size[1] // initial_scale_factor

        # Expanded fully connected layers for increased complexity
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(n_latent_dims, 256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(256, num_hidden_channels * self.init_height * self.init_width)
            # Output layer to match the size for convolutions
        )

        # Dynamically create the upsampling layers based on initial_scale_factor
        layers = []
        current_scale_factor = initial_scale_factor
        while current_scale_factor > 1:
            layers += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(num_hidden_channels, num_hidden_channels,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(num_hidden_channels),
                torch.nn.LeakyReLU(0.2, inplace=True)
            ]
            # num_hidden_channels = 64 if current_scale_factor == 2 else num_hidden_channels
            current_scale_factor //= 2

        # Final layer to produce the output image
        layers += [
            torch.nn.Conv2d(num_hidden_channels, img_channels, kernel_size=3, stride=1, padding=1),
            # torch.nn.Tanh()
            torch.nn.ReLU()
        ]

        self.conv_blocks = torch.nn.Sequential(*layers)

    def forward(self, z):
        out = self.fc_layers(z)
        # Adjusted view operation for potentially non-square initial sizes
        out = out.view(-1, self.num_hidden_channels, self.init_height, self.init_width)
        img = self.conv_blocks(out)
        return img


class RewardPredictor(torch.nn.Module):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims + n_actions, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims + n_actions, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.LeakyReLU(inplace=True), ] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])

        self.reward_predictor = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1, a):
        a_logits = F.one_hot(a, num_classes=self.n_actions).float()
        context = torch.cat((z0, z1, a_logits), -1)
        reward = self.reward_predictor(context).squeeze(-1)
        return reward


class InvNet(torch.nn.Module):
    def __init__(self, n_actions, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, n_actions)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.inv_model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        a_logits = self.inv_model(context)
        return a_logits


class ContrastiveNet(torch.nn.Module):
    def __init__(self, n_latent_dims=4, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.frozen = False

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_latent_dims, 1)])
        else:
            self.layers.extend(
                [torch.nn.Linear(2 * n_latent_dims, n_units_per_layer),
                 torch.nn.Tanh()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.Tanh()] * (n_hidden_layers - 1))
            self.layers.extend([torch.nn.Linear(n_units_per_layer, 1)])
        self.layers.extend([torch.nn.Sigmoid()])
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        fakes = self.model(context).squeeze()
        return fakes


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:23]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return torch.nn.functional.mse_loss(vgg_input_features, vgg_target_features)


class FeatureNet(torch.nn.Module):
    def __init__(
            self,
            n_actions,
            n_latent_dims=4,
            n_hidden_layers=1,
            n_units_per_layer=32,
            lr=0.001,
            img_size=(128, 128),
            initial_scale_factor=2,
            weights=None,
            device=torch.device('cpu')
    ):
        super().__init__()
        if weights is None:
            weights = {'inv': 1.0, 'dis': 1.0, 'neighbour': 1.0, 'dec': 0.0, 'rwd': 0.0, 'demo': 0.0, }
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.n_units_per_layer = n_units_per_layer
        self.n_hidden_layers = n_hidden_layers
        self.lr = lr
        self.device = device
        self.weights = weights

        self.phi = FlexibleImageEncoder(
            input_channels=3,
            output_size=n_latent_dims,
        ).to(device)

        if weights['inv'] > 0.0:
            self.inv_model = InvNet(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_units_per_layer=n_units_per_layer,
                n_hidden_layers=n_hidden_layers,
            ).to(device)
        else:
            self.inv_model = None
        if weights['dis'] > 0.0:
            self.discriminator = ContrastiveNet(
                n_latent_dims=n_latent_dims,
                n_hidden_layers=1,
                n_units_per_layer=n_units_per_layer,
            ).to(device)
        else:
            self.discriminator = None
        if weights['dec'] > 0.0:
            self.decoder = FlexibleImageDecoder(
                n_latent_dims=n_latent_dims,
                img_channels=3,
                img_size=img_size,
                initial_scale_factor=initial_scale_factor,
            ).to(device)
        else:
            self.decoder = None
        if weights['rwd'] > 0.0:
            self.reward_predictor = RewardPredictor(
                n_actions=n_actions,
                n_latent_dims=n_latent_dims,
                n_units_per_layer=n_units_per_layer,
                n_hidden_layers=n_hidden_layers,
            ).to(device)
        else:
            self.reward_predictor = None

        self.cross_entropy = torch.nn.CrossEntropyLoss().to(device)
        self.bce_loss = torch.nn.BCELoss().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.mae = torch.nn.L1Loss().to(device)
        self.perceptual = PerceptualLoss().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        pass

    def inverse_loss(self, z0, z1, a):
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def ratio_loss(self, z0, z1):
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0).to(self.device)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def demo_loss(self, z0, z1, same_optimal_policy: torch.Tensor):
        # same_optimal_policy = 1 - same_optimal_policy  # inverse the logic so that same is 0.
        # different is 0, same is 1.
        z0_mul = torch.mul(z0, same_optimal_policy.view(z0.shape[0], 1,))
        z1_mul = torch.mul(z1, same_optimal_policy.view(z0.shape[0], 1,))
        return self.mse(z0_mul, z1_mul)

    def pixel_loss(self, x, z):
        self.decoder.train()
        fake_x = self.decoder(z)

        # Check if x needs to be resized to match fake_x's size
        if x.size() != fake_x.size():
            x = torch.nn.functional.interpolate(x, size=fake_x.size()[2:], mode='bilinear', align_corners=False)

        return self.mse(fake_x, x)  # + self.perceptual(fake_x, x)

    def reward_loss(self, z0, z1, a, r):
        reward_pred = self.reward_predictor(z0, z1, a)
        return self.mse(reward_pred, r)

    def forward(self, x):
        return self.phi(x)

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, x0, x1, z0, z1, a, r=None, d=None):
        loss = torch.tensor(0.0).to(self.device)
        inv_loss = self.inverse_loss(z0, z1, a) if self.weights['inv'] > 0.0 else torch.tensor(0.0)
        ratio_loss = self.ratio_loss(z0, z1) if self.weights['dis'] > 0.0 else torch.tensor(0.0)
        neighbour_loss = self.mse(z0, z1) if self.weights['neighbour'] > 0.0 else torch.tensor(0.0)
        pixel_loss = 0.5 * (self.pixel_loss(x0, z0) + self.pixel_loss(x1, z1)) if self.weights[
                                                                                      'dec'] > 0.0 else torch.tensor(
            0.0)
        loss += self.weights['inv'] * inv_loss
        loss += self.weights['dis'] * ratio_loss
        loss += self.weights['neighbour'] * neighbour_loss
        loss += self.weights['dec'] * pixel_loss

        # Initialize reward_loss and demo_loss to zero
        reward_loss = torch.tensor(0.0)
        demo_loss = torch.tensor(0.0)

        # Compute reward_loss if 'r' is not None and add it to the total loss
        if r is not None:
            reward_loss = self.reward_loss(z0, z1, a, r) if self.weights['rwd'] > 0.0 else torch.tensor(0.0)
            loss += self.weights['rwd'] * reward_loss

        # Compute demo_loss if 'd' is not None and add it to the total loss
        if d is not None:
            demo_loss = self.demo_loss(z0, z1, d) if self.weights['demo'] > 0.0 else torch.tensor(0.0)
            loss += self.weights['demo'] * demo_loss

        # Return the total loss and the individual components
        return loss, inv_loss, neighbour_loss, ratio_loss, pixel_loss, reward_loss, demo_loss

    def train_batch(self, x0, x1, a, r=None, d=None):
        self.train()
        self.phi.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        # z1_hat = self.fwd_model(z0, a)
        loss, inv_loss, neighbour_loss, ratio_loss, pixel_loss, reward_loss, demo_loss = self.compute_loss(x0, x1, z0, z1, a, r, d)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().item(), inv_loss.detach().cpu().item(), neighbour_loss.detach().cpu().item(), ratio_loss.detach().cpu().item(), pixel_loss.detach().cpu().item(), reward_loss.detach().cpu().item(), demo_loss.detach().cpu().item()

    def save(self, checkpoint_path, counter=-1, _counter=-1, performance=0.0):
        torch.save(
            {
                'counter': counter,
                '_counter': _counter,
                'phi': self.phi.state_dict(),
                'inv_model': self.inv_model.state_dict() if self.weights['inv'] > 0.0 else None,
                'discriminator': self.discriminator.state_dict() if self.weights['dis'] > 0.0 else None,
                'decoder': self.decoder.state_dict() if self.weights['dec'] > 0.0 else None,
                'reward_predictor': self.reward_predictor.state_dict() if self.weights['rwd'] > 0.0 else None,
                'optimizer': self.optimizer.state_dict(),
                'performance': performance,
                'weights': self.weights,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        weights = checkpoint['weights']  # not self
        self.phi.load_state_dict(checkpoint['phi'])
        if weights['inv'] > 0.0:
            self.inv_model.load_state_dict(checkpoint['inv_model'])
        if weights['dis'] > 0.0:
            self.discriminator.load_state_dict(checkpoint['discriminator'])
        if weights['dec'] > 0.0:
            self.decoder.load_state_dict(checkpoint['decoder'])
        if weights['rwd'] > 0.0:
            self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['counter'], checkpoint['_counter'], checkpoint['performance']
