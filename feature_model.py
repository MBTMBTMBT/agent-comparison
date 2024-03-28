from collections import defaultdict
import torch
import torch.nn


class SimpleCNN(torch.nn.Module):
    def __init__(self, input_channels, num_features=64):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, num_features, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x


class FlexibleImageEncoder(torch.nn.Module):
    def __init__(self, input_channels, output_size):
        super(FlexibleImageEncoder, self).__init__()
        self.feature_extractor = SimpleCNN(input_channels)
        self.adapt_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, output_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.adapt_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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


class FeatureNet(torch.nn.Module):
    def __init__(
            self,
            n_actions,
            n_latent_dims=4,
            n_hidden_layers=1,
            n_units_per_layer=32,
            lr=0.001,
            device=torch.device('cpu')
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.device = device

        self.phi = FlexibleImageEncoder(
            input_channels=3,
            output_size=n_latent_dims,
        ).to(device)
        self.inv_model = InvNet(
            n_actions=n_actions,
            n_latent_dims=n_latent_dims,
            n_units_per_layer=n_units_per_layer,
            n_hidden_layers=n_hidden_layers,
        ).to(device)
        self.discriminator = ContrastiveNet(
            n_latent_dims=n_latent_dims,
            n_hidden_layers=1,
            n_units_per_layer=n_units_per_layer,
        ).to(device)

        self.cross_entropy = torch.nn.CrossEntropyLoss().to(device)
        self.bce_loss = torch.nn.BCELoss().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def inverse_loss(self, z0, z1, a):
        if self.coefs['L_inv'] == 0.0:
            return torch.tensor(0.0)
        a_hat = self.inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)

    def ratio_loss(self, z0, z1):
        if self.coefs['L_rat'] == 0.0:
            return torch.tensor(0.0)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N), torch.ones(N)], dim=0)

        # Compute which ones are fakes
        fakes = self.discriminator(z0_extended, z1_pos_neg)
        return self.bce_loss(input=fakes, target=is_fake.float())

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def predict_a(self, z0, z1):
        raise NotImplementedError
        # a_logits = self.inv_model(z0, z1)
        # return torch.argmax(a_logits, dim=-1)

    def compute_loss(self, z0, z1, a, d):
        loss = 0
        loss += 1.0 * self.inverse_loss(z0, z1, a)
        loss += 1.0 * self.ratio_loss(z0, z1)
        return loss

    def train_batch(self, x0, x1, a, d):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        # z1_hat = self.fwd_model(z0, a)
        loss = self.compute_loss(z0, z1, a, d)
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, checkpoint_path, counter=-1, performance=0.0):
        torch.save(
            {
                'counter': counter,
                'phi': self.phi.state_dict(),
                'inv_model': self.inv_model.state_dict(),
                'discriminator': self.discriminator.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'performance': performance,
            },
            checkpoint_path,
        )

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.phi.load_state_dict(checkpoint['phi'])
        self.inv_model.load_state_dict(checkpoint['inv_model'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['counter'], checkpoint['performance']
