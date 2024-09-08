import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np


class StaticComp(pl.LightningModule):
    def __init__(self, params, min_db=-80):
        super(StaticComp, self).__init__()
        self.curve_type = params.pop('type')
        self.params = params

        if self.curve_type == 'hk':
            self.model = SimpleHardKnee(**self.params)
        elif self.curve_type == 'sk':
            self.model = SimpleSoftKnee(**self.params)

        self.min_db = min_db
        self.eps = 10 ** (self.min_db / 20)
        self.test_in = torch.linspace(self.min_db, 0, 1000, device=self.device)

    def get_leg(self, cond):
        return self.model.get_leg(cond)

    def forward(self, x, conds=None):
        # Convert to dB/abs
        x = 20 * torch.log10(torch.clamp(torch.abs(x), self.eps))
        # Get static gain curve
        g = self.model(x, conds)
        # Return the gain curve
        return g

    def make_static_in(self):
        return torch.linspace(self.min_db, 0, 1000, device=self.device).unsqueeze(0).unsqueeze(2)

    def get_curve(self, cond_val):
        static_in = self.make_static_in()
        x_ret = self.model(static_in, cond_val.unsqueeze(0))
        x_ret += static_in
        return static_in, x_ret.squeeze()

    def export(self, out_dir: str, class_name: str, sub_class_name: str):
        with open(os.path.join(out_dir, sub_class_name + ".h"), 'w') as header_file:
            header_file.write(f'#pragma once\n\n')
            header_file.write(f'#include "{class_name}.h"\n\n')
            header_file.write(f'struct {sub_class_name}\n')
            header_file.write('{\n')
            header_file.write(f'    {sub_class_name}();\n\n')
            header_file.write(f'    static const size_t INPUT_SIZE = {self.model.cond_size} ;\n')
            header_file.write(f'    static const size_t OUTPUT_SIZE = {self.model.out_features} ;\n')
            header_file.write(f'    static const size_t HIDDEN_SIZE = {self.model.hidden_size} ;\n')
            header_file.write(f'    static const size_t NUM_LAYERS = 4 ;\n')
            header_file.write('\n')
            header_file.write(f'    {class_name}<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS> params;\n')
            header_file.write('};\n')
        # write source file
        with open(os.path.join(out_dir, sub_class_name + ".cpp"), "w") as source_file:
            source_file.write(f'#include "{sub_class_name}.h"\n\n')
            source_file.write(f'{sub_class_name}::{sub_class_name}() : params ')
            source_file.write('{\n')

            weight_ih = self.model.HCNet[0].weight.detach().clone().cpu().numpy().flatten()
            bias_ih = self.model.HCNet[0].bias.detach().clone().cpu().numpy().flatten()

            weight_hh1 = self.model.HCNet[1].weight.detach().clone().cpu().numpy().flatten()
            bias_hh1 = self.model.HCNet[1].bias.detach().clone().cpu().numpy().flatten()

            weight_hh2 = self.model.HCNet[3].weight.detach().clone().cpu().numpy().flatten()
            bias_hh2 = self.model.HCNet[3].bias.detach().clone().cpu().numpy().flatten()

            weight_output = self.model.HCNet[5].weight.detach().clone().cpu().numpy().flatten()
            bias_output = self.model.HCNet[5].bias.detach().clone().cpu().numpy().flatten()
            
            all_params = [weight_ih, bias_ih, np.append(weight_hh1, weight_hh2), np.append(bias_hh1, bias_hh2), weight_output, bias_output]

            for i, param in enumerate(all_params):
                source_file.write('{\n')
                for j, value in enumerate(param):
                    source_file.write('{:.15e}'.format(value))
                    if j < len(param) - 1:
                        source_file.write(',')
                    source_file.write('\n')
                if i < len(all_params) - 1:
                    source_file.write('},\n')
                else:
                    source_file.write('}\n')
            source_file.write('} {}')

# A simple two parameter static-compression curve generator
class SimpleHardKnee(pl.LightningModule):
    def __init__(self, cond_size=1, HC_hidden=20, min_db=-80):
        super(SimpleHardKnee, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = HC_hidden
        self.out_features = 2
        self.HCNet = nn.Sequential(
            nn.Linear(in_features=cond_size, out_features=HC_hidden),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.ReLU(),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.ReLU(),
            nn.Linear(in_features=HC_hidden, out_features=self.out_features)
        )

        self.min_db = min_db

    def forward(self, x, cond):
        params = self.HCNet(cond)
        threshold, ratio = self.get_pars(params)

        xsc = torch.where(x >= threshold, threshold + ((x - threshold) / ratio), x)
        return xsc - x

    def get_leg(self, cond):
        params = self.HCNet(cond)
        threshold, ratio = self.get_pars(params)

        T = str(-int(torch.round(threshold).item()))
        R = str(min(int(torch.round(ratio).item()), 30))
        T = T + ','
        R = R + ','
        T = T + '  ' if len(T) == 2 else T
        R = R + '  ' if len(R) == 2 else R
        return ' T=' + T + ' R=' + R

    # Function to convert outputs of HCNet to compression curve parameters
    def get_pars(self, params):
        threshold = params[:, 0:1].unsqueeze(1)
        threshold = self.min_db * torch.sigmoid(threshold)

        ratio = params[:, 1:2].unsqueeze(1)
        ratio = 30 * (torch.sigmoid(ratio)) + 1

        return threshold, ratio

class SimpleSoftKnee(pl.LightningModule):
    def __init__(self, cond_size=1, HC_hidden=20,  log_dom=True, min_db=-80):
        super(SimpleSoftKnee, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = HC_hidden
        self.out_features = 3
        self.HCNet = nn.Sequential(
            nn.Linear(in_features=cond_size, out_features=HC_hidden),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.ReLU(),
            nn.Linear(in_features=HC_hidden, out_features=HC_hidden),
            nn.ReLU(),
            nn.Linear(in_features=HC_hidden, out_features=self.out_features)
        )
        self.log_dom = log_dom
        self.min_db = min_db

    def forward(self, xdb, cond):
        params = self.HCNet(cond)
        threshold, ratio, kw = self.get_pars(params)

        xind = 2*(xdb-threshold)

        out1 = torch.where(xind < -kw, xdb, torch.zeros(1, device=self.device))
        out2 = torch.where(torch.abs(xind) <= kw,
                           xdb + (((1/ratio) - 1) *((xdb - threshold + (kw/2))**2) / (2*kw)), torch.zeros(1, device=self.device))
        out3 = torch.where(xind > kw, threshold + ((xdb - threshold)/ratio), torch.zeros(1, device=self.device))
        return out1 + out2 + out3 - xdb

    def get_leg(self, cond):
        params = self.HCNet(cond)
        threshold, ratio, kw = self.get_pars(params)

        T = str(-int(torch.round(threshold).item()))
        R = str(int(torch.round(ratio).item()))
        W = str(int(torch.round(kw).item()))
        T = T + ','
        R = R + ','
        W = W + ','
        T = T + '  ' if len(T) == 2 else T
        R = R + '  ' if len(R) == 2 else R
        W = W + '  ' if len(W) == 2 else W
        return ' T=' + T + ' R=' + R + ' W=' + W

    # Function to convert outputs of HCNet to compression curve parameters
    def get_pars(self, params):
        threshold = params[:, 0:1].unsqueeze(1)
        threshold = self.min_db * torch.sigmoid(threshold)

        ratio = params[:, 1:2].unsqueeze(1)
        ratio = 30 * (torch.sigmoid(ratio)) + 1

        kw = params[:, 2:3].unsqueeze(1)
        kw = 30*(torch.sigmoid(kw))

        return threshold, ratio, kw
