import os
import torch
import torch.nn as nn
import pytorch_lightning as pl


class MakeUp(pl.LightningModule):
    def __init__(self, params):
        super(MakeUp, self).__init__()
        self.type = params.pop('type')
        self.params = params
        if self.type == 'GRU':
            self.model = GRUAmp(**self.params)
        if self.type == 'Static':
            self.model = StaticAmp()

    def forward(self, x):
        return self.model(x)

    def reset_state(self, batch_size):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()


class GRUAmp(pl.LightningModule):
    def __init__(self, hidden_size):
        super(GRUAmp, self).__init__()
        self.rec = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)
        self.hidden_size = hidden_size
        self.state = None

    def forward(self, x):
        res = x
        x, self.state = self.rec(x, self.state)
        x = self.lin(x)
        return x + res

    def reset_state(self, batch_size=None):
        self.state = None

    def detach_state(self):
        self.state = self.state.detach()

    def export(self, out_dir: str, class_name: str, sub_class_name: str):
        with open(os.path.join(out_dir, sub_class_name + ".h"), 'w') as header_file:
            header_file.write(f'#pragma once\n\n')
            header_file.write(f'#include "{class_name}.h"\n\n')
            header_file.write(f'struct {sub_class_name}\n')
            header_file.write('{\n')
            header_file.write(f'    {sub_class_name}();\n\n')
            header_file.write(f'    static const unsigned int INPUT_SIZE = 1 ;\n')
            header_file.write(f'    static const unsigned int OUTPUT_SIZE = 1 ;\n')
            header_file.write(f'    static const unsigned int HIDDEN_SIZE = {self.hidden_size} ;\n')
            header_file.write('\n')
            header_file.write(f'    {class_name}<INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE> params;\n')
            header_file.write('};\n')
        # write source file
        with open(os.path.join(out_dir, sub_class_name + ".cpp"), "w") as source_file:
            source_file.write(f'#include "{sub_class_name}.h"\n\n')
            source_file.write(f'{sub_class_name}::{sub_class_name}() : params ')
            source_file.write('{\n')

            weight_ih_r = self.rec.weight_ih_l0[:self.hidden_size].detach().clone().cpu().numpy().flatten()
            weight_ih_z = self.rec.weight_ih_l0[self.hidden_size:self.hidden_size*2].detach().clone().cpu().numpy().flatten()
            weight_ih_n = self.rec.weight_ih_l0[self.hidden_size*2:].detach().clone().cpu().numpy().flatten()

            bias_ih_r = self.rec.bias_ih_l0[:self.hidden_size].detach().clone().cpu().numpy().flatten()
            bias_ih_z = self.rec.bias_ih_l0[self.hidden_size:self.hidden_size*2].detach().clone().cpu().numpy().flatten()
            bias_ih_n = self.rec.bias_ih_l0[self.hidden_size*2:].detach().clone().cpu().numpy().flatten()

            weight_hh_r = self.rec.weight_hh_l0[:self.hidden_size].detach().clone().cpu().numpy().flatten()
            weight_hh_z = self.rec.weight_hh_l0[self.hidden_size:self.hidden_size*2].detach().clone().cpu().numpy().flatten()
            weight_hh_n = self.rec.weight_hh_l0[self.hidden_size*2:].detach().clone().cpu().numpy().flatten()

            bias_hh_r = self.rec.bias_hh_l0[:self.hidden_size].detach().clone().cpu().numpy().flatten()
            bias_hh_z = self.rec.bias_hh_l0[self.hidden_size:self.hidden_size*2].detach().clone().cpu().numpy().flatten()
            bias_hh_n = self.rec.bias_hh_l0[self.hidden_size*2:].detach().clone().cpu().numpy().flatten()

            weight_output = self.lin.weight.detach().clone().cpu().numpy().flatten()
            bias_output = self.lin.bias.detach().clone().cpu().numpy().flatten()
            
            all_params = [weight_ih_r, weight_ih_z, weight_ih_n, bias_ih_r, bias_ih_z, bias_ih_n, weight_hh_r, weight_hh_z, weight_hh_n, bias_hh_r, bias_hh_z, bias_hh_n, weight_output, bias_output]

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


class StaticAmp(pl.LightningModule):
    def __init__(self):
        super(StaticAmp, self).__init__()
        self.cond = False
        self.gain = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=self.device))

    def forward(self, x, cond1=None, cond2=None):
        x = x + torch.tanh(self.bias)
        x = x*(torch.tanh(self.gain) + 1)
        return x

    def reset_state(self, batch_size=None):
        pass

    def detach_state(self):
        pass

    def export(self, header_path: str, source_path: str, class_name: str):
        print("Static gain is", self.forward(1.0).detach().clone().cpu().numpy().flatten())