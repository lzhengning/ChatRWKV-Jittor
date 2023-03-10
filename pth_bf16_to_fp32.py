import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, required=True)
args = parser.parse_args()
param = torch.load(args.input,map_location="cpu")

for key, val in param.items():
    if val.dtype == torch.bfloat16:
        param[key] = val.float()
out0, out1 = args.input.split('.')
output = out0 + "-fp32." + out1
torch.save(param, output, _use_new_zipfile_serialization=False)