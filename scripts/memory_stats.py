#!/usr/bin/env python
import torch

def show_memory():
    prop = torch.cuda.get_device_properties(0)
    total = round(prop.total_memory/1024**3,3)
    reserved = round(torch.cuda.max_memory_reserved()/1024**3,3)
    print(f"GPU: {prop.name} | Total: {total} GB | Reserved: {reserved} GB")

if __name__=='__main__':
    show_memory()