#!/usr/bin/env python
import argparse, subprocess, sys
from pathlib import Path

if __name__=='__main__':
    parser = argparse.ArgumentParser("Export LoRA adapter to GGUF via llama.cpp")
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--adapter-folder', required=True)
    parser.add_argument('--outfile', default=None)
    parser.add_argument('--convert-script', default='llama.cpp/convert_lora_to_gguf.py')
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_folder)
    outfile = Path(args.outfile) if args.outfile else adapter_dir.with_suffix('.gguf')

    cmd = [
        sys.executable, args.convert_script,
        '--base-model-id', args.base_model,
        '--outfile', str(outfile),
        str(adapter_dir)
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    print('Saved GGUF to', outfile)