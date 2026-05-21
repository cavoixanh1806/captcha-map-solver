import torch
import sys

def inspect_checkpoint(ckpt_path):
    print(f"Inspecting: {ckpt_path}")
    try:
        # We need to map to cpu if cuda is not available
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            print("\nHyperparameters found:")
            if 'cfg' in hparams:
                cfg = hparams['cfg']
                print(f"  solver.pretrained_model: {cfg.get('solver', {}).get('pretrained_model')}")
                print(f"  solver.batch_size: {cfg.get('solver', {}).get('batch_size')}")
            else:
                print(f"  hparams keys: {list(hparams.keys())}")
        else:
            print("\nNo 'hyper_parameters' found in checkpoint.")
            
        print("\nState Dict Keys (first 20):")
        sd_keys = list(checkpoint['state_dict'].keys())
        for k in sd_keys[:20]:
            print(f"  {k}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        inspect_checkpoint("best-epoch032.ckpt")
