import os
import dill

def get_savepath():
    # work out where to save outputs:
    if os.path.isdir("output"): # running in an ARC job (using my submission script)
        save_path = "output"
    elif os.path.isdir("outputs/manual"): # running from the root directory of the git repo, not in a job
        save_path = "outputs/manual"
    else: # somewhere else?
        save_path = "."
    return save_path


def save_args(name, args):
        
    with open(f'{get_savepath()}/{name}_args.dill', 'wb+') as f:
        dill.dump(args, f)


def decoder_filename(file_code, args, suffix="_decoder"):
    """Return a file name that reflects the params used to generate the saved weights. If the structure of args changes, this will gracefully fail,
     as it uses a default value if any of the params change."""
    
    
    param_names = "hidden_dim1", "hidden_dim2", "latent_dim", "vae_var", "num_epochs", "learning_rate", "batch_size", "train_num_batches"
    vals = []
    for p in param_names:
        vals.append(str(args.get(p, ".")))

    return f"{file_code}_" + "_".join(vals) + suffix