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




