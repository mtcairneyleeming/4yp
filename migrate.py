import os
import dill
import signal
from flax import serialization
from flax.core.frozen_dict import freeze
import jax.numpy as jnp




def old_save_args(name, args):

    with open(f"{__get_savepath()}/{name}_args.dill", "wb+") as f:
        dill.dump(args, f)


def decoder_filename(file_code, args, suffix="_decoder", leave_out=[]):
    """Return a file name that reflects the params used to generate the saved weights. If the structure of args changes, this will gracefully fail,
    as it uses a default value if any of the params change."""

    param_names = [
        x
        for x in [
            "hidden_dim1",
            "hidden_dim2",
            "latent_dim",
            "vae_var",
            "num_epochs",
            "learning_rate",
            "batch_size",
            "train_num_batches",
        ]
        if x not in leave_out
    ]

    vals = []
    for p in param_names:
        vals.append(str(args.get(p, ".")))

    return f"{file_code}_" + "_".join(vals) + suffix



def old_save_training(path, final_state, metrics_history):
    if final_state is not None:
        with open(path, "wb") as file:
            file.write(serialization.to_bytes(freeze({"params": final_state.params})))

    with open(path + "_metrics_hist", "wb") as file:
        dill.dump(metrics_history, file)
    print(f"Saved {path}")


from reusable.util import save_args, __get_savepath, save_training, gen_file_name, save_samples, update_args_11
from reusable.vae import VAE, VAE_Decoder
from reusable.train_nn import SimpleTrainState
import optax
import jax.random as random


def migrate_old(code, name, args):
        


    rng_key_init = random.PRNGKey(2135)

    
    module = VAE(
        hidden_dim1=args["hidden_dim1"],
        hidden_dim2=args["hidden_dim2"],
        latent_dim=args["latent_dim"],
        out_dim=args["n"] ** args["dim"],
        conditional=True,
    )
    params = module.init(rng_key_init, jnp.ones((args["batch_size"], args["n"] ** args["dim"] + 1,)))[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.adam(args["learning_rate"])
    state = SimpleTrainState.create(apply_fn=module.apply, params=params, tx=tx, key=rng_key_init)


    #os.remove(folder + "/" + file)

    # if "loss_functions" not in args:
    #     print("HELP !!!! No loss functions to load, skipping!!!!!!")
    #     error_count += 1
    #     continue

    # for loss_fn in args["loss_functions"]:

    file_path_A = f'{__get_savepath()}/{code}/{decoder_filename(code, args, suffix="_decoder_metrics_hist")}'

    try:

        with open(file_path_A, 'rb') as file:
            metrics_history = dill.load(file)

        file_path = f'{__get_savepath()}/{code}/{decoder_filename(code, args)}'

        with open(file_path, 'rb') as file:
            bytes = file.read()
            
            decoder_params = serialization.from_bytes(freeze({"params": state.params}), bytes)

        save_training(code, gen_file_name(code, args), decoder_params, metrics_history)

        os.remove(file_path_A)
        os.remove(file_path)

    except FileNotFoundError as e:
        print("HELP!!!!! Couldn#t find a file", e)
        error_count += 1

    print("Samples next")
    with open(f'{__get_savepath()}/{decoder_filename("14", args, suffix=f"inference_true_ls_mcmc")}' + "_samples", "rb") as file:
        mcmc_samples = dill.load(file)
    save_samples(code, gen_file_name(code, args, "inference_true_ls_mcmc"), mcmc_samples)
    with open(f'{__get_savepath()}/{decoder_filename("14", args, suffix=f"inference_all_ls_mcmc")}' + "_samples", "rb") as file:
        mcmc_samples = dill.load(file)
    save_samples(code, gen_file_name(code, args, "inference_all_ls_mcmc"), mcmc_samples)

    os.remove(f'{__get_savepath()}/{decoder_filename("14", args, suffix=f"inference_true_ls_mcmc")}' + "_samples")
    os.remove(f'{__get_savepath()}/{decoder_filename("14", args, suffix=f"inference_all_ls_mcmc")}' + "_samples")

def migrate13(code, path, args):
    aL = len(args["Arange"])
    bL = len(args["Brange"])

    loss_fn = "RCL+KLD"

    for i, a in enumerate(args["Arange"]):
   
        for j, b in enumerate(args["Brange"]):
            with open(path, "rb") as f:
                new_args = dill.load(f)

            new_args = update_args_11(new_args, new_args, i, j)

            index = i + j * aL
        
            name = f"{loss_fn}_13_{index}"
        
            file_path = f'{__get_savepath()}/{decoder_filename("13/13", new_args, suffix=name)}'
            try: 
                with open(file_path + "_metrics_hist", "rb") as file:
                    metrics_history = dill.load(file)


                dummy = VAE(
                    hidden_dim1=args["hidden_dim1"],
                    hidden_dim2=args["hidden_dim2"],
                    latent_dim=args["latent_dim"],
                    out_dim=args["n"],
                    conditional=False,
                )
                dummy_params = dummy.init(random.PRNGKey(0), jnp.ones((args["n"],)))

                with open(file_path, "rb") as file:
                    bytes = file.read()
                    # serialization.to_bytes(freeze({"params": final_state.params["VAE_Decoder_0"]})))
                    # new_dummy  =  freeze({"params" : dummy_params})
                    new_state = serialization.from_bytes(dummy_params, bytes)

                save_training(code, gen_file_name(code, new_args, str(index)), new_state, metrics_history)

                os.remove(file_path + "_metrics_hist")
                os.remove(file_path)
            except FileNotFoundError as e:
                print(e)



def migrate12(code, path, args):


    for loss_fn in args["loss_fns"]:
        
        loss_fn = str.replace(loss_fn, ":", "-")
        for k in [0,1]:
            suffix2 = "_shuffle" if k ==1 else "_standard"
            

        
        
            file_path = f'{__get_savepath()}/{decoder_filename(f"{code}/{code}", args, suffix=loss_fn + suffix2)}'
            try: 
                with open(file_path + "_metrics_hist", "rb") as file:
                    metrics_history = dill.load(file)


                dummy = VAE(
                    hidden_dim1=args["hidden_dim1"],
                    hidden_dim2=args["hidden_dim2"],
                    latent_dim=args["latent_dim"],
                    out_dim=args["n"],
                    conditional=False,
                )
                dummy_params = dummy.init(random.PRNGKey(0), jnp.ones((args["n"],)))

                with open(file_path, "rb") as file:
                    bytes = file.read()
                    # serialization.to_bytes(freeze({"params": final_state.params["VAE_Decoder_0"]})))
                    # new_dummy  =  freeze({"params" : dummy_params})
                    new_state = serialization.from_bytes(dummy_params, bytes)

                save_training(code, gen_file_name(code, args, loss_fn + suffix2), new_state, metrics_history)

                os.remove(file_path + "_metrics_hist")
                os.remove(file_path)
            except FileNotFoundError as e:
                print(e)


def migrate11(path, exp_args):
    code = exp_args["experiment"]
    aL = len(exp_args[exp_args["experiment"]]["Arange"])
    bL = len(exp_args[exp_args["experiment"]]["Brange"])

    
    for loss_fn in exp_args["loss_fns"]:
        loss_fn = str.replace(loss_fn, ":", "-")

        for i, a in enumerate(exp_args[exp_args["experiment"]]["Arange"]):
    
            for j, b in enumerate(exp_args[exp_args["experiment"]]["Brange"]):
                with open(path, "rb") as f:
                    new_args = dill.load(f)

                new_args = update_args_11(new_args, exp_args[exp_args["experiment"]], i, j)

                index = i + j * aL
            
                name = f"{loss_fn}_13_{index}"
                index = i + j * aL

                old_num_epochs = new_args["num_epochs"]

                if args["experiment"] == "11_exp3":
                    name = f"{loss_fn}_{code}_{i}_{j}"
                    
                else:
                    name = f"{loss_fn}_{code}_{index}"

                if args["experiment"] == "11_exp3" or args["experiment"] == "11_exp7":
                    print("chaning",  exp_args["num_epochs"],  new_args["num_epochs"])
                    new_args["num_epochs"] = 150 # exp_args["num_epochs"]  # due to error updating arguments
            
                file_path = f'{__get_savepath("11")}/{decoder_filename("11", new_args, suffix=name)}'
            
                try: 
                    with open(file_path + "_metrics_hist", "rb") as file:
                        metrics_history = dill.load(file)


                    dummy = VAE(
                        hidden_dim1=args["hidden_dim1"],
                        hidden_dim2=args["hidden_dim2"],
                        latent_dim=args["latent_dim"],
                        out_dim=args["n"],
                        conditional=False,
                    )
                    dummy_params = dummy.init(random.PRNGKey(0), jnp.ones((args["latent_dim"],)))

                    with open(file_path, "rb") as file:
                        bytes = file.read()
                        new_state = serialization.from_bytes(dummy_params, bytes)

                    new_args["num_epochs"] = old_num_epochs

                    save_training("11", gen_file_name("11", new_args, f"{code}_{index}_{loss_fn}"), new_state, metrics_history)

                    os.remove(file_path + "_metrics_hist")
                    os.remove(file_path)
                except FileNotFoundError as e:
                    print ("!" * 400)
                    print(e)


code = "11"
folder = f"learnt_models/{code}"

error_count = 0
for file in os.listdir(folder):
    if file.endswith(".dill"):
        p = folder  + "/" +file
        print(p)
        with open(p, "rb") as f:
            args = dill.load(f)
        
        name = file.replace(".dill", "").replace(f"{code}_", "")



        if "loss_fns" in args:
            old_lfs = args["loss_fns"]
            args["loss_fns"] = [str.replace(loss_fn, ":", "-") for loss_fn in args["loss_fns"]]

        if "loss_fns" not in args:
            old_lfs = ["RCL+KLD", "0.01RCL+KLD+10mmd_rbf_sum:4.0"]
            args["loss_fns"] =["RCL+KLD", "0.01RCL+KLD+10mmd_rbf_sum-4.0"]

        save_args(code, name, args)
        
        migrate11(p, args, )

        os.remove(p)

print(error_count)
            
