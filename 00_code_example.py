# V quick example of what to add when running on ARC


args = {
    # GP prior configuration
    "n": 100,
    # ...
}
args.update({ 
    
    "pretrained_vae": False


})

# add here!

import reusable.util as util
util.save_args("01", args)


# Do work....


# Save params, using

path = util.get_savepath()

from flax.core.frozen_dict import freeze

if not args["pretrained_vae"]:
    decoder_params = svi.get_params(svi_state)["decoder$params"]
    #print(decoder_params)
    decoder_params = freeze({"params": decoder_params})
    args["decoder_params"] = decoder_params
    with open(f'{util.get_savepath()}/01_decoder_1d_n{args["n"]}', 'wb') as file:
       file.write(serialization.to_bytes(decoder_params))
