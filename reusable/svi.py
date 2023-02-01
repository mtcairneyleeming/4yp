"""
Generic methods to run SVI like normal training.
"""


import jax
import jax.lax as lax
import jax.random as random





def svi_training_epoch(svi, rng_key, svi_state, epoch_size, gp_batch_gen, batch_size):

    def body_fn(i, val):
        rng_key_i = random.fold_in(rng_key, i)

        loss_sum, svi_state = val
    
        batch = gp_batch_gen(rng_key_i)
        svi_state, loss = svi.update(svi_state, batch["y"])
        loss_sum += loss/ batch_size

        return loss_sum, svi_state

    return lax.fori_loop(0, epoch_size, body_fn, (0.0, svi_state))



def svi_test_eval(svi, rng_key, svi_state, num_test, gp_batch_gen, batch_size):
    def body_fn(i, val):
        rng_key_i = random.fold_in(rng_key, i)

    
        batch = gp_batch_gen(rng_key_i)
        loss = svi.evaluate(svi_state, batch["y"]) 
        val += loss/ batch_size

        return val

    loss =  lax.fori_loop(0, num_test, body_fn, 0.0)
    return loss / num_test



