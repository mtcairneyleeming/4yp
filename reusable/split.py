import jax.numpy as jnp
import jax.random as random
import numpy as onp
import spacv

def calculate_obs_fracs(obs_fracs, n, rng_key):
    obs_idx_lst = []
    for i, frac in enumerate(obs_fracs):
        num_obs = int(frac * n)
        obs_mask = jnp.concatenate((jnp.full((num_obs), True), jnp.full((n - num_obs), False)))
        obs_mask = random.permutation(random.fold_in(rng_key, i), obs_mask)

        obs_idx_lst.append(jnp.array([x for x in range(n) if obs_mask[x] == True]))

    return obs_idx_lst
        

def calculate_spatial_cv(num_splits, geometry, n, rng_key):
    obs_idx_lst = []
    skcv = spacv.SKCV(n_splits=num_splits, random_state=onp.random.RandomState(rng_key)).split(
       geometry
    )

    for (train, test) in skcv:
        train = train[:-1]
        assert train.size + test.size == n and jnp.array_equiv(
            jnp.sort(jnp.concatenate((train, test))), jnp.arange(n)
        )
        obs_idx_lst.append(jnp.array(train))

    return obs_idx_lst

def generate_split_titles(obs_fracs, n, num_cv_splits):
    titles_list = []

    for i, frac in enumerate(obs_fracs):
        num_obs = int(frac * n)
        titles_list.append(f"{num_obs} ({int(obs_fracs[i]*100)}%) randomly chosen observations ")

    for i in range(num_cv_splits):
        titles_list.append(f"cross validation split {i+1}/{num_cv_splits}")

    return titles_list