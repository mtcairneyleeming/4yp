""" Unbiased sample moments up to order ???"""

import jax.numpy as jnp

def sample_central_moment(order, sample):
    mean = jnp.mean(sample, axis=0)
    if order == 1:
        return mean
    return jnp.mean(jnp.power(sample -mean, order), axis=0)

def moment(order, sample):
    """Given a sample, calculate the h-statistic for it, https://mathworld.wolfram.com/h-Statistic.html 
     also https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7546107/pdf/nihms-1546397.pdf - both agree 
      
    Note the division have to be repeated to avoid overflowing numpy's int32 type """
    
    n = sample.shape[0]
    
    match order:
        case 0:
            return 1
        case 1:
            return sample_central_moment(1, sample)
        case 2:
            return n / (n-1) * sample_central_moment(2, sample)
        case 3:
            return n ** 2 / ((n-2)*(n-1)) * sample_central_moment(3, sample)
        case 4:
            return (
                3 * n * (3 - 2 * n)      / ((n-3)*(n-2)*(n-1)) * sample_central_moment(2, sample) ** 2 + 
                n * (n ** 2 - 2 * n + 3) / ((n-3)*(n-2)*(n-1)) * sample_central_moment(4, sample)
            )
        case 5:
            return ( # note simplified out (n-2) below!
                (-10 * n * n )           / ((n-4)*(n-3)*1    *(n-1)) * sample_central_moment(2, sample) * sample_central_moment(3, sample) 
                + n**2 * (n**2 -5*n +10) / ((n-4)*(n-3)*(n-2)*(n-1)) * sample_central_moment(5, sample)
            )
        case 6:
            return (
                15 * n**2 * (3*n-10)                               / ((n-5)*(n-4)*(n-3)*(n-2)*(n-1))  * sample_central_moment(2, sample) ** 3 
                - 15 * n * (n ** 3 - 8 * n**2 + 29 * n -40)        / ((n-5)*(n-4)*(n-3)*(n-2)*(n-1))  * sample_central_moment(2, sample) * sample_central_moment(4, sample)
                - 40 * n * (n ** 2 - 6 * n +10)                    / ((n-5)*(n-4)*(n-3)*(n-2)*(n-1))  * sample_central_moment(3, sample) ** 2
                + n * (n ** 4 - 9 * n **3 +31 * n **2 -39 * n +40) / ((n-5)*(n-4)*(n-3)*(n-2)*(n-1))  * sample_central_moment(6, sample)
            )
        case _:
            raise NotImplementedError()
         

def covariance(sample):
    """Treating the sample as args["n"] RVs:"""
    centred = sample - sample_central_moment(1, sample)
    variance = sample_central_moment(2, sample)
    #return jnp.divide(centred.T @ centred, variance.T @ variance)
    return centred.T @ centred

def cokurtosis(sample, order=2):
    centred = sample - sample_central_moment(1, sample)
    squared = jnp.power(centred, order)
    variance = sample_central_moment(order, sample)
    return jnp.divide(covariance(squared) , variance.T @ variance)

# def moment_matrix(order, sample):
#     print(sample.shape)
#     n = sample.shape[0]

    

#     match order:
#         case 2:
#             return jnp.cov(sample.T)
#         case 4:
#             # eq 3.12 in Mardia 1970
#             sample_cov = jnp.cov(sample)
#             print("Cov", sample_cov.shape)
#             print("Sample T", centred.T.shape)
#             print("Cov inv", jnp.linalg.inv(sample_cov.T).shape) 
#             inte = centred.T @ jnp.linalg.inv(sample_cov.T) @ centred
#             print(inte.shape)
#             return jnp.mean( jnp.power(inte, 2), axis = )
#         case _:
#             raise NotImplementedError()

