"""
Number theory utilities for Dawn Field Theory.

Provides prime number tools, Mobius function, and Mertens function
used in PAC arithmetic and Fibonacci analysis.

Usage:
    from fracton.number_theory import sieve, is_prime, prime_factors
    from fracton.number_theory import mobius_function, mertens_function
"""

from .primes import sieve, is_prime, prime_factors, nth_prime
from .arithmetic import mobius_function, mertens_function, euler_totient

__all__ = [
    "sieve", "is_prime", "prime_factors", "nth_prime",
    "mobius_function", "mertens_function", "euler_totient",
]
