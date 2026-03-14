"""
Number-theoretic arithmetic functions.

Mobius function, Mertens function, and Euler totient — used in
PAC arithmetic and the FibonacciMobius class.
"""

from typing import Dict

from .primes import prime_factors


def mobius_function(n: int) -> int:
    """Compute the Mobius function mu(n).

    mu(n) = 1 if n is a product of an even number of distinct primes
    mu(n) = -1 if n is a product of an odd number of distinct primes
    mu(n) = 0 if n has a squared prime factor

    Args:
        n: Positive integer.

    Returns:
        -1, 0, or 1.

    Examples:
        >>> mobius_function(1)
        1
        >>> mobius_function(6)   # 2 × 3, two distinct primes
        1
        >>> mobius_function(12)  # 2² × 3, has squared factor
        0
        >>> mobius_function(30)  # 2 × 3 × 5, three distinct primes
        -1
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return 1

    factors = prime_factors(n)

    # Check for squared factors
    for exp in factors.values():
        if exp > 1:
            return 0

    # Count distinct prime factors
    k = len(factors)
    return (-1) ** k


def mertens_function(n: int) -> int:
    """Compute the Mertens function M(n) = sum(mu(k) for k=1..n).

    The Mertens function tracks the cumulative Mobius function.
    M(n) = O(sqrt(n)) on the Riemann Hypothesis.

    Args:
        n: Upper bound.

    Returns:
        M(n).

    Examples:
        >>> mertens_function(10)
        -1
    """
    if n < 1:
        return 0
    return sum(mobius_function(k) for k in range(1, n + 1))


def euler_totient(n: int) -> int:
    """Compute Euler's totient function phi(n).

    phi(n) = count of integers 1..n coprime to n.
    phi(n) = n * product((1 - 1/p) for p in prime_factors(n)).

    Args:
        n: Positive integer.

    Returns:
        phi(n).

    Examples:
        >>> euler_totient(12)
        4
        >>> euler_totient(13)  # 13 is prime
        12
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n == 1:
        return 1

    result = n
    factors = prime_factors(n)
    for p in factors:
        result -= result // p
    return result
