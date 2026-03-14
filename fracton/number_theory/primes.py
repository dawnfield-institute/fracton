"""
Prime number utilities.

Efficient sieve, primality testing, and factorization for use
in PAC arithmetic and Fibonacci number theory.
"""

import math
from typing import Dict, List


def sieve(n: int) -> List[int]:
    """Sieve of Eratosthenes: all primes up to n.

    Args:
        n: Upper bound (inclusive).

    Returns:
        Sorted list of primes <= n.

    Examples:
        >>> sieve(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]


def is_prime(n: int) -> bool:
    """Test whether n is prime.

    Uses trial division for small n, sufficient for PAC applications.

    Args:
        n: Non-negative integer.

    Returns:
        True if n is prime.

    Examples:
        >>> is_prime(13)
        True
        >>> is_prime(15)
        False
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_factors(n: int) -> Dict[int, int]:
    """Compute prime factorization of n.

    Args:
        n: Positive integer.

    Returns:
        Dict mapping prime factors to their exponents.

    Examples:
        >>> prime_factors(360)
        {2: 3, 3: 2, 5: 1}
    """
    if n <= 1:
        return {}
    factors: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def nth_prime(n: int) -> int:
    """Return the nth prime (1-indexed: nth_prime(1) = 2).

    Uses a simple sieve with an upper bound estimate.

    Args:
        n: Which prime to return (1-indexed).

    Returns:
        The nth prime number.

    Examples:
        >>> nth_prime(1)
        2
        >>> nth_prime(7)
        17
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return 2
    # Upper bound: p_n < n * (ln(n) + ln(ln(n))) + 3 for n >= 6
    if n < 6:
        return [2, 3, 5, 7, 11][n - 1]
    upper = int(n * (math.log(n) + math.log(math.log(n))) + 3)
    primes = sieve(upper)
    while len(primes) < n:
        upper = int(upper * 1.5)
        primes = sieve(upper)
    return primes[n - 1]
