"""
Tests for fracton.number_theory module.
"""

import pytest


class TestSieve:
    """Test prime sieve."""

    def test_small_sieve(self):
        from fracton.number_theory import sieve
        assert sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_sieve_empty(self):
        from fracton.number_theory import sieve
        assert sieve(1) == []

    def test_sieve_2(self):
        from fracton.number_theory import sieve
        assert sieve(2) == [2]

    def test_sieve_count(self):
        from fracton.number_theory import sieve
        # There are 25 primes below 100
        assert len(sieve(100)) == 25


class TestIsPrime:
    """Test primality testing."""

    def test_small_primes(self):
        from fracton.number_theory import is_prime
        for p in [2, 3, 5, 7, 11, 13]:
            assert is_prime(p), f"{p} should be prime"

    def test_composites(self):
        from fracton.number_theory import is_prime
        for n in [4, 6, 8, 9, 10, 12, 15]:
            assert not is_prime(n), f"{n} should not be prime"

    def test_fibonacci_primes(self):
        from fracton.number_theory import is_prime
        # F7=13 is prime, F6=8 is not, F4=3 is prime
        assert is_prime(13)
        assert not is_prime(8)
        assert is_prime(3)

    def test_edge_cases(self):
        from fracton.number_theory import is_prime
        assert not is_prime(0)
        assert not is_prime(1)
        assert is_prime(2)


class TestPrimeFactors:
    """Test prime factorization."""

    def test_basic(self):
        from fracton.number_theory import prime_factors
        assert prime_factors(360) == {2: 3, 3: 2, 5: 1}

    def test_prime(self):
        from fracton.number_theory import prime_factors
        assert prime_factors(13) == {13: 1}

    def test_power_of_two(self):
        from fracton.number_theory import prime_factors
        assert prime_factors(64) == {2: 6}

    def test_one(self):
        from fracton.number_theory import prime_factors
        assert prime_factors(1) == {}

    def test_183(self):
        from fracton.number_theory import prime_factors
        # 183 = 3 × 61 (gravity depth)
        assert prime_factors(183) == {3: 1, 61: 1}


class TestNthPrime:
    """Test nth prime lookup."""

    def test_first_primes(self):
        from fracton.number_theory import nth_prime
        assert nth_prime(1) == 2
        assert nth_prime(2) == 3
        assert nth_prime(3) == 5
        assert nth_prime(7) == 17

    def test_invalid(self):
        from fracton.number_theory import nth_prime
        with pytest.raises(ValueError):
            nth_prime(0)


class TestMobiusFunction:
    """Test Mobius function."""

    def test_mu_1(self):
        from fracton.number_theory import mobius_function
        assert mobius_function(1) == 1

    def test_mu_primes(self):
        from fracton.number_theory import mobius_function
        # mu(p) = -1 for any prime
        assert mobius_function(2) == -1
        assert mobius_function(13) == -1

    def test_mu_squarefree_even(self):
        from fracton.number_theory import mobius_function
        # 6 = 2×3, two distinct primes → mu = 1
        assert mobius_function(6) == 1
        # 30 = 2×3×5, three primes → mu = -1
        assert mobius_function(30) == -1

    def test_mu_squared_factor(self):
        from fracton.number_theory import mobius_function
        # 12 = 2²×3, squared factor → mu = 0
        assert mobius_function(12) == 0
        assert mobius_function(4) == 0


class TestMertensFunction:
    """Test Mertens function."""

    def test_mertens_10(self):
        from fracton.number_theory import mertens_function
        # M(10) = 1+(-1)+(-1)+0+(-1)+1+(-1)+0+0+1 = -1
        assert mertens_function(10) == -1

    def test_mertens_1(self):
        from fracton.number_theory import mertens_function
        assert mertens_function(1) == 1


class TestEulerTotient:
    """Test Euler's totient function."""

    def test_prime(self):
        from fracton.number_theory import euler_totient
        assert euler_totient(13) == 12  # prime: phi(p) = p-1

    def test_composite(self):
        from fracton.number_theory import euler_totient
        assert euler_totient(12) == 4  # {1,5,7,11}

    def test_power_of_two(self):
        from fracton.number_theory import euler_totient
        assert euler_totient(8) == 4  # 2^3: phi = 2^2

    def test_one(self):
        from fracton.number_theory import euler_totient
        assert euler_totient(1) == 1
