def is_power_of_2(n: int):
    """Check if n is a power of 2."""
    return n and not (n & (n - 1))


if __name__ == "__main__":
    assert is_power_of_2(256) and is_power_of_2(1)
    assert not is_power_of_2(100)
