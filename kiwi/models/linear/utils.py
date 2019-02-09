# -*- coding: utf-8 -*-
"""Several utility functions."""


def nearly_eq_tol(a, b, tol):
    """Checks if two numbers are equal up to a tolerance."""
    return (a - b) * (a - b) <= tol


def nearly_binary_tol(a, tol):
    """Checks if a number is binary up to a tolerance."""
    return nearly_eq_tol(a, 0.0, tol) or nearly_eq_tol(a, 1.0, tol)


def nearly_zero_tol(a, tol):
    """Checks if a number is zero up to a tolerance."""
    return (a <= tol) and (a >= -tol)
