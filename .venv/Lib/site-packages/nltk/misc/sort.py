# Natural Language Toolkit: List Sorting
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
This module provides a variety of list sorting algorithms, to
illustrate the many different algorithms (recipes) for solving a
problem, and how to analyze algorithms experimentally.
"""
# These algorithms are taken from:
# Levitin (2004) The Design and Analysis of Algorithms

##################################################################
# Selection Sort
##################################################################


def selection(a):
    """
    Selection Sort: scan the list to find its smallest element, then
    swap it with the first element.  The remainder of the list is one
    element smaller; apply the same method to this list, and so on.
    """
    count = 0

    for i in range(len(a) - 1):
        min = i

        for j in range(i + 1, len(a)):
            if a[j] < a[min]:
                min = j

            count += 1

        a[min], a[i] = a[i], a[min]

    return count


##################################################################
# Bubble Sort
##################################################################


def bubble(a):
    """
    Bubble Sort: compare adjacent elements of the list left-to-right,
    and swap them if they are out of order.  After one pass through
    the list swapping adjacent items, the largest item will be in
    the rightmost position.  The remainder is one element smaller;
    apply the same method to this list, and so on.
    """
    count = 0
    for i in range(len(a) - 1):
        for j in range(len(a) - i - 1):
            if a[j + 1] < a[j]:
                a[j], a[j + 1] = a[j + 1], a[j]
                count += 1
    return count


##################################################################
# Merge Sort
##################################################################


def _merge_lists(b, c):
    count = 0
    i = j = 0
    a = []
    while i < len(b) and j < len(c):
        count += 1
        if b[i] <= c[j]:
            a.append(b[i])
            i += 1
        else:
            a.append(c[j])
            j += 1
    if i == len(b):
        a += c[j:]
    else:
        a += b[i:]
    return a, count


def merge(a):
    """
    Merge Sort: split the list in half, and sort each half, then
    combine the sorted halves.
    """
    count = 0
    if len(a) > 1:
        midpoint = len(a) // 2
        b = a[:midpoint]
        c = a[midpoint:]
        count_b = merge(b)
        count_c = merge(c)
        result, count_a = _merge_lists(b, c)
        a[:] = result  # copy the result back into a.
        count = count_a + count_b + count_c
    return count


##################################################################
# Quick Sort
##################################################################


def _partition(a, l, r):
    p = a[l]
    i = l
    j = r + 1
    count = 0
    while True:
        while i < r:
            i += 1
            if a[i] >= p:
                break
        while j > l:
            j -= 1
            if j < l or a[j] <= p:
                break
        a[i], a[j] = a[j], a[i]  # swap
        count += 1
        if i >= j:
            break
    a[i], a[j] = a[j], a[i]  # undo last swap
    a[l], a[j] = a[j], a[l]
    return j, count


def _quick(a, l, r):
    count = 0
    if l < r:
        s, count = _partition(a, l, r)
        count += _quick(a, l, s - 1)
        count += _quick(a, s + 1, r)
    return count


def quick(a):
    return _quick(a, 0, len(a) - 1)


##################################################################
# Demonstration
##################################################################


def demo():
    from random import shuffle

    for size in (10, 20, 50, 100, 200, 500, 1000):
        a = list(range(size))

        # various sort methods
        shuffle(a)
        count_selection = selection(a)
        shuffle(a)
        count_bubble = bubble(a)
        shuffle(a)
        count_merge = merge(a)
        shuffle(a)
        count_quick = quick(a)

        print(
            ("size=%5d:  selection=%8d,  bubble=%8d,  " "merge=%6d,  quick=%6d")
            % (size, count_selection, count_bubble, count_merge, count_quick)
        )


if __name__ == "__main__":
    demo()
