# Natural Language Toolkit: Word Finder
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

# Simplified from PHP version by Robert Klein <brathna@gmail.com>
# http://fswordfinder.sourceforge.net/

import random


# reverse a word with probability 0.5
def revword(word):
    if random.randint(1, 2) == 1:
        return word[::-1]
    return word


# try to insert word at position x,y; direction encoded in xf,yf
def step(word, x, xf, y, yf, grid):
    for i in range(len(word)):
        if grid[xf(i)][yf(i)] != "" and grid[xf(i)][yf(i)] != word[i]:
            return False
    for i in range(len(word)):
        grid[xf(i)][yf(i)] = word[i]
    return True


# try to insert word at position x,y, in direction dir
def check(word, dir, x, y, grid, rows, cols):
    if dir == 1:
        if x - len(word) < 0 or y - len(word) < 0:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y - i, grid)
    elif dir == 2:
        if x - len(word) < 0:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y, grid)
    elif dir == 3:
        if x - len(word) < 0 or y + (len(word) - 1) >= cols:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y + i, grid)
    elif dir == 4:
        if y - len(word) < 0:
            return False
        return step(word, x, lambda i: x, y, lambda i: y - i, grid)


def wordfinder(words, rows=20, cols=20, attempts=50, alph="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """
    Attempt to arrange words into a letter-grid with the specified
    number of rows and columns.  Try each word in several positions
    and directions, until it can be fitted into the grid, or the
    maximum number of allowable attempts is exceeded.  Returns a tuple
    consisting of the grid and the words that were successfully
    placed.

    :param words: the list of words to be put into the grid
    :type words: list
    :param rows: the number of rows in the grid
    :type rows: int
    :param cols: the number of columns in the grid
    :type cols: int
    :param attempts: the number of times to attempt placing a word
    :type attempts: int
    :param alph: the alphabet, to be used for filling blank cells
    :type alph: list
    :rtype: tuple
    """

    # place longer words first
    words = sorted(words, key=len, reverse=True)

    grid = []  # the letter grid
    used = []  # the words we used

    # initialize the grid
    for i in range(rows):
        grid.append([""] * cols)

    # try to place each word
    for word in words:
        word = word.strip().upper()  # normalize
        save = word  # keep a record of the word
        word = revword(word)
        for attempt in range(attempts):
            r = random.randint(0, len(word))
            dir = random.choice([1, 2, 3, 4])
            x = random.randint(0, rows)
            y = random.randint(0, cols)
            if dir == 1:
                x += r
                y += r
            elif dir == 2:
                x += r
            elif dir == 3:
                x += r
                y -= r
            elif dir == 4:
                y += r
            if 0 <= x < rows and 0 <= y < cols:
                if check(word, dir, x, y, grid, rows, cols):
                    #                   used.append((save, dir, x, y, word))
                    used.append(save)
                    break

    # Fill up the remaining spaces
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == "":
                grid[i][j] = random.choice(alph)

    return grid, used


def word_finder():
    from nltk.corpus import words

    wordlist = words.words()
    random.shuffle(wordlist)
    wordlist = wordlist[:200]
    wordlist = [w for w in wordlist if 3 <= len(w) <= 12]
    grid, used = wordfinder(wordlist)

    print("Word Finder\n")
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=" ")
        print()
    print()

    for i in range(len(used)):
        print("%d:" % (i + 1), used[i])


if __name__ == "__main__":
    word_finder()
