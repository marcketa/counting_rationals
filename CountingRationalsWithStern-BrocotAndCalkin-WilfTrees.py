# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
#       jupytext_version: 0.8.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.5
# ---

# %%
import numpy as np
from typing import Any, List, Union, Optional, Tuple, Callable
from fractions import Fraction
from math import *
import matplotlib.pyplot as plt
from itertools import product

# %% [markdown]
# ### Binary tree   
# #### Levels
# In the following we'll represent the `m` first successive levels of a binary tree by a list of `m` lists each one of $1,2,\cdots,2^{m-1}$ nodes representing at level $0$ the root, at level $1$ the left and right sons of the root, and at the level $k+1$ the left and right sons of the $i^{th}$ node of the level $k$  by respectively the nodes with respective indices $2i$ and $2i+1$ of this level $k+1$ where $i \in \{0,1,\ldots,k-1\}$.  
# Below a function `bin_levels(lst)` that transforms a list `lst` in the levels of the corresponding binary tree.

# %% {"code_folding": [0]}
def bin_levels(lst: List[Any]) -> List[List[Any]]:
    """ divide a list in blocks (lists) corresponding to the first levels of a binary tree
    
    Args: 
        lst:  a list
    Returns:
        a list of levels: [lvls[0],...,lvls[k],..., lvls[-1]]] 
        length(lvls[0]) == 1, lvls[0][0] is the root of the binary tree
        for each level k > 0, len(lvls[k]) == 2*len(lvls[k-1]), 
        but for the last level, len(lvls[-1]) <= 2*len(lvls[-2]) depending on len(lst)
    Example:
        if lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        bin_levels(lst) -> [[1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]
    """
    n = len(lst)
    lvls = []
    m = 1
    while 2*m <= n:
        lvls.append(lst[m-1:2*m-1])
        m = 2*m
    lvls.append(lst[m-1:n+1])
    return lvls

# %%
l13 = list(range(1,14))
print('l13 = {}'.format(l13))
print('levels:             0    1       2             3')
print('bin_levels(l13):  {}'.format(bin_levels(l13)))


# %% [markdown]
# And `print_bintree(levels)` is a pretty print for binary trees expressed in this form

# %% {"code_folding": [0]}
def print_bintree(lvls : List[List[Any]],r: int = 1, fmt: Callable[[Any],str] = str) -> None:
    """ pretty print of a binary tree defined by lvls as in the result 
        of the bin_levels function
        
    Args:
        lvls: binary tree as a list of levels: [lvls[0],...,lvls[k],..., lvls[-1]]] 
              length(lvls[0]) == 1, lvls[0][0] is the root of the binary tree
              for each level k > 0, len(lvls[k]) == 2*len(lvls[k-1]), 
              but for the last level, len(lvls[-1]) <= 2*len(lvls[-2]) 
        r:    (int) w = 2*r+1 is the width of printed label for each node.
              default: r=1 => 3 characters for each node's label
        fmt:  a function defining the print format of each node. 
              default: the function str
    Returns:
        None: this function is just for printing the first levels of a binary tree 
    """
    def bn(klvl: int, r: int, n: int) -> int:
        """returns the half-distance between two nodes at level klvl
        
        Args:
            klvl: (int) the level number
            r:    (int) w = 2*r+1 is the width of label for each node
            n:    number of levels
        Returns:
            (int) the half-distance between two nodes at level klvl
        """
        return (2*r+1)*(2**(n-1-klvl)-1)
    n = len(lvls)
    # lbn: list of the half-distance lbn[k] between two nodes at each level k
    lbn = [bn(k,r,n) for k in range(n)]
    # lu[k]: length of horizontal branches in level k
    lu = [el//2 for el in lbn]
    # w: number of chars for each node
    w = 2*r+1
    for k in range(n):
        print(''.join([(lbn[k]+r)*' ' + '|' + (lbn[k]+3*r+1)*' ' for el in lvls[k]]))
        print(''.join([(lu[k]+(k!=(n-1)))*' ' + lu[k]*'_' + '{:^{}}'.format(fmt(el),w) + 
                       lu[k]*'_'+ (lu[k]+(k!=(n-1))+2*r+1)*' ' for el in lvls[k]]))

# %%
print_bintree(bin_levels(l13))

# %% [markdown]
# #### Path string   
# Let's use the letters L and R to stand for going down to the left or right branch as we proceed from the root of a binary tree to a particular node; then a string of L's and R's uniquely identifies a place in the tree.
# For example in the tree above, to reach the node with label 12, from the root 1 we are going right to 3, left to 6 and left to 12. The node with label 12 can be adressed by the "path string" `'RLL'`.  
# The function `paths_level(k)` returns the "path strings" ordered list describing all the nodes the $k^{th}$ level of a binary tree:
# *  `['']` for the root at level k = 0  
# *  `['L', 'R']` for the left and right sons of the root at the first level k = 1
# *  `['LL', 'LR', 'RL', 'RR']` at level k = 2, for the respective left and right sons of level 1 nodes
# and so on ...

# %%
def paths_level(k: int) -> List[str]:
    """ return the path strings describing the kth level of a binary tree:
    
    Args: 
        k: an integer
    Returns:
        a list of strings, defining the ordered list of the level's nodes
    Examples:
        [''] for the root at level k = 0
        ['L', 'R'] for the left and right sons of the root at the first level k = 1,
        ['LL', 'LR', 'RL', 'RR'] at level k = 2 for the respective left and right sons of level 1 nodes
        """
    return [''.join(t) for t in product(('L','R'),repeat=k)]

# %%
print(paths_level(3))

# %% [markdown]
# #### Bits representation  

# %%
def ints2bin(ints: List[int], nbits: Optional[int] = None) -> List[str]:
    """ return the list of the binary representations on nbits of all integers in the list ints.
    
    Args:
        ints: a list of integers 
        nbits: an integer, the fixed length for all the bits string representing the integers in ints.
        If nbits == None then for each integer in ints the binary representation is the string of minimal length
    Returns:
        The list of the binary representation of the integers in ints
    Example:
        ints2bin([1,2],nbits=3) ->  ['001', '010']
        ints2bin([1,2]) ->  ['1', '10']
    """
    return [np.binary_repr(k,nbits) for k in ints]

def rev_ints(ints: List[int], nbits: Optional[int] = None) -> List[int]:
    """return the list of integers resulting of the reversing of the binary repr of the integers in  ints
    
    Args:
        ints: a list of integers 
        nbits: an integer, the fixed length for all the bits string representing the integers in ints.
        If nbits == None then for each integer in ints the binary representation is the string of minimal length   
    Returns:
        the list of integers resulting of the reversing of the binary repr of the integers in  ints
    Example:
        rev_ints([1,2],nbits=3) -> [4, 2] (['001', '010'] -> ['100', '010'])
        rev_ints([1,2]) -> [1, 1] (['1', '10'] -> ['1', '1'])
    """
    return [int(bits[::-1],2) for bits in ints2bin(ints,nbits)]

def str_translate(s: str, fromchrs: str = '01', tochrs: str = '10') -> str:
    """substitute in the string s the characters in fromchrs to those in tochrs and return the resulting string
    
    Args:
        s: a string
        fromchrs: a string listing all the characters to be changed
        tochrs: a string listing the corresponding substitutes
    Returns:
        the string resulting from the substitutions in s.
    Examples:
       str_transl('1101') -> '0010'
       str_transl('LLRL','LR','01') -> '0010'
    """
    if s == '':
        return tochrs[0]
    return s.translate(dict(list(zip((map(ord,fromchrs)),tochrs))))

# %% [markdown]
# With the substitutions: `L -> 0` and `R -> 1`, we substitute the the path string indexation of a node for the numerical index of this node in the corresponding level:  
# For example the node with path string `'RRL'` is the node number `6` of level 3 (`len('RRL')`)   
# because `str_translate('RRL','LR','01')` is `'110'` and `int('110',2)` is `6`.
# Remember that node number `6` is the $7^{th}$ node, the Python indexation beginning with`0`.  
# We define two functions to pass from one indexation to the other: `level_idx(path_string)` and `path_str(level,idx)`

# %%
def level_idx(path_string: str) -> Tuple[int, int]:
    """ returns a pair of integers(level_number,idx) identifying the positition  
        idx of a node in the list representing the level in a binary tree.
    
    Args:
        path_string: a string of 'L' and 'R',describing the left and right from the root moves to reach a node
    Returns:
        a tuple of 2 integers: the node's level number and his index position in the corresponding list
    Example:
        level_idx('RRL') -> (3, 6)
        """
    return len(path_string), int(str_translate(path_string,'LR','01'), 2)

# %%
level_idx('RRL')

# %%
def path_str(level: int, idx: int) -> str:
    """ returns the path string of the node defined by level and idx
    
    Args:
        level: (int) node's level number of the node the pair of integers(level,idx) identifies the positition idx of a node 
        idx: (int) node's index in the list representing the level
    Returns:
        the node's corresponding path string
    Example:
        path_str(3,6) == 'RRL' 
    """
    return str_translate(np.binary_repr(idx,level),'01','LR')
    

# %%
path_str(3,6)

# %% [markdown]
# These functions are inverses of each other:

# %%
level_idx(path_str(*(3,6))) == (3,6) and path_str(*level_idx('RRL')) == 'RRL'

# %% [markdown]
# If we do the same substitution on all the path strings of a level as given for example by `paths_level(3)` we  obtain obviously the list of the first $2^3 = 8$ consecutives integers:

# %%
paths_lvl3 = paths_level(3)
print(paths_lvl3)
print([str_translate(s,'LR','01') for s in paths_lvl3])
print([level_idx(s)[1] for s in paths_lvl3])

# %% [markdown]
# ### Stern Brocot Tree
# The following text cells are a transcription of the chapter on Stern-Brocot tree in the Donald Knuth's book
# _Concrete Mathematics_ (p.129) and the Python cells a programming materialization of this chapter.

# %% [markdown]
# #### There's a beautiful way to construct the set of all nonnegative fractions m/n with m ⊥ n , called the Stern-Brocot tree.  
# Using the Donald Knuth's notation we'll write $m \perp n$ for a pair of two integers $m,n$ who are relatively prime (coprime). A fraction $m/n$ is in lowest terms if and only if $m \perp n$ .
#
# The idea is to start with the two fractions (0/1, 1/0) (here we add 1/0 for $\infty$!) and then to repeat the following operation as many times as desired:   
# Insert (m + m')/(n + n') between two adjacent fractions m/n and m'/n'.  
# The new fraction **(m+m')/(n+n')** is called the **mediant** of **m/n** and **m'/n'**.  
# The first step gives us one new entry 1/1, the next two more, 1/2 between 0/1 and 1/1, and 2/1 between 1/1 and 1/0 and so on, building a binary tree with root 1/1.

# %% [markdown]
# To build this construct we'll first examine a way to produce the tree corresponding to the numerators of these nonnegative fractions.  
# We'll start with the integers(0,1) and then we repeat the following operation as many times as desired, i.e depending on the level desired:    
# Insert m + m' between two adjacent integers m and m'.   
# Build each level successively and in the same time keep a kind of "cumulative" list:  
# level 0: `[0+1] = [1]`  
# cumul: `[1]`
# level 1: `[0+1,1+1] = [1,2]`  
# cumul:`[1,1,2]`  
# level 2: `[0+1,1+1,1+2,2+1] = [1,2,3,3]` 
# etc...    
# This construction is defined from different sources:

# %% [markdown]
# #### Stern (diatomique) sequence
#
# can be defined by 
# $$s(0) = 0,\  s(1) = 1 \  \textrm{and} \ \ 
# s(2n) = s(n),\  s(2n + 1) = s(n) + s(n + 1)\ 
# \textrm{when} \ (n ≥ 1)$$
# This sequence, which appears in different notations in the literature, is sequence A002487 in
# [_The On-Line Encyclopedia of Integer Sequences_](http://oeis.org), where numerous properties and references can be found. The first few nonzero terms of
# the sequence (1) are easily seen to be **1**, **1**, 2, **1**, 3, 2, 3, **1**, 4, 3, 5, 2, 5, 3, 4, **1**, 5,. . . , where
# those with an index that is a power of 2 are shown in bold.  
# See also [Rational Trees and Binary Partitions](http://www.luschny.de/math/seq/RationalTreesAndBinaryPartitions.html)  from Peter Luschny, March 2010.  
# The next function is a free adaptation in Python from a maple program adapted from E. Dijkstra, Selected Writings on Computing, Springer, 1982, p. 232.

# %%
def stern_levels(m: int, a: int = 0, b: int = 1) -> Tuple[List[List[int]], List[int]]:
    """ This function build the first m levels of the numerators of the Stern-Brocot binary tree
        and the list of the first 2**m-1 terms of the Stern sequence
    
    Args:
        m: (int) the desired number of levels for the Stern-Brocot binary tree
        a: (int) first initial value
        b: (int) second initial value
    Returns: a tuple t = (levels, l)
        t[0]: levels, a m terms list where levels[k] is a list of 2**k integers, 
              a representation of the tree of the numerators of kth level in the Stern-Brocot tree.
        t[1]: list of the first 2**m-1 terms of the Stern sequence
    """
    l = [a,b]
    levels = []
    for k in range(1,m+1):
        l_k = [l[0]]
        level_k = []
        for i in range(len(l)-1):
            si = l[i]+l[i+1]
            level_k.append(si)
            l_k.append(si)
            l_k.append(l[i+1])
        levels.append(level_k)
        l = l_k[:]
    return levels,l[1:-1]

# %%
SBnums_5,stern_list_5 = stern_levels(5)

# %%
for l in SBnums_5:
    print(l)

# %% [markdown]
# This first result give the five first levels of the numerators tree: 

# %%
print_bintree(SBnums_5)

# %%
print(stern_list_5)

# %% [markdown]
# And we'll note that this second result `stern_levels(5)[1]`,
# is the "ordered projection" on a line of all the nodes of the 5-levels tree

# %% [markdown]
# We'll note than the denominators will be obtained in the same way starting with the integers (1,0), using the 
# the same function `stern_levels(m,1,0)` or reversing each list in the first result of `SBnums(m,0,1)`

# %%
SBdens_5, _ = stern_levels(5,1,0)

# %%
for l in SBdens_5:
    print(l)

# %% [markdown]
# or using the `Python` notation to get the reversed `l[::-1]` of a list `l`

# %%
SBdens_5 = [l[::-1] for l in SBnums_5]

# %%
for l in SBdens_5:
    print(l)

# %% [markdown]
# And for each level we can build the nodes list, each node as a pair `(numerator,denominator)`.  
# For example the nodes of level 3: 

# %%
i=3
print(list(zip(SBnums_5[i],SBdens_5[i])))


# %% [markdown]
# Or as a list of the corresponding fractions:

# %%
i=3
print([str(Fraction(*pair)) for pair in zip(SBnums_5[i],SBdens_5[i])])

# %% [markdown]
# In this way we can build the Stern-Brocot tree up to a desired level `m`.

# %%
def SBpairs(m: int) -> List[List[Tuple[int, int]]]:
    """ return the first m levels of Stern-Brocot tree, the nodes being a pair (numerator,denominator)
    
    Args:
        m: (int) the desired number of levels for the Stern-Brocot binary tree
    Returns:
        the first m levels of Stern-Brocot tree, with the pairs (numerator, denominator) as nodes
    """
    sbnums_m, _ = stern_levels(m,0,1)
    sbdens_m, _ = stern_levels(m,1,0)
    return [list(zip(sbnums_m[k],sbdens_m[k])) for k in range(m)]

# %%
for l in SBpairs(5):
    print(l)

# %%
print_bintree(SBpairs(5),fmt=lambda pair:'{},{}'.format(*pair))

# %%
print_bintree(SBpairs(5),fmt=lambda pair:str(Fraction(*pair)))

# %% [markdown]
# And if we combine the second results `stern_levels(m,a=0,b=1)[1]` and `stern_levels(m,a=1,b=0)[1]`
# we'll get the "projection", as an ordered sequence of all the fractions of the `m` levels tree just above.

# %%
m=5
for frac in [Fraction(*pair) for pair in zip(stern_levels(m,a=0,b=1)[1],stern_levels(m,a=1,b=0)[1])]:
    print(frac,end=' ')

# %% [markdown]
# #### the fundamental fact:  
# If $m/n < m'/n'$ are consecutive fractions at any
# stage of the construction, we have:   $m'n − mn' = 1$  
# By Bezout identity, it means that $m,n$ are relatively prime (or coprime) and the same for the pair $m',n'$.
# Using the Donald Knuth's notation we'll write $m \perp n$ for such a pair. A fraction $m/n$ is in lowest terms if and only if $m \perp n$.
#
# 1. It's true for $0/1$ and $1/0$ for $( 1 · 1 − 0 · 0 = 1 )$.
# Note that we accept $1/0$ for $\infty$ but we are reasoning on the pair `(1,0)`...
# 2. when we insert a new mediant $(m + m')/(n + n')$ , the new cases that need to be checked are:   
# $$(m + m')n − m(n + n') = 1 $$ 
# $$m'(n + n') − (m + m')n' = 1$$ 
# Both of these equations are equivalent to the original condition.  
# So $m \perp n$, $m' \perp n'$ and we have the same for the mediant $(m + m') \perp (n + n')$
# 3. Furthermore as the integers are non negatives and as 
# $$m'n − mn' = 1 \Leftrightarrow \frac{m'}{n'} − \frac{m}{n} = \frac{1}{nn'}$$      
# we have 
# $$\frac{m}{n} < \frac{m + m'}{n + n'} < \frac{m'}{n'}$$ 
# Therefore the construction preserves order, explaining why the "projection" above is an ordered sequence of the considered fractions and we couldn't
# possibly get the same fraction in two different places.
# 4. One question still remains. Can any positive fraction $a/b$ with $a ⊥ b$
# possibly be omitted? The answer is no, because we can confine the construction to the immediate neighborhood of $a/b$ , and in this region the behavior is easy to analyze: Initially we have:
# $$\frac{m}{n} = \frac{0}{1} \  < \  (\frac{a}{b}) \  < \ \frac{1}{0} = \frac{m'}{n'}$$ 
# where we put parentheses around $a/b$ to indicate that it's not really present
# yet. Then if at some stage we have
# $$\frac{m}{n} \  < \  (\frac{a}{b}) \  < \  \frac{m'}{n'}$$ 
# the construction forms $(m + m')/(n + n')$ and there are three cases. Either
# $(m + m' )/(n + n') = a/b$ and we win; or $(m + m')/(n + n') < a/b$ and we
# can set $m ← m + m' , n ← n + n'$ ; or $(m + m')/(n + n') > a/b$ and we
# can set $m' ← m + m' , n' ← n + n'$ . This process cannot go on indefinitely,
# because the conditions:
# $$ \frac{a}{b} - \frac{m}{n} > 0 \quad  \textrm{   and   } \quad  \frac{m'}{n'} - \frac{a}{b} > 0 $$
# imply that 
# $$ an − bm \ \geq \ 1  \quad  \textrm{   and   } \quad   bm' − an' \ \geq \ 1 $$
# Hence
# $$ (m' + n')(an − bm) + (m + n)(bm' − an') \  \geq \ m' + n'+ m + n $$ 
# and this is the same than:  
# $$ a + b \  \geq \ m' + n'+ m + n$$ 
# Either $m$ or $n$ or $m'$ or $n'$ increases at each step, so we must win after at most $a + b$ steps.  
# Conclusion: **The nodes of the Stern-Brocot tree are all the nonnegative fractions m/n with m ⊥ n.**
#
#

# %% [markdown]
# ### Calkin-Wilf  tree
# In an article named [_Recounting the rationals_](https://fermatslibrary.com/s/recounting-the-rationals#email-newsletter) Neil Calkin and Herbert S. Wilf give another tree representation of the rationals using the **Stern (diatomic) sequence**  
# this sequence can be defined by 
# $$s(0) = 0,\  s(1) = 1 \  \textrm{and} \ \ 
# s(2n) = s(n),\  s(2n + 1) = s(n) + s(n + 1)\ 
# \textrm{when} \ (n ≥ 1)$$.
# This sequence, which appears in different notations in the literature, is sequence A002487 in
# [_The On-Line Encyclopedia of Integer Sequences_](http://oeis.org/A002487), where numerous properties and references can be found. The first few nonzero terms of
# the sequence (1) are easily seen to be **1**, **1**, 2, **1**, 3, 2, 3, **1**, 4, 3, 5, 2, 5, 3, 4, **1**, 5,. . . , where
# those with an index that is a power of 2 are shown in bold.
# [Rational Trees and Binary Partitions](http://www.luschny.de/math/seq/RationalTreesAndBinaryPartitions.html)  from Peter Luschny, March 2010.  
# But the second result of `stern_levels(m)` is just the list of the first $2^m-1$ terms of this sequence (without the leading 0) and represents the first $2^m-1$ numerators of the Calkin-Wilf representation according to their article.
# For example for $m = 5$:

# %%
stern_list_5 = stern_levels(5)[1]
print(stern_list_5)

# %% [markdown]
# If we build a binary tree from this sequence we get the binary tree of the numerators of the Calkin-Wilf tree 

# %%
print_bintree(bin_levels(stern_list_5))

# %% [markdown]
# and according to the same article the corresponding denominators list is:

# %%
print(stern_list_5[1:]+[1])

# %% [markdown]
# And as described in the same article [_Recounting the rationals_](https://fermatslibrary.com/s/recounting-the-rationals#email-newsletter) we can build Calkin-Wilf tree representing all the rationals. More exactly we build a new function `CWpairs(m)` returning the first $m$ levels with the first $2^m - 1$ nodes of this tree

# %%
def CWpairs(m):
    """ return the first m levels of Calkin-Wilf tree, the nodes being a pair (numerator,denominator)
    
    Args:
        m: (int) the desired number of levels for the Calkin-Wilf binary tree
    Returns:
        the first m levels of Calkin-Wilf tree, with the pairs (numerator, denominator) as nodes
    """
    nums = stern_levels(m)[1]
    dens = nums[1:]+[1]
    return bin_levels(list(zip(nums,dens)))

# %%
print_bintree(CWpairs(5),fmt=lambda pair:'{},{}'.format(*pair))

# %%
print_bintree(CWpairs(5),fmt=lambda pair:str(Fraction(*pair)))

# %% [markdown]
# **We remark that for each `m` the Stern-Brocot tree and the Calkin-Wilf tree,  up to level `m` contain the same fractions, because, for every `0<k<m` the levels `SBpairs(m)[k]` and  `CWpairs(m)[k]` contain two different permutations of the same pairs, all issued from the same sequence `stern_levels(m)[1]`.**  
# as we saw before with `m = 5`.

# %%
print(stern_levels(5)[1])

# %% [markdown]
# ### Stern-Brocot tree  or Calkin-Wilf tree as number system for representing rational numbers
# As we presented in the chapter on **Binary Trees** we can use use the letters L and R  for going down to the left or right branch as we proceed from the root of a tree to a particular fraction and a string of L's and R's uniquely identifies a place in the tree.  
# In the Stern-Brocot tree, for example, LRLL means that we go left from 1/1 down to 1/2 , then right to 2/3 , then left to 3/5 , then left to 4/7 .  
# In this tree we can consider LRLL to be a representation of 4/7 . Every positive fraction gets represented in this way as a unique string of L's and R's.  
# Well, actually there's a slight problem: The fraction 1/1 corresponds to the empty string, and we need a notation for that. Let's agree to call it I , because that looks something like 1 and it stands for "identity."
# This representation raises two natural questions:
# 1. Given positive integers $m$ and $n$ with $m \perp n$ , what is the string of L's and R's that corresponds to $m/n$ ? 
# 2. Given a string of L's and R's, what fraction corresponds to it?  
# Question 2 seems easier, so let's work on it first.  For Stern-Brocot the we define:  
#  `SBfrac(S) =` the fraction corresponding to the path string `S`in this tree.    
# For example, `SBfrac('LRLL') == Fraction(4,7)` .   
# And we do the same for the Calkin-Wilf tree: We define    
# `CWfrac(S) =` the fraction corresponding to the path string `S`in the Calkin-Wilf tree.
# In this case `CWfrac('LRLL') == Fraction(3,8)`.

# %% [markdown]
# #### Matrix representation  
# We define some functions using `numpy arrays` for 2x2 matrices corresponding  
# to strings composed of chars in 'ILR' describing pathes in a binary string:  
# ''→ I, 'I' → I, 'L' → L, 'R' → R   
# and a string, sequence of 'L' and 'R' by the product M of the corresponding matrices L and R   
# `powmat(M,n)`   for $M^n$ where $M$ is a matrix   
# `matprod(mats)` for the matrix product $M_0 M_1 \cdots M_n$ if  $\textrm{mats} = [M_0, M_1, \cdots, M_n]$ where, for $0 \leq i \leq n$,  $M_i$  is L or R.    
# `path2mat(S)`   for the matrix product of the elementary matrices L,R present in a binary string S node's path

# %%
""" Specific 2x2 matrices in the Stern-Brocot context"""
L = np.array([[1,1],[0,1]])  # left move, left root's son
R = np.array([[1,0],[1,1]])  # right move, right root's son
I = np.eye(2,dtype=int)      #identity, root

def powmat(M: np.array, n: int) -> np.array:
    """ return the n-th power of a matrix M 
    
    Args: 
        M: (np.array) a square matrix pxp
        n: (int) exponent
    Returns:
        the n-th power of M
    """    
    assert n >= 0, "{} is not a positive or null integer".format(n)
    if n == 0:
        return np.eye(M.shape[0],dtype=int)
    return M @ powmat(M,n-1)

def matprod(mats: List[np.array], n: int = 2)-> np.array:
    """ mats is a list of matrices pxp. By defaut n == 2
    
    Args:
        mats: a list of square matrices pxp
    Returns:
        the matrix (np.array) product of all the matrices in the list in the same order
    """
    if len(mats) == 0:
        return np.eye(n,dtype=int)
    if len(mats) == 1:
        return mats[0]
    return mats[0] @ matprod(mats[1:])

def path2mat(S: str) -> np.array:
    """return the matrix product corresponding to a path string in a Stern-Brocot binary tree 
    
    Args:
        S: (str) a node's path string
    Returns:
        the corresponding product matrix (np.array)
    """
    return matprod([eval(chr) for chr in S])

print('I = ')
print(I)
print('L = ')
print(L)
print('R = ')
print(R)

# %%
powmat(L,0)

# %%
L@L@L

# %%
path2mat('')

# %%
path2mat('LRLL')

# %%
M = L@R@L@L
print(M)

# %% [markdown]
# In the Stern-Brocot tree we'll note that the fraction $\dfrac{n}{d}$ is represented by the vector $[d,n]$. In this form the node left to $\dfrac{1}{2}$ is the node $\dfrac{1}{3}$ described by the vector `L@[2,1] = [3 1]` and the node right to $\dfrac{1}{2}$ is the node described by the vector `R@[2,1] = [2 3]` for the fraction $\dfrac{3}{2}$

# %% [markdown]
# and the product 
# $\begin{equation*}
# LRLL\begin{bmatrix} 1 \\ 1 \end{bmatrix}=\begin{bmatrix} 2 & 5 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 
# \begin{bmatrix} 7 \\ 4 \end{bmatrix}
# \end{equation*}$ represents the fraction $\dfrac{4}{7}$.

# %%
print(M@[1,1])

# %% [markdown]
# **Calkin-Wilf tree matrix representation**  
# If we consider a Calkin-Wilf tree's node $\dfrac{n}{d}$  represented by the line vector $\begin{bmatrix}n & d\end{bmatrix}$   
# The product $\begin{equation*}\begin{bmatrix}n & d\end{bmatrix}L = \begin{bmatrix}n & d\end{bmatrix}\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix}n & n + d\end{bmatrix}\end{equation*}$ represents the node $\dfrac{n}{n+d}$  
# The product $\begin{equation*}\begin{bmatrix}n & d\end{bmatrix}R = \begin{bmatrix}n & d\end{bmatrix}\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix}n + d & d \end{bmatrix}\end{equation*}$ represents the node $\dfrac{n+d}{d}$   
# These are exactly the rules of construction of the Calkin-Wilf tree. 
# In this way the same matrix $M$ representing a path $S$ in the Stern-Brocot tree give the value of a node in the Calkin-Wilf tree.  
# With the same matrix product M = LRLL we get the fraction $\dfrac{3}{8}$ in the Calkin-Wilf tree:

# %%
print([1,1]@L@R@L@L)
print([1,1]@M)

# %% [markdown]
# generalizing we can define the function `SBfrac(S)` in this way:

# %%
def SBfrac(S: str) -> Tuple[int, int]:  # -> Fraction[int, int] type hint doesn't recognize Fraction
    """ return the Stern-Brocot node value as the fraction corresponding to the string path S
    
    Args:
        S: (str) a Stern-Brocot node path string
    Returns:
        the Fraction value of the corresponding node
    Example:
        SBfrac('LRLL') -> Fraction(4, 7)
    """
    M = matprod([eval(chr) for chr in S])
    den, num = M@[1,1]
    return Fraction(num, den)

# %%
print(SBfrac('LRLL'))

# %%
SBfrac('LRLL')

# %% [markdown]
# and we can define the function `CWfrac(S)` returning the fraction value of Calkin-Wilf tree's node corresponding to the string path S by:

# %%
def CWfrac(S: str) -> Tuple[int, int]:  # -> Fraction[int, int] type hint doesn't recognize Fraction
    """ return the Calkin-Wilf node value as the fraction corresponding to the string path S
    
    Args:
        S: (str) a Calkin-Wilf node path string
    Returns:
        the Fraction value of the corresponding node
    Example:
        CWfrac('LRLL') -> Fraction(3, 7)
    """
    M = matprod([eval(chr) for chr in S])
    num,dem = [1,1]@M
    return Fraction(num,dem)

# %%
print(CWfrac('LRLL'))

# %% [markdown]
# Another way of interpreting this situation is to notice that a fraction $\dfrac{n}{d}$ corresponding to a certain string path `S` in one of the tree correspond to the reversed path `S[::-1]` in the another tree, or that the associated matrix `M` in one tree corresponds to the transpose matrix `M.T` in the other.  
# **That's the most important result establishing the relation between Stern-Brocot tree and Calkin-Wilf tree:**   
# For each path string `S`: **`SBfrac(S) == CWfrac(S[::-1])`**  
# Using the anterior examples: 

# %%
SBfrac('LRLL') == CWfrac('LRLL'[::-1]) == CWfrac('LLRL')

# %%
print(CWfrac('LLRL'))

# %%
print(CWfrac('LRLL'))
print(SBfrac('LLRL'))

# %% [markdown]
#  We can re-build, for example, the first five levels of the Stern-Brocot tree, by applying this function `SBfrac`to the corresponding string paths generated by the function `paths_level(k)` applied to each level `k`:

# %%
SBlevels5 = [[SBfrac(S) for S in paths_level(k)] for k in range(5)]
print_bintree(SBlevels5)

# %% [markdown]
# And in the same way we can build the first five levels of the Calkin-Wilf tree, by applying the function `CWfrac` to the corresponding string paths 

# %%
CWlevels5 = [[CWfrac(S) for S in paths_level(k)] for k in range(5)]
print_bintree(CWlevels5)

# %% [markdown]
# We have solved the question:  
# Given a string of L's and R's, what fraction corresponds to it?  
# in the Stern-Brocot tree representation and in the Calkin-Wilf tree representation of the rationals

# %% [markdown]
# #### string path corresponding to a given fraction

# %% [markdown]
# ##### Stern-Brocot tree first version  
# Each node fraction of Stern-Brocot tree is greater than the left son fraction and less than the right son fraction:  
# for each path `S`: `SBfrac(S + 'L') < SBfrac(S) < SBfrac(S + 'R')`  
# So, to reach a node carrying a fraction `n/d`, we move from the root `1` to the searched node depending if the current node's fraction is less or greater than `n/d`, as described in the function `SBpathDemo(frac)`.
#

# %%
def SBpathDemo(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Stern-Brocot path string corresponding to a fraction by a binary search 
        moving from the root down to the frac value
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        SBpathDemo(3/8) -> 'LLRL'        
    Demo: this version print the intermediate results of the binary search   
    """
    if type(frac) is tuple:
        frac = Fraction(*frac)
    else:
        frac = Fraction(frac)
    S = ''
    print('S = {:<7}: frac(S) = {:<3} '.format(S,str(SBfrac(S))),end=' -> ')
    while frac != SBfrac(S):
        if frac < SBfrac(S):
            print('{} < {:<3} go to left'.format(frac,str(SBfrac(S))))
            S += 'L'
        else:
            print('{} > {} go to right'.format(frac,SBfrac(S)))
            S += 'R'
        print('S = {0:<7}: frac(S) = {1:} ' .format(S,SBfrac(S)),end=' -> ')
    return S

# %%
SBpathDemo(3/8)

# %% [markdown]
# ##### Calkin-Wilf tree
# From the matrix representation of Calkin-Wilf tree we know that for a node carrying the fraction `n/d`:
# * if `d > n` this node is the left son of the node carrying the fraction `n/(d-n)`
# * if `d < n` this node is the right son of the node carrying the fraction `(n-d)/d`
# We can start from the node with `n/d` and move to the root using this rule and building the corresponding path
# string at each move, as in the function `CWpathDemo(frac)`:

# %% {"code_folding": []}
def CWpathDemo(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string corresponding to a fraction by a binary search 
        moving from the frac value up to the root
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        CWpathDemo(3/8) -> 'LRLL'        
    Demo: this version print the intermediate results of the binary search   
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    print('{}/{}'.format(num,den),end=': ')
    while num != den:
        if num > den:
            print('{} > {} coming from right ->'.format(num,den),end=' ')
            S = 'R' + S
            num -= den
            print('{}/{}: {}'.format(num,den,S))
        else:
            print('{} < {} coming from left  ->'.format(num,den),end=' ')   
            S = 'L' + S
            den -= num  
            print('{}/{}: {}'.format(num,den,S))      
        print('{}/{}'.format(num,den),end=': ')
    print('path_str({}) == {}'.format(str(frac),S))
    return S

# %%
CWpathDemo(3/8)

# %% [markdown]
# And without the demo printing, we define the function `CWpath(frac)`:

# %%
def CWpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string corresponding to a fraction by a binary search 
        moving from the frac value up to the root
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example:
        CWpathDemo(3/8) -> 'LRLL'          
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    while num != den:
        if num > den:
            S = 'R' + S
            num -= den
        else:
            S = 'L' + S
            den -= num
    return S    

# %%
CWpath(3/8)

# %% [markdown]
# ##### Stern-Brocot tree second version  
# We know that:  
# For each path string `S`: `SBfrac(S) == CWfrac(S[::-1])`  
# We deduce from this result than:  
# For each fraction `frac`: `SBpath(frac) == CWpath(frac)[::-1]`  

# %% [markdown]
# In this way, we can define `SBpath(frac)` by:

# %%
def SBpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Calkin-Wilf path string S corresponding to a fraction by a binary search 
        moving from the frac value up to the root and return the reverse path S[::-1],
        which is the corresponding Stern-Brocot path
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example: SBpath(3/8) -> CWpath(3/8)[::-1] == 'LRLL'[::-1] == 'LLRL'
    """
    return CWpath(frac)[::-1]

# %%
SBpath(3/8)

# %% [markdown]
# Or directly building the reversed path with the code of `CWpath(frac)`

# %%
def SBpath(frac: Union[Tuple[int, int], str]) -> str: 
    # -> Union[Tuple[int, int], str, Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the Stern-Brocot path string S corresponding to a fraction by a binary search 
        moving from the frac value up to the root on Calkin-Wilf and build on each move 
        the corresponding Stern-Brocot path string S

    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the path string S:
    Example: SBpath(3/8) -> 'LLRL'
    """
    if type(frac) is tuple:
        num,den = frac
    else:
        frac = Fraction(frac)
        num,den = frac.numerator,frac.denominator
    S = ''
    while num != den:
        if num > den:
            S += 'R'
            num -= den
        else:
            S += 'L'
            den -= num
    return S    

# %%
print(SBpath(3/8) == CWpath(3/8)[::-1] == 'LLRL')

# %% [markdown]
# ##### Approximation of real numbers  
# We can use the same algorithm to get an approximation of a real number $x$ by a Stern Brocot path of length $n$ and the corresponding fractions:

# %%
def SBrealpath(x: float, n: int) -> str:
    """ return a path string of length n as an approximation of infinite path string of x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) length of the desired path string
    Returns:
        a path string of length n
    """
    S = ''
    for k in range(n):
        if x < 1:
            S += 'L'
            x = x/(1-x)
        else:
            S += 'R'
            x = x - 1
    return S

# %% [markdown]
# For example, approximations of Euler number $e = 2.718281828459045\ldots$

# %%
SB_e20 = SBrealpath(e,20)

# %%
SB_e20

# %% [markdown]
# And using `SBfrac` it's easy to get a sequence of fractions approximating a real number `x`, where `n` can be a number or a slice as `slice(12,41,2)`, meaning to return only the fractions corresponding to the paths from 12 characters up to 40 characters included by steps of 2 (12,14,16,...,40):

# %%
def SBrealfrac(x: float, n: Union[int, slice]) -> Tuple[int, int]:  
    # -> Fraction[int, int] type hint doesn't recognize Fraction:
    """ return a list of n successive fractions approximating the real number x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) number of successive rational approximations of x
           or a slice defining the subset of desired rational approximations of x
    Returns:
        a list of n fractions approximating x
    """
    if isinstance(n, slice):
        SBpath = SBrealpath(x,n.stop)
        return [SBfrac(SBpath[:k]) for k in range(n.stop)[n]]
    SBpath = SBrealpath(x,n)
    return [SBfrac(SBpath[:k]) for k in range(n)]



# %%
print(SBrealfrac(e,20))

# %%
1264/465 == 2.718279569892473 < e < 2.718283582089552 == 1457/536

# %%
print(SBrealfrac(e,slice(5,21,5)))

# %% [markdown]
# Or a prettiest version returning a list of formatted strings

# %% {"code_folding": [0]}
def prettySBrealfrac(x, n: Union[int, slice] = 10, prec: str ='.10f') -> List[str]: 
    """ return a list of n successive fractions approximating the real number x
    
    Args:
        x: (float) the float representation of the real number x
        n: (int) number of successive rational approximations of x
           or a slice defining the subset of desired rational approximations of x
        prec: a string describing the desired float precision of the result
    Returns:
        a list of n formated strings each string describing an approximation of x and consists of:
        - position of each selected approximation in the list of max length (n.stop when n is a slice)
        - the fraction approximation
        - the float value of this fraction with the precision defined by prec
    """

    fracs = SBrealfrac(x,n)
    fmt = '{}:{}={:'+prec+'}'
    if isinstance(n, slice):
        return [fmt.format(n.start+n.step*k,frac,frac.numerator/frac.denominator) 
                for (k,frac) in enumerate(fracs)]
    return [fmt.format(k,frac,frac.numerator/frac.denominator) for (k,frac) in enumerate(fracs)]

# %%
print(prettySBrealfrac(e,slice(12,40,2)))

# %% [markdown]
# With the same method, approximations of $\pi$ = 3.141592653589793…

# %%
SB_pi =  SBrealpath(pi,400) 
SB_pi

# %%
print(prettySBrealfrac(pi,slice(21,401,20)))

# %% [markdown]
# ### The Stern-Brocot tree in $\mathbb{N^2}$

# %% {"code_folding": []}
#import nbimporter
#from SButilsNote import plot_pt2pts,plot_points

# %% [markdown]
# It is interesting to remark than in $\mathbb{N^2}$ the pairs `(n,d)` corresponding to the fractions `n/d` of the Stern-Brocot tree form a binary tree too.  
# But it's not the case for the pairs `(n,d)` corresponding to the fractions `n/d` of the Calkin-Wilf tree, they don't form a binary tree in $\mathbb{N^2}$; here the links are horizontal or vertical.  
# Below the seven first levels with different colors for each level in the two cases:

# %% {"code_folding": [0]}
#click to see  the code 
xmax = 21
plt.rcParams["figure.figsize"] =  [14.0, 6.0]
fig = plt.figure()
sub1 = fig.add_subplot(1,2, 1)
tree = SBpairs(7)
for k in range(0,len(tree)-1):
    level_colors=['blue','brown','green','navy','goldenrod','turquoise']
    for i in range(len(tree[k])):
        pt = np.array(tree[k][i])
        pts = np.array([tree[k+1][2*i],tree[k+1][2*i+1]])             
        plot_pt2pts(sub1,pt,pts,colors=[level_colors[k%len(level_colors)]])
        plot_points(sub1,pt.reshape(1,2))
        plot_points(sub1,pts)
gridticks(sub1,xmajticks=(0,xmax+1,1))
sub1.set_xlabel('Stern-Brocot: SBpairs(7)')

#plt.rcParams["figure.figsize"] =  [6.0, 14.0]
#fig = plt.figure()
sub4 = fig.add_subplot(1,2, 2)
tree = CWpairs(7)
for k in range(0,len(tree)-1):
    level_colors=['blue','brown','green','navy','goldenrod','turquoise']
    for i in range(len(tree[k])):
        pt = np.array(tree[k][i])
        pts = np.array([tree[k+1][2*i],tree[k+1][2*i+1]])             
        plot_pt2pts(sub4,pt,pts,colors=[level_colors[k%len(level_colors)]])
        plot_points(sub4,pt.reshape(1,2))
        plot_points(sub4,pts)    
gridticks(sub4,xmajticks=(0,xmax+1,1))
sub4.set_xlabel('Calkin-Wilf: CWpairs(7)')
plt.show()

# %% [markdown]
# We can put more nodes in the grid, plotting all the relatively prime integer's pairs present in the grid 

# %% {"code_folding": [0]}
def rel_prime(a: int, b: int) -> bool:
    """ return True if a and b are relatively prime
    
    Args:
        a: int
        b: int
    Returns:
        True or False
    """
    if a == b == 1:
        return False
    return gcd(a,b) == 1

# %%
rel_prime(3,8)

# %% [markdown]
# and adding more branches to the Stern-Brocot tree, plotting the branches linking a node to his father.  
# We define easily a function `SBfather(frac)` returning the father node of a node `(n,d)`:

# %% {"code_folding": [0]}
def SBfather(frac: Union[Tuple[int, int], str]) -> Tuple[int, int]: 
    # frac: Union[Tuple[int, int], Fraction[int, int], str] 
    # -> Union[Tuple[int, int], Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the father node of a Stern-Brocot node from the pair or fraction value or string fraction value
    
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        the pair value of the father of frac if the frac parameter value was a pair
        the fraction value of the father of frac if the frac parameter value was a fraction or a string fraction
    """
    father = SBfrac(SBpath(frac)[:-1])
    if type(frac) is tuple:
        return father.numerator,father.denominator
    return father

# %%
SBfather((3,8))

# %%
SBfather('3/8')

# %% [markdown]
# or again with more branches plotting the branches linking a node to his left and right sons. 

# %% {"code_folding": [0]}
def SBsons(frac: Union[Tuple[int, int], str]) -> Tuple[Tuple[int, int],Tuple[int, int]]: 
    # frac: Union[Tuple[int, int], Fraction[int, int], str] 
    # -> Union[Tuple[int, int], Fraction[int, int]] type hint doesn't recognize Fraction
    """ find the (left son, right son) nodes pair of a Stern-Brocot tree node
    
    Args:
        frac: a fraction as a tuple (numerator: int, denominator: int) as (3,8) 
        or a string as '3/8' or a fraction as Fraction(3,8)
    Returns:
        a pair of pair's values for the sons of frac if the frac parameter value was a pair
        a pair of fraction values for the sons of frac if the frac parameter value was a fraction or a string
    """
    path = SBpath(frac)
    sons = [SBfrac(path + 'L'),SBfrac(path + 'R' )]
    if type(frac) is tuple:
        return ((sons[0].numerator,sons[0].denominator),(sons[1].numerator,sons[1].denominator))
    return sons

# %%
SBsons((2, 5))

# %% [markdown]
# In this case the sons can be out of the grid and we have to clip the lines.

# %% {"code_folding": []}
def xylinefrom(*pts: np.array) -> Tuple[Tuple[float,float],Tuple[float,float]]:
    """ compute 2 pairs (x0,x1),(y0,y1) to be plotted by matlib.plot
    
    Args:
        pts : a np.array of 2 or 3 points in the plane
        the 2 pts pts[0] and pts[1] verifying:
           0 < pts[0][0] < pts[1][0]
           0 < pts[0][1] < pts[1][1]
        if present, the third point defines a clipping
        window (0,xmax=pts[2][0]),(0,ymax=pts[2][1])
        to the segment defined by pts[0]to pts[1].
    Returns:
        two pairs (x0,x1),(y0,y1) defining the segment
        ready to be plotted by matlib.plot
    """
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    if len(pts) == 2:
        return xs,ys
    else:
        m = (ys[1]-ys[0])/(xs[1]-xs[0])
        if xs[1] > xs[2]:
            xs[1] = xs[2]
            ys[1] = ys[0] + m*(xs[1]-xs[0])
        if ys[1] > ys[2]:
            ys[1] = ys[2]
            xs[1] = xs[0] + (xs[1]-xs[0])/m  
    return xs[:2],ys[:2]   

# %% {"code_folding": []}
#click to see the plot's code 
xmax = 22
plt.rcParams["figure.figsize"] =  [14.0, 6.0]
fig = plt.figure()

sub2 = fig.add_subplot(1,2, 1)
for lin in range(2,xmax+1): 
    xpts = [col for col in range(lin+1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        xpair,ypair = np.array((SBfather((xpts[i],lin)),(xpts[i],lin))).T
        sub2.plot(xpair, ypair, color='green', linewidth=1.2)
        sub2.plot(ypair, xpair, color='green', linewidth=1.2)
for lin in range(1,xmax+1):
    xpts = [col for col in range(1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        sub2.plot(xpts[i], ypts[i], color='red', marker='o')
gridticks(sub2,xmajticks=(0,xmax+1,1))
sub2.set_xlabel('Stern-Brocot: coprimes pairs and SBfathers')


#xmax = 22
#plt.rcParams["figure.figsize"] =  [6.0, 6.0]

sub3 = fig.add_subplot(1,2, 2)
for lin in range(1,xmax+1): 
    xpts = [col for col in range(lin+1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        left,right = SBsons((xpts[i],lin))
        #xpair,ypair = pts2xys(SBfather((xpts[i],lin)),(xpts[i],lin))
        xpair,ypair = xylinefrom((xpts[i],lin),left,(xmax,xmax))
        sub3.plot(xpair, ypair, color='green', linewidth=1.2)
        sub3.plot(ypair, xpair, color='green', linewidth=1.2)
        xpair,ypair = xylinefrom((xpts[i],lin),right,(xmax,xmax))
        sub3.plot(xpair, ypair, color='green', linewidth=1.2)
        sub3.plot(ypair, xpair, color='green', linewidth=1.2)
for lin in range(1,xmax+1):
    xpts = [col for col in range(1,xmax+1) if rel_prime(col,lin)]
    ypts = len(xpts)*[lin]
    for i in range(len(xpts)):
        sub3.plot(xpts[i], ypts[i], color='red', marker='o')
gridticks(sub3,xmajticks=(0,xmax+1,1))
sub3.set_xlabel('Stern-Brocot: coprimes pairs and SBsons')
plt.show()
