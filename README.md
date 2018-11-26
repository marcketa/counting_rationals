# counting_rationals
We know that the set <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{80}&space;\bg_white&space;\fn_cm&space;\large&space;\mathbb{Q}" target="_blank">
<img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{80}&space;\bg_white&space;\fn_cm&space;\large&space;\mathbb{Q}" title="\large \mathbb{Q}" /></a>

of rationals is countable. This notebook presents two implementations of this result in the form of binary tree whose nodes are all the positive fractions in lowest terms.  
I was impressed by the chapter of Donald E. Knuth's *Concrete Mathematics* book dedicated to this theme, by the construction of the Stern-Brocot binary tree. It seems to me, that transforming this chapter in a "dynabook", with Python permitting experiments, was a good idea.  
In the same time, a friend sent me an article of Calkin and Wilf about building another binary tree whose nodes are all the fractions in lowest terms. I was excited by the idea to find the relation between these two different versions. I was very happy to find an elegant relation, but looking on the web, I found that this was an old new...  
I hope that my pythonic version may please to someones...

This notebook use some extensions of Jupyter Notebooks that seem to me useful. You can look at this adress if you are interested: https://ndres.me/post/best-jupyter-notebook-extensions/  

I'm using Python 3.6.5 and I tried the Python type annotations imported from module `typing`. You can look at:  
https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

