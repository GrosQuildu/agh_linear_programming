# AGH linear programming
Solver for (not necessary) linear problems, using monte-carlo method. University project.

```
pip install -r requirements.txt
python ./main.py
```

### Linear problems
Linear programming is a method to obtain "the best" set of values restricted to some linear conditions.
It's may be useful in decision making. Let's consider example:

Firm can create n products. For production it uses r means of production, which are bounded to some limits. Let's put
```
a[i][j] - consumption of i-th means of production for creation j-th product (i=1..r, j=1..n)
b[i] - owned resource of i-th means of production
c[j] - income from j-th product sales
d[j]/g[j] - min/max amount of j-th product that can be sell
x[j] - volume of production of j-th product
```
The firm wants to find out amount of the products (x variables) it has to produce to maximalize it's profits.
We can construct math equations for given problem:
```
a[1][i]*x[1] + a[1][2]*x[2] + ... + a[1][n]*x[n] <= b[1]
........................................................
a[r][i]*x[1] + a[r][2]*x[2] + ... + a[r][n]*x[n] <= b[r]

d[1] <= x[1] <= g[1]
....................
d[n] <= x[n] <= g[n]

c[1]*x[1] + c[2]*x[2] + ... + c[n]*x[n] -> max
```

The most popular method for solving problems of that type is called simplex. However, this code uses monte-carlo method.
Therefore it can be used with non-linear equations.

### Solver

Run `main.py` for interactive session or `main.py --help` to pass args in command line.

Program works by drawing set of variables `x[j]` (each in range `<d[j], g[j]>`) and choosing the best set meeting all inequalities.
Then new set (narrower) of boundaries d[x],g[x] is constructed around each x[j] and the algorithm is repeated.

The program can use multiple processes to speed up computations. The work is spread by splitting the biggest boundary between them.

Parameters:
```
Epsilon - program will end if all boundaries sizes (g[j]-d[j]) are smaller than epsilon
Multiprocessing epsilon - end multiprocessing if biggest boundary size is smaller than this value
Recursion limit - you know
Delta - new boundaries are constructed as (x - boundary_size/delta, x + boundary_size/delta)

Number of variables - number of products (x)
Number of equations - number of equations restricting means of production
Goal function - this function will be (min)maximalized

Amount of random points at each level - how many sets of x-es to rand
Number of processes - 1 for one process
Type of multiprocessing - if breadth, processes will be created and joined at each level
(for computing the best variables with current boundaries), if deep, processes will be created at first level
and joined after each of them reach epsilon
```

Example run: `python main.py test_inputs/p1.txt 2000 -p 4`

### Reverse polish notation
Parsing of equations is done by translating them into prefix (rpn) notation. Implementation supports variables,
constants and functions (from math module).

Example usage:
```python
from rpn import RPN, RPNError

infix_equation_string = "A + 2*B + pi*sin(tan(3))"
rpn = RPN(infix_equation_string)
print rpn.infix()
print rpn.compute(A=3, B=4.1)
```

Or run `python rpn.py`
