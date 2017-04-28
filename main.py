# Linear programming, monte carlo method
# ~Gros

import multiprocessing
import os
import random
import sys
import traceback
import re
from rpn import RPN, RPNError, is_number
from decimal import *
getcontext().prec = 4

# http://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
numeric_pattern = r"""
[-+]?
(?:
    (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
    |
    (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
)
(?: [Ee] [+-]? \d+ ) ?
"""


def manually():
    number_of_variables = int(raw_input("Number of variables: "))

    # --------- BOUNDARIES
    boundaries = []
    rx = re.compile(numeric_pattern, re.VERBOSE)
    print "Boundaries (int the form: 10 <= x0 <= 50):"
    for i in xrange(number_of_variables):
        boundary_equation = raw_input("for x{}: ".format(i))
        numbers_eq = rx.findall(boundary_equation)

        if len(numbers_eq) < 2:
            raise IOError("Wrong amount of numbers id boundaries: {}".format(repr(boundary_equation)))

        a, b = map(float, (numbers_eq[0], numbers_eq[-1]))
        if a > b:
            raise IOError("Left bound is bigger than right one: {}".format(repr(boundary_equation)))
        boundaries.append((a, b))

    # ------- EQUATIONS
    number_of_equations = int(raw_input("Number of equations: "))
    equations = []
    for i in xrange(number_of_equations):
        equation = raw_input("Gimme equation {}: ".format(i))
        function, equality, bound = preparse_equation(equation)
        equations.append((RPN(function), equality, bound))

    # -------- GOAL
    goal = raw_input("Gimme goal function: ")
    goal = RPN(goal)

    goal_type = raw_input("Maximalize (max) or minimalize (min) goal function: ").lower()
    if goal_type not in ('min', 'max'):
        raise IOError("Incorrect goal type, must be one of (min, max)")

    return goal, goal_type, equations, boundaries


def preparse_equation(equation):
    """Parse equation to function and equality

    Args:
        equation(string): in form: 1 + 3^5*8 <= 5.5
    Returns:
        list: [function, equality, bound], ie. ['1 + 3^5*8', '<=', 5.5]
    """
    correct_equalities = ('<=', '>=')
    for equality in correct_equalities:
        equation = equation.replace(equality, ' '+equality+' ')
    equation = equation.split()

    if len(equation) < 3:
        raise Exception("Incorrect equation to preparse: {}".format(repr(equation)))

    if equation[-2] not in correct_equalities:
        raise IOError("Wrong equality, must be one of ({})".format(','.join(correct_equalities)))

    if not is_number(equation[-1]):
        raise IOError("Wrong boundary, must be numeric: {}".format(equation[-1]))

    return ' '.join(equation[:-2]), equation[-2], float(equation[-1])


def from_file(path):
    """Parse file

    Args:
        path(string): file format is:
            number_of_variables
            0 <= x0 <= 3.3
            10 <= x1 <= 323.3
            11.3 <= x2 <= 323
                ...
            0 <= xn <= 2
            number_of_equations
            x1 + x2 <= 3
            sin(tan(x3)) + (x0^4) >= 4
                ...
            x0 + x1 + x2 <= 5.5
            goal_function
            goal_type

    """
    data = open(path).readlines()
    number_of_variables = int(data[0])

    # --------- BOUNDARIES
    boundaries, equations = [], []
    rx = re.compile(numeric_pattern, re.VERBOSE)
    i = 1
    while i < number_of_variables+1:
        boundary_equation = data[i]
        numbers_eq = rx.findall(boundary_equation)

        if len(numbers_eq) < 2:
            raise IOError("Wrong amount of numbers id boundaries: {}".format(repr(boundary_equation)))

        a, b = map(float, (numbers_eq[0], numbers_eq[-1]))
        if a > b:
            raise IOError("Left bound is bigger than right one: {}".format(repr(boundary_equation)))

        boundaries.append((a, b))
        i += 1

    # ------- EQUATIONS
    number_of_equations = int(data[i])
    i += 1
    while i < number_of_equations + number_of_variables + 2:
        function, equality, bound = preparse_equation(data[i])
        equations.append((RPN(function), equality, bound))
        i += 1

    # -------- GOAL
    goal = RPN(data[i])
    goal_type = data[i+1]
    if goal_type not in ('min', 'max'):
        raise IOError("Incorrect goal type, must be one of (min, max)")

    return goal, goal_type, equations, boundaries


def print_data(goal, goal_type, equations, boundaries):
    print "Goal: ",
    if goal_type == 'min':
        print 'minimalize ',
    else:
        print 'maximialize ',
    print goal.infix()

    for i in xrange(len(boundaries)):
        print "{} <= x{} <= {}".format(boundaries[i][0], i, boundaries[i][1])

    for equation in equations:
        print "{} {} {}".format(equation[0].infix(), equation[1], equation[2])


def is_smaller(x, y):
    return x < y


def is_bigger(x, y):
    return x > y


def is_smaller_equal(x, y):
    return x <= y


def is_bigger_equal(x, y):
    return x >= y


def pass_all_equations(variables, equations):
    for equation in equations:
        if equation[1] == '<=':
            cmp_func = is_smaller_equal
        else:
            cmp_func = is_smaller_equal
        if not cmp_func(equation[0].compute(*variables), equation[2]):
            return False
    return True


def random_vector(boundaries):
    return [random.uniform(l, r) for l, r in boundaries]


def find_optimum_one_level_wrapper(*args):
    return find_optimum_one_level(*(args[0]))


def find_optimum_one_level(goal, goal_type, equations, boundaries):
    """Find max/min value of goal function in boundaries

    Returns:
        optimum(tuple): (list of optimum variables, optimum)
    """
    density = 2
    optimum_variables = None

    if goal_type == 'min':
        optimum = float('Inf')
        cmp_func = is_smaller
    else:
        optimum = -1
        cmp_func = is_bigger

    # amount_of_rands = reduce(lambda x, y: x*y, [int(r-l+1) for l, r in boundaries])
    amount_of_rands = 30
    max_iterations = 10
    max_counter = 0
    print os.getpid()
    while optimum_variables is None and max_counter < max_iterations:
        for i in xrange(amount_of_rands*density):
            random_variables = random_vector(boundaries)
            if pass_all_equations(random_variables, equations):
                print boundaries, os.getpid(), optimum, random_variables, goal.compute(*random_variables), pass_all_equations(random_variables, equations)
                to_check = goal.compute(*random_variables)
                if cmp_func(to_check, optimum):
                    optimum = to_check
                    optimum_variables = random_variables
        max_counter += 1

    if optimum_variables is None:
        print os.getpid()
        raise Exception("Can't find optimum value")

    return optimum_variables, optimum


def find_optimum(goal, goal_type, equations, boundaries, recursion_level=0):
    """Minimalize/maximalize goal function using monte-carlo method with respect to boundaries

    Args:
        goal(RPN): goal function
        goal_type(string): min or max
        equations(list of tuples): (RPN, string, float), i.e. (RPN('x1+x0'), '<=', 13.4)
        boundaries(list of tuples): (float, float) for every variable, such that boundaries[i]][0] <= xi <= boundaries[i]][1]
        recursion_level(int)

    Returns:
        optimum(tuple): (list of optimum variables, optimum)
    """
    epsilon = 5
    max_recursion = sys.getrecursionlimit() - 10
    delta = 2  # new boundary = (x - boundary_size/delta, x + boundary_size/delta)
    processes = 2

    # if recursion_level >= max_recursion:
    #     return find_optimum_one_level(goal, goal_type, equations, boundaries)
    if all([True if r-l < epsilon else False for l, r in boundaries]):
        return find_optimum_one_level(goal, goal_type, equations, boundaries)

    if processes > 1:
        bound_sizes = [(r-l)/processes for l, r in boundaries]
        new_boundaries = [[(l+(i*size), l+((i+1)*size)) if size > epsilon else (l, r) for (l, r), size in
                           zip(boundaries, bound_sizes)] for i in xrange(processes)]

        pool = multiprocessing.Pool(processes=processes)
        print "new_boundaries", new_boundaries
        optimum_vars = pool.map(find_optimum_one_level_wrapper, [[goal, goal_type, equations, bound] for bound in new_boundaries])

        optimum_index = None
        if goal_type == 'min':
            optimum = float('Inf')
            cmp_func = is_smaller
        else:
            optimum = -1
            cmp_func = is_bigger
        for i in xrange(len(optimum_vars)):
            if cmp_func(optimum_vars[i][1], optimum):
                optimum = optimum_vars[i][1]
                optimum_index = i

        if optimum_index is None:
            raise Exception("Can't find optimum value")
        optimum_vars = optimum_vars[optimum_index][0]
    else:
        optimum_vars, optimum = find_optimum_one_level(goal, goal_type, equations, boundaries)

    new_boundaries = [(max(l, x-((r-l)/delta)), min(r, x+((r-l)/delta))) for (l, r), x in zip(boundaries, optimum_vars)]
    print boundaries, new_boundaries
    print optimum_vars
    print ''
    raw_input("pause")
    return find_optimum(goal, goal_type, equations, new_boundaries, recursion_level+1)


if __name__ == "__main__":
    path = 'test_inputs/1p2.txt'
    parsed_data = from_file(path)
    print_data(*parsed_data)
    print find_optimum(*parsed_data)
    sys.exit(0)

    '''
    while True:
        print "-"*15
        input_type = raw_input("Input data manually (m), form file (f) or end it(end): ").lower()
        try:
            if input_type == 'm':
                parsed_data = manually()
            elif input_type == 'f':
                path = raw_input("Gimme path: ")
                parsed_data = from_file(path)
                print_data(*parsed_data)
            else:
                break
        except IOError, e:
            print "Input error:", e
            traceback.print_exc()
            sys.exit(1)
        except RPNError, e:
            print "RPN error:", e
            traceback.print_exc()
            sys.exit(1)
        except Exception, e:
            print "Unknown error:", e
            traceback.print_exc()
            sys.exit(1)

        find_optimum(*parsed_data)
        '''
