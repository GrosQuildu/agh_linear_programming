# Linear programming, monte carlo method
# ~Gros

import multiprocessing
import random
import sys
import traceback
import operator
import logging
import argparse
from collections import namedtuple
from rpn import RPN, RPNError, is_number
from time import time

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.ERROR)


class EquationError(Exception):
    pass


class UnsolvableError(Exception):
    pass


def manually():
    number_of_variables = int(raw_input("Number of variables: "))

    # --------- BOUNDARIES
    boundaries = []
    print "Boundaries (int the form: 10 <= x0 <= 50):"
    for i in xrange(number_of_variables):
        boundary_equation = raw_input("for x{}: ".format(i))
        bound = preparse_bound(boundary_equation)
        boundaries.append(bound)

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
        raise EquationError("Incorrect goal type, must be one of (min, max)")

    return goal, goal_type, equations, boundaries


def preparse_bound(boundary_equation):
    """Parse bound to bounds and variable

    Args:
        boundary_equation(string): in form: 12.2 <= x3 <= 24
    Returns:
        list: [variable(string), bound(tuple of floats)], ie. ['x3', (12.2, 24)]
    """
    bound_parts = boundary_equation.replace('<=', ' <= ')
    bound_parts = bound_parts.split()
    if len(bound_parts) != 5:
        raise EquationError("Wrong bound equation: {}".format(boundary_equation))

    bound = (float(bound_parts[0]), float(bound_parts[4]))

    if bound[0] > bound[1]:
        raise EquationError("Left bound is bigger than right one: {}".format(repr(boundary_equation)))

    if bound[0] < 0:
        raise EquationError("Left bound is smaller than zero: {}".format(repr(boundary_equation)))

    return bound[0], bound_parts[2], bound[1]


def preparse_equation(equation):
    """Parse equation to function and equality

    Args:
        equation(string): in form: 1 + 3^5*8 <= 5.5
    Returns:
        list: [function(string), equality(string), bound(float)], ie. ['1 + 3^5*8', '<=', 5.5]
    """
    correct_equalities = ('<=', '>=')
    for equality in correct_equalities:
        equation = equation.replace(equality, ' '+equality+' ')
    equation = equation.split()

    if len(equation) < 3:
        raise EquationError("Incorrect equation to pre-parsing: {}".format(repr(equation)))

    if equation[-2] not in correct_equalities:
        raise EquationError("Wrong equality, must be one of ({})".format(','.join(correct_equalities)))

    if not is_number(equation[-1]):
        raise EquationError("Wrong boundary, must be numeric: {}".format(equation[-1]))

    return ' '.join(equation[:-2]), equation[-2], float(equation[-1])


def from_file(path):
    """Parse file

    Args:
        path(string): file format is:
            number_of_variables(n)
            0 <= x0 <= 3.3
            10 <= x1 <= 323.3
            11.3 <= x2 <= 323
                ...
            0 <= xn <= 2
            number_of_equations(r)
            x1 + x2 <= 3
            sin(tan(x3)) + (x0^4) >= 4
                ...
            x0 + x1 + x2 <= 5.5
            goal_function
            goal_type

    """
    with open(path) as f:
        data = f.readlines()
    data = map(lambda x: x.strip(), data)
    data = filter(lambda x: x and not x.startswith('#'), data)
    number_of_variables = int(data[0])

    # --------- BOUNDARIES
    boundaries, equations = [], []
    i = 1
    while i < number_of_variables+1:
        boundary_equation = data[i]
        bound = preparse_bound(boundary_equation)
        boundaries.append(bound)
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
        raise EquationError("Incorrect goal type, must be one of (min, max): {}".format(goal_type))

    return goal, goal_type, equations, boundaries


def print_data(goal, goal_type, equations, boundaries):
    print "Goal:",
    print goal.infix(),
    if goal_type == 'min':
        print '-> min'
    else:
        print '-> max'

    for i in xrange(len(boundaries)):
        print "{} <= {} <= {}".format(boundaries[i][0], boundaries[i][1], boundaries[i][2])

    for equation in equations:
        print "{} {} {}".format(equation[0].infix(), equation[1], equation[2])


def prepare_from_equality(equality):
    if equality == "<=":
        cmp_func = operator.le
    elif equality == ">=":
        cmp_func = operator.ge
    elif equality == "<":
        cmp_func = operator.lt
    else:
        cmp_func = operator.gt
    return cmp_func


def prepare_from_goal_type(goal_type):
    if goal_type == 'min':
        optimum = float('Inf')
        cmp_func = operator.lt
    else:
        optimum = -1
        cmp_func = operator.ge
    return optimum, cmp_func


def pass_all_equations(variables, equations):
    for equation in equations:
        cmp_func = prepare_from_equality(equation[1])
        if not cmp_func(equation[0].compute(**variables), equation[2]):
            return False
    return True


def random_vector(boundaries):
    return {v: random.uniform(l, r) for l, v, r in boundaries}


def find_optimum_one_level_wrapper(*args):
    try:
        return find_optimum_one_level(*(args[0]))
    except Exception as e:
        print "Error in multiprocessing: {}".format(e)
        return None, None


def find_optimum_one_level(goal, goal_type, equations, boundaries, amount_of_rands, parameters=None):
    optimum, cmp_func = prepare_from_goal_type(goal_type)
    optimum_variables = None
    logger.info("one_level: rands {}, boundaries {}".format(amount_of_rands, boundaries))

    for i in xrange(amount_of_rands):
        random_variables = random_vector(boundaries)
        logger.debug('{} {} {}'.format(boundaries, optimum, random_variables))
        if pass_all_equations(random_variables, equations):
            to_check = goal.compute(**random_variables)
            if cmp_func(to_check, optimum):
                optimum = to_check
                optimum_variables = random_variables

    if optimum_variables is None:
        logger.info("optimum not found")
        return None, None

    logger.info("optimum found: {} {}".format(optimum_variables, optimum))
    return optimum_variables, optimum


def find_optimum_wrapper(*args):
    try:
        return find_optimum(*(args[0]))
    except Exception as e:
        print "Error in multiprocessing: {}".format(e)
        return None, None


MonteCarloParameters = namedtuple("MonteCarloParameters", ['epsilon', 'epsilon_multiprocessing', 'max_recursion', 'delta'])


def find_optimum(goal, goal_type, equations, boundaries, amount_of_rands, parameters=None, processes=1, deep=True, recursion_level=0):
    """Minimalize/maximalize goal function using monte-carlo method with respect to boundaries

    Args:
        goal(RPN): goal function
        goal_type(string): min or max
        equations(list of tuples): (RPN, string, float), i.e. (RPN('x1+x0'), '<=', 13.4)
        boundaries(list of tuples): (float, string, float) for every variable, such that
                                                boundaries[i][0] <= boundaries[i][1] <= boundaries[i][2]
        amount_of_rands(int): how many random points check on every level

        parameters(namedtuple MonteCarloParameters):
            epsilon(float): end condition, end if all boundaries sizes are smaller than epsilon
            epsilon_multiprocessing(float): end condition for multiprocessing, end multiprocessing if
                                                biggest boundary size is smaller than this
            max_recursion(int)
            delta(float): new boundary = (x - boundary_size/delta, x + boundary_size/delta)

        processes(int): number of processes to use
        deep(bool): how to arrange work for processes (if True, join final results, else join on each level)
        recursion_level(int)

    Returns:
        optimum(tuple): (dict of optimum variables, optimum)
    """
    logger.info("Start with boundaries {}".format(boundaries))
    if parameters is None:
        epsilon = 1.
        epsilon_multiprocessing = 100.
        max_recursion = sys.getrecursionlimit() - 10
        delta = 4
        parameters = MonteCarloParameters(epsilon, epsilon_multiprocessing, max_recursion, delta)
    else:
        epsilon, epsilon_multiprocessing, max_recursion, delta = parameters

    # end conditions
    if recursion_level >= max_recursion:
        logger.info("Max recursion level overflow")
        return find_optimum_one_level(goal, goal_type, equations, boundaries, amount_of_rands)

    if all(r-l < epsilon for l, _, r in boundaries):
        logger.info("Epsilon reached")
        return find_optimum_one_level(goal, goal_type, equations, boundaries, amount_of_rands)

    # split biggest bound across processes
    max_bound_index, max_bound = max(enumerate(boundaries), key=lambda bound: bound[1][2] - bound[1][0])
    max_bound_size = max_bound[2] - max_bound[0]
    if processes > 1 and max_bound_size >= epsilon_multiprocessing:
        new_boundaries = []
        for i in xrange(processes):
            new_boundaries.append(boundaries[:])
            new_boundaries[-1][max_bound_index] = (boundaries[max_bound_index][0] + (i * max_bound_size / processes),
                                                   boundaries[max_bound_index][1],
                                                   boundaries[max_bound_index][0] + ((i+1) * max_bound_size / processes))

        if deep:
            find_optimum_func = find_optimum_wrapper
        else:
            find_optimum_func = find_optimum_one_level_wrapper

        pool = multiprocessing.Pool(processes=processes)
        optimum_point_value_from_processes = pool.map(find_optimum_func, [(goal, goal_type, equations, bound,
                                                                           int(amount_of_rands/processes),
                                                                           parameters) for bound in new_boundaries])

        if goal_type == 'min':
            optimum_variables_result = min(optimum_point_value_from_processes, key=lambda optimum_vars: optimum_vars[1])
        else:
            optimum_variables_result = max(optimum_point_value_from_processes, key=lambda optimum_vars: optimum_vars[1])

        if deep:
            return optimum_variables_result

    else:
        optimum_variables_result = find_optimum_one_level(goal, goal_type, equations, boundaries, amount_of_rands)

    optimum_variables = optimum_variables_result[0]
    if optimum_variables is None:
        return None, None

    new_boundaries = [(max(l, optimum_variables[v]-((r-l)/delta)), v, min(r, optimum_variables[v]+((r-l)/delta))) for
                      (l, v, r), x in zip(boundaries, optimum_variables)]
    return find_optimum(goal, goal_type, equations, new_boundaries, amount_of_rands, parameters,
                        processes=processes, recursion_level=recursion_level+1)


def quick_run(path, number_of_tests):
    parsed_data = from_file(path)
    print_data(*parsed_data)
    start_time_whole = time()
    for i in xrange(number_of_tests):
        start_time = time()
        print find_optimum(*parsed_data, amount_of_rands=1000, processes=1, deep=True)
        print "Duration {}: {}".format(i, time()-start_time)
    end_time_whole = time()
    print "Duration whole: {}".format(end_time_whole - start_time_whole)
    sys.exit(0)


if __name__ == "__main__":
    # ----- COMMAND LINE
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Solver for linear problems, uses monte-carlo method Edit')
        parser.add_argument('file', type=str, help='Path to file to parse')
        parser.add_argument('amount', type=int, help='Amount of random points at each level')
        parser.add_argument('-t', '--tests', type=int, help='Number of tests', default=1)
        parser.add_argument('-p', '--processes', type=int, help='Number of processes', default=1)
        parser.add_argument('-d', '--deep', action='store_true', help='Breath or deep type of multiprocessing')
        args = parser.parse_args()

        parsed_data = from_file(args.file)
        print_data(*parsed_data)
        print ''

        start_time_total = time()
        for i in xrange(args.tests):
            start_time = time()
            try:
                print "Result {}: {}".format(i, find_optimum(*parsed_data, amount_of_rands=args.amount, processes=args.processes, deep=args.deep))
            except EquationError as e:
                print "Input error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
                sys.exit(1)
            except RPNError as e:
                print "RPN error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
                sys.exit(1)
            except Exception as e:
                print "Unknown error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
                sys.exit(1)

            if args.tests > 1:
                print "Duration {}: {}".format(i, time() - start_time)
                print ''

        end_time_total = time()
        print "Duration total: {}".format(end_time_total - start_time_total)
        sys.exit(0)

    # ------- SET PARAMS
    parameters = None
    while parameters is None:
        set_params = raw_input("Set parameters manually (y/N)").lower()
        if set_params in ("y", "yes"):
            try:
                epsilon = float(raw_input("Gimme epsilon: "))
                epsilon_multiprocessing = float(raw_input("Gimme multiprocessing epsilon: "))
                max_recursion = int(raw_input("Gimme recursion limit (max is {}): ".format(sys.getrecursionlimit())))
                delta = float(raw_input("Gimme delta (for new boundaries): "))
                parameters = MonteCarloParameters(epsilon, epsilon_multiprocessing, max_recursion, delta)
            except ValueError as e:
                print "Input error: {}".format(e)
            except Exception as e:
                print "Unknown error: {}".format(e)
        else:
            print "Parameters set to default"
            break

    while True:
        parsed_data = None
        while parsed_data is None:
            # ------ READ DATA
            input_type = raw_input("Input data manually (m), form file (f) or end it(end): ").lower()
            try:
                if input_type == 'm':
                    parsed_data = manually()
                elif input_type == 'f':
                    path = raw_input("Gimme path: ")
                    parsed_data = from_file(path)
                    print ''
                    print_data(*parsed_data)
                    print ''
                else:
                    sys.exit(0)
            except EquationError as e:
                print "Input error: {}".format(e)
                traceback.print_exc()
            except RPNError as e:
                print "RPN error: {}".format(e)
                traceback.print_exc()
            except Exception as e:
                print "Unknown error: {}".format(e)
                traceback.print_exc()

            # ------ SET MULTIPROCESSING
            amount_of_rands, processes, deep = None, None, None
            while amount_of_rands is None:
                try:
                    amount_of_rands = int(raw_input("Gimme amount of random points at each level: "))
                    processes = int(raw_input("Gimme number of processes (threads) to use: "))
                    deep = True
                    if processes > 1:
                        deep = raw_input("Type of multiprocessing: breadth (b) or deep (d): ").lower()
                        if deep == 'b':
                            deep = False
                except ValueError as e:
                    print "Input error: {}".format(e)
                except Exception as e:
                    print "Unknown error: {}".format(e)

            # ------ COMPUTE OPTIMUM
            try:
                start_time = time()
                optimum_variables, optimum_value = find_optimum(*parsed_data, amount_of_rands=1000, parameters=parameters,
                                                                processes=processes, deep=deep)
                end_time = time()
                print ''
                if optimum_variables is None:
                    print "Optimum didn't found"
                else:
                    print "Optimum is {}".format(optimum_value)
                    for var_name, var_value in optimum_variables.iteritems():
                        print "{} = {}".format(var_name, var_value)
                    print ''
                    print "Duration: {}".format(end_time - start_time)
                    print "-" * 15
                    print ''
            except EquationError as e:
                print "Input error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
            except RPNError as e:
                print "RPN error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
            except Exception as e:
                print "Unknown error: {}".format(e)
                if logger.level <= logging.DEBUG:
                    traceback.print_exc()
