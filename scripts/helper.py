import re, json, numbers

## answer normalization function 
# provided by Minerva P18 https://arxiv.org/pdf/2206.14858.pdf

# YS: add ('$.', '$')
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('$.', '$'), ('\\$', ''), (r'\ ', ''), (' ', ''), ('mbox', 'text'), 
    (',\\text{and}', ','), ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]

# YS: add 'days'
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'days',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer
# ------------------------------------------------------------------------------------


def find_num_from_str(a):
    try:
        num = json.loads(a)
        if isinstance(num, numbers.Number):
            return ("Number", num)
        elif isinstance(num, list) and len(num)>0 and all([isinstance(el, numbers.Number) for el in num]):
            return ('MultiNumbers', num)
    except json.decoder.JSONDecodeError: pass
    
    fractions = re.findall(r'-?[0-9]+/0*[1-9][0-9]*', a)
    
    _a = a
    for fr in fractions: 
        _a = _a.replace(fr, " ") # avoid numerator and denominator from being detected as separate numbers
    nums = re.findall(r'(-?\d*\.?\d+)', _a)
    
    for fr in fractions: # convert fractions to float
        nu, de = fr.split("/")
        nums.append(str(float(nu) / float(de)))
    
    if len(nums) == 1: return('Number', float(nums[0]))
    elif len(nums) > 1: return('MultiNumbers', [float(n) for n in nums])
    else: return("String", a)

def count_numbers_in_str(a):

    fractions = re.findall(r'-?[0-9]+/0*[1-9][0-9]*', a)
    
    _a = a
    for fr in fractions: 
        _a = _a.replace(fr, " ") # avoid numerator and denominator from being detected as separate numbers
    nums = re.findall(r'(-?\d*\.?\d+)', _a)
    
    for fr in fractions: # convert fractions to float
        nu, de = fr.split("/")
        nums.append(str(float(nu) / float(de)))
    
    res = []
    if len(nums) > 0: res = [float(n) for n in nums]
    
    return res

# ---------------------------------------------------------------------------
# Create a decorator that limits execution time of a function
from functools import wraps
import multiprocessing

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

def function_runner(*args, **kwargs):
    """Used as a wrapper function to handle
    returning results on the multiprocessing side"""

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return
    send_end.send(result)


@parametrized
def run_with_timer(func, max_execution_time):
    @wraps(func)
    def wrapper(*args, **kwargs):
        recv_end, send_end = multiprocessing.Pipe(False)
        kwargs["__send_end"] = send_end
        kwargs["__function"] = func
        
        ## PART 2
        p = multiprocessing.Process(target=function_runner, args=args, kwargs=kwargs)
        p.start()
        p.join(max_execution_time)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError
        result = recv_end.recv()

        if isinstance(result, Exception):
            raise result

        return result

    return wrapper
# ---------------------------------------------------------------------------

# Editing distance
def LD(s, t):
    """ 
        iterative_levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings 
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein 
        distance between the first i characters of s and the 
        first j characters of t
    """

    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    # source prefixes can be transformed into empty strings 
    # by deletions:
    for i in range(1, rows):
        dist[i][0] = i

    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution    
 
    return dist[row][col]

def lcs(X, Y): 
    m = len(X) 
    n = len(Y) 
 
    L = [[None]*(n + 1) for i in range(m + 1)] 

    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
    return L[m][n] 