import csv
#import tabulate
import numpy as np
from itertools import groupby
from operator import itemgetter
from math import log10, floor

def write_header_to_tex_file(filename, landscape=False):
    with open(filename, 'w') as f:
        f.write("""\\documentclass{article}\n""")
        if landscape:
            f.write("""\\usepackage[landscape,margin=1cm]{geometry}\n\n""")
        else:
            f.write("""\\usepackage[margin=1cm]{geometry}\n\n""")

        f.write("""\\usepackage{graphicx}% http://ctan.org/pkg/graphicx\n""")
        f.write("""\\usepackage{booktabs}% http://ctan.org/pkg/booktabs\n""")
        f.write("""\\usepackage{xparse}\n\n""")

        f.write("""\\renewcommand\\floatpagefraction{.9}""")
        f.write("""\\renewcommand\\topfraction{.9}""")
        f.write("""\\renewcommand\\bottomfraction{.9}""")
        f.write("""\\renewcommand\\textfraction{.1}""")
        f.write("""\\setcounter{totalnumber}{50}""")
        f.write("""\\setcounter{topnumber}{50}""")
        f.write("""\\setcounter{bottomnumber}{50}\n\n""")

        f.write("""\\NewDocumentCommand{\\rot}{O{45} O{1em} m}{\makebox[#2][l]{\\rotatebox{#1}{#3}}}%\n\n""")

        f.write("""\\begin{document}\n\n""")

def append_footer_to_tex_file(filename):
    with open(filename, 'a') as f:
        f.write("""\\end{document}\n""")

def import_dict_from_csv_file(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        l = [dict([a, convert_to_float_if_applicable(x)] for a, x in b.items()) for b in reader]
    return l

def write_dict_to_csv_file(filename, dictionary, fields=None):
    """
        list_of_dicts --
        filename --
        dictionary -- list of dictionaries
        fields -- ordered list of fields to write
    """
    with open(filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=(dictionary[0].keys() if fields is None else fields),
            extrasaction='ignore')
        writer.writeheader()
        writer.writerows(dictionary)

def aggregate_data(list_of_dicts, keys, aggr):
    """
        list_of_dicts --
        keys -- list of keys to sort by
        aggr --  list of (name, key, function) tuples specifiying how to
        aggregate key and under what name

    """
    grouper = itemgetter(*keys)
    result = []
    for key, grp in groupby(sorted(list_of_dicts, key = grouper), grouper):
        grp = [item for item in grp]
        temp_dict = dict(zip(keys, key))
        temp_dict['number of cases'] = len(grp)
        for n,k,f,t in aggr:
            temp_dict[n] = round_to_sig_digits(f([t(np.float128(item[k])) for item in grp]), 3)
        result.append(temp_dict)

    return result

def convert_to_float_if_applicable(item):
    try:
        item = float(item)
    except ValueError:
        pass
    return item

def round_to_sig_digits(x, n):
    try:
        return round(x, -int(floor(log10(abs(x)))) + (n - 1))
    except ValueError:
        return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def identity(x):
    return x

def sorted_lod_by_key(lod, keys):
    return sorted(lod, key=itemgetter(*keys))

def create_table(dictionary, header=None, rot=False):
    if rot:
        return tabulate.tabulate(dictionary, headers=('keys' if header is None else header), tablefmt="latex_rot")
    else:
        return tabulate.tabulate(dictionary, headers=('keys' if header is None else header), tablefmt="latex")

if __name__ == '__main__':

    write_header_to_tex_file('out.tex', landscape=True)
    d = import_dict_from_csv_file('results.csv')

    # write aggregated results
    keep = [
        'forgetting',
        'novelty',
        'lambda',
        'n-features'
    ]

    aggr = [
        ('mean of log product ratio', 'ratio (PROD)', np.mean, np.log),
        ('variance of log product ratio', 'ratio (PROD)', np.var, np.log),
        ('mean of sigmoid sum ratio', 'ratio (SUM)', np.mean, sigmoid),
        ('variance of sigmoid sum ratio', 'ratio (SUM)', np.var, sigmoid),
        ('mean ref. prob. of novel referent (PROD)', 'novel referent (MUL)', np.mean, identity),
        ('variance of ref. prob. of novel referent (PROD)', 'novel referent (MUL)', np.var, identity),
        ('mean ref. prob. of familiar referent (PROD)', 'familiar referent (MUL)', np.mean, identity),
        ('variance of ref. prob. of familiar referent (PROD)', 'familiar referent (MUL)', np.var, identity),
        ('mean ref. prob. of novel referent (SUM)', 'novel referent (SUM)', np.mean, identity),
        ('variance of ref. prob. of novel referent (SUM)', 'novel referent (SUM)', np.var, identity),
        ('mean ref. prob. of familiar referent (SUM)', 'familiar referent (SUM)', np.mean, identity),
        ('variance of ref. prob. of familiar referent (SUM)', 'familiar referent (SUM)', np.var, identity),
    ]

    aggregated = aggregate_data(d, keep, aggr)

    keys = [
        'number of cases',
        'forgetting',
        'novelty',
        'lambda',
        'n-features',
        'mean of log product ratio',
        'variance of log product ratio',
        'mean of sigmoid sum ratio',
        'variance of sigmoid sum ratio',
        'mean ref. prob. of novel referent (PROD)',
        'variance of ref. prob. of novel referent (PROD)',
        'mean ref. prob. of familiar referent (PROD)',
        'variance of ref. prob. of familiar referent (PROD)',
        'mean ref. prob. of novel referent (SUM)',
        'variance of ref. prob. of novel referent (SUM)',
        'mean ref. prob. of familiar referent (SUM)',
        'variance of ref. prob. of familiar referent (SUM)'
    ]

    s = create_table(aggregated, header=keys, rot=True)
    with open('out.tex', 'a') as f:
        f.write(s)
    append_footer_to_tex_file('out.tex')

    # examine the outliers
    fields = [
        'n-features',
        'novelty',
        'forgetting',
        'lambda',
        'novel word',
        'familiar object',
        'scene',
        'overlapping feature(s)',
        'ratio (SUM)',
        'ratio (PROD)',
        'novel referent (SUM)',
        'familiar referent (SUM)',
        'novel referent (MUL)',
        'familiar referent (MUL)',
    ]

    keys = [
        'ratio (PROD)',
    ]
    l = sorted_lod_by_key(d, keys)
    write_dict_to_csv_file('sorted_by_prod.csv', l, fields)

    keys = [
        'ratio (SUM)',
    ]
    l = sorted_lod_by_key(d, keys)
    write_dict_to_csv_file('sorted_by_sum.csv', l, fields)

