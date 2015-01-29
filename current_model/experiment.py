"""
experiment.py

Generic class to conduct experiments and write results to a CSV file.
===============================================================================
Adapted from https://github.com/rueckstiess/expsuite:

Copyright (c) 2010, Thomas Ruckstiess
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* The name of the copyright holders and authors may not be used to endorse
or promote products derived from this software without specific prior
written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDERS AND AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
from multiprocessing import Process, Pool, cpu_count
from optparse import OptionParser
from ConfigParser import ConfigParser
import numpy as np
import itertools
import types

import datetime
import csv

def mp_runrep(args):
    return Experiment.run_rep(*args)

class Experiment(object):

    def __init__(self):
        self.parse_cmd_line()
        self.parse_config()
        self.id = datetime.date.today().isoformat()
        self.csv_header = None # header for results CSV file
        #print 'Default number of cores is ', cpu_count()
        pass

    def parse_cmd_line(self):
        optparser = OptionParser()
        optparser.add_option('-c', '--config',
            action='store', dest='config', type='string', default='experiments.cfg',
            help="The experiment config file.")
        optparser.add_option('-n', '--numcores',
            action='store', dest='ncores', type='int', default=cpu_count(),
            help="Number of processes used; default is %i"%cpu_count())
        optparser.add_option('-o', '--outputfile',
            action='store', dest='outputfile', type='string', default='results.csv',
            help="The output csv file.")

        options, args = optparser.parse_args()
        self.options = options
        return options, args

    def parse_config(self):
        self.config_parser = ConfigParser()
        if not self.config_parser.read(self.options.config):
            raise SystemExit('config file %s not found.'%self.options.config)

    def mkdir(self, path): # TODO automatically create directory
        if not os.path.exists(path):
            os.makedirs(path)

    def items_to_params(self, items):
        params = {}
        for t,v in items:
            try: # evaluating the parameter
                params[t] = eval(v)
                if isinstance(params[t], np.ndarray):
                    params[t] = params[t].tolist()
            except (NameError, SyntaxError):
                params[t] = v
        return params

    def generate_conditions(self, paramlist):
        if type(paramlist) == types.DictType:
            paramlist = [paramlist]

        iparamlist = []
        for params in paramlist:
            if ('experiment' in params and params['experiment'] == 'single'):
                iparamlist.append(params) # only do one repetition of this experiment
            else:
                iterparams = [p for p in params if hasattr(params[p], '__iter__')]
                if len(iterparams) > 0:
                    iterfunc = itertools.product
                    for il in iterfunc(*[params[p] for p in iterparams]):
                        par = params.copy() # keep the params which have only one value
                        for i, ip in enumerate(iterparams):
                            par[ip] = il[i]
                        iparamlist.append(par)
                else:
                    iparamlist.append(params)

        return iparamlist

    def start(self):
        paramlist = []
        for exp in self.config_parser.sections():
            params = self.items_to_params(self.config_parser.items(exp))
            params['name'] = exp
            paramlist.append(params)

        np.random.shuffle(paramlist) # randomise the order of experiment conditions

        self.outputs = self.run_experiment(paramlist)

        self.finalize(params)

    def run_experiment(self, params):
        paramlist = self.generate_conditions(params)
        for pl in paramlist:
            if ('iterations' in pl) and ('repetitions' in pl):
                pass
            else:
                raise SystemExit('parameter set does not contain all required keys: iterations, repetitions')

        explist = []

        for p in paramlist:
            explist.extend(zip( [self]*p['repetitions'], [p]*p['repetitions'], xrange(p['repetitions']) ))

        setup_params = {}
        params = params[0]
        for key in params:
            if not isinstance(params[key], list):
                setup_params[key] = params[key]
            else:
                try:
                    if len(params[key]) == 1:
                        setup_params[key] = params[key][0]
                except TypeError:
                    pass
        self.success = self.pre_setup(setup_params)

        if self.success is True:

            if self.options.ncores == 1:
                outputs = []
                for e in explist:
                    output = mp_runrep(e)
                    outputs.append(output)

            else:
                pool = Pool(processes=self.options.ncores, maxtasksperchild=2)
                outputs = pool.map(mp_runrep, explist)
                pool.close()

            return outputs

        else:
            raise Exception

    def run_rep(self, params, rep):
        self.success = self.setup(params, rep)

        results = []

        if self.success is True:

            if params['iterations'] == 1:
                iter_dict = params.copy()
                return_dict = self.iterate(params, rep, 1)
                if return_dict is not None:
                    iter_dict.update(return_dict)
                    results.append(iter_dict)

            else:

                for it in xrange(params['iterations']):
                    iter_dict = params.copy()
                    if return_dict is not None:
                        iter_dict.update(return_dict)
                        results.append(iter_dict)

            self.finalize_rep(params, rep)

        # write intermediate results
        try:
            if self.csv_header is None:
                self.csv_header = list(results[0].keys())
                with open(self.options.outputfile, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
                    writer.writeheader()

            with open(self.options.outputfile, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
                for row in results:
                    writer.writerow(row)
        except IndexError: # there are no results to record
            print "No results to record."
            pass

        return results

    def pre_setup(self, params):
        """ Implemented by the subclass. """
        return True

    def setup(self, params, rep):
        """ Implemented by the subclass. """
        return True

    def iterate(self, params, rep, n):
        """ Implemented by the subclass. """
        ret = {'iteration':n, 'repetition':rep}
        return ret

    def finalize_rep(self, params, rep):
        """ Optionally implemented by the subclass. """
        pass

    def finalize(self, params):
        """ Optionally implemented by the subclass. """
        pass

    def query_output(self, filter=None, sort_keys=None):
        """
        Return a list of dictionaries corresponding to experiment trials,
        filtered by filter.
        e.g., query_output(lambda r:1997 <= int(r['Year']) <= 2002))
        will return a list of trials whose 'Year' parameter satisfies
        the given condition.

        """
        if filter is not None:
            ret = (r for r in self.outputs if filter(r))
        if sort_keys is not None:
            ret = sorted(self.outputs, key=lambda r:[r[k] for k in sort_keys])
        else:
            ret = list(self.outputs)
        return ret

    def lookup_output(self, **kw):
        """
        Return a list of dictionaries corresponding to experiment trials
        that have the parameter value given by kw.
        e.g., lookup_output(Name='Experiment1')
        will return a list of trials whose name is Experiment1.

        """
        for row in self.outputs:
            for k,v in kw.iteritems():
                if row[k] != str(v): break
            else:
                return row
        return None
