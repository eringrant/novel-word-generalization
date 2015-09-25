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
            action='store', dest='outputfile', type='string', default=None,
            help="The output csv file.")

        options, args = optparser.parse_args()
        self.options = options
        return options, args

    def parse_config(self):

    def mkdir(self, path): # TODO automatically create directory


    def start(self):


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
                if outputfile is not None:
                    with open(self.options.outputfile, 'a') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=self.csv_header)
                        writer.writeheader()

            if outputfile is not None:
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
