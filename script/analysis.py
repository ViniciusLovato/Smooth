#!/usr/bin/env python3
# encoding: utf-8

import os
import pprint
import re
import sys
import math
import collections


def main():
    # Generate regex for file names
    p = re.compile('(?P<type>(par|seq))_(?P<name>[A-Za-z0-9]*)\.out')

    avg = collections.defaultdict(dict)
    error = collections.defaultdict(dict)
    

    for f in os.listdir('output'):
        # Get information from filename regex match
        m = p.match(f)
        if not m:
            print('invalid filename: ' + f + ', skipping...', sys.stderr)
            continue
        
        name = m.group('name')

        f_ = os.path.join('output', f)
        
        sum_ = 0.0
        sum_error = 0.0
        with open(f_) as f:
            floats = []
            for line in f:
                float_line = line.split()
                for num in float_line:
                    floats.append(float(num))


            total = 0
            for num in floats:
                sum_ += num
                total += 1
            sum_ /= total
            avg[m.group('type')][name] = sum_

            for num in floats:
                sum_error += (num - sum_) * (num - sum_)
            sum_error /= total
            sum_error = math.sqrt(sum_error)
            error[m.group('type')][name] = sum_error

    for key in avg['seq']:
    	print('Name: ' + key)
    	print('seq: ' + str(avg['seq'][key]) + ' +- ' + str(error['seq'][key]))
    	print('par: ' + str(avg['par'][key]) + ' +- ' + str(error['par'][key]))

    return

      

if __name__ == "__main__":
    main()