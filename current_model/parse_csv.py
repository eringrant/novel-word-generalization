import operator
import csv

def sort_by_column(csv_cont, col, reverse=False):
    """
    Sorts CSV contents by column name (if col argument is type <str>)
    or column index (if col argument is type <int>).

    """
    header = csv_cont[0]
    body = csv_cont[1:]
    if isinstance(col, str):
        col_index = header.index(col)
    else:
        col_index = col
    body = sorted(body,
           key=operator.itemgetter(col_index),
           reverse=reverse)
    body.insert(0, header)
    return body

def csv_to_list(csv_file, delimiter=','):
    """
    Reads in a CSV file and returns the contents as list,
    where every row is stored as a sublist, and each element
    in the sublist represents 1 cell in the table.

    """
    with open(csv_file, 'r') as csv_con:
        reader = csv.reader(csv_con, delimiter=delimiter)
        return list(reader)

def convert_cells_to_floats(csv_cont):
    """
    Converts cells to floats if possible
    (modifies input CSV content list).

    """
    for row in range(len(csv_cont)):
        for cell in range(len(csv_cont[row])):
            try:
                csv_cont[row][cell] = float(csv_cont[row][cell])
            except ValueError:
                pass

def write_csv(dest, csv_cont):
    """ Writes a comma-delimited CSV file. """

    with open(dest, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=',')
        for row in csv_cont:
            writer.writerow(row)

