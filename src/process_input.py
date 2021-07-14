"""
Explorations in Data Science
Crypto buy/sell indicators project
"""
import calendar
import csv
import datetime


class ProcessInput:

    def __init__(self, infile, outfile):
        """ Initialize the class

        :param infile: filename of the data to be processed
        :param outfile: filename for the output processed data
        """
        self.infile = infile
        self.outfile = outfile
        self.headers = None
        self.btc_historical_data = []

    def read_btc_historical_csv(self):
        """ Read the BTC historical data and process it

        :return: None
        """
        with open(self.infile) as f:
            # First line is the headers
            self.headers = f.readline()

            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                # remove commas from the elements
                for val in range(len(row)):
                    row[val] = row[val].replace(',', "")

                # Process date
                d = row[0].split(' ')
                month = d[0]
                months = {month: index for index, month in enumerate(calendar.month_abbr) if month}

                row[0] = datetime.date(int(d[2]), months[month], int(d[1]))

                # Convert all numerical values to floats
                row[1] = float(row[1])
                row[2] = float(row[2])
                row[3] = float(row[3])
                row[4] = float(row[4])

                # Process trading volume
                if row[5][-1] == 'K':
                    row[5] = float(row[5][:-1]) * 1000
                elif row[5][-1] == 'M':
                    row[5] = float(row[5][:-1]) * 1000000

                # Remove percent sign from last element
                row[6] = float(row[6][:-1])

                self.btc_historical_data.insert(0, row)

    def write_btc_historical_csv(self):
        """ Write the processed BTC historical data back to file

        :return: None
        """
        with open(self.outfile, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in self.btc_historical_data:
                writer.writerow(row)
