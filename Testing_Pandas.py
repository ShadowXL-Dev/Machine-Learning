import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None, 'display.max_columns', None)

# df = pd.read_excel(r"S:\Common\Production Control\ProductionSchedules\LINE 3 PROD SCHED.xls", sheet_name='Main', usecols="A, C, D, F")
df = pd.read_excel(r"C:\Users\cfournier\NeuralDS.xls", sheet_name='Fascia WP')




#df1 = df.loc[df['LINE 3 PRODUCTION SCHEDULE '].str.contains('ZERV FASCIA', na=False)]
#df1 = df.iloc[df['Unnamed: 2'].str.contains('84742309', na=False)]


# df2 = df[df['Unnamed: 2'] == 71400100]
# df2 = df2.dropna()
# df2 = df1[df['Unnamed: 2'].astype('str').str.contains('84742309')]



# print(df)
# print(df2['Unnamed: 5'])

class ReadData():
    stored = []
    sheetnames = []
    def readNames(self, path, fileType):
        if fileType == 'excel':
            df = pd.ExcelFile(path)
            self.sheetnames = df.sheet_names
            for i in range(len(self.sheetnames)):
                self.stored.append(df.parse(self.sheetnames[i]))
        # for i in range(len(self.stored)):
        #     print(self.stored[2])
        #     print()
    
    def getData(self, ndx):
        return stored[ndx]

    def getSheet(self, ndx):
        return sheetnames[ndx]

    # def length(self):
    #     return len(sheetnames)

    def findNDX(self, searchString):
        for i in range(len(self.sheetnames)):
            if self.sheetnames[i] == searchString:
                return i
            else:
                print("Not Found, Returning NULL")
                pass

rd = ReadData()

rd.readNames(r"C:\Users\cfournier\NeuralDS.xls", 'excel')