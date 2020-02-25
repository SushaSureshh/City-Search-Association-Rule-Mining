
''' Author : Susha Pozhampallan Suresh,

The program takes as input the data, trains an apriori algorithm, and fp tree algorithm to find frequent dataset.

for any specific counrty. 

The program then creates an association rule to predict the next possible city to be searched.

'''

# Import all the libraries required

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules


class association_mining():

    ''' Define a model that will take in the training data of sequence
        of search and predict the next possible searches given 0 to n list of searches
    '''

    def __init__(self):
        self.city = {'Chicago IL'} # the list of cities searched
        self.country = 'US'
    def load_data(self):
        
        '''
        Load the json file to a dataframe
        '''
        
        # path of the data here
        self.data = pd.read_json('./city_search.json')
        self.data['user_id'] = self.data['user'].apply(lambda x: x[0][0]['user_id'])
        self.data['joining_date'] = self.data['user'].apply(lambda x: x[0][0]['joining_date'])
        self.data['country'] = self.data['user'].apply(lambda x: x[0][0]['country'])
        # drop the user column
        self.data.drop('user', axis=1, inplace=True)
        
        self.data_country = self.data[self.data['country'] == self.country]

    def sparse_transaction_encoder(self):
        
        '''
        Convert the training data to format that is suitable for Machine Learning models
        encode the data using transaction encoder
        '''
        
        # obtain list of cities from data frame
        self.data_cities = self.data_country['cities'].values.tolist()
        self.data_cities_list = [i[0].split(', ') for i in self.data_cities]
        
        # initialize the tansaction encoder 
        transactionencoder = TransactionEncoder()
        
        #  encode the into an array format suitable for typical machine learning
        transaction = transactionencoder.fit(
            self.data_cities_list).transform(self.data_cities_list, sparse=True)
        
        # convert to a sparse data frame
        self.data_transaction = pd.SparseDataFrame(
            transaction, columns=transactionencoder.columns_, default_fill_value=False)

    def association_rule(self):
        
        '''
        Grow an fptree or use apriori algorithm to find the frequent dataset
        '''
        # self.data_itemset = apriori(self.data_transaction, min_support=0.001, use_colnames = True)
        self.data_itemset = fpgrowth(self.data_transaction, min_support=0.001, use_colnames = True)
        # create association rules for the frequent item set
        self.data_rule = association_rules(self.data_itemset, metric="confidence", min_threshold=0.01)


    def prediction_cities(self, city):
        
        ''' 
        Takes a list of cities and cities searched as input and returns the next possible city 
        likely to be searched
        '''
        
        # Math the antecedents to the list of cities 
        self.match = self.data_rule[self.data_rule['antecedents'] == city]
        
        #Print the consequent or list of consequents corresponding to the antecedent with highest confidece value
        self.prediction = self.data_rule.loc[self.match['confidence'].idxmax()]['consequents']
        
        return self.prediction

    def run(self):
        
        '''
        Train and predicts the next search
        '''
        
        data = self.load_data()
        data_transaction = self.sparse_transaction_encoder()
        data_rules = self.association_rule()
        prediction = self.prediction_cities(self.city)
        print("Cities most likely to be searched next:" , prediction)


def main():
    
    ''' 
    Function that predicts next possible search
    
    '''
    association_mining().run()



if __name__ == '__main__':
    main()






