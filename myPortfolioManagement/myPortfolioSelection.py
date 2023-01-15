import itertools
from itertools import chain
from timebudget import timebudget




def all_combinations_inner_loop(min_num_of_funds:int, 
                                max_num_of_funds:int,
                                assets_to_consider:list):
    '''
    This function takes as inputs a list of strings and returns 
    another list with all possible combination that one can make given a 
    set of strings. 
    
    For example. The list of strings could be different tickers of stocks
    The function will find all the possible portfolios you can make given 
    a certain number of stocks one wants to hold (i.e. up to 15 stocks)
    
    It is mostly a helper function. 
    '''
    
    all_combinations = []
    for i in range(min_num_of_funds, max_num_of_funds+1):
        comb = list(itertools.combinations(assets_to_consider,i))
        all_combinations.append(comb)
        
    return all_combinations

@timebudget
def possible_combinations(assets_to_consider:list , min_assets=3, 
                          max_assets=20, must_have = []):
    
    # sanity checks 
    if min_assets> max_assets:
        raise ValueError("max assets langer than minimum")
    if max_assets>20:
        raise ValueError('''Current version does not support
                         more than 20 assets''')
    
    # get combinations     
    all_combinations = all_combinations_inner_loop(min_assets, 
                                                   max_assets,
                                                   assets_to_consider)
    all_portf = list(chain(*all_combinations))
    
    # when not empty 
    if  must_have:
       # consider to raise errors if the user does not pass a subset of 
       # assets_to_consider
       must_have_set = set(must_have)
       all_portf = [x for x in all_portf 
                    if set(x) & must_have_set == must_have_set]
     
    # convert list of lists     
    all_portf = [[*all_portf[i]] for i in range(len(all_portf))] 
    
    return all_portf


