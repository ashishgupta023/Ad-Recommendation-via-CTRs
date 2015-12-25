import numpy as np
import csv

ads_set = np.array(["A1","A2","A3","A4","A5","A6","A7","A8","B1","B2","B3","B4","B5","B6","B7","B8",
                    "C1","C2","C3","C4","C5","C6","C7","C8","D1","D2","D3","D4","D5","D6","D7","D8"])

next_recommendation = ''

rows_to_process = 200

ctrs = np.zeros(32,dtype=np.float64) # Stores click through rates
ads_impressions = np.zeros(32) # Stores impressions of ads
ads_clicks = np.zeros(32) # Stores clicks of ads

minimax = np.zeros((3,32),dtype=np.float64)
# Using the minimax approach for ad recommendation
# https://en.wikipedia.org/wiki/Minimax
# 0th row : CTRs increase -> ad show + hit
# 1st row : CTRs decrease -> ad show + miss
# 2nd row : CTRs same -> ad no show

max_regret = np.zeros(32) # Stores the maximum regret for all ads
                          # calculated for all the states of nature an action can take.
                          # Action corresponds to an Ad being shown or not.

with open('Data-Scientist-Task-Test-data.csv') as datafile:
     csvreader = csv.reader(datafile)
     row_num = 0
     for row in csvreader:
        if row_num != 0: 
            count = 0
            processed_columns = set([])
            while len(processed_columns) != 32:  # For each impression
                column = 0
                while column in processed_columns:
                    column = np.random.randint(0,32) # randomizing the page views
                processed_columns.add(column)
                ads_impressions[column] += 1 # Increment impression for current Ad variant
                ads_clicks[column] += np.int(row[column]) # Store the click data for the current ad variant
                ctrs[column] = ads_clicks[column] * 100 /  ads_impressions[column] # Update click-through rate for current ad variant
                
                #The Minimax algorithm
                # Consider:  payoff directly proportional to the clicks
                minimax[0] = (((ads_clicks + 1 ) * 100) / (ads_impressions + 1)) 
                minimax[1] = ((ads_clicks * 100) / (ads_impressions + 1)) 
                minimax[2] = ctrs
                #state of nature - 0 CTRs increase  - ad show + hit   
                maxPayoff = np.amax(minimax[0])
                minimax[0] = maxPayoff - minimax[0]
                #state of nature - 1 CTRs decrease - ad show + miss
                maxPayoff = np.amax(minimax[1])
                minimax[1] = maxPayoff - minimax[1] 
                #state of nature - 2 CTRs same - ad no show
                maxPayoff = np.amax(minimax[2])
                minimax[2] = maxPayoff - minimax[2] 
                
                max_regret = np.amax(minimax,axis=0) # Find maximum regret for all actions
                
                next_recommendation = ads_set[np.argmin(max_regret)] # Choose the action with minimum regret
                
                print "Ad Recommendation: " + next_recommendation
        row_num = row_num+1
        if row_num == rows_to_process:
            break     
