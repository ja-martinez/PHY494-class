import numpy as np

#tuple array of South, West, East and Midwest Regionals respectively, listed in #1 - 16 seed order
R1teams = (("Kansas", "Villanova", "Miami", "California", "Maryland", "Arizona", "Iowa", "Colorado", "Connecticut", "Temple", "Wichita State", "South Dakota State", "Hawaii", "Buffalo", "UNC Asheville", "Austin Peay"),
            ("Oregon", "Oklahoma", "Texan A&M", "Duke", "Baylor", "Texas", "Oregon State", "Saint Joseph's", "Cincinnatti", "VCU", "Northern Iowa", "Yale", "UNC Wilmington", "Green Bay", "Cal State Bakersfield", "Holy Cross"),
            ("North Carolina", "Xavier", "West Virginia", "Kentucky", "Indiana", "Notre Dame", "Wisconsin", "USC", "Providence", "Pittsburgh", "Michigan", "Chattanooga", "Stony Brook", "Stephen F Austin", "Weber State", "Florida Gulf Coast"), 
            ("Virginia", "Michigan State", "Utah", "Iowa State", "Purdue", "Seton Hall", "Dayton", "Texas Tech", "Butler", "Syracuse", "Gonzaga", "Ankansas-Little Rock", "Iona", "Fresno State", "Middle Tennessee", "Hampton"))

# blank placeholder for each round winners
R1Winners=[["", "", "", "", "", "", "", ""],
          ["", "", "", "", "", "", "", ""],
          ["", "", "", "", "", "", "", ""],
          ["", "", "", "", "", "", "", ""]]

R2Winners=[["", "", "", ""],
          ["", "", "", ""],
          ["", "", "", ""],
          ["", "", "", ""]]

R3Winners=[["", ""],
          ["", ""],
          ["", ""],
          ["", ""]]

R4Winners=[[""],
          [""],
          [""],
          [""]]

R5Winners=["", ""]

Champion=[""]

#Random number generator for first round of games (8 games x 4 conferences)
Rand1 = np.random.randint(100, size=(4, 8))

# Round 1 simulation
for i in range(8):
    for j in range(4):
        if Rand1[j][i] > 49:
            R1Winners[j][i] = R1teams[j][2*i]
        else:
            R1Winners[j][i] = R1teams[j][2*i+1]

Rand2 = np.random.randint(100, size=(4, 4))  #Random generator for second round (4 games x 4 conferences)

# Round 2 simulation
for i in range(4):
    for j in range(4):
        if Rand2[j][i] > 49:
            R2Winners[j][i] = R1Winners[j][2*i]
        else:
            R2Winners[j][i] = R1Winners[j][2*i+1]

Rand3 = np.random.randint(100, size=(4, 2))

# Round 3 (Sweet 16) simulation 
for i in range(2):
    for j in range(4):
        if Rand3[j][i] > 49:
            R3Winners[j][i] = R2Winners[j][2*i]
        else:
            R3Winners[j][i] = R2Winners[j][2*i+1]

Rand4 = np.random.randint(100, size=(4, 1))

# Round 4 (Elite 8) simulation 
for i in range(1):
    for j in range(4):
        if Rand4[j][i] > 49:
            R4Winners[j][i] = R3Winners[j][2*i]
        else:
            R4Winners[j][i] = R3Winners[j][2*i+1]

Rand5 = np.random.randint(100, size=(2, 1))

# Round 5 (Final 4) simulation 
for i in range(2):
    if Rand5[i][0] > 49:
        R5Winners[i] = R4Winners[2*i][0]
    else:
        R5Winners[i] = R4Winners[2*i+1][0]
            
Rand6 = np.random.randint(100, size=(1, 1))

# Round 6 (Championship) simulation 

if Rand6[0][0] > 49:
    Champion = R5Winners[0]
else:
    Champion = R5Winners[1]

