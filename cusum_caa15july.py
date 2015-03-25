import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import numpy.random as np_random
import copy

DEBUG = False

def read_intel_data_to_database(filename, database_name):
    line_number = 1
    try:

        # Creates or opens a db file"
        db = sqlite3.connect(database_name)

        print "Creating database..."

        # Get a cursor object
        cursor = db.cursor()

        cursor.execute('''DROP TABLE IF EXISTS labdata''')
        cursor.execute('''
            CREATE TABLE labdata(id INTEGER PRIMARY KEY, date TEXT,
                           time TEXT, epoch INTEGER, moteid INTEGER,
                           temperature REAL, humidity REAL, light REAL, voltage REAL)
            ''')
        db.commit()


        # Inserting the data from the .txt file
        with open(filename) as f:
            # goes through every record/row
            for line in f:
                x = line.split(" ")
                cursor = db.cursor()
                date_data = x[0]
                time_data = x[1]
                epoch_data = x[2]
                moteid_data = x[3]
                temperature_data = x[4]
                humidity_data = x[5]
                light_data = x[6]
                voltage_data = x[7]
        
                # Insert data into table
                cursor.execute('''INSERT INTO labdata(date, time, epoch, moteid, temperature, humidity, light, voltage)
                                  VALUES(?,?,?,?,?,?,?,?)''', (date_data,time_data, epoch_data, moteid_data, temperature_data, humidity_data, light_data, voltage_data))
                 
                line_number += 1
            db.commit()
            print "Data loaded into database"

    except Exception as e:
        print "Error on line number: " + str(line_number)
        db.rollback()
        raise e

    finally:
        db.close()

# given original data, corrupts the data
def corrupt_data(z):

    # desired temperature
    #desired_temp = 100

    # number of trials
    no_of_trials = 100

    # makes a copy
    corrupted_array = copy.deepcopy(z)

    # start from certain point
    starting_point = 200

    # corrupt first sample data at starting point
    corrupted_array[starting_point] += np_random.uniform(0, 0.2)

    count = 0

    # corrupt subsequent data
    for i in range(starting_point+1, len(z)):
        corrupted_array[i] = corrupted_array[i-1] + np_random.uniform(0, 0.2)

        if (count >= no_of_trials):
            break
        count += 1
    return corrupted_array

# gets an array of temperature in chronological order for a particular sensor
def get_temperature_array_from_database(database_file, moteid):
    line_number = 1
    temperature_array = []

    # Retrieving data
    try:
        print "Connecting to database..."
        # Creates or opens a db file"
        db = sqlite3.connect(database_file)

        # Get a cursor object
        cursor = db.cursor()

        print "Connected to database"

        print "Querying the database..."
    
        cursor.execute('''SELECT * FROM labdata WHERE moteid = ? AND temperature != "" AND id >= 400 AND id <= 1000 ORDER BY date, time''', (str(moteid),))

        print "Loading data into array..."
        
        for row in cursor:
            #print('{0} : {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}'.format(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]))
            if row[5] == "":
                temperature_array.append(None)
            else:
                temperature_array.append(row[5])
            
    except Exception as e:
        print "Error reading from database in line " + str(line_number)
        db.rollback()
        raise e

    finally:
        db.close()

    print "Data loaded into array..."
    return temperature_array

# takes an array of temperature plotted against time for a particular sensor
def plot_graph(z, label):
    plt.plot(z)
    plt.ylabel(label)
    plt.show()

# prediction algorithm
# takes in an array of the measured values z_k
# returns an array of predictions x_minus [x_0-, x_1-, x_2-, ...]
def cusum(z):

    # h is the predefined threshold
    h = 0.4

    # w is the window size used to estimate parameters
    w = 10
    
    k_size = len(z)
    
    # an array of all the x_plus(s) with arbitrary initial values = 0
    # index of array represents the time k in x_k_plus
    x_plus = [0] * k_size

    # an array of all the x_minus(s)
    # index represents the time k in x_k_minus
    x_minus = [0] * k_size
    x_minus[0] = z[0]

    # an array of all the P_plus(s)
    # index represents the time k in P_k_plus
    P_plus = [0] * k_size

    # an array of all the P_minus(s)
    # index represents the time k in P_k_minus
    P_minus = [0] * k_size

    # an array of all the K(s)
    # index represents the time k in K_k
    K = [0] * k_size

    # an array of all the y(s)
    # index represents the time k in y_k
    y = [0] * k_size

    # S_N_array keeps track of all the S_N in each iteration of k
    S_N_array = [0] * k_size

    # calculate R, the variance between z_k and z_(k-1)
    z_diff_by_one = [z[0]]
    
    for index in range(1, k_size):
        z_diff_by_one.append(z[index] - z[index-1])

    #R = np.var(z_diff_by_one)
    R = 0.016438594892

    # constants, Q
    Q = 0.2809  # estimated variance for Intel Dataset
    
    for k in range(0, k_size):

        if (DEBUG == True):
            print "k = " + str(k)
            print "Sensor Values:"
            print z
            
        
        # updates x- with the (previous) value of x+
        if k > 0:
            x_minus[k] = x_plus[k-1]  # Eqn 3

        if (DEBUG == True):
            print "x_minus[k] = x_plus[k-1] = " + str(x_minus[k])
            print "x_minus:"
            print x_minus


    
        # computes y_k, the difference between the measurement
        # and the estimated state
        y[k] = z[k] - x_minus[k]

        if (DEBUG == True):
            print "y[k] = z[k] - x_minus[k] = " + str(y[k])
            print "y:"
            print y

        # implementing line 2 of algorithm 2
        # miu_1 is the computation of line 2 of algorithm 2
        miu_one = 0

        if (k >= w - 1):
            # compute miu_1

            for i in range(k-w+1, k+1):
               miu_one += y[i]
            
        miu_one /= w

        # implementing line 3 of algorithm 2
        # S_N is the computation of line 3 of algorithm 2        
        S_N = 0

        for i in range(0, k):
            var = (y[i] - miu_one/2.0)
            #print var
            S_N += var
        var2 = miu_one/Q
        #print var2
        S_N = S_N * var2
        
        # stores the S_N value into an array for plotting to track its evolution
        S_N_array[k] = S_N

        # implementing line 4 of algorithm 2
        #if (S_N > h):
            #print "Alert is raised on time k = " + str(k)


        # Start with P+ = P- = 0
        if k > 0:
            P_minus[k] = P_plus[k-1] + Q  # Eqn 4
        else:
            P_minus[k] = Q

        if (DEBUG == True):
            print "P_minus[k] = P_plus[k-1] + Q = " + str(P_minus[k])
            print "P_minus:"
            print P_minus


        # computes K, the factor which multiplies the noise
        # K = P-/(P- + R)
        K[k] = P_minus[k]/(P_minus[k] + R)  # Eqn 5

        if (DEBUG == True):
            print "K[k] = P_minus[k]/(P_minus[k] + R) = " + str(K[k])
            print "K:"
            print K



        # update x+ with the measurement
        # x+ = x- + K * (y_k)
        x_plus[k] = x_minus[k] + K[k] * y[k]  # Eqn 6

        if (DEBUG == True):
            print "x_plus[k] = x_minus[k] + K[k] * y[k] = " + str(x_plus[k])
            print "x_plus:"
            print x_plus


        
        # update P+
        P_plus[k] = (1-K[k]) * P_minus[k]  # Eqn 7

        if (DEBUG == True):
            print "P_plus[k] = (1-K[k]) * P_minus[k] = " + str(P_plus[k])
            print "P_plus:"
            print P_plus
            print " ----- "


    # returns an array of predictions [x_0-, x_1-, x_2-, ...]
    
    if (DEBUG == True):
        print "x_minus predictions:"
        print x_minus

    #print S_N_array

    return (x_minus, S_N_array)


# MAIN PROGRAM

# SELECT SENSOR ID
sensor_id = 1

# READS DATA FROM TXT TO DATABASE FILE
# read_intel_data_to_database("intel_data.txt", "intel_lab_data.db")

# GETS DATA FROM DATABASE TO ARRAY
z = get_temperature_array_from_database("intel_lab_data.db", sensor_id)

# CORRUPT THE DATA
corrupted_data = corrupt_data(z)

(x_minus, S_N_array) = cusum(z)
(x_minus2, S_N_array2) = cusum(corrupted_data)

plt.plot(corrupted_data)
plt.plot(S_N_array, 'r')
plt.plot(S_N_array2, 'b')
plt.show()

# PERFORM PREDICTION ALGORITHM
#(x_minus, S_N_array) = cusum(z)

#plt.plot(S_N_array)
#plt.show()

#plot_graph(z, "Temperature for Sensor " + str(sensor_id))
#plot_graph(x_minus, "x_minus Temperature for Sensor " + str(sensor_id))

#plt.plot(z, 'g')
#plt.plot(x_minus, 'r', alpha=0.5)
#plt.plot(S_N_array, 'b', alpha=0.5)

#plt.ylim([24,25])
#plt.ylabel("Temperature")
#plt.show()
