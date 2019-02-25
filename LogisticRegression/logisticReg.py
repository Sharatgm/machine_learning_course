from random import seed
from random import randrange
from csv import reader
import math
import numpy
import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)


cost = []
# Load a CSV file
def load_csv(filename):
	# Opens a csv file and saves each row in a list with the value of b added (1)
	dataset = list()
	with open(filename, 'r') as file:
		csv_data = reader(file)
		for row in csv_data:
			if not row:
				continue
			dataset.append([1] + [float(item) for item in row])
	return dataset

def separate_y(dataset):
	# Separates the last column (y values) in a new list
	y = list()
	for i in range(len(dataset)):
		y.append(dataset[i].pop())
	return dataset, y

def standard_deviation(data, avg):
	sum = 0
	for i in range(len(data)):
		sum += (data[i] - avg)**2
	return sum/len(data)

def get_minmax(dataset):
	# Find the min and max values for each column
	minmax = list()
	for i in range(1,len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

def mean_scaling(dataset):
	# Scales the dataset received
	# This distribution will have values between -1 and 1
	acum =0
	dataset = numpy.asarray(dataset).T.tolist()
	for i in range(1,len(dataset)):
		for j in range(len(dataset[i])):
			acum=+ dataset[i][j]
		avg = acum/(len(dataset[i]))
		maxV = max(dataset[i])
		minV = min(dataset[i])
		print("avg %f" % avg)
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - avg)/maxV, 6)  #Mean scaling?
	return numpy.asarray(dataset).T.tolist()

def standarization_scaling(dataset):
	# Scales the dataset received
	acum =0
	dataset = numpy.asarray(dataset).T.tolist()
	for i in range(1,len(dataset)):
		for j in range(len(dataset[i])):
			acum=+ dataset[i][j]
		avg = acum/(len(dataset[i]))
		sd = standard_deviation(dataset[i], avg)
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - avg)/sd, 6)  #Mean scaling
	return numpy.asarray(dataset).T.tolist()

def maxmin_scaling(dataset):
	# This scaling brings the value between 0 and 1.
	minmax = [[0, 1.0]]
	minmax += get_minmax(dataset)
	print(minmax)
	acum =0
	dataset = numpy.asarray(dataset).T.tolist()
	for i in range(1,len(dataset)):
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - minmax[i][0])/  (minmax[i][1] - minmax[i][0]), 6)
	return numpy.asarray(dataset).T.tolist()

def hypothesis(data, params):
	sum = 0
	for i in range(len(params)):
		sum += params[i] * data[i]
	return 1 / (1 + math.exp((-1) * sum))

def stochastic_gradient_descent(dataset, alfa, params, y):
	newParams = list(params)
	for i in range(len(params)):
		sum = 0
		for j in range(len(dataset)):
			prediction = hypothesis(dataset[j], params)
			sum = sum + (prediction - y[j]) * dataset[j][i]
		# Update
		newParams[i] = params[i] - alfa * (1 / len(dataset)) * sum
		#print(newParams)
	return newParams

def cross_entropy(dataset, params, y):
	sum = 0
	for i  in range(len(dataset)):
		prediction = hypothesis(dataset[i], params)
		if y[i] == 1:
			if (prediction == 0):
				prediction = 0.00001
			# -log (h(xi))
			error = math.log(prediction) * -1
		else:
			if(prediction == 1):
				prediction = 0.99999
			# -log (1 - h(xi))
			error = math.log(1-prediction) * -1
		sum = sum + error
	c = sum / len(dataset)
	print("Error: ", c)
	cost.append(c)
	return c

def main():
	s = True

	filename = 'pima-indians-diabetes.csv'
	# Load csv file
	dataset = load_csv(filename)

	# Separate last column (y)
	dataset,y = separate_y(dataset)

	# Initialize params = 0
	params = [0 for i in range(len(dataset[0]))]

	# If scale = True, scale the data
	if s:
		dataset = maxmin_scaling(dataset)

	# Define old params, epochs and alfa
	oldparams = list()
	epochs = 0
	alfa = 0.3

	keepTrying = True
	while(keepTrying):
		prev = list(params)
		params = stochastic_gradient_descent(dataset, alfa, params, y)
		error = cross_entropy(dataset, params, y)
		epochs += 1
		#print("EPOCH = ", epochs)
		if (error < 0.00001 or prev == params or epochs == 1000):
			keepTrying = False
	#print("Final params:")
	#print(params)
	plt.plot(cost)
	plt.show()

if __name__ == "__main__":
    main()
