from random import seed
from random import randrange
from csv import reader
from math import exp
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
	for i in range(len(dataset[0])):
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
		print("avg %f" % avg)
		sd = standardDeviation(dataset[i], avg)
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - avg)/sd, 6)  #Mean scaling
	return numpy.asarray(dataset).T.tolist()

def maxmin_scaling(dataset):
	# This scaling brings the value between 0 and 1.
	minmax = get_minmax(dataset)
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def hypothesis(data, params):
	sum = 0
	for i in range(len(params)):
		sum += params[i] * data[i]
	return sum

def gradient_descent(dataset, alfa, params, y):
	newParams = list(params)
	for i in range(len(params)):
		sum = 0
		for j in range(len(dataset)):
			sum = sum + (hypothesis(dataset[j], params) - y[j]) * dataset[j][i]
		# Update
		newParams[i] = params[i] - alfa * (1 / len(dataset)) * sum
		#print(newParams)
	return newParams

def calculate_errors(dataset, params, y):
	sum = 0
	for i in range(len(dataset)):
		hyp = hypothesis(dataset[i], params)
		#print("Predicted: ", hyp, "\t Real: ", y[i])
		sum = sum + (hyp - y[i]) ** 2
	c = sum / len(dataset)
	print("Error: ", c)
	cost.append(c)
	return c

def main():
	s = True
	""" bodyWeight_data details:
			I,  the index,
			A1, the brain weight;
			B,  the body weight.
		We seek a model of the form:
			B = A1 * X1. """
	#filename = 'bodyWeight_data.csv'

	""" admision_data details:
		GRE
		Score
		TOEFL Score
		University Rating
		SOP
		LOR
		CGPA
		Research
		Chance of Admit """
	filename = 'admission_data2.csv'
	# Load csv file
	dataset = load_csv(filename)

	# Separate y
	dataset,y = separate_y(dataset)

	# Initialize params = 0
	params = [0 for i in range(len(dataset[0]))]
	print(dataset)
	#params = [0,0,0]
	#dataset = [[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5]]
	#y = [2,4,6,8,10]
	# If scale = True, scale the data
	if s:
		dataset = mean_scaling(dataset)

	# Define old params, epochs and error
	oldparams = list()
	epochs = 0
	alfa = 0.01
	#print("oldparams: ", oldparams, "epochs: ", epochs)

	keepTrying = True
	while(keepTrying):
		oldparams = list(params)
		params = gradient_descent(dataset, alfa, params, y)
		error = calculate_errors(dataset, params, y)
		epochs += 1
		#print("EPOCH = ", epochs)
		if (error < 0.00001 or oldparams == params or epochs == 1000):
			keepTrying = False
	print("Final params:")
	print(params)
	plt.plot(cost)
	plt.show()

if __name__ == "__main__":
    main()
