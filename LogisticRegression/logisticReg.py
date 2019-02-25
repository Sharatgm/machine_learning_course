from random import seed
from random import randrange
import csv
import math
import numpy
import matplotlib.pyplot as plt


cost = []
accuracy = []
# Load a CSV file
def load_csv(filename, addBias):
	# Opens a csv file and saves each row in a list with the value of b added (1)
	dataset = list()
	with open(filename, 'r') as file:
		csv_data = csv.reader(file)
		for row in csv_data:
			if not row:
				continue
			if addBias:
				dataset.append([1] + [float(item) for item in row])
			else:
				dataset.append([float(item) for item in row])
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

def standardization_scaling(dataset):
	# Standardization: Scales the dataset received
	#to have a mean = 0 and a standardDeviation = 1
	acum =0
	dataset = numpy.asarray(dataset).T.tolist()
	for i in range(1,len(dataset)):
		for j in range(len(dataset[i])):
			acum=+ dataset[i][j]
		avg = acum/(len(dataset[i]))
		sd = standard_deviation(dataset[i], avg)
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - avg)/sd, 6)
	return numpy.asarray(dataset).T.tolist()

def maxmin_scaling(dataset):
	# Normalization: This scaling brings the value between 0 and 1.
	minmax = [[0, 1.0]]
	minmax += get_minmax(dataset)
	print(minmax)
	acum =0
	dataset = numpy.asarray(dataset).T.tolist()
	for i in range(1,len(dataset)):
		for j in range(len(dataset[i])):
			dataset[i][j] = round((dataset[i][j] - minmax[i][0])/
							(minmax[i][1] - minmax[i][0]), 6)
	return numpy.asarray(dataset).T.tolist()

def hypothesis(data, coefficients):
	yhat = 0
	for i in range(len(coefficients)):
		yhat += coefficients[i] * data[i]
	return 1.0 / (1.0 + math.exp(-yhat))

def stochastic_gradient_descent(dataset, alfa, coefficients, y):
	newCoef = list(coefficients)
	for i in range(len(coefficients)):
		sum = 0
		for j in range(len(dataset)):
			prediction = hypothesis(dataset[j], coefficients)
			sum = sum + (prediction - y[j]) * dataset[j][i]
		# Update
		newCoef[i] = coefficients[i] - alfa * (1 / len(dataset)) * sum
		#print(newParams)
	return newCoef

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
	""" Titanic prediction:
			pclass -> socieconomic status
			sex -> 0 (male), 1 (female)
			age
			passenger fare
			port of embarkation -> 0 = Cherbourg, 1 = Queenstown, 2 = Southampton
			survived """
	filename = 'titanic/train.csv'

	# Load csv file
	dataset = load_csv(filename, True)

	# Separate last column (y)
	dataset,y = separate_y(dataset)

	# Initialize params = 0
	coefficients = [0.0 for i in range(len(dataset[0]))]

	# If scale = True, scale the data
	if s:
		dataset = standardization_scaling(dataset)

	# Define epochs and alfa
	n_epoch = 100000
	alfa = 0.3

	for epoch in range(n_epoch):
		temp = list(coefficients)
		coefficients = stochastic_gradient_descent(dataset, alfa, coefficients, y)
		error = cross_entropy(dataset, coefficients, y)
		#print("EPOCH = ", epochs)
		if (error < 0.001 or temp == coefficients):
			break

	print("Final params:")
	print(coefficients)
	plt.plot(cost)
	plt.show()

	filename = 'titanic/test.csv'
	# Load csv file
	test = load_csv(filename, False)

	# If scale = True, scale the data
	if s:
		test = standardization_scaling(test)

	print("Executing test...")

	with open('testSubmission.csv', mode='w') as submission_test:
		test_writer = csv.writer(submission_test, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		test_writer.writerow(['PassengerId', 'Survived'])
		for i in range(len(test)):
			prediction = hypothesis([1] + test[i][1:], coefficients)
			if (prediction > 0.5):
				test_writer.writerow([int(test[i][0]), '1'])
			else:
				test_writer.writerow([int(test[i][0]), '0'])

	print("Test executed! Results available in testSubmission.csv")
if __name__ == "__main__":
    main()
