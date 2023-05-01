import socket, time
import json
import csv
import os;
from datetime import datetime


port = 8080
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)        
print ("Socket successfully created") 

# TODO Change server IP here
s.bind(('172.26.63.39', port))          
print ("socket binded to %s" %(port))

s.listen(5)      
print ("socket is listening")  

try:
	while 1:
		# Connect to processing socket
		newSocket, address = s.accept()
		print ("Connection from ", address)
		
		
		while 1:
			cmd = ""

			# new rx buffer from socket
			receivedData = newSocket.recv(1024)
			if not receivedData:
			    break
			# decode to utf-8 unicode characters
			data = json.loads(receivedData.decode('utf-8'))
			print(data)


			if(data["type"] == "reset"):
				if(os.path.isfile('personal.csv')):
					os.remove('personal.csv')
				if(os.path.isfile('enviro.csv')):
					os.remove('enviro.csv')
			elif(data["type"] == "personal"):
				with open('personal.csv', mode='a', newline='') as file:
					writer = csv.writer(file)
					current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
					time_write = {"Time":current_time}
					data.update(time_write)
					writer.writerow(data.values())
			elif(data['type'] == "Training"):
				os.system("python particle_server.py")
			elif(data['type'] == 'request'):

				# Open the CSV file
				with open('opt_temp.csv', 'r') as csvfile:
					# Create a CSV reader object
					reader = csv.reader(csvfile)

					# Iterate over the rows in the file
					rows = list(reader)

					# Extract the last row from the list of rows
					last_val = rows[-1]

					# Extract the value from the last row (which is a list with only one element)
					opt_temp = last_val[0]

				message = "Optimum temperature for this zone is " + opt_temp +" degrees Celcius"
				newSocket.send(message.encode('utf-8'))
				print("Sent curr temp out")
			else:
				with open('enviro.csv', mode='a', newline='') as file:
					writer = csv.writer(file)
					current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
					time_write = {"Time":current_time}
					data.update(time_write)
					writer.writerow(data.values())
			

			# make sure that data matches expected protocol
			# if not data.startswith("msg:"):
			# 	continue
			# else:
			# 	print ("Received (" + str(len(data)) + "): " + data)

			# prepare reply from user
			# while (cmd == ""):
			# 	cmd = input("Name: ")

			# # format string as byte array
			# cmd_b = bytearray()
			# cmd_b.extend(map(ord, cmd))
			# print ("Sending: " + cmd_b.decode('utf-8'))

			# send reply
			# newSocket.send(cmd_b)
		
		# close connection
		newSocket.close()
		print("Disconnected from ", address)
finally:
	s.close()