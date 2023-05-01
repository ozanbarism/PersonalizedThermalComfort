import socket
import threading
import csv
import os
import time

def handle_client(client_socket, client_address):
    print("Connection from ", client_address)

    while True:
        # Receive data from client
        received_data = client_socket.recv(1024)

        if not received_data:
            break

        # Decode received data
        data = received_data.decode('utf-8')
        
        if data=='\r\n':
            pass
        else:
            
            values = data.split(',')
            csv_file = values[0]+"_temp_hum.csv"
            #print("Writing to file:", csv_file)
            
            
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                
                print("Received data:", data)
                if file.tell() == 0:
                    
                    writer.writerow(['time','temp', 'hum'])
                writer.writerow(values[1:4])

    # Close connection
    client_socket.close()
    print("Disconnected from ", client_address)

def start_server():
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind socket to IP address and port
    server_socket.bind(("172.26.63.39", 8008))

    # Listen for incoming connections
    server_socket.listen()

    print("Server started. Listening on port 8080...")

    # Handle incoming connections
    while True:
        # Accept incoming connection
        client_socket, client_address = server_socket.accept()

        # Start a new thread to handle the client connection
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        client_thread.start()

if __name__ == "__main__":
    start_server()
