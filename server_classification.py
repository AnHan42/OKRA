import socket
import pickle
from sklearn import svm
import Kernel_lib
import numpy as np
import time 
import argparse
import zlib
import os
from sklearn.datasets import make_classification
#from cuml import SVC as cumlSVC
from threading import Thread
import struct 

def deserialize_matrix_server(serialized_data):
    header_length = 4 * 4 + 1 * 8  # 4 integers and 1 long integer
    client_id, rows, cols, matrix_type, data_length = struct.unpack('!4I1Q', serialized_data[:header_length])
    matrix_data = serialized_data[header_length:]
    del serialized_data
    if matrix_type == 0:  # 1D array
        matrix = np.frombuffer(matrix_data, dtype=np.float32)
    elif matrix_type in [1, 2, 3]:  # 2D array, Row vector, Column vector
        matrix = np.frombuffer(matrix_data, dtype=np.float32).reshape(rows, cols if cols else 1)
    elif matrix_type == 4:  # CSR
        matrix = pickle.loads(matrix_data)

    elif matrix_type == 5:  # CSC
        matrix = pickle.loads(matrix_data)

    elif matrix_type == 6:  # list of 1D arrays
        matrix = []
        offset = 0
        for _ in range(rows):  # rows here represents the number of 1D arrays in the list
            # Parse the header for each 1D array
            sub_client_id, sub_rows, _, sub_matrix_type, sub_data_length = struct.unpack(
                '!4I1Q', matrix_data[offset:offset + header_length]
            )
            offset += header_length

            # Deserialize the 1D array
            sub_matrix_data = matrix_data[offset:offset + sub_data_length]
            offset += sub_data_length
            matrix.append(np.frombuffer(sub_matrix_data, dtype=np.float32))
    else:
        raise ValueError("Unsupported matrix type.")

    return client_id, matrix



def get_labels(n_samples, num_clients):
    """
    Get the labels for a data set.

    :param n_samples: the total number of samples in the data set
    :param num_clients: the number of clients in the data set
    :return: the labels for the data set
    """
    temp2 = n_samples * num_clients
    training_data, y = make_classification(n_samples=temp2, random_state=42, shuffle=False)
    return y


def listen_for_clients(num_clients, port):
    """
    Listen for incoming connections from clients and receive their data

    :param num_clients: the number of clients to listen for
    :param base_port: the base port number to use for assigning ports to clients
    :return: a list of the data received from the clients
    """
    #print("gets server adress")
    # The server IP address and port
    server_address = ('localhost', port)
    #print("creates socket")

    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #print("created socket, binds it to adress")

    # Bind the socket to the address
    sock.bind(server_address)

    # Set the socket to listen for incoming connections
    #print("Server is listening for incoming connections")
    sock.listen(num_clients)  # listen for up to num_clients clients

    # Initialize a list to store the data received from the clients
    data = []
    
    # Accept incoming connections one at a time
    while len(data) < num_clients:
        # Wait for a connection
        connection, client_address = sock.accept()
        try:

            # Receive the data in small chunks and retransmit it
            data_chunks = b''
            while True:
                chunk = connection.recv(1024)
                if not chunk:
                    break
                data_chunks += chunk

            # Decompress the data
            data_chunks = zlib.decompress(data_chunks)
            
            # Deserialize the data using pickle
            matrix = pickle.loads(data_chunks)
            data.append(matrix)

        finally:
            # Clean up the connection
            connection.close()

    return data


def receive_data(client_socket):
    data_size_bytes = client_socket.recv(8)

    if len(data_size_bytes) != 8:
        raise ConnectionError("Client disconnected prematurely while sending data size.")
    tt=time.time()
    data_size = struct.unpack('!Q', data_size_bytes)[0]
    tt=time.time()
    chunks = []
    received_size = 0

    while received_size < data_size:
        chunk = client_socket.recv(min(8192, data_size - received_size))
        if not chunk:
            break
        chunks.append(chunk)
        received_size += len(chunk)

    if received_size < data_size:
        raise ConnectionError(f"Expected {data_size} bytes but received only {received_size} bytes.")
    return b''.join(chunks)


def receive_matrices(client_sockets):
    raw_data = [None] * len(client_sockets)
    threads = []

    def threaded_receive_data(client_socket, results, index):
        results[index] = receive_data(client_socket)

    for i, client_socket in enumerate(client_sockets):
        thread = Thread(target=threaded_receive_data, args=(client_socket, raw_data, i))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    matrices = []
    for data in raw_data:
        _, matrix = deserialize_matrix_server(data)
        # Deserialize the data using pickle
        #matrix = pickle.loads(data)
        matrices.append(matrix)

    return matrices

def listen_for_clients_multithreading(num_clients, base_port):
    client_sockets = []
    server_sockets = []
    print('starting to connect')
    for _ in range(num_clients):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #hostname = socket.gethostbyname(socket.gethostname())
        server.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        server.bind(('localhost', base_port + _ ))
        #with open(f'./server_ready_{_+1}.txt', 'w') as f:
        #    f.write('Server is ready')
        server.listen(1)
        print(f"Server is listening on port {base_port + _ }")
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_sockets.append(client_socket)
        server_sockets.append(server)

    #server_address = ('localhost', port)

    # Create a TCP/IP socket
    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #sock.bind(server_address)
    #sock.listen(num_clients)

    # Accept incoming connections one at a time
    #client_sockets = []
    #while len(client_sockets) < num_clients:
        # Wait for a connection
    #    connection, client_address = sock.accept()
    #    client_sockets.append(connection)

    # Use the receive_matrices function to handle the received data
    data = receive_matrices(client_sockets)

    # Clean up all client connections
    for connection in client_sockets:
        connection.close()

    return data


def train_kernel(args): 
    """
    Train a kernel-based ML method on data received from clients.

    :param args: arguments containing the number of clients and the base port to use for listening for incoming connections
    """
    ts1 = time.time()
    
    data = listen_for_clients_multithreading(args.num_clients, args.base_port)

    ts2 = time.time()

    '''
    if args.FL == True: 
        file_path_data = "temp_data.npy"
        file_path_gram = "temp_gram.npy"
        #wenn eine Gram Matrix schon abgelegt wurde == nte FL Runde
        if os.path.exists(file_path_data):
        # load the data matrix from the file
            old_data = np.load(file_path_data) 
            gram = np.load(file_path_gram)
            full = Kernel_lib.update_gram_matrix(gram, old_data, data)
        else: #erste FL Runde
            full = Kernel_lib.compute_gram_matrix(data)
            np.save(file_path_data, data)
            np.save(file_path_gram, full)
    else: 

    '''
    full = Kernel_lib.compute_gram_matrix(data)
    ts3 = time.time()

    #training the classifier
    #if args.GPU == True:
    #    clf = cumlSVC(kernel = 'precomputed', probability=True)
    #else: 
    
    #clf = svm.SVC(kernel = 'precomputed', probability=True)
    #clf1.fit(sklearn.metrics.pairwise.polynomial_kernel(full, Y=None, degree=3, gamma=None, coef0=1), y)

    #y = get_labels(no_samples, args.num_clients)

    #clf.fit(Kernel_lib.polynomial_kernel(full, degree=3), y)

    #ts4 = time.time()
    gram_time=ts3-ts2
    whole_time=ts3-ts1
    print("Gram computation time (serverside) for {}: {}".format(args.num_clients, gram_time)) 
    print("Whole time (serverside) for {}: {}".format(args.num_clients, whole_time)) 
       # Print the result
    #return gram_time, whole_time
    #print('Performance on Custom (Polynomial) Gram Kernel: ')
    #Kernel_lib.metrics_micro(clf, full, y, cv=3)
    #path = "results/results_server_classification_{}_{}.xlsx".format(args.num_clients, no_samples)
    #Kernel_lib.dump_serverside_results(clf, full, y, cv=3, times=time_list, path=path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--run", type=int, default=3, help="the number of clients to listen for")
    parser.add_argument("--num_clients", type=int, default=3, help="the number of clients to listen for")
    parser.add_argument("--run", type=int, default=42, help="number of current run")
    parser.add_argument("--base_port", type=int, default=8000, help="port to connect to")
    parser.add_argument("--GPU", type=bool, default=False, help="use GPU support for training the svm")
    #parser.add_argument("--FL", type=bool, default=False, help="multiple training iterations")
    #parser.add_argument("--n_samples", type=int, default=1000, help="dataset size for one party")
    args = parser.parse_args()
    #gram_time, whole_time = 
    train_kernel(args)

    #times_dict = {"run_no": args.run, "gram_time": gram_time, "whole_time": whole_time}

    #result_string = "results_server_classification_{}_{}".format(args.num_clients, no_samples)

    #Kernel_lib.create_and_save_latex_table(times_dict, result_string, "results/")



