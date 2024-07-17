import socket
import numpy as np
import zlib
import pickle
import Kernel_lib
import time
import argparse
from sklearn.datasets import make_classification
from threading import Thread
import struct 
from scipy.sparse import csr_matrix, vstack, hstack, diags, save_npz, csc_matrix, issparse

def chunk_compress_data(data, sock, chunk_size=4096):
    """
    Send data over a socket in small chunks and compress it before sending.

    :param data: the data to send
    :param sock: the socket to send the data over
    :param chunk_size: the size of each data chunk in bytes
    """
    # Compress the data using zlib
    compressed_data = zlib.compress(pickle.dumps(data))

    # Send the data in small chunks
    total_sent = 0
    while total_sent < len(compressed_data):
        sent = sock.send(compressed_data[total_sent:total_sent + chunk_size])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent

def send_data_to_server(client_socket, matrix, client_id):
    tt=time.time()
    data = serialize_matrix_client(client_id, matrix)
    #data = pickle.dumps((client_id, matrix))
    data_size = struct.pack('!Q', len(data))
    tt=time.time()
    client_socket.sendall(data_size)
    client_socket.send(data)
    #print("client sent data to server", client_id, client_socket)


def serialize_matrix_client(client_id, matrix):
    if isinstance(matrix, np.ndarray):
        if len(matrix.shape) == 1:
            matrix_type = 0  # 1D array
            rows = len(matrix)
            cols = 0
            matrix_data = matrix.astype(np.float32).tobytes()
        else:
            rows, cols = matrix.shape
            matrix_data = matrix.astype(np.float32).tobytes()
            if rows > 1 and cols > 1:
                matrix_type = 3  # General 2D array
            elif rows == 1:
                matrix_type = 2  # Row vector
            else:
                matrix_type = 1  # Column vector
    elif isinstance(matrix, (csr_matrix, csc_matrix)):
        rows, cols = matrix.shape
        matrix_type = 4 if isinstance(matrix, csr_matrix) else 5  # CSR or CSC
        matrix_data = pickle.dumps(matrix)
    elif isinstance(matrix, list):
        if all(isinstance(x, np.ndarray) and len(x.shape) == 1 for x in matrix):
            matrix_type = 6  # list of 1D arrays
            serialized_matrices = []
            for mat in matrix:
                serialized_matrices.append(serialize_matrix_client(client_id, mat))
            matrix_data = b''.join(serialized_matrices)
            rows = len(matrix)
            cols = 0  # not applicable for list of 1D arrays
        else:
            raise ValueError("Unsupported list elements.")
    else:
        raise ValueError("Unsupported matrix type.")

    header = struct.pack('!4I1Q', client_id, rows, cols, matrix_type, len(matrix_data))
    return header + matrix_data
    
def get_data(n_samples, party_id, max_parties, n_features): 
    """
    Get a partition of data for a particular party.

    :param n_samples: the total number of samples in the data set
    :param party_id: the identifier of the party requesting the data
    :return: a partition of the data set containing n_samples/party_id samples
    """
    temp = n_samples * max_parties
    training_data, y = make_classification(n_samples=temp, n_features=n_features, random_state=42, shuffle=False)
    partitions = np.split(training_data, max_parties)
    x = partitions[party_id-1]
    return x


def chunk_compress_data(data, sock, chunk_size=4096):
    """
    Send data over a socket in small chunks and compress it before sending.

    :param data: the data to send
    :param sock: the socket to send the data over
    :param chunk_size: the size of each data chunk in bytes
    """
    # Compress the data using zlib
    compressed_data = zlib.compress(pickle.dumps(data))

    # Send the data in small chunks
    total_sent = 0
    while total_sent < len(compressed_data):
        sent = sock.send(compressed_data[total_sent:total_sent + chunk_size])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent

def send_data(args):
    """
    Send the training data to a server on a port determined by the base port and the client number

    :param base_port: the base port number to use for determining the port to send data on
    :param client_number: the number of the client sending the data
    :param training_data: the training data to send to the server
    """

    #load data 
    training_data = get_data(n_samples=args.n_samples, party_id=args.party_id, n_features=args.n_features)

    ts1 = time.time()

    n_features = training_data.shape[1] 
    if args.mask == "FLAKE":
        N = Kernel_lib.random_matrix(n_features + args.k, n_features) 
        data_prime = Kernel_lib.generate_data_prime(training_data, N)
    else:
        gamma = Kernel_lib.get_gamma(n_features, args.k, seed=42)
        data_prime = training_data @ gamma.T
    
    ts2 = time.time()
    masking_time = ts2-ts1

    # The server IP address and port
    port = args.base_port + args.party_id - 1 #because the client_id starts from 1 in bash script
    server_address = ('localhost', port)


    #client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    #print(f'CLient {p} connecting via {args.base_port + p}')
    #client_socket.connect((server_hostname, args.base_port + p))


    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    print(f'client {args.party_id} tries to connect via {server_address}')

    time.sleep(10)

    #try:
    #    sock.connect(server_address)
    #except ConnectionRefusedError:
    #    print("Connection was refused. Check the server address and ensure the server is running.")
    #except Exception as e:
    #    print(f"An unexpected error occurred: {e}")
        
    # Connect the socket to the server
    sock.connect(server_address)

    #time.sleep(5)
    send_data_to_server(sock, data_prime, args.party_id)

    #chunk_compress_data(data_prime, sock)
    #sock.close()    

    # Dump the masking time to a file
    #print("Masking computation times (clientside) for {} {} {} {}: {}".format(args.mask, args.party_id, args.n_samples, args.n_features, masking_time))    # Print the result

    #path = "results/results_client_classification_{}_{}_{}.xlsx".format(args.party_id, args.n_samples, args.mask)
    #Kernel_lib.dump_clientside_results(masking_time, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--run", type=int, default=3, help="the number of clients to listen for")
    parser.add_argument("--mask", type=str, default="OKRA", help="FLAKE or OKRA")
    parser.add_argument("--n_features", type=int, default=1000, help="no of features")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--base_port", type=int, default=8000, help="port to connect to")
    parser.add_argument("--party_id", type=int, default=1, help="ID of input party this python file simulates")
    parser.add_argument("--n_samples", type=int, default=1000, help="dataset size for one party")
    parser.add_argument("--max_parties", type=int, default=3, help="number of clients in total")
    args = parser.parse_args()
    send_data(args)
