from sklearn.datasets import make_classification
import Kernel_lib
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--run", type=int, default=3, help="the number of clients to listen for")
    parser.add_argument("--mask", type=str, default="FLAKE", help="FLAKE or OKRA")
    parser.add_argument("--n_features", type=int, default=1000, help="no of features")
    parser.add_argument("--k", type=int, default=10)
    #parser.add_argument("--no_parties", type=int, default=3, help="number of parties")
    parser.add_argument("--n_samples", type=int, default=400, help="dataset size for one party")
    args = parser.parse_args()


    x,y = make_classification(n_samples=args.n_samples, n_features=args.n_features, random_state=42, shuffle=False)

    dataset_partitions, y = Kernel_lib.partition_dataset(x, y, 3)

    n_features = dataset_partitions[1].shape[1] 

    ts1 = time.time()

    if args.mask == "FLAKE":
                
        N = Kernel_lib.random_matrix(args.n_features + args.k, args.n_features) 

        A_prime = Kernel_lib.generate_data_prime(dataset_partitions[0], N)
        ts2 = time.time()

        #B_prime = Kernel_lib.generate_data_prime(dataset_partitions[1], N)
        #C_prime = Kernel_lib.generate_data_prime(dataset_partitions[2], N)
        
    else:
        gamma = Kernel_lib.get_gamma_gpu(n_features, args.k, seed=42)

        A_prime = dataset_partitions[0]@gamma.T
        ts2 = time.time()

        #B_prime = dataset_partitions[1]@gamma.T
        #C_prime = dataset_partitions[2]@gamma.T
    

    full = Kernel_lib.compute_gram_matrix(A_prime, A_prime, A_prime)

    ts3 = time.time()

    masking_time = ts2-ts1
    gram_time = ts3-ts2

    print("Masking computation times for {} {}: {}".format(args.mask, args.n_features, masking_time))    # Print the result
    print("Gram computation times for {} {}: {}".format(args.mask, args.n_features, gram_time))    # Print the result